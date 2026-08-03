"""Worker's own, narrow, sync database access.

Deliberately NOT importing services/api/app/db/models.py: the api and worker
are separate Docker images with separate dependency sets (the worker's image
carries ~1.5GB of ML libraries the api image should never need, and vice
versa the api's async SQLAlchemy stack has no reason to live in the worker
image). Sharing one ORM module across a real package boundary would mean
either publishing an internal pip package or symlinking source across build
contexts -- both are more machinery than this project's size justifies.

Instead, this module REFLECTS the tables the worker actually touches
directly from the live database via SQLAlchemy's autoload_with. This means
the worker can never have a stale, hand-copied version of a column
definition -- if services/api/app/db/models.py adds a column via a
migration, this file sees it automatically, at the cost of needing a real
DB connection at import time (acceptable: this module is only imported
inside the worker process, which always has one).

Phase 6 addition: `enrollment`, `class_session`, `course`, `detected_cluster`,
`cluster_match` -- the match stage (pipeline/match.py's run_match_stage)
needs to look up which students are enrolled in a session's course, read
each one's cached gallery vector (already reflected via `student`), and
write its own DetectedCluster/ClusterMatch rows.

KNOWN TRADE-OFF, flagged rather than hidden: the *business rule* "a consent
row exists and isn't revoked" is expressed twice -- once in
services/api/app/services/consent.py (async, for the api's own consent
endpoints, not yet built) and once in services/worker/consent.py (sync, for
this file's use inside the enrollment job). If that rule ever changes,
both places need updating. A future refactor could extract it into a
tiny shared package; not done now because it would be the ONLY thing in
that shared package, which is a lot of packaging ceremony for one function.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.engine import Engine

from config import settings


def _sync_database_url() -> str:
    """The api's DATABASE_URL uses the asyncpg driver (postgresql+asyncpg://);
    this process uses plain psycopg2, so swap the driver segment.
    """
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


@lru_cache
def get_engine() -> Engine:
    return create_engine(_sync_database_url(), pool_pre_ping=True)


@lru_cache
def _metadata() -> MetaData:
    metadata = MetaData()
    metadata.reflect(
        bind=get_engine(),
        only=[
            "consent", "student", "gallery_photo", "gallery_embedding", "processing_job", "video_upload",
            "enrollment", "class_session", "course", "detected_cluster", "cluster_match",
        ],
    )
    return metadata


def table(name: str) -> Table:
    return _metadata().tables[name]
