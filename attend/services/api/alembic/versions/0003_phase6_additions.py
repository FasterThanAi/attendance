"""phase 6 additions: preflight status persistence, draft summary cache

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-04

Two additions Phase 6 needs that earlier phases didn't:
  - video_upload.preflight_status_json: Phase 2 computed the pre-flight
    result but never stored it anywhere durable -- only returned it in the
    upload-complete HTTP response. Phase 6's session_health needs to know
    "did this session's pre-flight have warnings" well after that response
    was sent.
  - class_session.draft_summary_json: the computed session summary object
    (Phase 6 deliverable 4), stored once at match-stage completion rather
    than re-derived on every GET /sessions/{id}/draft request.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("video_upload", sa.Column("preflight_status_json", sa.Text(), nullable=True))
    op.add_column("class_session", sa.Column("draft_summary_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("class_session", "draft_summary_json")
    op.drop_column("video_upload", "preflight_status_json")
