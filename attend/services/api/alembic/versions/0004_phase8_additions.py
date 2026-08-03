"""phase 8 additions: commit idempotency key, current_attendance view

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-04

Two additions Phase 8 needs:
  - class_session.commit_request_id: idempotency key for POST
    /sessions/{id}/commit (Phase 8 deliverable 3) -- a double tap on a bad
    network must not double-commit.
  - the current_attendance view (Phase 8 deliverable 4): resolves an
    attendance_record correction chain of any depth down to the one row
    per (class_session_id, student_id) with no successor. The exact same
    SQL is registered against Base.metadata's after_create/before_drop
    events in app/db/views.py, so services/api/tests/conftest.py's
    Base.metadata.create_all()-based SQLite fixture gets the identical
    view -- imported from there rather than redefined here, so the two can
    never drift apart.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from app.db.views import _CREATE_VIEW_SQL, _DROP_VIEW_SQL

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("class_session", sa.Column("commit_request_id", sa.String(length=64), nullable=True))
    op.create_unique_constraint("uq_class_session_commit_request_id", "class_session", ["commit_request_id"])
    op.execute(_CREATE_VIEW_SQL)


def downgrade() -> None:
    op.execute(_DROP_VIEW_SQL)
    op.drop_constraint("uq_class_session_commit_request_id", "class_session", type_="unique")
    op.drop_column("class_session", "commit_request_id")
