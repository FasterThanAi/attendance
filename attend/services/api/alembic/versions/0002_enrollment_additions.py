"""enrollment additions: pose_bucket, gallery mean vector cache

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-03

Phase 1 needs two things the original Phase 0 schema (transcribed directly
from the roadmap) didn't include:
  - gallery_photo.pose_bucket: which of left/frontal/right an enrollment
    photo was bucketed into, so the enrollment status endpoint can report
    pose coverage without re-running pose estimation.
  - student.gallery_mean_vector / gallery_updated_at: the cached, quality-
    independent mean embedding Phase 1's deliverable 4 asks for ("a cached
    per-student mean vector... recomputed on enrollment change. This is what
    matching will use").
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("gallery_photo", sa.Column("pose_bucket", sa.String(16), nullable=True))
    op.add_column("student", sa.Column("gallery_mean_vector", sa.LargeBinary(), nullable=True))
    op.add_column("student", sa.Column("gallery_updated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("student", "gallery_updated_at")
    op.drop_column("student", "gallery_mean_vector")
    op.drop_column("gallery_photo", "pose_bucket")
