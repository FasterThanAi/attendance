"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-03

Written by hand against services/api/app/db/models.py rather than via
`alembic revision --autogenerate`, because this environment had no network
access to a running Postgres instance to autogenerate against. Column types,
names, nullability and constraints below are a direct, careful transcription
of models.py -- cross-check the two if you add a table before running this.

Enum columns intentionally use literal value lists here (not an import of
the Python enum classes from app.db.models) so that migrations never depend
on application code -- migration history must remain valid even if a future
refactor renames or moves an enum class.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "institution",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "department",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "institution_id",
            sa.BigInteger(),
            sa.ForeignKey("institution.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(32), nullable=False),
        sa.UniqueConstraint("institution_id", "code", name="uq_department_institution_code"),
    )
    op.create_index("ix_department_institution_id", "department", ["institution_id"])

    op.create_table(
        "course",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "department_id",
            sa.BigInteger(),
            sa.ForeignKey("department.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code", sa.String(32), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("semester", sa.String(32), nullable=False),
    )
    op.create_index("ix_course_department_id", "course", ["department_id"])

    op.create_table(
        "student",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "department_id",
            sa.BigInteger(),
            sa.ForeignKey("department.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("roll_number", sa.String(64), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("admission_year", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("department_id", "roll_number", name="uq_student_department_roll_number"),
    )
    op.create_index("ix_student_department_id", "student", ["department_id"])

    op.create_table(
        "teacher",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "department_id",
            sa.BigInteger(),
            sa.ForeignKey("department.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
    )
    op.create_index("ix_teacher_department_id", "teacher", ["department_id"])

    op.create_table(
        "enrollment",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("student_id", sa.BigInteger(), sa.ForeignKey("student.id", ondelete="CASCADE"), nullable=False),
        sa.Column("course_id", sa.BigInteger(), sa.ForeignKey("course.id", ondelete="CASCADE"), nullable=False),
        sa.UniqueConstraint("student_id", "course_id", name="uq_enrollment_student_course"),
    )
    op.create_index("ix_enrollment_student_id", "enrollment", ["student_id"])
    op.create_index("ix_enrollment_course_id", "enrollment", ["course_id"])

    op.create_table(
        "consent",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("student_id", sa.BigInteger(), sa.ForeignKey("student.id", ondelete="CASCADE"), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consent_version", sa.String(32), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("evidence_uri", sa.String(1024), nullable=False),
    )
    op.create_index("ix_consent_student_id", "consent", ["student_id"])

    op.create_table(
        "gallery_photo",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("student_id", sa.BigInteger(), sa.ForeignKey("student.id", ondelete="CASCADE"), nullable=False),
        sa.Column("storage_uri", sa.String(1024), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
    )
    op.create_index("ix_gallery_photo_student_id", "gallery_photo", ["student_id"])

    op.create_table(
        "gallery_embedding",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("student_id", sa.BigInteger(), sa.ForeignKey("student.id", ondelete="CASCADE"), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column(
            "source_photo_id",
            sa.BigInteger(),
            sa.ForeignKey("gallery_photo.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_gallery_embedding_student_id", "gallery_embedding", ["student_id"])
    op.create_index("ix_gallery_embedding_source_photo_id", "gallery_embedding", ["source_photo_id"])

    op.create_table(
        "class_session",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("course_id", sa.BigInteger(), sa.ForeignKey("course.id", ondelete="CASCADE"), nullable=False),
        sa.Column("teacher_id", sa.BigInteger(), sa.ForeignKey("teacher.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("room", sa.String(64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "scheduled", "recording", "processing", "awaiting_review", "committed", "failed",
                name="class_session_status", native_enum=False, length=32,
            ),
            nullable=False,
        ),
    )
    op.create_index("ix_class_session_course_id", "class_session", ["course_id"])
    op.create_index("ix_class_session_teacher_id", "class_session", ["teacher_id"])

    op.create_table(
        "video_upload",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "class_session_id",
            sa.BigInteger(),
            sa.ForeignKey("class_session.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("storage_uri", sa.String(1024), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_video_upload_class_session_id", "video_upload", ["class_session_id"])

    op.create_table(
        "processing_job",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "class_session_id",
            sa.BigInteger(),
            sa.ForeignKey("class_session.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "video_upload_id",
            sa.BigInteger(),
            sa.ForeignKey("video_upload.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "state",
            sa.Enum(
                "queued", "running", "succeeded", "failed",
                name="processing_job_state", native_enum=False, length=32,
            ),
            nullable=False,
        ),
        sa.Column("stage", sa.String(64), nullable=True),
        sa.Column("params_json", sa.Text(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_processing_job_class_session_id", "processing_job", ["class_session_id"])
    op.create_index("ix_processing_job_video_upload_id", "processing_job", ["video_upload_id"])

    op.create_table(
        "detected_cluster",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "processing_job_id",
            sa.BigInteger(),
            sa.ForeignKey("processing_job.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("representative_vector", sa.LargeBinary(), nullable=False),
        sa.Column("crop_count", sa.Integer(), nullable=False),
        sa.Column("mean_quality", sa.Float(), nullable=False),
        sa.Column("best_crop_uri", sa.String(1024), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_detected_cluster_processing_job_id", "detected_cluster", ["processing_job_id"])

    op.create_table(
        "cluster_match",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "cluster_id",
            sa.BigInteger(),
            sa.ForeignKey("detected_cluster.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("student_id", sa.BigInteger(), sa.ForeignKey("student.id", ondelete="SET NULL"), nullable=True),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column("runner_up_similarity", sa.Float(), nullable=True),
        sa.Column(
            "decision",
            sa.Enum(
                "confident", "uncertain", "unmatched",
                name="cluster_match_decision", native_enum=False, length=32,
            ),
            nullable=False,
        ),
    )
    op.create_index("ix_cluster_match_cluster_id", "cluster_match", ["cluster_id"])
    op.create_index("ix_cluster_match_student_id", "cluster_match", ["student_id"])

    # Append-only: see the docstring on AttendanceRecord in models.py.
    op.create_table(
        "attendance_record",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "class_session_id",
            sa.BigInteger(),
            sa.ForeignKey("class_session.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("student_id", sa.BigInteger(), sa.ForeignKey("student.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "status",
            sa.Enum("present", "absent", name="attendance_status", native_enum=False, length=32),
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.Enum(
                "auto", "teacher_confirmed", "teacher_override",
                name="attendance_source", native_enum=False, length=32,
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "decided_by_teacher_id",
            sa.BigInteger(),
            sa.ForeignKey("teacher.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "supersedes_id",
            sa.BigInteger(),
            sa.ForeignKey("attendance_record.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_attendance_record_class_session_id", "attendance_record", ["class_session_id"])
    op.create_index("ix_attendance_record_student_id", "attendance_record", ["student_id"])
    op.create_index("ix_attendance_record_decided_by_teacher_id", "attendance_record", ["decided_by_teacher_id"])
    op.create_index("ix_attendance_record_supersedes_id", "attendance_record", ["supersedes_id"])
    op.create_index(
        "ix_attendance_record_session_student", "attendance_record", ["class_session_id", "student_id"]
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("actor_type", sa.String(32), nullable=False),
        sa.Column("actor_id", sa.String(64), nullable=False),
        sa.Column("action", sa.String(128), nullable=False),
        sa.Column("subject_type", sa.String(64), nullable=False),
        sa.Column("subject_id", sa.String(64), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("attendance_record")
    op.drop_table("cluster_match")
    op.drop_table("detected_cluster")
    op.drop_table("processing_job")
    op.drop_table("video_upload")
    op.drop_table("class_session")
    op.drop_table("gallery_embedding")
    op.drop_table("gallery_photo")
    op.drop_table("consent")
    op.drop_table("enrollment")
    op.drop_table("teacher")
    op.drop_table("student")
    op.drop_table("course")
    op.drop_table("department")
    op.drop_table("institution")
