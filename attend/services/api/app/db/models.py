"""SQLAlchemy 2.0 async models for the full Attend schema (Phase 0 design).

ASSUMPTIONS I MADE (per the global brief's "when you are unsure" rule):
  - Primary keys are auto-incrementing BigInteger. The roadmap only says "id"
    for every table without specifying a type. Integers are simpler to work
    with in early phases (readable in logs, sortable). If this needs to
    become UUIDs later (e.g. to avoid leaking row counts, or for multi-tenant
    sharding), that is a mechanical migration, not a design change -- flag it
    if it matters to you.
  - Enum columns are stored as VARCHAR + CHECK constraint (SQLAlchemy
    Enum(native_enum=False)) rather than native Postgres ENUM types. Native
    enums make "add one more status value" an ALTER TYPE migration headache;
    VARCHAR+CHECK is a plain column-constraint change. Values are identical
    either way.

Non-negotiable rules from the global brief that this file exists to satisfy:
  Rule 4: attendance_record is append-only. Enforced at the service layer
          (see services/api/app/services/attendance.py, built in Phase 8) --
          NOT by revoking UPDATE/DELETE grants here, because the dev/test
          environment needs normal DB access. The comment on
          AttendanceRecord below is the canonical reminder.
  Rule 5: biometric data (gallery_embedding, detected_cluster) lives in its
          own tables, never inline in logs/errors, and carries created_at +
          retention_expires_at.
  Rule 6: all timestamps are timezone-aware UTC, set in Python
          (datetime.now(timezone.utc)) rather than relying on DB-side
          defaults, so behavior is identical across Postgres and the SQLite
          test database.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------------
# Enums (stored as VARCHAR + CHECK constraint -- see module docstring)
# --------------------------------------------------------------------------


class ClassSessionStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    RECORDING = "recording"
    PROCESSING = "processing"
    AWAITING_REVIEW = "awaiting_review"
    COMMITTED = "committed"
    FAILED = "failed"


class ProcessingJobState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ClusterMatchDecision(str, enum.Enum):
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    UNMATCHED = "unmatched"


class AttendanceStatus(str, enum.Enum):
    PRESENT = "present"
    ABSENT = "absent"


class AttendanceSource(str, enum.Enum):
    AUTO = "auto"
    TEACHER_CONFIRMED = "teacher_confirmed"
    TEACHER_OVERRIDE = "teacher_override"


def _enum_column(enum_cls: type[enum.Enum], name: str):
    return mapped_column(
        SAEnum(
            enum_cls,
            name=name,
            native_enum=False,
            length=32,
            values_callable=lambda e: [member.value for member in e],
        ),
        nullable=False,
    )


# --------------------------------------------------------------------------
# Institution / people
# --------------------------------------------------------------------------


class Institution(Base):
    __tablename__ = "institution"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    departments: Mapped[list["Department"]] = relationship(back_populates="institution")


class Department(Base):
    __tablename__ = "department"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    institution_id: Mapped[int] = mapped_column(
        ForeignKey("institution.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str] = mapped_column(String(32), nullable=False)

    institution: Mapped["Institution"] = relationship(back_populates="departments")

    __table_args__ = (UniqueConstraint("institution_id", "code", name="uq_department_institution_code"),)


class Course(Base):
    __tablename__ = "course"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    department_id: Mapped[int] = mapped_column(
        ForeignKey("department.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    semester: Mapped[str] = mapped_column(String(32), nullable=False)


class Student(Base):
    __tablename__ = "student"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    department_id: Mapped[int] = mapped_column(
        ForeignKey("department.id", ondelete="CASCADE"), nullable=False, index=True
    )
    roll_number: Mapped[str] = mapped_column(String(64), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    admission_year: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("department_id", "roll_number", name="uq_student_department_roll_number"),
    )


class Teacher(Base):
    __tablename__ = "teacher"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    department_id: Mapped[int] = mapped_column(
        ForeignKey("department.id", ondelete="CASCADE"), nullable=False, index=True
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)


class Enrollment(Base):
    """Which students are in which course. (Distinct from `gallery` enrollment,
    Phase 1's terminology for building a student's biometric reference set --
    an unfortunate name collision in the source roadmap, kept as-is here.)
    """

    __tablename__ = "enrollment"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("student.id", ondelete="CASCADE"), nullable=False, index=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("course.id", ondelete="CASCADE"), nullable=False, index=True)

    __table_args__ = (UniqueConstraint("student_id", "course_id", name="uq_enrollment_student_course"),)


# --------------------------------------------------------------------------
# Consent (Phase 0 ethics requirement -- must exist before any face capture)
# --------------------------------------------------------------------------


class Consent(Base):
    __tablename__ = "consent"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("student.id", ondelete="CASCADE"), nullable=False, index=True)
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    consent_version: Mapped[str] = mapped_column(String(32), nullable=False)
    scope: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_uri: Mapped[str] = mapped_column(String(1024), nullable=False)


# --------------------------------------------------------------------------
# Gallery (Phase 1 reference set) -- biometric data, retention-tracked
# --------------------------------------------------------------------------


class GalleryPhoto(Base):
    __tablename__ = "gallery_photo"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("student.id", ondelete="CASCADE"), nullable=False, index=True)
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)


class GalleryEmbedding(Base):
    """Biometric data (rule 5): dedicated table, own retention clock.

    `vector` is a raw bytea blob of 512 float32 values (2048 bytes), matching
    ArcFace r100's output dimensionality. Stored as bytes rather than a
    pgvector column for Phase 0-1 simplicity, per the tech-stack table's note
    that pgvector is "available later if the gallery grows" -- not required
    on day one.
    """

    __tablename__ = "gallery_embedding"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("student.id", ondelete="CASCADE"), nullable=False, index=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    source_photo_id: Mapped[int] = mapped_column(
        ForeignKey("gallery_photo.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# --------------------------------------------------------------------------
# Sessions, uploads, jobs
# --------------------------------------------------------------------------


class ClassSession(Base):
    __tablename__ = "class_session"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("course.id", ondelete="CASCADE"), nullable=False, index=True)
    teacher_id: Mapped[int] = mapped_column(ForeignKey("teacher.id", ondelete="RESTRICT"), nullable=False, index=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    room: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[ClassSessionStatus] = _enum_column(ClassSessionStatus, "class_session_status")


class VideoUpload(Base):
    __tablename__ = "video_upload"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    class_session_id: Mapped[int] = mapped_column(
        ForeignKey("class_session.id", ondelete="CASCADE"), nullable=False, index=True
    )
    storage_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ProcessingJob(Base):
    __tablename__ = "processing_job"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    class_session_id: Mapped[int] = mapped_column(
        ForeignKey("class_session.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_upload_id: Mapped[int] = mapped_column(
        ForeignKey("video_upload.id", ondelete="CASCADE"), nullable=False, index=True
    )
    state: Mapped[ProcessingJobState] = _enum_column(ProcessingJobState, "processing_job_state")
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# --------------------------------------------------------------------------
# Pipeline output: clusters and matches
# --------------------------------------------------------------------------


class DetectedCluster(Base):
    """Biometric data (rule 5): representative_vector is derived directly
    from face embeddings, so this table carries the same handling rules as
    gallery_embedding even though the roadmap schema doesn't list a
    retention_expires_at column on it explicitly -- added here for
    consistency; delete-on-job-retention-expiry cascades to this table too.
    """

    __tablename__ = "detected_cluster"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    processing_job_id: Mapped[int] = mapped_column(
        ForeignKey("processing_job.id", ondelete="CASCADE"), nullable=False, index=True
    )
    representative_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    crop_count: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_quality: Mapped[float] = mapped_column(Float, nullable=False)
    best_crop_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ClusterMatch(Base):
    __tablename__ = "cluster_match"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("detected_cluster.id", ondelete="CASCADE"), nullable=False, index=True
    )
    student_id: Mapped[int | None] = mapped_column(
        ForeignKey("student.id", ondelete="SET NULL"), nullable=True, index=True
    )
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    runner_up_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[ClusterMatchDecision] = _enum_column(ClusterMatchDecision, "cluster_match_decision")


# --------------------------------------------------------------------------
# Attendance -- append-only
# --------------------------------------------------------------------------


class AttendanceRecord(Base):
    """APPEND-ONLY (non-negotiable rule 4). Never UPDATE or DELETE a row here.

    A correction is a new row with `supersedes_id` pointing at the row being
    corrected. `current_attendance` (a SQL view built in Phase 8) resolves to
    whichever row in a chain has no successor. The service layer must never
    expose an update/delete path for this model -- there is deliberately no
    such method anywhere in services/api/app/services/.
    """

    __tablename__ = "attendance_record"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    class_session_id: Mapped[int] = mapped_column(
        ForeignKey("class_session.id", ondelete="CASCADE"), nullable=False, index=True
    )
    student_id: Mapped[int] = mapped_column(ForeignKey("student.id", ondelete="CASCADE"), nullable=False, index=True)
    status: Mapped[AttendanceStatus] = _enum_column(AttendanceStatus, "attendance_status")
    source: Mapped[AttendanceSource] = _enum_column(AttendanceSource, "attendance_source")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    decided_by_teacher_id: Mapped[int | None] = mapped_column(
        ForeignKey("teacher.id", ondelete="SET NULL"), nullable=True, index=True
    )
    supersedes_id: Mapped[int | None] = mapped_column(
        ForeignKey("attendance_record.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    __table_args__ = (
        Index("ix_attendance_record_session_student", "class_session_id", "student_id"),
    )


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(64), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
