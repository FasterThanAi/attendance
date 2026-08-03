"""API-side enrollment business logic: status queries, deletion, and
enqueueing the worker job. The actual video processing lives entirely in
services/worker/enrollment.py -- this module never touches ML code.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog, GalleryEmbedding, GalleryPhoto, Student

# Duplicated from services/worker/pipeline/params.py's PipelineParams.gallery_min_embeddings
# default. Not imported directly: that module lives in a separate Docker
# image/dependency set from this one (see services/worker/db.py's docstring
# for the same trade-off applied to table definitions). If you change the
# real default in params.py, update this constant too.
GALLERY_MIN_EMBEDDINGS = 5
POSE_BUCKETS = ("left", "frontal", "right")


async def get_enrollment_status(db: AsyncSession, student_id: int) -> dict:
    pose_counts_result = await db.execute(
        select(GalleryPhoto.pose_bucket, func.count())
        .where(GalleryPhoto.student_id == student_id)
        .group_by(GalleryPhoto.pose_bucket)
    )
    pose_coverage = {bucket: 0 for bucket in POSE_BUCKETS}
    for bucket, count in pose_counts_result.all():
        if bucket in pose_coverage:
            pose_coverage[bucket] = count

    total_embeddings_result = await db.execute(
        select(func.count()).select_from(GalleryEmbedding).where(GalleryEmbedding.student_id == student_id)
    )
    total_embeddings = total_embeddings_result.scalar_one()

    student_result = await db.execute(select(Student.gallery_updated_at).where(Student.id == student_id))
    gallery_updated_at = student_result.scalar_one_or_none()

    is_sufficient = total_embeddings >= GALLERY_MIN_EMBEDDINGS and all(c > 0 for c in pose_coverage.values())

    return {
        "student_id": student_id,
        "total_embeddings": total_embeddings,
        "pose_coverage": pose_coverage,
        "gallery_updated_at": gallery_updated_at,
        "is_sufficient": is_sufficient,
    }


async def delete_enrollment(db: AsyncSession, student_id: int, actor_teacher_id: int | None = None) -> dict:
    """Deletes all biometric data for a student (Phase 1 deliverable 5's
    DELETE endpoint). Does NOT touch attendance_record -- those are academic
    records, not biometric data (see docs/CONSENT.md and, later,
    docs/DATA_PROTECTION.md in Phase 9).
    """
    embeddings_deleted = (
        await db.execute(delete(GalleryEmbedding).where(GalleryEmbedding.student_id == student_id))
    ).rowcount
    photos_deleted = (
        await db.execute(delete(GalleryPhoto).where(GalleryPhoto.student_id == student_id))
    ).rowcount

    # Existence of the student is checked by the router before calling this
    # (a 404 there is more useful than a silent no-op delete here).
    await db.execute(
        update(Student)
        .where(Student.id == student_id)
        .values(gallery_mean_vector=None, gallery_updated_at=None)
    )

    db.add(
        AuditLog(
            actor_type="teacher" if actor_teacher_id else "system",
            actor_id=str(actor_teacher_id) if actor_teacher_id else "api",
            action="enrollment.delete",
            subject_type="student",
            subject_id=str(student_id),
            payload_json=f'{{"photos_deleted": {photos_deleted}, "embeddings_deleted": {embeddings_deleted}}}',
            created_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()

    return {"student_id": student_id, "photos_deleted": photos_deleted, "embeddings_deleted": embeddings_deleted}
