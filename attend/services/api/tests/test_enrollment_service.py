"""Tests for the api-side enrollment status/delete logic (not the POST
upload+enqueue endpoint, which needs a running Redis and isn't worth mocking
out for a unit test -- see attend/README.md's Phase 1 verification notes for
how to exercise POST for real, on your Mac, against the docker-compose stack).
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Department, GalleryEmbedding, GalleryPhoto, Institution, Student
from app.services.enrollment import delete_enrollment, get_enrollment_status

UTC_NOW = datetime.now(timezone.utc)


@pytest.fixture
async def student(db_session: AsyncSession) -> Student:
    institution = Institution(name="Test Institute")
    db_session.add(institution)
    await db_session.flush()

    department = Department(institution_id=institution.id, name="CSE", code="CSE")
    db_session.add(department)
    await db_session.flush()

    student = Student(
        department_id=department.id,
        roll_number="CSE003",
        full_name="Chitra Devi",
        admission_year=2023,
        is_active=True,
    )
    db_session.add(student)
    await db_session.flush()
    return student


async def _add_gallery_row(db_session: AsyncSession, student_id: int, bucket: str, quality: float):
    photo = GalleryPhoto(
        student_id=student_id,
        storage_uri=f"/data/jobs/enrollment/{student_id}/{bucket}.jpg",
        captured_at=UTC_NOW,
        quality_score=quality,
        pose_bucket=bucket,
    )
    db_session.add(photo)
    await db_session.flush()

    embedding = GalleryEmbedding(
        student_id=student_id,
        vector=b"\x00" * 2048,
        source_photo_id=photo.id,
        model_version="arcface-r100-buffalo_l",
        retention_expires_at=UTC_NOW + timedelta(days=180),
    )
    db_session.add(embedding)
    await db_session.flush()


async def test_status_insufficient_when_no_gallery(db_session: AsyncSession, student: Student):
    status = await get_enrollment_status(db_session, student.id)
    assert status["total_embeddings"] == 0
    assert status["is_sufficient"] is False
    assert status["pose_coverage"] == {"left": 0, "frontal": 0, "right": 0}


async def test_status_sufficient_when_all_poses_covered(db_session: AsyncSession, student: Student):
    for bucket in ("left", "frontal", "right"):
        for i in range(2):  # 2 per bucket = 6 total, >= GALLERY_MIN_EMBEDDINGS(5)
            await _add_gallery_row(db_session, student.id, bucket, quality=0.8)

    status = await get_enrollment_status(db_session, student.id)
    assert status["total_embeddings"] == 6
    assert status["pose_coverage"] == {"left": 2, "frontal": 2, "right": 2}
    assert status["is_sufficient"] is True


async def test_status_insufficient_when_a_pose_missing(db_session: AsyncSession, student: Student):
    for bucket in ("frontal", "right"):
        for i in range(3):
            await _add_gallery_row(db_session, student.id, bucket, quality=0.8)

    status = await get_enrollment_status(db_session, student.id)
    assert status["total_embeddings"] == 6  # meets the raw count...
    assert status["is_sufficient"] is False  # ...but "left" pose is missing


async def test_delete_enrollment_removes_gallery_rows(db_session: AsyncSession, student: Student):
    await _add_gallery_row(db_session, student.id, "frontal", quality=0.8)
    await _add_gallery_row(db_session, student.id, "left", quality=0.7)

    result = await delete_enrollment(db_session, student.id)

    assert result["photos_deleted"] == 2
    assert result["embeddings_deleted"] == 2

    status = await get_enrollment_status(db_session, student.id)
    assert status["total_embeddings"] == 0
