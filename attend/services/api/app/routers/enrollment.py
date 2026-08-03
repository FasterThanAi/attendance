"""Phase 1 API endpoints: POST/GET/DELETE /students/{id}/enrollment.

POST accepts the video and enqueues the actual processing (services/worker/
enrollment.py::enroll_student) via RQ -- this router never runs any ML code,
matching the api image's deliberately light dependency set.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import redis
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from rq import Queue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Student
from app.db.session import get_db
from app.schemas.enrollment import (
    EnrollmentDeletedResponse,
    EnrollmentQueuedResponse,
    EnrollmentStatusResponse,
    PoseCoverage,
)
from app.schemas.errors import ErrorResponse
from app.services import enrollment as enrollment_service
from app.services.consent import assert_consent_valid

router = APIRouter(prefix="/students/{student_id}/enrollment", tags=["enrollment"])

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


async def _get_active_student(db: AsyncSession, student_id: int) -> Student:
    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one_or_none()
    if student is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(code="student_not_found", message=(
            f"No student with id={student_id}."
        )).model_dump())
    return student


@router.post("", response_model=EnrollmentQueuedResponse, status_code=202)
async def submit_enrollment_video(
    student_id: int,
    video: UploadFile,
    db: AsyncSession = Depends(get_db),
) -> EnrollmentQueuedResponse:
    await _get_active_student(db, student_id)

    # Fail fast with a clear 403 before accepting a large upload or queueing
    # any work -- the worker job re-checks this too (defense in depth,
    # not a substitute: the worker must never trust the API layer's checks
    # alone, since it can be invoked by other future callers).
    await assert_consent_valid(db, student_id)

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=422, detail=ErrorResponse(
            code="unsupported_video_format",
            message=f"Unsupported file type '{suffix}'. Use one of: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}.",
        ).model_dump())

    upload_dir = Path(settings.job_data_dir) / "enrollment" / str(student_id) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"

    with open(video_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            f.write(chunk)

    work_dir = Path(settings.job_data_dir) / "enrollment" / str(student_id) / "work" / uuid.uuid4().hex

    redis_conn = redis.from_url(settings.redis_url)
    queue = Queue("attend", connection=redis_conn)
    job = queue.enqueue_call(
        func="enrollment.enroll_student",
        args=(student_id, str(video_path), str(work_dir)),
        # Enrollment videos are short (~5s), but detector/embedding model
        # load time on a cold worker can be tens of seconds -- generous
        # timeout so a slow model load doesn't get killed mid-job.
        timeout=600,
    )

    return EnrollmentQueuedResponse(student_id=student_id, job_id=job.id)


@router.get("", response_model=EnrollmentStatusResponse)
async def get_enrollment_status(student_id: int, db: AsyncSession = Depends(get_db)) -> EnrollmentStatusResponse:
    await _get_active_student(db, student_id)
    status_dict = await enrollment_service.get_enrollment_status(db, student_id)
    return EnrollmentStatusResponse(
        student_id=status_dict["student_id"],
        total_embeddings=status_dict["total_embeddings"],
        pose_coverage=PoseCoverage(**status_dict["pose_coverage"]),
        gallery_updated_at=status_dict["gallery_updated_at"],
        is_sufficient=status_dict["is_sufficient"],
    )


@router.delete("", response_model=EnrollmentDeletedResponse)
async def delete_enrollment(student_id: int, db: AsyncSession = Depends(get_db)) -> EnrollmentDeletedResponse:
    await _get_active_student(db, student_id)
    result = await enrollment_service.delete_enrollment(db, student_id)
    return EnrollmentDeletedResponse(**result)
