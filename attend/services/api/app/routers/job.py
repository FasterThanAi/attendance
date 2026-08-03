"""Phase 2 deliverable 5: POST /sessions/{id}/process, GET /jobs/{id}."""

from __future__ import annotations

import json

import redis
from fastapi import APIRouter, Depends, HTTPException
from rq import Queue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ClassSession, ClassSessionStatus, ProcessingJob, ProcessingJobState, VideoUpload
from app.db.session import get_db
from app.schemas.errors import ErrorResponse
from app.schemas.job import ProcessingJobResponse, ProcessRequest
from app.services.job import DEFAULT_PIPELINE_PARAMS

router = APIRouter(tags=["processing"])


@router.post("/sessions/{session_id}/process", response_model=ProcessingJobResponse, status_code=202)
async def process_session_endpoint(
    session_id: int, body: ProcessRequest | None = None, db: AsyncSession = Depends(get_db)
) -> ProcessingJobResponse:
    session_row = (await db.execute(select(ClassSession).where(ClassSession.id == session_id))).scalar_one_or_none()
    if session_row is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(
            code="session_not_found", message=f"No class session with id={session_id}."
        ).model_dump())

    video_upload_id = body.video_upload_id if body else None
    if video_upload_id is not None:
        video_upload = (
            await db.execute(select(VideoUpload).where(VideoUpload.id == video_upload_id))
        ).scalar_one_or_none()
    else:
        video_upload = (
            await db.execute(
                select(VideoUpload)
                .where(VideoUpload.class_session_id == session_id)
                .order_by(VideoUpload.uploaded_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    if video_upload is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(
            code="video_upload_not_found",
            message="No video has been uploaded for this session yet.",
        ).model_dump())

    job = ProcessingJob(
        class_session_id=session_id,
        video_upload_id=video_upload.id,
        state=ProcessingJobState.QUEUED,
        params_json=json.dumps(DEFAULT_PIPELINE_PARAMS),
    )
    db.add(job)

    session_row.status = ClassSessionStatus.PROCESSING
    await db.commit()
    await db.refresh(job)

    redis_conn = redis.from_url(settings.redis_url)
    queue = Queue("attend", connection=redis_conn)
    queue.enqueue_call(func="pipeline.run.process_session", args=(job.id,), timeout=3600)

    return ProcessingJobResponse.model_validate(job)


@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
async def get_job(job_id: int, db: AsyncSession = Depends(get_db)) -> ProcessingJobResponse:
    job = (await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(
            code="job_not_found", message=f"No processing job with id={job_id}."
        ).model_dump())
    return ProcessingJobResponse.model_validate(job)
