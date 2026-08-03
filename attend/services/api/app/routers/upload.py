"""Phase 2 deliverable 1-2: resumable chunked upload, plus deliverable 2b's
pre-flight check wired into /complete.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ClassSession, Enrollment, VideoUpload
from app.db.session import get_db
from app.schemas.errors import ErrorResponse
from app.schemas.upload import (
    PreflightCheckResult,
    PreflightResult,
    UploadChunkResponse,
    UploadCompleteResponse,
    UploadCreateRequest,
    UploadCreateResponse,
    UploadStatusResponse,
    VideoUploadResponse,
)
from app.services import upload as upload_service
from app.services.preflight_client import PreflightTimeoutError, run_preflight_and_wait

router = APIRouter(prefix="/uploads", tags=["uploads"])


def _validation_error_to_http(exc: upload_service.UploadValidationError) -> HTTPException:
    # chunks_missing / upload_not_found are "try again" client errors (422);
    # everything else about the assembled video is also a 422 with a
    # plain-language message per non-negotiable rule #7.
    return HTTPException(status_code=422, detail=ErrorResponse(code=exc.code, message=str(exc)).model_dump())


@router.post("", response_model=UploadCreateResponse, status_code=201)
async def create_upload(body: UploadCreateRequest, db: AsyncSession = Depends(get_db)) -> UploadCreateResponse:
    session_exists = (
        await db.execute(select(ClassSession).where(ClassSession.id == body.class_session_id))
    ).scalar_one_or_none()
    if session_exists is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(
            code="session_not_found", message=f"No class session with id={body.class_session_id}."
        ).model_dump())

    result = upload_service.create_upload_session(body.class_session_id, body.filename, body.total_size_bytes)
    return UploadCreateResponse(**result)


@router.put("/{upload_id}/chunks/{chunk_index}", response_model=UploadChunkResponse)
async def put_chunk(upload_id: str, chunk_index: int, request: Request) -> UploadChunkResponse:
    data = await request.body()
    try:
        upload_service.write_chunk(upload_id, chunk_index, data)
    except upload_service.UploadValidationError as exc:
        raise _validation_error_to_http(exc) from exc
    return UploadChunkResponse(upload_id=upload_id, chunk_index=chunk_index)


@router.get("/{upload_id}", response_model=UploadStatusResponse)
async def get_upload(upload_id: str) -> UploadStatusResponse:
    try:
        status = upload_service.get_upload_status(upload_id)
    except upload_service.UploadValidationError as exc:
        raise _validation_error_to_http(exc) from exc
    return UploadStatusResponse(**status)


@router.post("/{upload_id}/complete", response_model=UploadCompleteResponse)
async def complete_upload(upload_id: str, db: AsyncSession = Depends(get_db)) -> UploadCompleteResponse:
    manifest = upload_service.get_manifest(upload_id)

    try:
        assembled_path, probed = upload_service.assemble_and_validate(upload_id)
    except upload_service.UploadValidationError as exc:
        raise _validation_error_to_http(exc) from exc

    video_upload = VideoUpload(
        class_session_id=manifest["class_session_id"],
        storage_uri=str(assembled_path),
        duration_seconds=probed.duration_seconds,
        width=probed.width,
        height=probed.height,
        fps=probed.fps,
        bytes=probed.bytes,
    )
    db.add(video_upload)
    await db.commit()
    await db.refresh(video_upload)

    # expected_students: how many students are enrolled in this session's
    # course -- used by the pre-flight face-yield check as "roughly how many
    # faces should we be finding." Falls back to 0 (which skips that one
    # check) if the session's course has no enrollments recorded yet.
    session_row = (
        await db.execute(select(ClassSession).where(ClassSession.id == manifest["class_session_id"]))
    ).scalar_one()
    expected_students = (
        await db.execute(
            select(func.count()).select_from(Enrollment).where(Enrollment.course_id == session_row.course_id)
        )
    ).scalar_one()

    try:
        preflight_dict = await run_preflight_and_wait(str(assembled_path), expected_students)
    except PreflightTimeoutError:
        # Fail open with a warning rather than blocking the teacher
        # indefinitely -- an unusually slow preflight isn't itself a reason
        # to force a re-shoot.
        preflight_dict = {
            "status": "warn",
            "checks": [{
                "code": "preflight_timed_out",
                "severity": "warn",
                "message": "Could not finish the quality check in time. Processing will continue anyway.",
            }],
        }

    # Phase 6 addition: persist this result -- session_health (Phase 6) needs
    # to know whether THIS session's pre-flight had warnings long after this
    # response has been sent and forgotten. See VideoUpload.preflight_status_json.
    video_upload.preflight_status_json = json.dumps(preflight_dict)
    await db.commit()

    return UploadCompleteResponse(
        video_upload=VideoUploadResponse.model_validate(video_upload),
        preflight=PreflightResult(
            status=preflight_dict["status"],
            checks=[PreflightCheckResult(**c) for c in preflight_dict["checks"]],
        ),
    )
