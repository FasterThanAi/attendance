"""Phase 8 deliverable 3-4: POST /sessions/{id}/commit and POST
/sessions/{id}/attendance/{student_id}/correct.

Thin wrappers over app/services/attendance.py -- all the actual logic
(progressive-review validation, idempotency, append-only correction) lives
there and is unit-tested directly against it, same split as every other
router in this codebase.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.attendance import CommitRequest, CommitResponse, CorrectionRequest, CorrectionResponse
from app.schemas.errors import ErrorResponse
from app.services.attendance import CommitError
from app.services.attendance import commit_session as commit_session_service
from app.services.attendance import correct_attendance as correct_attendance_service

router = APIRouter(prefix="/sessions", tags=["attendance"])

# Maps CommitError.code -> HTTP status. 404 for "doesn't exist", 409 for
# "exists but is in the wrong state for this action" (non-negotiable rule
# #7: every error is this structured ErrorResponse shape, never a bare string).
_ERROR_STATUS_CODES: dict[str, int] = {
    "session_not_found": 404,
    "draft_not_ready": 409,
    "already_committed": 409,
    "not_awaiting_review": 409,
    "decision_for_unenrolled_student": 422,
    "needs_review_incomplete": 422,
    "audit_log_missing": 500,
    "not_committed": 409,
    "attendance_not_found": 404,
    "no_change": 409,
}


def _http_exception_for(exc: CommitError) -> HTTPException:
    status_code = _ERROR_STATUS_CODES.get(exc.code, 400)
    return HTTPException(status_code=status_code, detail=ErrorResponse(code=exc.code, message=exc.message).model_dump())


@router.post("/{session_id}/commit", response_model=CommitResponse)
async def commit_session(session_id: int, body: CommitRequest, db: AsyncSession = Depends(get_db)) -> CommitResponse:
    """Phase 8 deliverable 3. Idempotent on `body.request_id` -- retrying
    with the SAME request_id after a dropped response returns the same
    result rather than erroring or double-committing; retrying with a
    DIFFERENT request_id against an already-committed session is rejected
    (409 `already_committed`), since a session commits exactly once --
    further changes go through the correction endpoint below.
    """
    try:
        result = await commit_session_service(db, session_id, body.teacher_id, body.request_id, body.decisions)
    except CommitError as exc:
        raise _http_exception_for(exc) from exc

    return CommitResponse(
        class_session_id=result.class_session_id,
        status=result.status,
        counts=result.counts,
        committed_at=result.committed_at,
        idempotent_replay=result.idempotent_replay,
    )


@router.post("/{session_id}/attendance/{student_id}/correct", response_model=CorrectionResponse)
async def correct_attendance(
    session_id: int, student_id: int, body: CorrectionRequest, db: AsyncSession = Depends(get_db)
) -> CorrectionResponse:
    """Phase 8 deliverable 4. Only valid on an already-committed session.
    Always inserts a new attendance_record row (supersedes_id set to
    whatever `current_attendance` currently resolves to for this student) --
    never an update, per non-negotiable rule #4.
    """
    try:
        new_record = await correct_attendance_service(db, session_id, student_id, body.status, body.teacher_id)
    except CommitError as exc:
        raise _http_exception_for(exc) from exc

    return CorrectionResponse(
        attendance_record_id=new_record.id,
        student_id=new_record.student_id,
        status=new_record.status,
        supersedes_id=new_record.supersedes_id,
        created_at=new_record.created_at,
    )
