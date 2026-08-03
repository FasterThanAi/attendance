"""Minimal class_session CRUD.

Not an explicit Phase 2 deliverable in the roadmap -- Phase 2's own
integration contract says it "consumes a class_session from Phase 0" as
though one already exists. Nothing before this point actually creates one,
though, and the upload endpoints (this phase's real deliverable) need a real
class_session_id to attach a video to. Added as the minimal thing needed to
make Phase 2 testable end-to-end; full scheduling/"Today" screen UX is
Phase 8-9's job, not this endpoint's.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ClassSession, ClassSessionStatus, Course, Teacher
from app.db.session import get_db
from app.schemas.errors import ErrorResponse
from app.schemas.session import ClassSessionCreate, ClassSessionResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=ClassSessionResponse, status_code=201)
async def create_session(body: ClassSessionCreate, db: AsyncSession = Depends(get_db)) -> ClassSessionResponse:
    if (await db.execute(select(Course).where(Course.id == body.course_id))).scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(
            code="course_not_found", message=f"No course with id={body.course_id}."
        ).model_dump())
    if (await db.execute(select(Teacher).where(Teacher.id == body.teacher_id))).scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(
            code="teacher_not_found", message=f"No teacher with id={body.teacher_id}."
        ).model_dump())

    session_row = ClassSession(
        course_id=body.course_id,
        teacher_id=body.teacher_id,
        scheduled_at=body.scheduled_at,
        room=body.room,
        status=ClassSessionStatus.SCHEDULED,
    )
    db.add(session_row)
    await db.commit()
    await db.refresh(session_row)
    return ClassSessionResponse.model_validate(session_row)


@router.get("/{session_id}", response_model=ClassSessionResponse)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)) -> ClassSessionResponse:
    result = await db.execute(select(ClassSession).where(ClassSession.id == session_id))
    session_row = result.scalar_one_or_none()
    if session_row is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(
            code="session_not_found", message=f"No class session with id={session_id}."
        ).model_dump())
    return ClassSessionResponse.model_validate(session_row)
