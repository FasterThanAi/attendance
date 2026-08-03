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

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ClassSession,
    ClassSessionStatus,
    ClusterMatch,
    ClusterMatchDecision,
    Course,
    DetectedCluster,
    Enrollment,
    GalleryPhoto,
    ProcessingJob,
    ProcessingJobState,
    Student,
    Teacher,
)
from app.db.session import get_db
from app.schemas.errors import ErrorResponse
from app.schemas.session import ClassSessionCreate, ClassSessionResponse
from app.schemas.session_draft import (
    DraftAbsentStudent,
    DraftClusterMatch,
    DraftSessionSummary,
    SessionDraftResponse,
)

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


@router.get("/{session_id}/draft", response_model=SessionDraftResponse)
async def get_session_draft(session_id: int, db: AsyncSession = Depends(get_db)) -> SessionDraftResponse:
    """Phase 6 deliverable 5. Never returns a raw embedding vector -- only
    URIs (a cluster's best crop, a student's best enrollment photo) and the
    scalar numbers (similarity, margin) a teacher's review UI needs to
    explain each proposed decision. See schemas/session_draft.py.
    """
    session_row = (await db.execute(select(ClassSession).where(ClassSession.id == session_id))).scalar_one_or_none()
    if session_row is None:
        raise HTTPException(status_code=404, detail=ErrorResponse(
            code="session_not_found", message=f"No class session with id={session_id}."
        ).model_dump())

    if session_row.draft_summary_json is None:
        raise HTTPException(status_code=409, detail=ErrorResponse(
            code="draft_not_ready",
            message=(
                "This session has no draft roster yet -- processing hasn't reached the match "
                "stage (or hasn't been started at all). Check GET /jobs/{job_id} for progress."
            ),
        ).model_dump())

    summary_dict = json.loads(session_row.draft_summary_json)

    # The match stage (pipeline/match.py's run_match_stage) is the last
    # stage in STAGE_ORDER and is what wrote draft_summary_json -- so "the
    # most recently SUCCEEDED job for this session" is, by construction,
    # the run that produced the summary we just parsed above.
    job_row = (
        await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.class_session_id == session_id, ProcessingJob.state == ProcessingJobState.SUCCEEDED)
            .order_by(ProcessingJob.finished_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if job_row is None:
        raise HTTPException(status_code=409, detail=ErrorResponse(
            code="draft_inconsistent",
            message=(
                f"class_session {session_id} has a draft_summary_json but no succeeded "
                "processing_job -- this is an inconsistent state, not a normal 'not ready yet'."
            ),
        ).model_dump())

    # One row per detected_cluster for this job -- run_match_stage inserts
    # exactly one cluster_match row per cluster, whether or not it resolved
    # to a student (see match.py's ClusterMatchRow / UNMATCHED handling).
    rows = (
        await db.execute(
            select(DetectedCluster, ClusterMatch, Student)
            .join(ClusterMatch, ClusterMatch.cluster_id == DetectedCluster.id)
            .outerjoin(Student, Student.id == ClusterMatch.student_id)
            .where(DetectedCluster.processing_job_id == job_row.id)
        )
    ).all()

    async def _enrollment_photo_uri(student_id: int) -> str | None:
        # ASSUMPTION: "the enrollment photo" for a student is their
        # highest-quality_score gallery_photo -- the schema has no explicit
        # "primary photo" flag, and quality_score (Phase 1) is exactly the
        # signal that would pick the most representative one anyway.
        photo_row = (
            await db.execute(
                select(GalleryPhoto)
                .where(GalleryPhoto.student_id == student_id)
                .order_by(GalleryPhoto.quality_score.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return photo_row.storage_uri if photo_row else None

    confident: list[DraftClusterMatch] = []
    needs_review: list[DraftClusterMatch] = []
    unrecognised_clusters: list[DraftClusterMatch] = []
    matched_student_ids: set[int] = set()

    for cluster, match, student in rows:
        entry = DraftClusterMatch(
            cluster_id=cluster.id,
            best_crop_uri=cluster.best_crop_uri,
            student_id=student.id if student else None,
            student_name=student.full_name if student else None,
            roll_number=student.roll_number if student else None,
            similarity=match.similarity,
            runner_up_similarity=match.runner_up_similarity,
            enrollment_photo_uri=(await _enrollment_photo_uri(student.id)) if student else None,
        )
        if match.decision == ClusterMatchDecision.CONFIDENT:
            confident.append(entry)
            if student is not None:
                matched_student_ids.add(student.id)
        elif match.decision == ClusterMatchDecision.UNCERTAIN:
            needs_review.append(entry)
            if student is not None:
                matched_student_ids.add(student.id)
        else:
            unrecognised_clusters.append(entry)

    # proposed_absent: every student enrolled in this session's course who
    # isn't accounted for by a confident or needs_review match above --
    # mirrors pipeline.match.build_session_summary's own partition exactly.
    enrolled_rows = (
        await db.execute(
            select(Student)
            .join(Enrollment, Enrollment.student_id == Student.id)
            .where(Enrollment.course_id == session_row.course_id)
        )
    ).scalars().all()

    proposed_absent = [
        DraftAbsentStudent(student_id=s.id, student_name=s.full_name, roll_number=s.roll_number)
        for s in enrolled_rows
        if s.id not in matched_student_ids
    ]

    return SessionDraftResponse(
        session_id=session_id,
        status=session_row.status,
        summary=DraftSessionSummary(**summary_dict),
        confident=confident,
        needs_review=needs_review,
        proposed_absent=proposed_absent,
        unrecognised_clusters=unrecognised_clusters,
    )
