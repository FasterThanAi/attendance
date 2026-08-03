"""Phase 8: attendance commit + correction business logic.

Split the same way every other service module in this codebase is: plain
functions here, called by app/routers/attendance.py, directly unit-testable
against the in-memory SQLite fixture (services/api/tests/conftest.py) with
no HTTP layer involved -- same pattern as services/enrollment.py.

Two non-negotiable rules from the global brief this module exists to
satisfy:
  Rule 3 (Phase 8's own "teacher's total interaction under 30 seconds"
          design principle): commit must be possible after handling ONLY
          the needs-your-check queue -- see the "progressive review" check
          in commit_session, which blocks on exactly that queue and
          nothing else.
  Rule 4: attendance_record is append-only. Nothing in this file ever
          UPDATEs or DELETEs an attendance_record row -- correct_attendance
          only ever INSERTs a new one with supersedes_id set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AttendanceRecord,
    AttendanceSource,
    AttendanceStatus,
    AuditLog,
    ClassSession,
    ClassSessionStatus,
    ClusterMatch,
    ClusterMatchDecision,
    DetectedCluster,
    Enrollment,
    ProcessingJob,
    ProcessingJobState,
    Student,
)
from app.db.views import CURRENT_ATTENDANCE_VIEW_NAME
from app.schemas.attendance import CommitCounts, CommitDecision


class CommitError(Exception):
    """Raised for any commit/correction precondition failure. `code` is a
    stable, machine-readable string (non-negotiable rule #7) the router
    maps to an HTTP status; `message` is the human-readable explanation.
    """

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class CommitResult:
    class_session_id: int
    status: ClassSessionStatus
    counts: CommitCounts
    committed_at: datetime
    idempotent_replay: bool


@dataclass(frozen=True)
class SessionBands:
    """The three bands GET /sessions/{id}/draft already computes, re-derived
    here rather than imported: routers/session.py's draft endpoint builds
    richer per-student display objects (photo URIs, names) this function
    doesn't need, and duplicating the query keeps that already-verified
    Phase 6 endpoint untouched. If these two ever need to share code, that's
    a refactor for whoever touches this next, not a risk worth taking now
    for either the already-shipped, already-verified draft endpoint.
    """

    confident_similarity_by_student_id: dict[int, float]
    needs_review_similarity_by_student_id: dict[int, float]
    enrolled_students: list[Student]
    processing_job_id: int
    params_json: str


async def _load_session_bands(db: AsyncSession, session_row: ClassSession) -> SessionBands:
    job_row = (
        await db.execute(
            select(ProcessingJob)
            .where(
                ProcessingJob.class_session_id == session_row.id,
                ProcessingJob.state == ProcessingJobState.SUCCEEDED,
            )
            .order_by(ProcessingJob.finished_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if job_row is None:
        raise CommitError(
            "draft_not_ready",
            f"class_session {session_row.id} has no succeeded processing_job yet -- nothing to commit.",
        )

    match_rows = (
        await db.execute(
            select(ClusterMatch)
            .join(DetectedCluster, ClusterMatch.cluster_id == DetectedCluster.id)
            .where(DetectedCluster.processing_job_id == job_row.id)
        )
    ).scalars().all()

    confident: dict[int, float] = {}
    needs_review: dict[int, float] = {}
    for m in match_rows:
        if m.student_id is None:
            continue
        if m.decision == ClusterMatchDecision.CONFIDENT:
            confident[m.student_id] = m.similarity
        elif m.decision == ClusterMatchDecision.UNCERTAIN:
            needs_review[m.student_id] = m.similarity

    enrolled_students = (
        await db.execute(
            select(Student)
            .join(Enrollment, Enrollment.student_id == Student.id)
            .where(Enrollment.course_id == session_row.course_id)
        )
    ).scalars().all()

    return SessionBands(
        confident_similarity_by_student_id=confident,
        needs_review_similarity_by_student_id=needs_review,
        enrolled_students=list(enrolled_students),
        processing_job_id=job_row.id,
        params_json=job_row.params_json,
    )


async def _counts_from_current_attendance(db: AsyncSession, class_session_id: int) -> CommitCounts:
    """Used for an idempotent-replay response -- recomputes counts from
    what's ACTUALLY committed (the view), rather than trusting whatever the
    original commit's in-memory counters said, so a replay reflects reality
    even if a correction happened between the original commit and the retry.
    """
    rows = (
        await db.execute(
            text(
                f"SELECT status, source FROM {CURRENT_ATTENDANCE_VIEW_NAME} WHERE class_session_id = :sid"
            ),
            {"sid": class_session_id},
        )
    ).all()

    present = sum(1 for r in rows if r[0] == AttendanceStatus.PRESENT.value)
    absent = sum(1 for r in rows if r[0] == AttendanceStatus.ABSENT.value)
    auto_count = sum(1 for r in rows if r[1] == AttendanceSource.AUTO.value)
    teacher_confirmed_count = sum(1 for r in rows if r[1] == AttendanceSource.TEACHER_CONFIRMED.value)
    teacher_override_count = sum(1 for r in rows if r[1] == AttendanceSource.TEACHER_OVERRIDE.value)

    return CommitCounts(
        total_enrolled=len(rows), present=present, absent=absent, auto_count=auto_count,
        teacher_confirmed_count=teacher_confirmed_count, teacher_override_count=teacher_override_count,
    )


async def _original_commit_timestamp(db: AsyncSession, class_session_id: int) -> datetime:
    """For an idempotent-replay response: class_session itself only stores
    `commit_request_id`, not a `committed_at` column, so the ORIGINAL
    commit's timestamp is recovered from the audit_log row commit_session
    always writes -- reusing data that already exists rather than adding a
    column whose only purpose would be answering this one question.
    Ordered ASC so a (should-never-happen, but don't trust that) duplicate
    audit_log row for the same commit doesn't shift the answer.
    """
    audit_row = (
        await db.execute(
            select(AuditLog)
            .where(
                AuditLog.action == "attendance.commit",
                AuditLog.subject_type == "class_session",
                AuditLog.subject_id == str(class_session_id),
            )
            .order_by(AuditLog.created_at.asc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if audit_row is None:
        raise CommitError(
            "audit_log_missing",
            f"class_session {class_session_id} is committed but has no 'attendance.commit' audit_log row -- "
            "this points at a bug (commit_session always writes one in the same transaction), not a normal state.",
        )
    return audit_row.created_at


async def commit_session(
    db: AsyncSession,
    class_session_id: int,
    teacher_id: int,
    request_id: str,
    decisions: list[CommitDecision],
) -> CommitResult:
    """Phase 8 deliverable 3. Writes exactly one attendance_record row for
    EVERY enrolled student (present or absent, no omissions):
      - a student with an explicit CommitDecision gets exactly that
        status/source, confidence=None (a human decision, not a machine
        score);
      - a student with NO decision who was a CONFIDENT match gets
        AUTO/PRESENT, confidence=that match's similarity;
      - a student with NO decision and no confident match gets AUTO/ABSENT
        (this covers both "not found" and "unrecognised cluster only"
        students who the teacher chose not to touch -- proposed_absent
        defaults to absent, per Phase 8 deliverable 1c, verbatim: "Default
        state is absent").
    Progressive review (deliverable 2): every UNCERTAIN ("needs your check")
    student MUST have an explicit decision -- that is the one non-negotiable
    minimum before committing; every other section may be left untouched.
    """
    session_row = (
        await db.execute(select(ClassSession).where(ClassSession.id == class_session_id))
    ).scalar_one_or_none()
    if session_row is None:
        raise CommitError("session_not_found", f"No class_session with id={class_session_id}.")

    if session_row.status == ClassSessionStatus.COMMITTED:
        if session_row.commit_request_id == request_id:
            counts = await _counts_from_current_attendance(db, class_session_id)
            committed_at = await _original_commit_timestamp(db, class_session_id)
            return CommitResult(
                class_session_id=class_session_id,
                status=ClassSessionStatus.COMMITTED,
                counts=counts,
                committed_at=committed_at,
                idempotent_replay=True,
            )
        raise CommitError(
            "already_committed",
            f"class_session {class_session_id} was already committed with a different request_id -- "
            "a session can only be committed once; use the correction endpoint to change a committed decision.",
        )

    if session_row.status != ClassSessionStatus.AWAITING_REVIEW:
        raise CommitError(
            "not_awaiting_review",
            f"class_session {class_session_id} is '{session_row.status.value}', not 'awaiting_review' -- "
            "there is no draft roster to commit yet.",
        )

    bands = await _load_session_bands(db, session_row)
    enrolled_ids = {s.id for s in bands.enrolled_students}

    decisions_by_student = {d.student_id: d for d in decisions}

    unknown_students = set(decisions_by_student) - enrolled_ids
    if unknown_students:
        raise CommitError(
            "decision_for_unenrolled_student",
            f"Decision(s) given for student_id(s) {sorted(unknown_students)}, who are not enrolled in this "
            "session's course.",
        )

    missing_needs_review = [
        sid for sid in bands.needs_review_similarity_by_student_id if sid not in decisions_by_student
    ]
    if missing_needs_review:
        raise CommitError(
            "needs_review_incomplete",
            f"{len(missing_needs_review)} student(s) in the needs-your-check queue have no decision yet: "
            f"{sorted(missing_needs_review)}. Every uncertain match must be confirmed or rejected before "
            "committing (Phase 8's progressive review rule) -- every other section may be left untouched.",
        )

    now = datetime.now(timezone.utc)
    present_count = absent_count = auto_count = teacher_confirmed_count = teacher_override_count = 0

    for student in bands.enrolled_students:
        decision = decisions_by_student.get(student.id)
        if decision is not None:
            status, source, confidence = decision.status, decision.source, None
        elif student.id in bands.confident_similarity_by_student_id:
            status = AttendanceStatus.PRESENT
            source = AttendanceSource.AUTO
            confidence = bands.confident_similarity_by_student_id[student.id]
        else:
            # Covers both "not found" (proposed_absent) and "unrecognised
            # cluster only" students the teacher chose not to touch --
            # default state is absent (deliverable 1c, verbatim).
            status = AttendanceStatus.ABSENT
            source = AttendanceSource.AUTO
            confidence = None

        db.add(
            AttendanceRecord(
                class_session_id=class_session_id,
                student_id=student.id,
                status=status,
                source=source,
                confidence=confidence,
                decided_by_teacher_id=teacher_id if source != AttendanceSource.AUTO else None,
                supersedes_id=None,
                created_at=now,
            )
        )

        present_count += status == AttendanceStatus.PRESENT
        absent_count += status == AttendanceStatus.ABSENT
        auto_count += source == AttendanceSource.AUTO
        teacher_confirmed_count += source == AttendanceSource.TEACHER_CONFIRMED
        teacher_override_count += source == AttendanceSource.TEACHER_OVERRIDE

    commit_counts = CommitCounts(
        total_enrolled=len(bands.enrolled_students),
        present=present_count,
        absent=absent_count,
        auto_count=auto_count,
        teacher_confirmed_count=teacher_confirmed_count,
        teacher_override_count=teacher_override_count,
    )

    # "the params_hash used" (deliverable 3): processing_job has no single
    # stored params_hash column of its own (pipeline.run.compute_stage_hashes
    # computes PER-STAGE hashes, never persisted as one value) -- a sha256 of
    # the exact params_json this job ran with is a reproducible fingerprint
    # of "what parameters produced this draft," which is what an auditor
    # actually wants to know.
    params_hash = hashlib.sha256(bands.params_json.encode("utf-8")).hexdigest()

    db.add(
        AuditLog(
            actor_type="teacher",
            actor_id=str(teacher_id),
            action="attendance.commit",
            subject_type="class_session",
            subject_id=str(class_session_id),
            payload_json=json.dumps({
                "request_id": request_id,
                "counts": commit_counts.model_dump(),
                "params_hash": params_hash,
                "processing_job_id": bands.processing_job_id,
            }),
            created_at=now,
        )
    )

    session_row.status = ClassSessionStatus.COMMITTED
    session_row.commit_request_id = request_id

    await db.commit()

    return CommitResult(
        class_session_id=class_session_id,
        status=ClassSessionStatus.COMMITTED,
        counts=commit_counts,
        committed_at=now,
        idempotent_replay=False,
    )


async def correct_attendance(
    db: AsyncSession,
    class_session_id: int,
    student_id: int,
    new_status: AttendanceStatus,
    teacher_id: int,
) -> AttendanceRecord:
    """Phase 8 deliverable 4. Only valid once a session is committed (a
    non-committed session has no "current" decision to correct -- that's
    what commit_session is for). Always INSERTs a new attendance_record row
    with supersedes_id pointing at whatever current_attendance currently
    resolves to for this student -- NEVER an UPDATE (rule 4).
    """
    session_row = (
        await db.execute(select(ClassSession).where(ClassSession.id == class_session_id))
    ).scalar_one_or_none()
    if session_row is None:
        raise CommitError("session_not_found", f"No class_session with id={class_session_id}.")

    if session_row.status != ClassSessionStatus.COMMITTED:
        raise CommitError(
            "not_committed",
            f"class_session {class_session_id} is '{session_row.status.value}', not 'committed' -- "
            "corrections only apply to an already-committed session; commit it first.",
        )

    current_row = (
        await db.execute(
            text(
                f"SELECT id, status FROM {CURRENT_ATTENDANCE_VIEW_NAME} "
                "WHERE class_session_id = :sid AND student_id = :stid"
            ),
            {"sid": class_session_id, "stid": student_id},
        )
    ).mappings().first()
    if current_row is None:
        raise CommitError(
            "attendance_not_found",
            f"No current attendance_record for student_id={student_id} in class_session_id={class_session_id}. "
            "Either this student isn't enrolled in this session's course, or (if they are) commit_session "
            "should have written a row for them -- worth checking which before assuming this is bad input.",
        )

    if current_row["status"] == new_status.value:
        raise CommitError(
            "no_change",
            f"student_id={student_id} is already marked {new_status.value} in class_session_id={class_session_id} "
            "-- nothing to correct.",
        )

    now = datetime.now(timezone.utc)
    new_record = AttendanceRecord(
        class_session_id=class_session_id,
        student_id=student_id,
        status=new_status,
        source=AttendanceSource.TEACHER_OVERRIDE,
        confidence=None,
        decided_by_teacher_id=teacher_id,
        supersedes_id=current_row["id"],
        created_at=now,
    )
    db.add(new_record)

    db.add(
        AuditLog(
            actor_type="teacher",
            actor_id=str(teacher_id),
            action="attendance.correct",
            subject_type="attendance_record",
            subject_id=str(current_row["id"]),
            payload_json=json.dumps({
                "student_id": student_id,
                "old_status": current_row["status"],
                "new_status": new_status.value,
            }),
            created_at=now,
        )
    )

    await db.commit()
    await db.refresh(new_record)
    return new_record
