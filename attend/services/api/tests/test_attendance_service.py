"""Phase 8 deliverable 6 tests: idempotent commit; the supersede chain
resolves to the latest row; current_attendance view correctness with a
three-deep correction chain; committing writes a row for every enrolled
student with no omissions.

Same style as test_enrollment_service.py: service functions called directly
against the in-memory SQLite `db_session` fixture, no HTTP layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
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
    Course,
    Department,
    DetectedCluster,
    Enrollment,
    Institution,
    ProcessingJob,
    ProcessingJobState,
    Student,
    Teacher,
    VideoUpload,
)
from app.db.views import CURRENT_ATTENDANCE_VIEW_NAME
from app.schemas.attendance import CommitDecision
from app.services.attendance import CommitError, commit_session, correct_attendance

UTC_NOW = datetime.now(timezone.utc)


@pytest.fixture
async def course_with_teacher(db_session: AsyncSession) -> tuple[Course, Teacher]:
    institution = Institution(name="Test Institute")
    db_session.add(institution)
    await db_session.flush()

    department = Department(institution_id=institution.id, name="CSE", code="CSE")
    db_session.add(department)
    await db_session.flush()

    course = Course(department_id=department.id, code="CS101", title="Intro to CS", semester="2026-1")
    teacher = Teacher(
        department_id=department.id, full_name="Ms. Rao", email="rao@example.test", password_hash="x",
    )
    db_session.add_all([course, teacher])
    await db_session.flush()
    return course, teacher


async def _make_student(db_session: AsyncSession, course: Course, roll_number: str) -> Student:
    student = Student(
        department_id=course.department_id, roll_number=roll_number, full_name=f"Student {roll_number}",
        admission_year=2023, is_active=True,
    )
    db_session.add(student)
    await db_session.flush()
    db_session.add(Enrollment(student_id=student.id, course_id=course.id))
    await db_session.flush()
    return student


@pytest.fixture
async def class_session(db_session: AsyncSession, course_with_teacher: tuple[Course, Teacher]) -> ClassSession:
    course, teacher = course_with_teacher
    session_row = ClassSession(
        course_id=course.id, teacher_id=teacher.id, scheduled_at=UTC_NOW, room="101",
        status=ClassSessionStatus.AWAITING_REVIEW,
    )
    db_session.add(session_row)
    await db_session.flush()
    return session_row


async def _make_succeeded_job(db_session: AsyncSession, session_row: ClassSession) -> ProcessingJob:
    video_upload = VideoUpload(
        class_session_id=session_row.id, storage_uri="/data/videos/session.mp4", duration_seconds=60.0,
        width=1920, height=1080, fps=30.0, bytes=1_000_000,
    )
    db_session.add(video_upload)
    await db_session.flush()

    job = ProcessingJob(
        class_session_id=session_row.id, video_upload_id=video_upload.id, state=ProcessingJobState.SUCCEEDED,
        params_json='{"match_threshold": 0.38}', finished_at=UTC_NOW,
    )
    db_session.add(job)
    await db_session.flush()
    return job


async def _add_cluster_match(
    db_session: AsyncSession, job: ProcessingJob, student: Student | None, decision: ClusterMatchDecision,
    similarity: float = 0.5,
) -> None:
    cluster = DetectedCluster(
        processing_job_id=job.id, representative_vector=b"\x00" * 2048, crop_count=5, mean_quality=0.8,
        best_crop_uri="crop.jpg", retention_expires_at=UTC_NOW + timedelta(days=180),
    )
    db_session.add(cluster)
    await db_session.flush()

    db_session.add(
        ClusterMatch(
            cluster_id=cluster.id, student_id=student.id if student else None, similarity=similarity,
            runner_up_similarity=0.1, decision=decision,
        )
    )
    await db_session.flush()


async def _attendance_row_count(db_session: AsyncSession) -> int:
    return len((await db_session.execute(select(AttendanceRecord))).scalars().all())


# --------------------------------------------------------------------------
# commit_session
# --------------------------------------------------------------------------


async def test_commit_writes_a_row_for_every_enrolled_student_no_omissions(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    confident_untouched = await _make_student(db_session, course, "R1")
    confident_overridden = await _make_student(db_session, course, "R2")
    uncertain_confirmed = await _make_student(db_session, course, "R3")
    never_matched = await _make_student(db_session, course, "R4")

    job = await _make_succeeded_job(db_session, class_session)
    await _add_cluster_match(db_session, job, confident_untouched, ClusterMatchDecision.CONFIDENT, similarity=0.9)
    await _add_cluster_match(db_session, job, confident_overridden, ClusterMatchDecision.CONFIDENT, similarity=0.85)
    await _add_cluster_match(db_session, job, uncertain_confirmed, ClusterMatchDecision.UNCERTAIN, similarity=0.4)
    # never_matched gets no cluster_match row at all.

    result = await commit_session(
        db_session, class_session.id, teacher.id, "req-1",
        decisions=[
            CommitDecision(student_id=confident_overridden.id, status=AttendanceStatus.ABSENT, source=AttendanceSource.TEACHER_OVERRIDE),
            CommitDecision(student_id=uncertain_confirmed.id, status=AttendanceStatus.PRESENT, source=AttendanceSource.TEACHER_CONFIRMED),
        ],
    )

    assert result.counts.total_enrolled == 4
    assert result.counts.present == 2  # confident_untouched (auto), uncertain_confirmed (teacher_confirmed)
    assert result.counts.absent == 2  # confident_overridden (teacher_override), never_matched (auto)
    assert await _attendance_row_count(db_session) == 4

    rows = {
        r.student_id: r
        for r in (await db_session.execute(select(AttendanceRecord))).scalars().all()
    }
    assert rows[confident_untouched.id].status == AttendanceStatus.PRESENT
    assert rows[confident_untouched.id].source == AttendanceSource.AUTO
    assert rows[confident_untouched.id].confidence == pytest.approx(0.9)

    assert rows[confident_overridden.id].status == AttendanceStatus.ABSENT
    assert rows[confident_overridden.id].source == AttendanceSource.TEACHER_OVERRIDE
    assert rows[confident_overridden.id].confidence is None

    assert rows[uncertain_confirmed.id].status == AttendanceStatus.PRESENT
    assert rows[uncertain_confirmed.id].source == AttendanceSource.TEACHER_CONFIRMED

    assert rows[never_matched.id].status == AttendanceStatus.ABSENT
    assert rows[never_matched.id].source == AttendanceSource.AUTO

    session_row = (await db_session.execute(select(ClassSession).where(ClassSession.id == class_session.id))).scalar_one()
    assert session_row.status == ClassSessionStatus.COMMITTED

    audit_rows = (await db_session.execute(select(AuditLog).where(AuditLog.action == "attendance.commit"))).scalars().all()
    assert len(audit_rows) == 1


async def test_commit_requires_every_needs_review_student_to_have_a_decision(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    uncertain_student = await _make_student(db_session, course, "R1")
    job = await _make_succeeded_job(db_session, class_session)
    await _add_cluster_match(db_session, job, uncertain_student, ClusterMatchDecision.UNCERTAIN, similarity=0.4)

    with pytest.raises(CommitError) as exc_info:
        await commit_session(db_session, class_session.id, teacher.id, "req-1", decisions=[])
    assert exc_info.value.code == "needs_review_incomplete"


async def test_commit_is_idempotent_on_the_same_request_id(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    student = await _make_student(db_session, course, "R1")
    job = await _make_succeeded_job(db_session, class_session)
    await _add_cluster_match(db_session, job, student, ClusterMatchDecision.CONFIDENT, similarity=0.9)

    first = await commit_session(db_session, class_session.id, teacher.id, "same-request-id", decisions=[])
    assert first.idempotent_replay is False
    assert await _attendance_row_count(db_session) == 1

    second = await commit_session(db_session, class_session.id, teacher.id, "same-request-id", decisions=[])
    assert second.idempotent_replay is True
    assert second.counts.total_enrolled == first.counts.total_enrolled
    sec_dt = second.committed_at.replace(tzinfo=None)
    first_dt = first.committed_at.replace(tzinfo=None)
    assert sec_dt == first_dt
    # No duplicate rows -- a retried commit must not double-write.
    assert await _attendance_row_count(db_session) == 1


async def test_commit_rejects_a_different_request_id_once_already_committed(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    student = await _make_student(db_session, course, "R1")
    job = await _make_succeeded_job(db_session, class_session)
    await _add_cluster_match(db_session, job, student, ClusterMatchDecision.CONFIDENT, similarity=0.9)

    await commit_session(db_session, class_session.id, teacher.id, "request-a", decisions=[])

    with pytest.raises(CommitError) as exc_info:
        await commit_session(db_session, class_session.id, teacher.id, "request-b", decisions=[])
    assert exc_info.value.code == "already_committed"


async def test_commit_rejects_a_decision_for_a_student_not_enrolled(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    student = await _make_student(db_session, course, "R1")
    job = await _make_succeeded_job(db_session, class_session)
    await _add_cluster_match(db_session, job, student, ClusterMatchDecision.CONFIDENT, similarity=0.9)

    with pytest.raises(CommitError) as exc_info:
        await commit_session(
            db_session, class_session.id, teacher.id, "req-1",
            decisions=[CommitDecision(student_id=999999, status=AttendanceStatus.PRESENT, source=AttendanceSource.TEACHER_OVERRIDE)],
        )
    assert exc_info.value.code == "decision_for_unenrolled_student"


# --------------------------------------------------------------------------
# correct_attendance + current_attendance view
# --------------------------------------------------------------------------


async def test_correct_attendance_requires_a_committed_session(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    student = await _make_student(db_session, course, "R1")
    # class_session is still AWAITING_REVIEW -- never committed.
    with pytest.raises(CommitError) as exc_info:
        await correct_attendance(db_session, class_session.id, student.id, AttendanceStatus.PRESENT, teacher.id)
    assert exc_info.value.code == "not_committed"


async def test_correct_attendance_inserts_a_new_row_never_updates(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    student = await _make_student(db_session, course, "R1")
    job = await _make_succeeded_job(db_session, class_session)
    await _add_cluster_match(db_session, job, student, ClusterMatchDecision.CONFIDENT, similarity=0.9)
    await commit_session(db_session, class_session.id, teacher.id, "req-1", decisions=[])

    assert await _attendance_row_count(db_session) == 1
    original_row = (await db_session.execute(select(AttendanceRecord))).scalars().one()
    assert original_row.status == AttendanceStatus.PRESENT

    corrected = await correct_attendance(db_session, class_session.id, student.id, AttendanceStatus.ABSENT, teacher.id)

    assert await _attendance_row_count(db_session) == 2  # a NEW row, the old one still exists untouched
    assert corrected.supersedes_id == original_row.id
    assert corrected.status == AttendanceStatus.ABSENT
    assert corrected.source == AttendanceSource.TEACHER_OVERRIDE

    # The original row must be byte-for-byte unchanged (append-only, rule 4).
    reloaded_original = (
        await db_session.execute(select(AttendanceRecord).where(AttendanceRecord.id == original_row.id))
    ).scalar_one()
    assert reloaded_original.status == AttendanceStatus.PRESENT


async def test_correct_attendance_rejects_no_op_correction(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    student = await _make_student(db_session, course, "R1")
    job = await _make_succeeded_job(db_session, class_session)
    await _add_cluster_match(db_session, job, student, ClusterMatchDecision.CONFIDENT, similarity=0.9)
    await commit_session(db_session, class_session.id, teacher.id, "req-1", decisions=[])

    with pytest.raises(CommitError) as exc_info:
        await correct_attendance(db_session, class_session.id, student.id, AttendanceStatus.PRESENT, teacher.id)
    assert exc_info.value.code == "no_change"


async def test_current_attendance_view_resolves_a_three_deep_correction_chain(
    db_session: AsyncSession, class_session: ClassSession, course_with_teacher: tuple[Course, Teacher],
):
    course, teacher = course_with_teacher
    student = await _make_student(db_session, course, "R1")
    job = await _make_succeeded_job(db_session, class_session)
    await _add_cluster_match(db_session, job, student, ClusterMatchDecision.CONFIDENT, similarity=0.9)
    await commit_session(db_session, class_session.id, teacher.id, "req-1", decisions=[])  # row 1: present

    correction_1 = await correct_attendance(db_session, class_session.id, student.id, AttendanceStatus.ABSENT, teacher.id)  # row 2
    correction_2 = await correct_attendance(db_session, class_session.id, student.id, AttendanceStatus.PRESENT, teacher.id)  # row 3
    correction_3 = await correct_attendance(db_session, class_session.id, student.id, AttendanceStatus.ABSENT, teacher.id)  # row 4 -- three corrections deep

    assert await _attendance_row_count(db_session) == 4  # every row still exists -- append-only

    view_rows = (
        await db_session.execute(
            text(f"SELECT id, status FROM {CURRENT_ATTENDANCE_VIEW_NAME} WHERE class_session_id = :sid AND student_id = :stid"),
            {"sid": class_session.id, "stid": student.id},
        )
    ).all()

    assert len(view_rows) == 1, "exactly one current row, no matter how deep the correction chain"
    assert view_rows[0][0] == correction_3.id
    assert view_rows[0][1] == AttendanceStatus.ABSENT.value
    assert correction_1.supersedes_id is not None
    assert correction_2.supersedes_id == correction_1.id
    assert correction_3.supersedes_id == correction_2.id
