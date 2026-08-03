"""One test per model: can it be created and queried back (deliverable 9).

`base_graph` builds the minimal chain of parent rows (institution ->
department -> course/student/teacher) that almost every other table hangs
off of, so each individual test only has to create the ONE row it's actually
testing.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
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
    Consent,
    Course,
    Department,
    DetectedCluster,
    Enrollment,
    GalleryEmbedding,
    GalleryPhoto,
    Institution,
    ProcessingJob,
    ProcessingJobState,
    Student,
    Teacher,
    VideoUpload,
)

UTC_NOW = datetime.now(timezone.utc)


@pytest.fixture
async def base_graph(db_session: AsyncSession):
    institution = Institution(name="Test Institute")
    db_session.add(institution)
    await db_session.flush()

    department = Department(institution_id=institution.id, name="Computer Science", code="CSE")
    db_session.add(department)
    await db_session.flush()

    course = Course(department_id=department.id, code="CS301", title="Operating Systems", semester="5")
    student = Student(
        department_id=department.id,
        roll_number="CSE001",
        full_name="Asha Kumar",
        admission_year=2023,
        is_active=True,
    )
    teacher = Teacher(
        department_id=department.id,
        full_name="Dr. Rao",
        email="rao@example.edu",
        password_hash="not-a-real-hash",
    )
    db_session.add_all([course, student, teacher])
    await db_session.flush()

    return {
        "institution": institution,
        "department": department,
        "course": course,
        "student": student,
        "teacher": teacher,
    }


async def test_institution(db_session: AsyncSession, base_graph):
    institution = base_graph["institution"]
    result = await db_session.execute(select(Institution).where(Institution.id == institution.id))
    fetched = result.scalar_one()
    assert fetched.name == "Test Institute"


async def test_department(db_session: AsyncSession, base_graph):
    department = base_graph["department"]
    result = await db_session.execute(select(Department).where(Department.id == department.id))
    fetched = result.scalar_one()
    assert fetched.code == "CSE"
    assert fetched.institution_id == base_graph["institution"].id


async def test_course(db_session: AsyncSession, base_graph):
    course = base_graph["course"]
    result = await db_session.execute(select(Course).where(Course.id == course.id))
    fetched = result.scalar_one()
    assert fetched.title == "Operating Systems"


async def test_student(db_session: AsyncSession, base_graph):
    student = base_graph["student"]
    result = await db_session.execute(select(Student).where(Student.id == student.id))
    fetched = result.scalar_one()
    assert fetched.roll_number == "CSE001"
    assert fetched.is_active is True


async def test_teacher(db_session: AsyncSession, base_graph):
    teacher = base_graph["teacher"]
    result = await db_session.execute(select(Teacher).where(Teacher.id == teacher.id))
    fetched = result.scalar_one()
    assert fetched.email == "rao@example.edu"


async def test_enrollment(db_session: AsyncSession, base_graph):
    enrollment = Enrollment(student_id=base_graph["student"].id, course_id=base_graph["course"].id)
    db_session.add(enrollment)
    await db_session.flush()

    result = await db_session.execute(select(Enrollment).where(Enrollment.id == enrollment.id))
    fetched = result.scalar_one()
    assert fetched.student_id == base_graph["student"].id
    assert fetched.course_id == base_graph["course"].id


async def test_consent(db_session: AsyncSession, base_graph):
    consent = Consent(
        student_id=base_graph["student"].id,
        granted_at=UTC_NOW,
        revoked_at=None,
        consent_version="v1",
        scope="classroom attendance via face recognition",
        evidence_uri="s3://attend-media/consent/student-1.pdf",
    )
    db_session.add(consent)
    await db_session.flush()

    result = await db_session.execute(select(Consent).where(Consent.id == consent.id))
    fetched = result.scalar_one()
    assert fetched.revoked_at is None
    assert fetched.consent_version == "v1"


async def test_gallery_photo(db_session: AsyncSession, base_graph):
    photo = GalleryPhoto(
        student_id=base_graph["student"].id,
        storage_uri="s3://attend-media/gallery/student-1/photo-1.jpg",
        captured_at=UTC_NOW,
        quality_score=0.82,
    )
    db_session.add(photo)
    await db_session.flush()

    result = await db_session.execute(select(GalleryPhoto).where(GalleryPhoto.id == photo.id))
    fetched = result.scalar_one()
    assert fetched.quality_score == pytest.approx(0.82)


async def test_gallery_embedding(db_session: AsyncSession, base_graph):
    photo = GalleryPhoto(
        student_id=base_graph["student"].id,
        storage_uri="s3://attend-media/gallery/student-1/photo-1.jpg",
        captured_at=UTC_NOW,
        quality_score=0.82,
    )
    db_session.add(photo)
    await db_session.flush()

    embedding = GalleryEmbedding(
        student_id=base_graph["student"].id,
        vector=b"\x00" * 2048,  # 512 float32 values, zeroed for the test
        source_photo_id=photo.id,
        model_version="arcface-r100-buffalo_l",
        retention_expires_at=UTC_NOW + timedelta(days=180),
    )
    db_session.add(embedding)
    await db_session.flush()

    result = await db_session.execute(select(GalleryEmbedding).where(GalleryEmbedding.id == embedding.id))
    fetched = result.scalar_one()
    assert len(fetched.vector) == 2048
    assert fetched.retention_expires_at > UTC_NOW


async def test_class_session(db_session: AsyncSession, base_graph):
    session_row = ClassSession(
        course_id=base_graph["course"].id,
        teacher_id=base_graph["teacher"].id,
        scheduled_at=UTC_NOW,
        room="Room 204",
        status=ClassSessionStatus.SCHEDULED,
    )
    db_session.add(session_row)
    await db_session.flush()

    result = await db_session.execute(select(ClassSession).where(ClassSession.id == session_row.id))
    fetched = result.scalar_one()
    assert fetched.status == ClassSessionStatus.SCHEDULED
    assert fetched.room == "Room 204"


async def test_video_upload(db_session: AsyncSession, base_graph):
    session_row = ClassSession(
        course_id=base_graph["course"].id,
        teacher_id=base_graph["teacher"].id,
        scheduled_at=UTC_NOW,
        room="Room 204",
        status=ClassSessionStatus.RECORDING,
    )
    db_session.add(session_row)
    await db_session.flush()

    upload = VideoUpload(
        class_session_id=session_row.id,
        storage_uri="s3://attend-media/sessions/1/video.mp4",
        duration_seconds=62.0,
        width=3840,
        height=2160,
        fps=30.0,
        bytes=310_000_000,
    )
    db_session.add(upload)
    await db_session.flush()

    result = await db_session.execute(select(VideoUpload).where(VideoUpload.id == upload.id))
    fetched = result.scalar_one()
    assert fetched.width == 3840
    assert fetched.height == 2160


async def test_processing_job(db_session: AsyncSession, base_graph):
    session_row = ClassSession(
        course_id=base_graph["course"].id,
        teacher_id=base_graph["teacher"].id,
        scheduled_at=UTC_NOW,
        room="Room 204",
        status=ClassSessionStatus.PROCESSING,
    )
    db_session.add(session_row)
    await db_session.flush()

    upload = VideoUpload(
        class_session_id=session_row.id,
        storage_uri="s3://attend-media/sessions/1/video.mp4",
        duration_seconds=62.0,
        width=3840,
        height=2160,
        fps=30.0,
        bytes=310_000_000,
    )
    db_session.add(upload)
    await db_session.flush()

    job = ProcessingJob(
        class_session_id=session_row.id,
        video_upload_id=upload.id,
        state=ProcessingJobState.QUEUED,
        stage=None,
        params_json="{}",
    )
    db_session.add(job)
    await db_session.flush()

    result = await db_session.execute(select(ProcessingJob).where(ProcessingJob.id == job.id))
    fetched = result.scalar_one()
    assert fetched.state == ProcessingJobState.QUEUED


async def test_detected_cluster(db_session: AsyncSession, base_graph):
    session_row = ClassSession(
        course_id=base_graph["course"].id,
        teacher_id=base_graph["teacher"].id,
        scheduled_at=UTC_NOW,
        room="Room 204",
        status=ClassSessionStatus.PROCESSING,
    )
    db_session.add(session_row)
    await db_session.flush()

    upload = VideoUpload(
        class_session_id=session_row.id,
        storage_uri="s3://attend-media/sessions/1/video.mp4",
        duration_seconds=62.0,
        width=3840,
        height=2160,
        fps=30.0,
        bytes=310_000_000,
    )
    db_session.add(upload)
    await db_session.flush()

    job = ProcessingJob(
        class_session_id=session_row.id,
        video_upload_id=upload.id,
        state=ProcessingJobState.RUNNING,
        stage="cluster",
        params_json="{}",
    )
    db_session.add(job)
    await db_session.flush()

    cluster = DetectedCluster(
        processing_job_id=job.id,
        representative_vector=b"\x00" * 2048,
        crop_count=42,
        mean_quality=0.74,
        best_crop_uri="s3://attend-media/jobs/1/clusters/0/best.jpg",
        retention_expires_at=UTC_NOW + timedelta(days=180),
    )
    db_session.add(cluster)
    await db_session.flush()

    result = await db_session.execute(select(DetectedCluster).where(DetectedCluster.id == cluster.id))
    fetched = result.scalar_one()
    assert fetched.crop_count == 42


async def test_cluster_match(db_session: AsyncSession, base_graph):
    session_row = ClassSession(
        course_id=base_graph["course"].id,
        teacher_id=base_graph["teacher"].id,
        scheduled_at=UTC_NOW,
        room="Room 204",
        status=ClassSessionStatus.PROCESSING,
    )
    db_session.add(session_row)
    await db_session.flush()

    upload = VideoUpload(
        class_session_id=session_row.id,
        storage_uri="s3://attend-media/sessions/1/video.mp4",
        duration_seconds=62.0,
        width=3840,
        height=2160,
        fps=30.0,
        bytes=310_000_000,
    )
    db_session.add(upload)
    await db_session.flush()

    job = ProcessingJob(
        class_session_id=session_row.id,
        video_upload_id=upload.id,
        state=ProcessingJobState.RUNNING,
        stage="match",
        params_json="{}",
    )
    db_session.add(job)
    await db_session.flush()

    cluster = DetectedCluster(
        processing_job_id=job.id,
        representative_vector=b"\x00" * 2048,
        crop_count=42,
        mean_quality=0.74,
        best_crop_uri="s3://attend-media/jobs/1/clusters/0/best.jpg",
        retention_expires_at=UTC_NOW + timedelta(days=180),
    )
    db_session.add(cluster)
    await db_session.flush()

    match = ClusterMatch(
        cluster_id=cluster.id,
        student_id=base_graph["student"].id,
        similarity=0.51,
        runner_up_similarity=0.22,
        decision=ClusterMatchDecision.CONFIDENT,
    )
    db_session.add(match)
    await db_session.flush()

    result = await db_session.execute(select(ClusterMatch).where(ClusterMatch.id == match.id))
    fetched = result.scalar_one()
    assert fetched.decision == ClusterMatchDecision.CONFIDENT


async def test_attendance_record(db_session: AsyncSession, base_graph):
    session_row = ClassSession(
        course_id=base_graph["course"].id,
        teacher_id=base_graph["teacher"].id,
        scheduled_at=UTC_NOW,
        room="Room 204",
        status=ClassSessionStatus.COMMITTED,
    )
    db_session.add(session_row)
    await db_session.flush()

    record = AttendanceRecord(
        class_session_id=session_row.id,
        student_id=base_graph["student"].id,
        status=AttendanceStatus.PRESENT,
        source=AttendanceSource.AUTO,
        confidence=0.51,
    )
    db_session.add(record)
    await db_session.flush()

    result = await db_session.execute(select(AttendanceRecord).where(AttendanceRecord.id == record.id))
    fetched = result.scalar_one()
    assert fetched.status == AttendanceStatus.PRESENT
    assert fetched.supersedes_id is None

    # A correction: a NEW row referencing the old one, never an UPDATE.
    correction = AttendanceRecord(
        class_session_id=session_row.id,
        student_id=base_graph["student"].id,
        status=AttendanceStatus.ABSENT,
        source=AttendanceSource.TEACHER_OVERRIDE,
        decided_by_teacher_id=base_graph["teacher"].id,
        supersedes_id=record.id,
    )
    db_session.add(correction)
    await db_session.flush()
    assert correction.supersedes_id == record.id


async def test_audit_log(db_session: AsyncSession, base_graph):
    entry = AuditLog(
        actor_type="teacher",
        actor_id=str(base_graph["teacher"].id),
        action="attendance.commit",
        subject_type="class_session",
        subject_id="1",
        payload_json='{"present": 72, "absent": 18}',
    )
    db_session.add(entry)
    await db_session.flush()

    result = await db_session.execute(select(AuditLog).where(AuditLog.id == entry.id))
    fetched = result.scalar_one()
    assert fetched.action == "attendance.commit"
