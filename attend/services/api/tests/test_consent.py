from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Consent, Department, Institution, Student
from app.services.consent import ConsentError, assert_consent_valid

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
        roll_number="CSE002",
        full_name="Bala Krishnan",
        admission_year=2023,
        is_active=True,
    )
    db_session.add(student)
    await db_session.flush()
    return student


async def test_missing_consent_raises(db_session: AsyncSession, student: Student):
    with pytest.raises(ConsentError) as exc_info:
        await assert_consent_valid(db_session, student.id)
    assert exc_info.value.code == "consent_missing"


async def test_revoked_consent_raises(db_session: AsyncSession, student: Student):
    consent = Consent(
        student_id=student.id,
        granted_at=UTC_NOW - timedelta(days=10),
        revoked_at=UTC_NOW - timedelta(days=1),
        consent_version="v1",
        scope="classroom attendance via face recognition",
        evidence_uri="s3://attend-media/consent/student-2.pdf",
    )
    db_session.add(consent)
    await db_session.flush()

    with pytest.raises(ConsentError) as exc_info:
        await assert_consent_valid(db_session, student.id)
    assert exc_info.value.code == "consent_revoked"


async def test_valid_consent_passes(db_session: AsyncSession, student: Student):
    consent = Consent(
        student_id=student.id,
        granted_at=UTC_NOW - timedelta(days=1),
        revoked_at=None,
        consent_version="v1",
        scope="classroom attendance via face recognition",
        evidence_uri="s3://attend-media/consent/student-2.pdf",
    )
    db_session.add(consent)
    await db_session.flush()

    result = await assert_consent_valid(db_session, student.id)
    assert result.consent_version == "v1"


async def test_most_recent_consent_wins(db_session: AsyncSession, student: Student):
    """If a student has an old revoked consent and a newer valid one (e.g. they
    revoked, then re-enrolled and re-consented), the newer grant must win.
    """
    old = Consent(
        student_id=student.id,
        granted_at=UTC_NOW - timedelta(days=30),
        revoked_at=UTC_NOW - timedelta(days=20),
        consent_version="v1",
        scope="classroom attendance via face recognition",
        evidence_uri="s3://attend-media/consent/student-2-old.pdf",
    )
    new = Consent(
        student_id=student.id,
        granted_at=UTC_NOW - timedelta(days=5),
        revoked_at=None,
        consent_version="v2",
        scope="classroom attendance via face recognition",
        evidence_uri="s3://attend-media/consent/student-2-new.pdf",
    )
    db_session.add_all([old, new])
    await db_session.flush()

    result = await assert_consent_valid(db_session, student.id)
    assert result.consent_version == "v2"
