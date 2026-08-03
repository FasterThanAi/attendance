"""Consent enforcement (deliverable 8, Phase 0).

This is deliberately the smallest possible module: one function, one
exception. It must be imported and called at the START of every operation
that touches a specific student's face -- enrollment (Phase 1) and gallery
matching (Phase 6) both call `assert_consent_valid` before doing anything
else. See Phase 1's prompt: "calls assert_consent_valid(student_id) FIRST and
aborts if it fails."

Do not weaken this into a warning or a soft check. A missing or revoked
consent row is a hard stop.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Consent


class ConsentError(Exception):
    """Raised when a student has no valid consent on file.

    Carries a machine-readable `code` so callers (routers) can translate this
    into the structured error model required by non-negotiable rule #7,
    instead of leaking a raw exception string to an API response.
    """

    def __init__(self, student_id: int, code: str, message: str) -> None:
        self.student_id = student_id
        self.code = code
        super().__init__(message)


async def assert_consent_valid(db: AsyncSession, student_id: int) -> Consent:
    """Raise ConsentError unless the student has a granted, non-revoked consent.

    Returns the Consent row on success so callers can read consent_version /
    scope if needed, without a second query.
    """
    result = await db.execute(
        select(Consent)
        .where(Consent.student_id == student_id)
        .order_by(Consent.granted_at.desc())
        .limit(1)
    )
    consent = result.scalar_one_or_none()

    if consent is None:
        raise ConsentError(
            student_id=student_id,
            code="consent_missing",
            message=f"No consent record exists for student_id={student_id}.",
        )

    if consent.revoked_at is not None:
        raise ConsentError(
            student_id=student_id,
            code="consent_revoked",
            message=f"Consent for student_id={student_id} was revoked at {consent.revoked_at.isoformat()}.",
        )

    return consent
