"""Sync mirror of services/api/app/services/consent.py's business rule, for
the worker process. See db.py's docstring for why this is a separate,
flagged duplication rather than a shared import.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Connection

from db import table


class ConsentError(Exception):
    def __init__(self, student_id: int, code: str, message: str) -> None:
        self.student_id = student_id
        self.code = code
        super().__init__(message)


def assert_consent_valid(conn: Connection, student_id: int) -> None:
    """Raise ConsentError unless the student has a granted, non-revoked
    consent row. Must be called FIRST, before any other enrollment work --
    see enrollment.py.
    """
    consent = table("consent")
    row = conn.execute(
        select(consent.c.revoked_at)
        .where(consent.c.student_id == student_id)
        .order_by(consent.c.granted_at.desc())
        .limit(1)
    ).first()

    if row is None:
        raise ConsentError(student_id, "consent_missing", f"No consent record exists for student_id={student_id}.")

    (revoked_at,) = row
    if revoked_at is not None:
        raise ConsentError(student_id, "consent_revoked", f"Consent for student_id={student_id} was revoked.")
