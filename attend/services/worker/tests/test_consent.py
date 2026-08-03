"""Worker-side consent gate test (Phase 1 deliverable 7: "consent gate
rejects an unconsented student").

Uses an in-memory SQLite table standing in for the reflected `consent` table
db.py would normally hand back from a live Postgres connection -- monkey-
patching `consent.table` means this test never needs a real database, per
non-negotiable rule #1.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import Column, DateTime, Integer, MetaData, Table, create_engine, insert

import consent as consent_module
from consent import ConsentError, assert_consent_valid

UTC_NOW = datetime.now(timezone.utc)


@pytest.fixture
def consent_table(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    consent_tbl = Table(
        "consent",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("student_id", Integer, nullable=False),
        Column("granted_at", DateTime(timezone=True), nullable=False),
        Column("revoked_at", DateTime(timezone=True), nullable=True),
    )
    metadata.create_all(engine)

    monkeypatch.setattr(consent_module, "table", lambda name: consent_tbl)

    with engine.connect() as conn:
        yield conn, consent_tbl


def test_missing_consent_raises(consent_table):
    conn, _ = consent_table
    with pytest.raises(ConsentError) as exc_info:
        assert_consent_valid(conn, student_id=1)
    assert exc_info.value.code == "consent_missing"


def test_revoked_consent_raises(consent_table):
    conn, tbl = consent_table
    conn.execute(
        insert(tbl).values(
            student_id=1,
            granted_at=UTC_NOW - timedelta(days=10),
            revoked_at=UTC_NOW - timedelta(days=1),
        )
    )
    conn.commit()

    with pytest.raises(ConsentError) as exc_info:
        assert_consent_valid(conn, student_id=1)
    assert exc_info.value.code == "consent_revoked"


def test_valid_consent_passes(consent_table):
    conn, tbl = consent_table
    conn.execute(
        insert(tbl).values(
            student_id=1,
            granted_at=UTC_NOW - timedelta(days=1),
            revoked_at=None,
        )
    )
    conn.commit()

    assert_consent_valid(conn, student_id=1)  # must not raise
