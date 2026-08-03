"""Phase 8: commit + correction request/response shapes.

Two endpoints use these: POST /sessions/{id}/commit and POST
/sessions/{id}/attendance/{student_id}/correct (see
app/routers/attendance.py, app/services/attendance.py).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, field_validator

from app.db.models import AttendanceSource, AttendanceStatus, ClassSessionStatus


class CommitDecision(BaseModel):
    """One explicit teacher decision, sent only for students the teacher
    actually acted on (Phase 8 deliverable 2, "progressive review" -- a
    student the teacher never touched gets the system's own AUTO default,
    computed server-side, not sent by the client).
    """

    student_id: int
    status: AttendanceStatus
    source: AttendanceSource

    @field_validator("source")
    @classmethod
    def source_must_be_a_human_decision(cls, v: AttendanceSource) -> AttendanceSource:
        if v == AttendanceSource.AUTO:
            raise ValueError(
                "CommitDecision.source must be teacher_confirmed or teacher_override -- "
                "'auto' is a server-computed default for untouched students, never something "
                "the client asserts about its own decision."
            )
        return v


class CommitRequest(BaseModel):
    # Client-generated (a UUID is the expected shape, not enforced here --
    # any stable, unique-per-attempt string works). Phase 8 deliverable 3:
    # "the commit endpoint must be idempotent on a client-supplied request
    # id, so a double tap on a bad network cannot double-commit."
    request_id: str
    teacher_id: int
    decisions: list[CommitDecision] = []


class CommitCounts(BaseModel):
    total_enrolled: int
    present: int
    absent: int
    auto_count: int
    teacher_confirmed_count: int
    teacher_override_count: int


class CommitResponse(BaseModel):
    class_session_id: int
    status: ClassSessionStatus
    counts: CommitCounts
    committed_at: datetime
    # True when this response is a REPLAY of an already-completed commit
    # (the same request_id was seen before) -- not a signal of anything
    # wrong, just honesty about what happened, useful for the frontend to
    # decide whether to show "committed!" vs "already committed."
    idempotent_replay: bool


class CorrectionRequest(BaseModel):
    status: AttendanceStatus
    teacher_id: int


class CorrectionResponse(BaseModel):
    attendance_record_id: int
    student_id: int
    status: AttendanceStatus
    supersedes_id: int
    created_at: datetime
