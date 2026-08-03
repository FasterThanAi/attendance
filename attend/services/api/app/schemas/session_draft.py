"""GET /sessions/{id}/draft (Phase 6 deliverable 5).

Deliberately excludes raw embedding vectors from every shape here -- the
roadmap's rule for this endpoint, verbatim: never expose a face embedding
over the API, only URIs a teacher's browser can render (a cluster's best
crop, a student's enrollment photo) plus the scalar numbers (similarity,
margin) needed to explain a decision.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.db.models import ClassSessionStatus


class DraftSessionSummary(BaseModel):
    total_enrolled: int
    proposed_present: int
    needs_review: int
    proposed_absent: int
    unrecognised_clusters: int
    coverage_percent: float
    mean_confident_similarity: float | None
    session_health: str


class DraftClusterMatch(BaseModel):
    cluster_id: int
    best_crop_uri: str
    student_id: int | None
    student_name: str | None
    roll_number: str | None
    similarity: float | None
    runner_up_similarity: float | None
    enrollment_photo_uri: str | None


class DraftAbsentStudent(BaseModel):
    student_id: int
    student_name: str
    roll_number: str
    enrollment_photo_uri: str | None


class SessionDraftResponse(BaseModel):
    session_id: int
    status: ClassSessionStatus
    summary: DraftSessionSummary
    confident: list[DraftClusterMatch]
    needs_review: list[DraftClusterMatch]
    proposed_absent: list[DraftAbsentStudent]
    unrecognised_clusters: list[DraftClusterMatch]
