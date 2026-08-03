from datetime import datetime

from pydantic import BaseModel


class EnrollmentQueuedResponse(BaseModel):
    student_id: int
    job_id: str
    message: str = "Enrollment video queued for processing."


class PoseCoverage(BaseModel):
    left: int
    frontal: int
    right: int


class EnrollmentStatusResponse(BaseModel):
    student_id: int
    total_embeddings: int
    pose_coverage: PoseCoverage
    gallery_updated_at: datetime | None
    is_sufficient: bool  # total_embeddings >= gallery_min_embeddings AND all 3 poses covered


class EnrollmentDeletedResponse(BaseModel):
    student_id: int
    photos_deleted: int
    embeddings_deleted: int
