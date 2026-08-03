from datetime import datetime

from pydantic import BaseModel

from app.db.models import ProcessingJobState


class ProcessRequest(BaseModel):
    video_upload_id: int | None = None  # defaults to the session's most recent upload


class ProcessingJobResponse(BaseModel):
    id: int
    class_session_id: int
    video_upload_id: int
    state: ProcessingJobState
    stage: str | None
    error_text: str | None
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}
