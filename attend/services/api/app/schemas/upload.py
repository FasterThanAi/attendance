from datetime import datetime

from pydantic import BaseModel


class UploadCreateRequest(BaseModel):
    class_session_id: int
    filename: str
    total_size_bytes: int


class UploadCreateResponse(BaseModel):
    upload_id: str
    chunk_size_bytes: int
    total_chunks: int


class UploadChunkResponse(BaseModel):
    upload_id: str
    chunk_index: int
    received: bool = True


class UploadStatusResponse(BaseModel):
    upload_id: str
    total_chunks: int
    received_chunks: list[int]
    is_complete: bool


class PreflightCheckResult(BaseModel):
    code: str
    severity: str  # "info" | "warn" | "fail"
    message: str


class PreflightResult(BaseModel):
    status: str  # "pass" | "warn" | "fail"
    checks: list[PreflightCheckResult]


class VideoUploadResponse(BaseModel):
    id: int
    class_session_id: int
    storage_uri: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    bytes: int
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class UploadCompleteResponse(BaseModel):
    video_upload: VideoUploadResponse
    preflight: PreflightResult
