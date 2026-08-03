from pydantic import BaseModel


class DependencyStatus(BaseModel):
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    service: str
    version: str
    environment: str
    database: DependencyStatus
    redis: DependencyStatus
