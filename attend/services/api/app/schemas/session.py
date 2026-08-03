from datetime import datetime

from pydantic import BaseModel

from app.db.models import ClassSessionStatus


class ClassSessionCreate(BaseModel):
    course_id: int
    teacher_id: int
    scheduled_at: datetime
    room: str


class ClassSessionResponse(BaseModel):
    id: int
    course_id: int
    teacher_id: int
    scheduled_at: datetime
    room: str
    status: ClassSessionStatus

    model_config = {"from_attributes": True}
