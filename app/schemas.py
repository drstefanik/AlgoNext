from typing import Optional, Dict
from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    video_url: str
    role: str
    category: str
    shirt_number: Optional[int] = Field(default=None, ge=0, le=99)

class JobOut(BaseModel):
    job_id: str
    status: str

class JobStatusOut(BaseModel):
    id: str
    status: str
    progress: Dict
    error: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]