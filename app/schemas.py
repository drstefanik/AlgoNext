from typing import Optional, Dict, List

from pydantic import BaseModel, Field, conlist


class JobCreate(BaseModel):
    video_url: str
    role: str
    category: str
    team_name: str = Field(min_length=1)
    player_name: Optional[str] = None
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


class SelectionBox(BaseModel):
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    t: float = Field(ge=0)


class SelectionPayload(BaseModel):
    selections: conlist(SelectionBox, min_items=2, max_items=5)
