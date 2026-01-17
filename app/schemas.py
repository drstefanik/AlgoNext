from typing import Optional, Dict, List, Literal

from pydantic import BaseModel, Field, conlist, root_validator


class JobCreate(BaseModel):
    video_url: Optional[str] = None
    video_key: Optional[str] = None
    role: str
    category: str
    team_name: str = Field(min_length=1)
    player_name: Optional[str] = None
    shirt_number: Optional[int] = Field(default=None, ge=0, le=99)

    @root_validator
    def require_video_source(cls, values: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        if not values.get("video_url") and not values.get("video_key"):
            raise ValueError("video_url or video_key is required")
        return values

class JobOut(BaseModel):
    job_id: str
    status: str

class JobStatusOut(BaseModel):
    id: str
    status: str
    progress: Dict
    error: Optional[str]
    failure_reason: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class SelectionBox(BaseModel):
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    t: float = Field(ge=0)


class SelectionPayload(BaseModel):
    selections: conlist(SelectionBox, min_length=2, max_length=5)


class PlayerRefBBox(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class PlayerRefPayload(BaseModel):
    frame_key: str = Field(min_length=1)
    time_sec: float = Field(ge=0)
    bbox: PlayerRefBBox
    team_hint: Optional[Literal["home", "away", "unknown"]] = None
    shirt_number: Optional[int] = Field(default=None, ge=0, le=99)
