from typing import Optional, Dict

from pydantic import BaseModel, Field, conlist, model_validator, ConfigDict


class JobCreate(BaseModel):
    video_url: Optional[str] = None
    video_bucket: Optional[str] = None
    video_key: Optional[str] = None
    role: str
    category: str
    team_name: str = Field(min_length=1)
    player_name: Optional[str] = None
    shirt_number: Optional[int] = Field(default=None, ge=0, le=99)

    @model_validator(mode="after")
    def require_video_source(self) -> "JobCreate":
        if not self.video_url and not self.video_key:
            raise ValueError("video_url or video_key is required")
        if self.video_url and self.video_key:
            raise ValueError("Provide either video_url or video_key, not both")
        return self

class JobOut(BaseModel):
    job_id: str
    status: str

class JobStatusOut(BaseModel):
    job_id: str
    id: Optional[str] = None
    status: str
    progress: Dict
    error: Optional[str]
    failure_reason: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class SelectionBox(BaseModel):
    frame_time_sec: float = Field(ge=0)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class SelectionPayload(BaseModel):
    selections: conlist(SelectionBox, min_length=1, max_length=5)


class PlayerRefPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    t: float = Field(ge=0, alias="frameTimeSec")
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)
