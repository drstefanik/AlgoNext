from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from app.core.deps import get_db
from app.core.models import AnalysisJob

router = APIRouter()


class PlayerProfilePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    player_name: Optional[str] = Field(
        default=None,
        alias="playerName",
        max_length=120,
    )
    team_name: Optional[str] = Field(
        default=None,
        alias="teamName",
        max_length=120,
    )
    shirt_number: Optional[int] = Field(
        default=None,
        alias="shirtNumber",
        ge=0,
        le=99,
    )

    @field_validator("player_name", "team_name", mode="before")
    @classmethod
    def normalize_optional_text(cls, value):
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


def _meta(request: Request) -> dict:
    return {
        "request_id": getattr(request.state, "request_id", None),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/jobs/{job_id}/player-profile")
def save_player_profile(
    job_id: str,
    payload: PlayerProfilePayload,
    request: Request,
    db: Session = Depends(get_db),
):
    """Attach descriptive data to the player chosen in the visual selection step.

    The bounding box and track remain the source of truth for identity. Name, team
    and shirt number are optional labels and never auto-select a candidate.
    """

    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": "Job not found",
                }
            },
        )

    target = dict(job.target or {})
    player = dict(target.get("player") or {})
    updates = payload.model_dump(exclude_unset=True, by_alias=False)
    for field_name, value in updates.items():
        player[field_name] = value

    target["player"] = player
    job.target = target

    if isinstance(job.player_ref, dict) and job.player_ref:
        player_ref = dict(job.player_ref)
        player_ref["profile"] = dict(player)
        job.player_ref = player_ref

    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)

    return {
        "ok": True,
        "data": {
            "job_id": job.id,
            "player": player,
        },
        "meta": _meta(request),
    }
