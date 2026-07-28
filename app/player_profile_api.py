from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.analysis_attempt_precondition import require_analysis_attempt
from app.core.deps import get_db
from app.core.models import AnalysisJob

router = APIRouter()
ACTIVE_ANALYSIS_STATUSES = frozenset({"QUEUED", "RUNNING", "PROCESSING"})


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


def _has_visual_selection(player_ref: object) -> bool:
    if not isinstance(player_ref, dict):
        return False
    if player_ref.get("track_id") is None:
        return False
    if player_ref.get("selection_source") or player_ref.get("best_preview_frame_key"):
        return True
    bbox = player_ref.get("bbox")
    if isinstance(bbox, dict) and {"x", "y", "w", "h"}.issubset(bbox):
        return True
    if {"x", "y", "w", "h"}.issubset(player_ref):
        return True
    sample_frames = player_ref.get("sample_frames")
    return isinstance(sample_frames, list) and len(sample_frames) > 0


def _load_job_for_update(db: Session, job_id: str) -> AnalysisJob | None:
    execute = getattr(db, "execute", None)
    if callable(execute):
        statement = (
            select(AnalysisJob)
            .where(AnalysisJob.id == job_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return execute(statement).scalar_one_or_none()
    try:
        return db.get(AnalysisJob, job_id, populate_existing=True)
    except TypeError:
        return db.get(AnalysisJob, job_id)


def _analysis_attempt_id(job: AnalysisJob) -> str | None:
    target = job.target if isinstance(job.target, dict) else {}
    return str(target.get("analysis_attempt_id") or "").strip() or None


def _reject_active_analysis(job: AnalysisJob) -> None:
    status = str(job.status or "").upper()
    if status not in ACTIVE_ANALYSIS_STATUSES:
        return
    raise HTTPException(
        status_code=409,
        detail={
            "code": "ANALYSIS_IN_PROGRESS",
            "message": "Player details cannot change during an active analysis.",
            "details": {
                "status": status,
                "analysis_attempt_id": _analysis_attempt_id(job),
            },
        },
    )


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

    job = _load_job_for_update(db, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={"code": "JOB_NOT_FOUND", "message": "Job not found"},
        )
    require_analysis_attempt(
        job,
        request,
        mutation="player-profile",
    )
    _reject_active_analysis(job)
    if not _has_visual_selection(job.player_ref):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "PLAYER_SELECTION_REQUIRED",
                "message": "Select a player visually before saving player details.",
            },
        )

    target = dict(job.target or {})
    player = dict(target.get("player") or {})
    updates = payload.model_dump(exclude_unset=True, by_alias=False)
    for field_name, value in updates.items():
        player[field_name] = value

    target["player"] = player
    job.target = target

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
