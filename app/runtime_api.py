from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.analysis_attempt_precondition import require_analysis_attempt
from app.core.deps import get_db
from app.core.models import AnalysisJob
from app.core.normalizers import normalize_failure_reason
from app.core.runtime_health import build_metadata, inspect_runtime

logger = logging.getLogger(__name__)
router = APIRouter()

ACTIVE_STATUSES = {"QUEUED", "RUNNING", "PROCESSING"}
SUCCESS_STATUSES = {"DONE", "COMPLETED", "PARTIAL"}
PRESERVED_RESULT_KEYS = {
    "candidates",
    "framesProcessed",
    "totalTracks",
    "rawTracks",
    "primaryCount",
    "secondaryCount",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _meta(request: Request) -> dict[str, Any]:
    return {
        "request_id": getattr(request.state, "request_id", None),
        "timestamp": _now().isoformat(),
        "revision": build_metadata("algonext-api")["revision"],
    }


def _ok(data: dict[str, Any], request: Request) -> dict[str, Any]:
    return {"ok": True, "data": data, "meta": _meta(request)}


def _error_detail(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return payload


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


def _has_player_ref(player_ref: Any) -> bool:
    if not isinstance(player_ref, dict) or not player_ref:
        return False
    if player_ref.get("track_id") is not None:
        return True
    if {"t", "x", "y", "w", "h"}.issubset(player_ref):
        return True
    bbox = player_ref.get("bbox")
    return isinstance(bbox, dict) and {"x", "y", "w", "h"}.issubset(bbox)


def _retry_history(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    raw = result.get("retry_history")
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def _retry_count(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    reported = result.get("retry_count")
    try:
        reported_count = max(0, int(reported or 0))
    except (TypeError, ValueError):
        reported_count = 0
    history_attempts = []
    for entry in _retry_history(result):
        try:
            history_attempts.append(max(0, int(entry.get("attempt") or 0)))
        except (TypeError, ValueError):
            continue
    return max([reported_count, len(_retry_history(result)), *history_attempts])


def _input_video_asset(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    assets = result.get("assets")
    if not isinstance(assets, dict):
        return None
    input_video = assets.get("input_video")
    return dict(input_video) if isinstance(input_video, dict) else None


def _preserve_retry_inputs(
    result: Any, history_entry: dict[str, Any]
) -> dict[str, Any]:
    source = result if isinstance(result, dict) else {}
    preserved = {key: source[key] for key in PRESERVED_RESULT_KEYS if key in source}

    input_video = _input_video_asset(source)
    if input_video is not None:
        preserved["assets"] = {"input_video": input_video}

    history = _retry_history(source)
    history.append(history_entry)
    preserved["retry_history"] = history[-10:]
    preserved["retry_count"] = int(history_entry["attempt"])
    return preserved


def _require_worker_ready() -> dict[str, Any]:
    snapshot = inspect_runtime()
    if snapshot.get("ready") is True:
        return snapshot
    raise HTTPException(
        status_code=503,
        detail=_error_detail(
            "WORKER_NOT_READY",
            "The analysis worker is not ready for a new run.",
            {
                "dependencies": snapshot.get("dependencies"),
                "worker": snapshot.get("worker"),
                "worker_age_seconds": snapshot.get("worker_age_seconds"),
                "worker_revision_matches_api": snapshot.get(
                    "worker_revision_matches_api"
                ),
            },
        ),
    )


@router.get("/runtime", include_in_schema=False)
def runtime(request: Request):
    snapshot = inspect_runtime()
    return _ok(
        {
            **build_metadata("algonext-api"),
            **snapshot,
        },
        request,
    )


@router.post("/jobs/{job_id}/retry")
def retry_job(
    job_id: str,
    request: Request,
    payload: dict | None = Body(default=None),
    db: Session = Depends(get_db),
):
    job = _load_job_for_update(db, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=_error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    request_payload = payload if isinstance(payload, dict) else {}
    force = bool(request_payload.get("force"))
    current_status = str(job.status or "").upper()
    current_retry_count = _retry_count(job.result)
    current_analysis_attempt_id = require_analysis_attempt(
        job,
        request,
        request_payload,
        mutation="retry",
    )

    if current_status in ACTIVE_STATUSES:
        return _ok(
            {
                "job_id": job.id,
                "id": job.id,
                "status": current_status,
                "retry_count": current_retry_count,
                "already_active": True,
                "analysis_attempt_id": (current_analysis_attempt_id),
            },
            request,
        )

    if current_status in SUCCESS_STATUSES and not force:
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "RETRY_NOT_ALLOWED",
                "Completed jobs are not retried unless force=true.",
                {"status": current_status},
            ),
        )

    if current_status != "FAILED" and not force:
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "RETRY_NOT_ALLOWED",
                "Only failed jobs can be retried.",
                {"status": current_status},
            ),
        )

    missing: list[str] = []
    if not _has_player_ref(job.player_ref):
        missing.append("player_ref")
    if not bool((job.target or {}).get("confirmed")):
        missing.append("target")
    if missing:
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "RETRY_NOT_READY",
                "The saved player and target are required before retrying.",
                {"missing": missing},
            ),
        )

    try:
        max_retries = max(1, int(os.environ.get("MAX_JOB_RETRIES", "3") or 3))
    except ValueError:
        max_retries = 3
    if current_retry_count >= max_retries and not force:
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "RETRY_LIMIT_REACHED",
                "The retry limit for this job has been reached.",
                {
                    "retry_count": current_retry_count,
                    "max_retries": max_retries,
                },
            ),
        )

    runtime_snapshot = _require_worker_ready()
    retry_id = str(uuid4())
    analysis_attempt_id = str(uuid4())
    next_retry_count = current_retry_count + 1
    now = _now()
    previous_target = job.target if isinstance(job.target, dict) else {}
    history_entry = {
        "attempt": next_retry_count,
        "retry_id": retry_id,
        "analysis_attempt_id": analysis_attempt_id,
        "previous_analysis_attempt_id": (
            str(previous_target.get("analysis_attempt_id") or "").strip() or None
        ),
        "requested_at": now.isoformat(),
        "previous_status": current_status,
        "previous_failure_reason": job.failure_reason,
        "previous_error": job.error,
        "previous_progress": dict(job.progress or {}),
        "worker_revision": (runtime_snapshot.get("worker") or {}).get("revision"),
    }

    input_video = _input_video_asset(job.result)
    if not job.video_key and input_video and input_video.get("key"):
        job.video_key = str(input_video["key"])
        if input_video.get("bucket"):
            job.video_bucket = str(input_video["bucket"])

    job.result = {
        **_preserve_retry_inputs(job.result, history_entry),
        "analysis_attempt_id": analysis_attempt_id,
    }
    job.status = "QUEUED"
    job.error = None
    job.failure_reason = normalize_failure_reason(None)
    job.warnings = []
    job.ai_report = None
    job.report = None
    job.report_status = "PENDING"
    job.report_error = None

    target = dict(job.target or {})
    target["analysis_attempt_id"] = analysis_attempt_id
    target["tracking"] = {
        "status": "PENDING",
        "retry_id": retry_id,
        "analysis_attempt_id": analysis_attempt_id,
    }
    job.target = target
    job.progress = {
        "step": "QUEUED",
        "phase": "QUEUE",
        "pct": 20,
        "message": "Retry queued",
        "updated_at": now.isoformat(),
        "retry_count": next_retry_count,
        "retry_id": retry_id,
        "analysis_attempt_id": analysis_attempt_id,
        "worker_revision": (runtime_snapshot.get("worker") or {}).get("revision"),
    }
    job.updated_at = now
    db.commit()
    db.refresh(job)

    try:
        from app.workers.pipeline import run_analysis

        run_analysis.delay(job.id, analysis_attempt_id)
    except Exception as exc:
        logger.exception(
            "Retry dispatch raised; reconciling delivery state job_id=%s",
            job.id,
        )
        db.rollback()
        expire_all = getattr(db, "expire_all", None)
        if callable(expire_all):
            expire_all()

        current_job = _load_job_for_update(db, job.id)
        if current_job is None:
            raise HTTPException(
                status_code=503,
                detail=_error_detail(
                    "RETRY_ENQUEUE_FAILED", "Unable to enqueue the retry."
                ),
            ) from exc

        current_target = (
            current_job.target if isinstance(current_job.target, dict) else {}
        )
        current_progress = (
            current_job.progress if isinstance(current_job.progress, dict) else {}
        )
        current_attempt_id = (
            str(current_target.get("analysis_attempt_id") or "").strip() or None
        )
        progress_attempt_id = (
            str(current_progress.get("analysis_attempt_id") or "").strip() or None
        )
        analysis_task_id = (
            str(current_progress.get("analysis_task_id") or "").strip() or None
        )
        current_status = str(current_job.status or "").upper()
        still_unclaimed_attempt = (
            current_attempt_id == analysis_attempt_id
            and progress_attempt_id == analysis_attempt_id
            and current_status == "QUEUED"
            and analysis_task_id is None
        )

        if not still_unclaimed_attempt:
            logger.warning(
                "Retry dispatch outcome is ambiguous; preserving authoritative "
                "job state job_id=%s expected_attempt=%s current_attempt=%s "
                "status=%s analysis_task_id=%s",
                current_job.id,
                analysis_attempt_id,
                current_attempt_id,
                current_status,
                analysis_task_id,
            )
            return _ok(
                {
                    "job_id": current_job.id,
                    "id": current_job.id,
                    "status": current_status,
                    "retry_count": _retry_count(current_job.result),
                    "retry_id": retry_id,
                    "analysis_attempt_id": current_attempt_id,
                    "worker_revision": (runtime_snapshot.get("worker") or {}).get(
                        "revision"
                    ),
                    "dispatch_ambiguous": True,
                },
                request,
            )

        failed_at = _now()
        current_job.status = "FAILED"
        current_job.error = f"Retry enqueue failed: {exc}"
        current_job.failure_reason = normalize_failure_reason("RETRY_ENQUEUE_FAILED")
        current_job.progress = {
            **current_progress,
            "step": "FAILED",
            "phase": "QUEUE",
            "pct": 100,
            "message": "Retry enqueue failed",
            "updated_at": failed_at.isoformat(),
        }
        current_job.updated_at = failed_at
        db.commit()
        raise HTTPException(
            status_code=503,
            detail=_error_detail(
                "RETRY_ENQUEUE_FAILED", "Unable to enqueue the retry."
            ),
        ) from exc

    return _ok(
        {
            "job_id": job.id,
            "id": job.id,
            "status": job.status,
            "retry_count": next_retry_count,
            "retry_id": retry_id,
            "analysis_attempt_id": analysis_attempt_id,
            "worker_revision": (runtime_snapshot.get("worker") or {}).get("revision"),
        },
        request,
    )
