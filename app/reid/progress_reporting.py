from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timezone
from typing import Any

from app.core.tracking_outcome import StaleAnalysisAttemptError

logger = logging.getLogger(__name__)

_profiles: dict[tuple[str, str | None], Any] = {}
_profiles_lock = threading.Lock()
_ACTIVE_ANALYSIS_STATUSES = frozenset({"QUEUED", "RUNNING", "PROCESSING"})


def _window_count(profile: Any) -> int:
    duration = float(getattr(profile, "duration_sec", 0.0) or 0.0)
    window_sec = float(getattr(profile, "window_sec", 45.0) or 45.0)
    overlap_sec = float(getattr(profile, "overlap_sec", 10.0) or 10.0)
    if duration <= 0:
        return 0
    if duration <= window_sec:
        return 1
    step = max(1.0, window_sec - overlap_sec)
    return 1 + int(math.ceil((duration - window_sec) / step))


def _profile_payload(profile: Any) -> dict[str, Any]:
    to_payload = getattr(profile, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        if isinstance(payload, dict):
            return payload
    return {
        "duration_sec": float(getattr(profile, "duration_sec", 0.0) or 0.0),
        "fps": int(getattr(profile, "fps", 1) or 1),
        "window_sec": float(getattr(profile, "window_sec", 45.0) or 45.0),
        "overlap_sec": float(getattr(profile, "overlap_sec", 10.0) or 10.0),
        "detector_model": str(getattr(profile, "detector_model", "unknown")),
        "target_samples": int(getattr(profile, "target_samples", 0) or 0),
        "estimated_samples": int(getattr(profile, "estimated_samples", 0) or 0),
    }


def _persist_progress(
    job_id: str,
    *,
    analysis_attempt_id: str | None,
    profile: Any,
    windows_completed: int,
    windows_total: int,
    window_progress_pct: float,
    message: str | None = None,
) -> None:
    try:
        from app.core.db import SessionLocal
        from app.core.models import AnalysisJob
        from sqlalchemy import select
    except Exception:
        logger.exception("Unable to import progress persistence")
        return

    db = None
    try:
        db = SessionLocal()
        execute = getattr(db, "execute", None)
        if callable(execute):
            statement = (
                select(AnalysisJob)
                .where(AnalysisJob.id == job_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            job = execute(statement).scalar_one_or_none()
        else:
            try:
                job = db.get(AnalysisJob, job_id, populate_existing=True)
            except TypeError:
                job = db.get(AnalysisJob, job_id)
        if not job:
            return
        target = job.target if isinstance(job.target, dict) else {}
        current_attempt_id = (
            str(target.get("analysis_attempt_id") or "").strip() or None
        )
        expected_attempt_id = str(analysis_attempt_id or "").strip() or None
        if current_attempt_id != expected_attempt_id:
            raise StaleAnalysisAttemptError(
                "ReID progress attempt differs from the current job target: "
                f"worker={expected_attempt_id or '<missing>'} "
                f"target={current_attempt_id or '<missing>'}"
            )
        status = str(job.status or "").strip().upper()
        if status not in _ACTIVE_ANALYSIS_STATUSES:
            raise StaleAnalysisAttemptError(
                "ReID progress cannot mutate a terminal or inactive job: "
                f"status={status or '<missing>'} "
                f"attempt={expected_attempt_id or '<missing>'}"
            )
        progress = dict(job.progress or {})
        stats = dict(progress.get("stats") or {})
        payload = _profile_payload(profile)
        stats.update(
            {
                "windows_completed": int(max(0, windows_completed)),
                "windows_total": int(max(0, windows_total)),
                "window_progress_pct": round(
                    max(0.0, min(100.0, window_progress_pct)), 1
                ),
                "tracking_fps": payload.get("fps"),
                "detector_model": payload.get("detector_model"),
                "estimated_samples": payload.get("estimated_samples"),
                "target_samples": payload.get("target_samples"),
                "window_sec": payload.get("window_sec"),
                "overlap_sec": payload.get("overlap_sec"),
            }
        )
        progress["stats"] = stats
        progress["runtime_profile"] = payload
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        if message:
            progress["message"] = message
        if expected_attempt_id is not None:
            progress["analysis_attempt_id"] = expected_attempt_id
        job.progress = progress
        db.commit()
    except StaleAnalysisAttemptError:
        if db is not None:
            db.rollback()
        raise
    except Exception:
        if db is not None:
            db.rollback()
        logger.exception("Unable to persist full-match progress job_id=%s", job_id)
    finally:
        if db is not None:
            db.close()


def begin_full_match_progress(
    job_id: str | None,
    profile: Any | None,
    *,
    analysis_attempt_id: str | None = None,
) -> None:
    if not job_id or profile is None:
        return
    total = _window_count(profile)
    _persist_progress(
        job_id,
        analysis_attempt_id=analysis_attempt_id,
        profile=profile,
        windows_completed=0,
        windows_total=total,
        window_progress_pct=0.0,
        message=f"Tracking player · 0/{total} finestre" if total else "Tracking player",
    )
    key = (job_id, str(analysis_attempt_id or "").strip() or None)
    with _profiles_lock:
        _profiles[key] = profile


def end_full_match_progress(
    job_id: str | None,
    *,
    analysis_attempt_id: str | None = None,
) -> None:
    if not job_id:
        return
    key = (job_id, str(analysis_attempt_id or "").strip() or None)
    with _profiles_lock:
        _profiles.pop(key, None)


def install_progress_stats_adapter(tracking_module: Any) -> None:
    current = getattr(tracking_module, "_update_tracking_progress", None)
    if not callable(current) or getattr(current, "__algonext_stats_adapter__", False):
        return

    def adapted(
        job_id: str,
        pct: int,
        message: str,
        *,
        analysis_attempt_id: str | None = None,
    ) -> Any:
        result = current(
            job_id,
            pct,
            message,
            analysis_attempt_id=analysis_attempt_id,
        )
        if message not in {
            "Tracking player with experimental ReID",
            "Tracking player (windowed)",
        }:
            return result
        key = (job_id, str(analysis_attempt_id or "").strip() or None)
        with _profiles_lock:
            profile = _profiles.get(key)
        if profile is None:
            return result
        ratio = max(0.0, min(1.0, (float(pct) - 10.0) / 30.0))
        total = _window_count(profile)
        completed = min(total, max(0, int(round(total * ratio)))) if total else 0
        exact_message = (
            f"{message} · {completed}/{total} finestre" if total else message
        )
        _persist_progress(
            job_id,
            analysis_attempt_id=analysis_attempt_id,
            profile=profile,
            windows_completed=completed,
            windows_total=total,
            window_progress_pct=ratio * 100.0,
            message=exact_message,
        )
        return result

    setattr(adapted, "__algonext_stats_adapter__", True)
    setattr(adapted, "__algonext_original_progress__", current)
    tracking_module._update_tracking_progress = adapted
