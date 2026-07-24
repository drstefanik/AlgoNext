from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_profiles: dict[str, Any] = {}
_profiles_lock = threading.Lock()


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
    profile: Any,
    windows_completed: int,
    windows_total: int,
    window_progress_pct: float,
    message: str | None = None,
) -> None:
    try:
        from app.core.db import SessionLocal
        from app.core.models import AnalysisJob
    except Exception:
        logger.exception("Unable to import progress persistence")
        return

    db = None
    try:
        db = SessionLocal()
        job = db.get(AnalysisJob, job_id)
        if not job:
            return
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
        job.progress = progress
        db.commit()
    except Exception:
        if db is not None:
            db.rollback()
        logger.exception("Unable to persist full-match progress job_id=%s", job_id)
    finally:
        if db is not None:
            db.close()


def begin_full_match_progress(job_id: str | None, profile: Any | None) -> None:
    if not job_id or profile is None:
        return
    with _profiles_lock:
        _profiles[job_id] = profile
    total = _window_count(profile)
    _persist_progress(
        job_id,
        profile=profile,
        windows_completed=0,
        windows_total=total,
        window_progress_pct=0.0,
        message=f"Tracking player · 0/{total} finestre" if total else "Tracking player",
    )


def end_full_match_progress(job_id: str | None) -> None:
    if not job_id:
        return
    with _profiles_lock:
        _profiles.pop(job_id, None)


def install_progress_stats_adapter(tracking_module: Any) -> None:
    current = getattr(tracking_module, "_update_tracking_progress", None)
    if not callable(current) or getattr(current, "__algonext_stats_adapter__", False):
        return

    def adapted(job_id: str, pct: int, message: str) -> Any:
        result = current(job_id, pct, message)
        if message not in {
            "Tracking player with experimental ReID",
            "Tracking player (windowed)",
        }:
            return result
        with _profiles_lock:
            profile = _profiles.get(job_id)
        if profile is None:
            return result
        ratio = max(0.0, min(1.0, (float(pct) - 10.0) / 30.0))
        total = _window_count(profile)
        completed = min(total, max(0, int(round(total * ratio)))) if total else 0
        exact_message = f"{message} · {completed}/{total} finestre" if total else message
        _persist_progress(
            job_id,
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
