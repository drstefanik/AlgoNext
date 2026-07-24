from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FullMatchRuntimeProfile:
    duration_sec: float
    fps: int
    window_sec: float
    overlap_sec: float
    detector_model: str
    target_samples: int
    estimated_samples: int

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def select_full_match_profile(
    *,
    video_duration_sec: Any,
    requested_fps: Any = 5,
    requested_window_sec: Any = 45.0,
    requested_overlap_sec: Any = 10.0,
    requested_detector_model: Any = "yolo11s.pt",
) -> FullMatchRuntimeProfile:
    """Choose a bounded CPU profile for a full-match tracking run."""

    duration = max(0.0, _safe_float(video_duration_sec, 0.0))
    requested_fps_value = max(1, int(round(_safe_float(requested_fps, 5.0))))
    requested_window = max(5.0, _safe_float(requested_window_sec, 45.0))
    requested_overlap = max(0.0, _safe_float(requested_overlap_sec, 10.0))
    requested_overlap = min(requested_overlap, requested_window - 1.0)
    requested_model = str(requested_detector_model or "yolo11s.pt").strip()

    target_samples = _env_int("FULL_MATCH_TARGET_SAMPLES", 6000, 1000, 50000)

    if duration < 900.0:
        fps = requested_fps_value
        window_sec = requested_window
        overlap_sec = requested_overlap
        detector_model = requested_model
    else:
        minimum_fps = _env_int("FULL_MATCH_MIN_FPS", 1, 1, 10)
        maximum_fps = _env_int("FULL_MATCH_MAX_FPS", 2, minimum_fps, 10)
        budget_fps = max(
            minimum_fps,
            int(math.floor(target_samples / max(1.0, duration))),
        )
        fps = max(
            minimum_fps,
            min(requested_fps_value, maximum_fps, budget_fps),
        )
        forced_fps = (os.environ.get("FULL_MATCH_TRACKING_FPS") or "").strip()
        if forced_fps:
            fps = _env_int(
                "FULL_MATCH_TRACKING_FPS",
                fps,
                minimum_fps,
                maximum_fps,
            )

        window_sec = _env_float("FULL_MATCH_WINDOW_SEC", 60.0, 20.0, 300.0)
        overlap_sec = _env_float(
            "FULL_MATCH_OVERLAP_SEC",
            5.0,
            0.0,
            max(0.0, window_sec - 1.0),
        )
        detector_model = (
            os.environ.get("FULL_MATCH_DETECTOR_MODEL") or "yolo11n.pt"
        ).strip() or "yolo11n.pt"

    step_sec = max(1.0, window_sec - overlap_sec)
    overlap_multiplier = window_sec / step_sec
    estimated_samples = (
        int(math.ceil(duration * float(fps) * overlap_multiplier))
        if duration > 0
        else 0
    )

    return FullMatchRuntimeProfile(
        duration_sec=round(duration, 3),
        fps=fps,
        window_sec=round(window_sec, 3),
        overlap_sec=round(overlap_sec, 3),
        detector_model=detector_model,
        target_samples=target_samples,
        estimated_samples=estimated_samples,
    )


def budget_full_match_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], FullMatchRuntimeProfile | None]:
    if "video_duration_sec" not in kwargs:
        return dict(kwargs), None

    profile = select_full_match_profile(
        video_duration_sec=kwargs.get("video_duration_sec"),
        requested_fps=kwargs.get("fps", 5),
        requested_window_sec=kwargs.get("window_sec", 45.0),
        requested_overlap_sec=kwargs.get("overlap_sec", 10.0),
        requested_detector_model=kwargs.get("detector_model", "yolo11s.pt"),
    )
    updated = dict(kwargs)
    updated.update(
        {
            "fps": profile.fps,
            "window_sec": profile.window_sec,
            "overlap_sec": profile.overlap_sec,
            "detector_model": profile.detector_model,
        }
    )
    return updated, profile


def install_progress_adapter(tracking_module: Any) -> None:
    current = getattr(tracking_module, "_update_tracking_progress", None)
    if not callable(current) or getattr(current, "__algonext_progress_adapter__", False):
        return

    def adapted(job_id: str, pct: int, message: str) -> Any:
        mapped_pct = int(pct)
        mapped_message = message
        if message in {
            "Tracking player with experimental ReID",
            "Tracking player (windowed)",
        }:
            stage_ratio = max(0.0, min(1.0, (float(pct) - 10.0) / 30.0))
            mapped_pct = 35 + int(round(stage_ratio * 35.0))
            mapped_message = (
                f"{message} · {int(round(stage_ratio * 100.0))}% finestre"
            )
        return current(job_id, mapped_pct, mapped_message)

    setattr(adapted, "__algonext_progress_adapter__", True)
    setattr(adapted, "__algonext_original_progress__", current)
    tracking_module._update_tracking_progress = adapted


def _estimate_window_count(profile: FullMatchRuntimeProfile | None) -> int:
    if profile is None or profile.duration_sec <= 0:
        return 0
    step = max(1.0, profile.window_sec - profile.overlap_sec)
    return max(1, int(math.ceil(profile.duration_sec / step)))


def partial_timeout_output(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    profile: FullMatchRuntimeProfile | None,
) -> dict[str, Any]:
    duration = (
        profile.duration_sec
        if profile is not None
        else max(0.0, _safe_float(kwargs.get("video_duration_sec"), 0.0))
    )
    fps = profile.fps if profile is not None else int(kwargs.get("fps") or 1)
    window_sec = (
        profile.window_sec
        if profile is not None
        else _safe_float(kwargs.get("window_sec"), 45.0)
    )
    overlap_sec = (
        profile.overlap_sec
        if profile is not None
        else _safe_float(kwargs.get("overlap_sec"), 10.0)
    )
    player_ref = args[2] if len(args) > 2 else None
    selections = args[3] if len(args) > 3 else []

    return {
        "mode": "full_match_windowed",
        "identity_mode": "appearance_reid_v1",
        "method": "yolo+bytetrack+appearance_reid",
        "fps": fps,
        "window_sec": window_sec,
        "overlap_sec": overlap_sec,
        "segments": [],
        "segments_total": _estimate_window_count(profile),
        "segments_with_player": 0,
        "coverage_pct_total": 0.0,
        "largest_gap_sec": round(duration, 2),
        "coverage_pct": 0.0,
        "bboxes": [],
        "lost_segments": [],
        "anchors_used": {"player_ref": player_ref, "selections": selections},
        "partial": True,
        "partial_reason": "TRACKING_TIMEOUT",
        "notes": (
            "The full-match tracking budget was exhausted before a complete "
            "tracking artifact was produced. The pipeline continued with "
            "partial diagnostics and no player score."
        ),
        "runtime_profile": profile.to_payload() if profile is not None else None,
        "reid_summary": {
            "status": "PARTIAL_TIMEOUT",
            "validated": False,
            "reason_codes": ["TRACKING_BUDGET_EXHAUSTED"],
        },
    }


def mark_partial_timeout(
    job_id: str | None,
    profile: FullMatchRuntimeProfile | None,
) -> None:
    if not job_id:
        return

    try:
        from app.core.db import SessionLocal
        from app.core.models import AnalysisJob
        from app.core.normalizers import normalize_failure_reason
    except Exception:
        logger.exception("Unable to import job persistence for partial tracking timeout")
        return

    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        if not job:
            return
        warnings = [
            code for code in list(job.warnings or []) if code != "TRACKING_TIMEOUT"
        ]
        if "TRACKING_PARTIAL_TIMEOUT" not in warnings:
            warnings.append("TRACKING_PARTIAL_TIMEOUT")
        job.warnings = warnings
        job.status = "RUNNING"
        job.error = None
        job.failure_reason = normalize_failure_reason(None)
        job.progress = {
            **(job.progress or {}),
            "step": "TRACKING_PARTIAL",
            "pct": 70,
            "message": (
                "Tracking budget exhausted; continuing with partial diagnostics"
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runtime_profile": profile.to_payload() if profile is not None else None,
        }
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Unable to convert tracking timeout into partial result")
    finally:
        db.close()
