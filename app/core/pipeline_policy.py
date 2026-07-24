from __future__ import annotations

import logging
import math
from typing import Any, Callable

logger = logging.getLogger(__name__)

ANALYSIS_STATUSES = {"QUEUED", "RUNNING", "PROCESSING", "PARTIAL"}
PRESERVED_PROGRESS_KEYS = {
    "stats",
    "runtime_profile",
    "retry_count",
    "retry_id",
    "worker_revision",
    "interrupted_progress",
}

PHASES = {
    "STARTING": "QUEUE",
    "DOWNLOADING": "PREPARE",
    "PROBING": "PREPARE",
    "UPLOADING_INPUT": "PREPARE",
    "EXTRACTING_FRAMES": "PREPARE",
    "PREVIEWS_READY": "PREPARE",
    "TRACKING_CANDIDATES": "DETECTION",
    "TRACKING": "TRACKING",
    "TRACKING_PARTIAL": "TRACKING",
    "EXTRACTING_FEATURES": "FEATURES",
    "EXTRACTING": "CLIPS",
    "UPLOADING_CLIPS": "CLIPS",
    "ANALYZING": "ANALYSIS",
    "FINALIZING": "FINALIZE",
    "DONE": "DONE",
    "FAILED": "FAILED",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_analysis_run(job: Any) -> bool:
    status = str(getattr(job, "status", "") or "").upper()
    target = getattr(job, "target", None)
    return status in ANALYSIS_STATUSES and isinstance(target, dict) and bool(
        target.get("confirmed")
    )


def _analysis_progress(step: str, pct: int) -> int:
    normalized = str(step or "").upper()
    if normalized == "STARTING":
        return 20
    if normalized == "DOWNLOADING":
        source = max(10, min(18, pct))
        return 21 + int(round(((source - 10) / 8.0) * 3.0))
    fixed = {
        "PROBING": 25,
        "UPLOADING_INPUT": 27,
        "EXTRACTING_FRAMES": 29,
        "PREVIEWS_READY": 30,
        "TRACKING_CANDIDATES": 32,
        "TRACKING": max(35, pct),
        "TRACKING_PARTIAL": max(70, pct),
        "EXTRACTING_FEATURES": 72,
        "EXTRACTING": 78,
        "UPLOADING_CLIPS": 84,
        "ANALYZING": 90,
        "FINALIZING": 96,
        "DONE": 100,
        "FAILED": 100,
    }
    return fixed.get(normalized, pct)


def _parse_rate(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        try:
            number = float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    else:
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return None
    return number if math.isfinite(number) and number > 0 else None


def _tracking_only_features(meta: Any) -> dict[str, Any]:
    metadata = meta if isinstance(meta, dict) else {}
    try:
        duration = float((metadata.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    duration = max(0.0, duration)

    streams = metadata.get("streams") or []
    video_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        {},
    )
    fps = _parse_rate(video_stream.get("avg_frame_rate")) or _parse_rate(
        video_stream.get("r_frame_rate")
    )
    try:
        frame_count = int(video_stream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count <= 0 and fps and duration > 0:
        frame_count = int(round(fps * duration))

    return {
        "duration_seconds": duration,
        "frame_count": max(0, frame_count),
        "fps": round(float(fps or 0.0), 3),
        "scene_change_count": None,
        "scene_change_rate": None,
        "feature_mode": "tracking_only",
        "validated": False,
        "reason_codes": [
            "SCENE_BASED_PLAYER_FEATURES_DISABLED",
            "BALL_AND_EVENTS_NOT_MODELLED",
        ],
    }


def install_pipeline_policy(pipeline_module: Any) -> bool:
    """Install truthful, monotonic and inexpensive post-tracking behavior."""

    current_progress = getattr(pipeline_module, "set_progress", None)
    if not callable(current_progress):
        raise RuntimeError("pipeline module does not expose set_progress")
    if getattr(current_progress, "__algonext_pipeline_policy__", False):
        return False

    def governed_progress(
        job: Any, step: str, pct: int, message: str = ""
    ) -> None:
        existing = dict(getattr(job, "progress", None) or {})
        requested = max(0, min(100, _safe_int(pct)))
        target = (
            _analysis_progress(step, requested)
            if _is_analysis_run(job)
            else requested
        )
        if str(step or "").upper() not in {"DONE", "FAILED"}:
            target = max(_safe_int(existing.get("pct")), target)

        current_progress(job, step, target, message)
        updated = dict(getattr(job, "progress", None) or {})
        merged = {**existing, **updated}
        for key in PRESERVED_PROGRESS_KEYS:
            if key in existing and key not in updated:
                merged[key] = existing[key]
        phase = PHASES.get(str(step or "").upper())
        if phase:
            merged["phase"] = phase
        job.progress = merged

    setattr(governed_progress, "__algonext_pipeline_policy__", True)
    setattr(governed_progress, "__algonext_original_progress__", current_progress)
    pipeline_module.set_progress = governed_progress

    original_extract = getattr(pipeline_module, "extract_video_features", None)

    def extract_tracking_only_features(_path: Any, meta: Any) -> dict[str, Any]:
        return _tracking_only_features(meta)

    setattr(extract_tracking_only_features, "__algonext_pipeline_policy__", True)
    setattr(extract_tracking_only_features, "__algonext_original__", original_extract)
    pipeline_module.extract_video_features = extract_tracking_only_features

    original_skills = getattr(pipeline_module, "compute_skill_scores", None)

    def abstain_from_player_skills(_features: Any):
        order = list(getattr(pipeline_module, "SKILLS_ORDER", []) or [])
        return {}, order

    setattr(abstain_from_player_skills, "__algonext_pipeline_policy__", True)
    setattr(abstain_from_player_skills, "__algonext_original__", original_skills)
    pipeline_module.compute_skill_scores = abstain_from_player_skills

    original_explain = getattr(pipeline_module, "_build_explain_text", None)

    def truthful_explain(
        _role: str,
        _radar: dict[str, Any],
        _evidence_metrics: dict[str, Any],
        tracking_output: dict[str, Any] | None,
    ) -> str:
        if tracking_output:
            return (
                "Diagnostica di computer vision basata sul tracking. Non sono disponibili "
                "eventi palla, calibrazione atletica o un punteggio tecnico-tattico validato."
            )
        return (
            "Evidenza di tracking insufficiente. Il sistema si astiene dalla valutazione "
            "del giocatore."
        )

    setattr(truthful_explain, "__algonext_pipeline_policy__", True)
    setattr(truthful_explain, "__algonext_original__", original_explain)
    pipeline_module._build_explain_text = truthful_explain

    logger.info(
        "Installed tracking-only pipeline policy: monotonic progress and scene scoring disabled"
    )
    return True
