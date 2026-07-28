from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.core.tracking_outcome import StaleAnalysisAttemptError

logger = logging.getLogger(__name__)

TRACKING_ARTIFACT_ROOT = Path("/tmp/fnh_jobs")
_ACTIVE_ANALYSIS_STATUSES = frozenset({"QUEUED", "RUNNING", "PROCESSING"})
_SAFE_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_DANGEROUS_JOB_COMPONENTS = {
    ".",
    "..",
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _validated_job_id_component(job_id: Any) -> str | None:
    if job_id is None:
        return None
    candidate = str(job_id)
    if (
        not candidate
        or candidate != candidate.strip()
        or "/" in candidate
        or "\\" in candidate
        or ".." in candidate
        or Path(candidate).is_absolute()
        or candidate.lower() in _DANGEROUS_JOB_COMPONENTS
        or any(ord(character) < 32 or ord(character) == 127 for character in candidate)
        or _SAFE_JOB_ID_PATTERN.fullmatch(candidate) is None
    ):
        return None
    return candidate


def fail_closed_legacy_fallback(
    output: Any,
    *,
    reason_code: str,
    summary_status: str = "FALLBACK_LEGACY",
    tracking_status: str = "REID_FALLBACK_LEGACY_UNVERIFIED",
    identity_mode: str = "unverified_legacy_fallback",
    notes: str | None = None,
) -> dict[str, Any]:
    """Discard legacy observations that cannot attest selected-player identity."""

    source = dict(output) if isinstance(output, Mapping) else {}
    reason_codes = list(
        dict.fromkeys(
            [
                reason_code,
                "IDENTITY_UNVERIFIED_LEGACY_FALLBACK",
            ]
        )
    )
    preserved = {
        key: source[key]
        for key in (
            "mode",
            "method",
            "fps",
            "window_sec",
            "overlap_sec",
            "runtime_profile",
        )
        if key in source
    }
    return {
        "mode": preserved.pop("mode", "full_match_windowed"),
        "identity_mode": identity_mode,
        **preserved,
        "segments": [],
        "segments_total": 0,
        "segments_with_player": 0,
        "autonomous_segments_with_player": 0,
        "autonomous_bboxes_count": 0,
        "tracking_scope_status": "EMPTY",
        "windows_processed": 0,
        "coverage_pct_total": 0.0,
        "coverage_pct": 0.0,
        "largest_gap_sec": None,
        "bboxes": [],
        "bboxes_count": 0,
        "lost_segments": [],
        "motion_segments": [],
        "anchor_reacquisitions": 0,
        "anchors_total": 0,
        "anchors_matched": 0,
        "anchor_matches": [],
        "anchors_used": {},
        "anchor_acquisition": {},
        "tracking_success": False,
        "partial": False,
        "partial_reason": None,
        "tracking_status": tracking_status,
        "action_required": "RETRY_ANALYSIS",
        "reid_summary": {
            "status": summary_status,
            "validated": False,
            "reason_codes": reason_codes,
            "anchors_total": 0,
            "anchors_matched": 0,
            "anchor_matches": [],
            "autonomous_segments_with_player": 0,
            "autonomous_bboxes_count": 0,
            "tracking_scope_status": "EMPTY",
            "windows_processed": 0,
        },
        "notes": notes
        or (
            "Legacy tracking output was discarded because selected-player "
            "identity could not be verified after the ReID failure."
        ),
    }


def persist_canonical_tracking_artifact(
    output: Any,
    *,
    job_id: Any,
    tracking_module: Any,
    analysis_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Persist tracking under its immutable analysis-attempt key.

    Existing asset references are always removed first. Invalid identifiers,
    missing configuration, or persistence failures therefore return a payload
    with no key or URL rather than pointing at an older, unguarded artifact.
    """

    payload = dict(output) if isinstance(output, Mapping) else {}
    payload.pop("tracking_key", None)
    payload.pop("tracking_url", None)
    normalized_attempt_id = str(analysis_attempt_id or "").strip() or None
    payload["analysis_attempt_id"] = normalized_attempt_id
    normalized_job_id = _validated_job_id_component(job_id)
    if normalized_job_id is None:
        return payload
    attempt_component = (
        _validated_job_id_component(normalized_attempt_id)
        if normalized_attempt_id is not None
        else "legacy"
    )
    if attempt_component is None:
        return payload

    endpoint_url = getattr(tracking_module, "S3_ENDPOINT_URL", None)
    bucket = (os.environ.get("S3_BUCKET") or "").strip()
    get_client = getattr(tracking_module, "_get_s3_client", None)
    ensure_bucket = getattr(tracking_module, "_ensure_bucket_exists", None)
    upload_file = getattr(tracking_module, "_upload_file", None)
    presign = getattr(tracking_module, "_presign_get_object", None)
    if (
        not endpoint_url
        or not bucket
        or not all(
            callable(callback)
            for callback in (get_client, ensure_bucket, upload_file, presign)
        )
    ):
        return payload

    tracking_key = (
        f"jobs/{normalized_job_id}/attempts/{attempt_component}/"
        "tracking/tracking.json"
    )
    try:
        expires_seconds = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))
    except (TypeError, ValueError):
        expires_seconds = 3600

    try:
        artifact_root = TRACKING_ARTIFACT_ROOT.resolve()
        tracking_dir = (
            artifact_root
            / normalized_job_id
            / "attempts"
            / attempt_component
            / "tracking"
        )
        resolved_tracking_dir = tracking_dir.resolve(strict=False)
    except OSError:
        return payload
    if (
        resolved_tracking_dir != artifact_root
        and artifact_root not in resolved_tracking_dir.parents
    ):
        return payload

    tracking_path = tracking_dir / "tracking.json"
    temporary_path = tracking_dir / "tracking.attempt.tmp"
    try:
        tracking_dir.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary_path.replace(tracking_path)

        client = get_client(endpoint_url)
        ensure_bucket(client, bucket)
        upload_file(
            client,
            bucket,
            tracking_path,
            tracking_key,
            "application/json",
        )
        tracking_url = presign(bucket, tracking_key, expires_seconds)
    except Exception:
        logger.exception(
            "Unable to persist attempt tracking artifact job_id=%s attempt_id=%s",
            normalized_job_id,
            normalized_attempt_id or "legacy",
        )
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.debug(
                "Unable to remove attempt tracking temporary file job_id=%s",
                normalized_job_id,
                exc_info=True,
            )
        return payload

    return {
        **payload,
        "tracking_key": tracking_key,
        "tracking_url": tracking_url,
    }


def persist_fail_closed_legacy_fallback(
    output: Any,
    *,
    reason_code: str,
    job_id: Any,
    tracking_module: Any,
    analysis_attempt_id: str | None = None,
    summary_status: str = "FALLBACK_LEGACY",
    tracking_status: str = "REID_FALLBACK_LEGACY_UNVERIFIED",
    identity_mode: str = "unverified_legacy_fallback",
    notes: str | None = None,
) -> dict[str, Any]:
    """Replace a legacy tracking artifact with its fail-closed representation."""

    sanitized = fail_closed_legacy_fallback(
        output,
        reason_code=reason_code,
        summary_status=summary_status,
        tracking_status=tracking_status,
        identity_mode=identity_mode,
        notes=notes,
    )
    return persist_canonical_tracking_artifact(
        sanitized,
        job_id=job_id,
        tracking_module=tracking_module,
        analysis_attempt_id=analysis_attempt_id,
    )


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

    target_samples = _env_int("FULL_MATCH_TARGET_SAMPLES", 12000, 1000, 50000)

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
            os.environ.get("FULL_MATCH_DETECTOR_MODEL") or "yolo11s.pt"
        ).strip() or "yolo11s.pt"

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
    if not callable(current) or getattr(
        current, "__algonext_progress_adapter__", False
    ):
        return

    def adapted(
        job_id: str,
        pct: int,
        message: str,
        *,
        analysis_attempt_id: str | None = None,
    ) -> Any:
        mapped_pct = int(pct)
        mapped_message = message
        if message in {
            "Tracking player with experimental ReID",
            "Tracking player (windowed)",
        }:
            stage_ratio = max(0.0, min(1.0, (float(pct) - 10.0) / 30.0))
            mapped_pct = 35 + int(round(stage_ratio * 35.0))
            mapped_message = f"{message} · {int(round(stage_ratio * 100.0))}% finestre"
        return current(
            job_id,
            mapped_pct,
            mapped_message,
            analysis_attempt_id=analysis_attempt_id,
        )

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
        "analysis_attempt_id": (
            str(kwargs.get("analysis_attempt_id") or "").strip() or None
        ),
        "mode": "full_match_windowed",
        "identity_mode": "appearance_reid_v1",
        "method": "yolo+bytetrack+appearance_reid",
        "fps": fps,
        "window_sec": window_sec,
        "overlap_sec": overlap_sec,
        "segments": [],
        "segments_total": _estimate_window_count(profile),
        "segments_with_player": 0,
        "autonomous_segments_with_player": 0,
        "autonomous_bboxes_count": 0,
        "tracking_scope_status": "EMPTY",
        "windows_processed": 0,
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
            "autonomous_segments_with_player": 0,
            "autonomous_bboxes_count": 0,
            "tracking_scope_status": "EMPTY",
            "windows_processed": 0,
            "reason_codes": ["TRACKING_BUDGET_EXHAUSTED"],
        },
    }


def mark_partial_timeout(
    job_id: str | None,
    profile: FullMatchRuntimeProfile | None,
    *,
    analysis_attempt_id: str | None = None,
) -> None:
    if not job_id:
        return

    try:
        from app.core.db import SessionLocal
        from app.core.models import AnalysisJob
        from app.core.normalizers import normalize_failure_reason
        from sqlalchemy import select
    except Exception:
        logger.exception(
            "Unable to import job persistence for partial tracking timeout"
        )
        return

    db = SessionLocal()
    try:
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
                "Partial-timeout attempt differs from the current job target: "
                f"worker={expected_attempt_id or '<missing>'} "
                f"target={current_attempt_id or '<missing>'}"
            )
        status = str(job.status or "").strip().upper()
        if status not in _ACTIVE_ANALYSIS_STATUSES:
            raise StaleAnalysisAttemptError(
                "Partial-timeout writer cannot mutate a terminal or inactive job: "
                f"status={status or '<missing>'} "
                f"attempt={expected_attempt_id or '<missing>'}"
            )
        warnings = [
            code for code in list(job.warnings or []) if code != "TRACKING_TIMEOUT"
        ]
        if "TRACKING_PARTIAL_TIMEOUT" not in warnings:
            warnings.append("TRACKING_PARTIAL_TIMEOUT")
        job.warnings = warnings
        job.status = "RUNNING"
        job.error = None
        job.failure_reason = normalize_failure_reason(None)
        updated_progress = {
            **(job.progress or {}),
            "step": "TRACKING_PARTIAL",
            "pct": 70,
            "message": (
                "Tracking budget exhausted; continuing with partial diagnostics"
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runtime_profile": profile.to_payload() if profile is not None else None,
        }
        if expected_attempt_id is not None:
            updated_progress["analysis_attempt_id"] = expected_attempt_id
        job.progress = updated_progress
        db.commit()
    except StaleAnalysisAttemptError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Unable to convert tracking timeout into partial result")
    finally:
        db.close()
