from __future__ import annotations

from typing import Any, Callable, Dict, List


_PLAYER_RESELECTION_WARNINGS = {
    "PLAYER_ANCHOR_NOT_FOUND",
    "PLAYER_RESELECTION_REQUIRED",
}


def _preview_result_state(result: Any) -> Dict[str, Any]:
    source = result if isinstance(result, dict) else {}
    preserved = {
        key: source[key]
        for key in (
            "candidates",
            "framesProcessed",
            "totalTracks",
            "rawTracks",
            "primaryCount",
            "secondaryCount",
        )
        if key in source
    }
    assets = source.get("assets")
    if isinstance(assets, dict) and isinstance(assets.get("input_video"), dict):
        preserved["assets"] = {"input_video": dict(assets["input_video"])}
    return preserved


def _tracking_reason_codes(tracking: Dict[str, Any]) -> List[str]:
    summary = tracking.get("reid_summary")
    raw_codes = summary.get("reason_codes") if isinstance(summary, dict) else []
    codes = [
        str(code).strip()
        for code in (raw_codes if isinstance(raw_codes, list) else [])
        if str(code).strip()
    ]
    status = str(tracking.get("tracking_status") or "").strip()
    if status and status not in codes:
        codes.append(status)
    return codes


def apply_tracking_outcome(
    job: Any,
    tracking_payload: Dict[str, Any],
    *,
    set_progress: Callable[[Any, str, int, str], None],
) -> bool:
    """Persist selected-player tracking and stop failed runs before scoring.

    Returns ``True`` when the caller must stop. A manual-anchor miss is
    recoverable, so the job returns to player selection with preview assets
    intact. Infrastructure/acquisition failures remain terminal.
    """

    reason_codes = _tracking_reason_codes(tracking_payload)
    observed_samples = max(0, int(tracking_payload.get("bboxes_count") or 0))
    tracking_failed = tracking_payload.get("tracking_success") is False
    tracking_incomplete = bool(tracking_payload.get("partial") is True)
    analysis_outcome = {
        "pipeline_state": (
            "STOPPED"
            if tracking_failed
            else "INCOMPLETE"
            if tracking_incomplete
            else "RUNNING"
        ),
        "tracking_state": (
            "FAILED"
            if tracking_failed
            else "INCOMPLETE"
            if tracking_incomplete
            else "SUCCEEDED"
        ),
        "metrics_scope": "selected_player",
        "observed_samples": observed_samples,
        "windows_processed": int(tracking_payload.get("windows_processed") or 0),
        "windows_total": int(tracking_payload.get("segments_total") or 0),
        "anchors_total": int(tracking_payload.get("anchors_total") or 0),
        "anchors_matched": int(tracking_payload.get("anchors_matched") or 0),
        "reason_codes": reason_codes,
        "action_required": tracking_payload.get("action_required"),
    }
    result_base = (
        _preview_result_state(job.result)
        if tracking_failed
        else dict(job.result or {})
    )
    job.result = {
        **result_base,
        "tracking": tracking_payload,
        "analysis_outcome": analysis_outcome,
    }
    if not tracking_failed:
        return False

    action_required = str(
        tracking_payload.get("action_required") or "RETRY_ANALYSIS"
    ).upper()
    warnings = [
        warning
        for warning in list(job.warnings or [])
        if warning not in _PLAYER_RESELECTION_WARNINGS
    ]

    if action_required == "RESELECT_PLAYER":
        warnings.extend(
            [
                "PLAYER_ANCHOR_NOT_FOUND",
                "PLAYER_RESELECTION_REQUIRED",
            ]
        )
        target = dict(job.target or {})
        target.pop("selection", None)
        target.pop("selections", None)
        target["confirmed"] = False
        target["tracking"] = {
            "status": "FAILED",
            "reason_codes": reason_codes,
            "action_required": action_required,
        }
        job.target = target
        job.player_ref = None
        job.anchor = {}
        job.status = "WAITING_FOR_PLAYER"
        job.error = None
        job.failure_reason = "PLAYER_RESELECTION_REQUIRED"
        job.warnings = list(dict.fromkeys(warnings))
        set_progress(
            job,
            "WAITING_FOR_PLAYER",
            35,
            "Player reference not found. Select a clearer frame.",
        )
        return True

    tracking_status = str(
        tracking_payload.get("tracking_status") or ""
    ).upper()
    anchor_acquisition_failed = tracking_status == "ANCHOR_ACQUISITION_ERROR"
    retry_warning = (
        "PLAYER_ANCHOR_ACQUISITION_FAILED"
        if anchor_acquisition_failed
        else "PLAYER_TRACKING_RETRY_REQUIRED"
    )
    warnings.append(retry_warning)
    job.status = "FAILED"
    job.error = (
        "Manual player references could not be processed."
        if anchor_acquisition_failed
        else "Selected-player tracking could not be completed."
    )
    job.failure_reason = retry_warning
    job.warnings = list(dict.fromkeys(warnings))
    set_progress(
        job,
        "FAILED",
        100,
        (
            "Player reference acquisition failed. Retry the analysis."
            if anchor_acquisition_failed
            else "Selected-player tracking failed technically. Retry the analysis."
        ),
    )
    return True
