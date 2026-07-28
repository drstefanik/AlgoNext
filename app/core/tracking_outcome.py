from __future__ import annotations

from typing import Any, Callable, Dict, List

_PLAYER_RESELECTION_WARNINGS = {
    "PLAYER_ANCHOR_NOT_FOUND",
    "PLAYER_RESELECTION_REQUIRED",
}


class StaleAnalysisAttemptError(RuntimeError):
    """A worker tried to publish evidence for a superseded enqueue attempt."""


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

    current_target = job.target if isinstance(job.target, dict) else {}
    payload_attempt_id = str(tracking_payload.get("analysis_attempt_id") or "").strip()
    target_attempt_id = str(current_target.get("analysis_attempt_id") or "").strip()
    if payload_attempt_id != target_attempt_id:
        raise StaleAnalysisAttemptError(
            "Tracking attempt does not match the job target: "
            f"payload={payload_attempt_id or '<missing>'} "
            f"target={target_attempt_id or '<missing>'}"
        )
    analysis_attempt_id = payload_attempt_id
    tracking_payload = dict(tracking_payload)
    if analysis_attempt_id:
        tracking_payload["analysis_attempt_id"] = analysis_attempt_id

    reason_codes = _tracking_reason_codes(tracking_payload)
    observed_samples = max(0, int(tracking_payload.get("bboxes_count") or 0))
    windows_processed = int(tracking_payload.get("windows_processed") or 0)
    windows_total = int(tracking_payload.get("segments_total") or 0)
    processing_completed = bool(
        windows_total > 0 and windows_processed >= windows_total
    )
    anchors_matched = int(tracking_payload.get("anchors_matched") or 0)
    pre_guard_anchor_diagnostics = (
        tracking_payload.get("pre_guard_anchor_diagnostics")
        if isinstance(
            tracking_payload.get("pre_guard_anchor_diagnostics"),
            dict,
        )
        and tracking_payload["pre_guard_anchor_diagnostics"].get(
            "diagnostic_only"
        )
        is True
        and tracking_payload["pre_guard_anchor_diagnostics"].get("validated")
        is False
        else {}
    )
    try:
        anchors_matched_before_guard = max(
            0,
            int(
                pre_guard_anchor_diagnostics.get(
                    "anchors_matched_before_guard",
                    0,
                )
                or 0
            ),
        )
    except (TypeError, ValueError):
        anchors_matched_before_guard = 0
    tracking_failed = tracking_payload.get("tracking_success") is False
    tracking_incomplete = bool(tracking_payload.get("partial") is True)
    analysis_outcome = {
        "pipeline_state": (
            "DONE"
            if processing_completed and tracking_failed
            else (
                "STOPPED"
                if tracking_failed
                else "INCOMPLETE" if tracking_incomplete else "RUNNING"
            )
        ),
        "tracking_state": (
            "FAILED"
            if tracking_failed
            else "INCOMPLETE" if tracking_incomplete else "SUCCEEDED"
        ),
        "metrics_scope": "selected_player",
        "observed_samples": observed_samples,
        "segments_with_player": int(tracking_payload.get("segments_with_player") or 0),
        "autonomous_segments_with_player": int(
            tracking_payload.get("autonomous_segments_with_player") or 0
        ),
        "autonomous_bboxes_count": int(
            tracking_payload.get("autonomous_bboxes_count") or 0
        ),
        "tracking_scope_status": tracking_payload.get("tracking_scope_status"),
        "windows_processed": windows_processed,
        "windows_total": windows_total,
        "anchors_total": int(tracking_payload.get("anchors_total") or 0),
        "anchors_matched": anchors_matched,
        "anchors_matched_before_guard": anchors_matched_before_guard,
        "reason_codes": reason_codes,
        "action_required": tracking_payload.get("action_required"),
        "analysis_attempt_id": analysis_attempt_id or None,
    }
    result_base = (
        _preview_result_state(job.result) if tracking_failed else dict(job.result or {})
    )
    job.result = {
        **result_base,
        "analysis_attempt_id": analysis_attempt_id or None,
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
        reference_was_matched = (
            anchors_matched > 0 or anchors_matched_before_guard > 0
        )
        if reference_was_matched:
            warnings.extend(
                code for code in reason_codes if code != "PLAYER_ANCHOR_NOT_FOUND"
            )
            warnings.append("PLAYER_RESELECTION_REQUIRED")
        else:
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
        target_tracking = {
            "status": "FAILED",
            "reason_codes": reason_codes,
            "action_required": action_required,
        }
        if analysis_attempt_id:
            target_tracking["analysis_attempt_id"] = analysis_attempt_id
        target["tracking"] = target_tracking
        job.target = target
        job.player_ref = None
        job.anchor = {}
        job.status = "WAITING_FOR_PLAYER"
        job.error = None
        if reference_was_matched:
            job.failure_reason = (
                str(tracking_payload.get("tracking_status") or "").strip()
                or (reason_codes[0] if reason_codes else "")
                or "PLAYER_RESELECTION_REQUIRED"
            )
        else:
            job.failure_reason = "PLAYER_RESELECTION_REQUIRED"
        job.warnings = list(dict.fromkeys(warnings))
        set_progress(
            job,
            "WAITING_FOR_PLAYER",
            100 if processing_completed else 35,
            (
                "Processing completed, but selected-player tracking was "
                "rejected. Select a clearer frame."
                if reference_was_matched and processing_completed
                else (
                    "Player reference matched, but selected-player tracking "
                    "was rejected. Select a clearer frame."
                    if reference_was_matched
                    else "Player reference not found. Select a clearer frame."
                )
            ),
        )
        return True

    tracking_status = str(tracking_payload.get("tracking_status") or "").upper()
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
