from __future__ import annotations

import os
from typing import Any, Mapping

from sqlalchemy import event

from app.core.evaluation_truth import apply_evaluation_truth_gate
from app.core.models import AnalysisJob

_INSTALLED = False

_OBSOLETE_SCORING_WARNINGS = {
    "INCOMPLETE_RADAR",
    "MISSING_OVERALL_SCORE",
    "MISSING_ROLE_SCORE",
}
_EVIDENCE_INSUFFICIENCY_REASONS = {
    "LOW_TRACKING_COVERAGE",
    "LOW_TRACKLET_CONTINUITY",
    "CONTINUITY_NOT_MEASURED",
    "INSUFFICIENT_TRACKING_SAMPLES",
    "LONG_TRACKING_GAPS",
}


def _enabled() -> bool:
    value = (os.environ.get("EVALUATION_TRUTH_GUARD") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _is_validated_player_evaluation(result: Mapping[str, Any]) -> bool:
    provenance = result.get("score_provenance")
    if not isinstance(provenance, Mapping):
        return False
    return bool(
        result.get("player_evaluation_available") is True
        and provenance.get("validated_player_score") is True
        and provenance.get("kind") == "player_evaluation"
    )


def _candidate_metrics(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    evidence = result.get("evidence_metrics")
    if not isinstance(evidence, Mapping):
        return None
    candidate = evidence.get("candidate_metrics") or evidence.get("candidateMetrics")
    return candidate if isinstance(candidate, Mapping) else None


def _tracking_payload(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tracking = result.get("tracking")
    return tracking if isinstance(tracking, Mapping) else None


def _tracking_only_warnings(
    result: Mapping[str, Any], existing: Any
) -> list[str]:
    warnings: list[str] = []
    for warning in existing if isinstance(existing, list) else []:
        if not isinstance(warning, str):
            continue
        normalized = warning.strip()
        if not normalized or normalized in _OBSOLETE_SCORING_WARNINGS:
            continue
        if normalized not in warnings:
            warnings.append(normalized)

    raw_reasons = result.get("reason_codes")
    reasons = (
        {reason for reason in raw_reasons if isinstance(reason, str)}
        if isinstance(raw_reasons, list)
        else set()
    )

    derived: list[str] = []
    if reasons.intersection(_EVIDENCE_INSUFFICIENCY_REASONS):
        derived.append("TRACKING_EVIDENCE_INSUFFICIENT")
    if "IDENTITY_NOT_VERIFIED_ACROSS_SHOTS" in reasons:
        derived.append("CROSS_SHOT_IDENTITY_UNVALIDATED")
    if "LONG_TRACKING_GAPS" in reasons:
        derived.append("LONG_TRACKING_GAPS")
    if result.get("player_evaluation_available") is not True:
        derived.append("PLAYER_EVALUATION_WITHHELD")

    for warning in derived:
        if warning not in warnings:
            warnings.append(warning)
    return warnings


def sanitize_analysis_job(job: AnalysisJob) -> None:
    if not _enabled():
        return
    result = job.result
    if not isinstance(result, Mapping) or not result:
        return
    if _is_validated_player_evaluation(result):
        return

    sanitized_result = apply_evaluation_truth_gate(
        result,
        candidate_metrics=_candidate_metrics(result),
        tracking=_tracking_payload(result),
        evidence_metrics=(
            result.get("evidence_metrics")
            if isinstance(result.get("evidence_metrics"), Mapping)
            else None
        ),
    )
    job.result = sanitized_result
    job.warnings = _tracking_only_warnings(sanitized_result, job.warnings)

    status = str(job.status or "").upper()
    if status in {"COMPLETED", "DONE"} and job.warnings:
        job.status = "PARTIAL"


def _before_write(_mapper, _connection, target: AnalysisJob) -> None:
    sanitize_analysis_job(target)


def _after_load(target: AnalysisJob, _context) -> None:
    sanitize_analysis_job(target)


def _after_refresh(target: AnalysisJob, _context, _attrs) -> None:
    sanitize_analysis_job(target)


def install_evaluation_guard() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    event.listen(AnalysisJob, "before_insert", _before_write, propagate=True)
    event.listen(AnalysisJob, "before_update", _before_write, propagate=True)
    event.listen(AnalysisJob, "load", _after_load, propagate=True)
    event.listen(AnalysisJob, "refresh", _after_refresh, propagate=True)
    _INSTALLED = True
