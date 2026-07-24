from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from app.benchmark.reid_schema import (
    CANDIDATE_ABSENT,
    CANDIDATE_PRESENT,
    CANDIDATE_UNVERIFIABLE,
    DECISION_ABSTAINED,
    DECISION_ACCEPTED,
    DECISION_FAILED,
    VISIBILITY_NOT_VISIBLE,
    VISIBILITY_UNCERTAIN,
    VISIBILITY_VISIBLE,
    ReIDSequenceAnnotation,
    ReIDSequencePrediction,
)


@dataclass(frozen=True)
class ReIDGateThresholds:
    minimum_scorable_windows: int = 30
    accepted_judgement_coverage_min: float = 0.90
    accepted_precision_min: float = 0.95
    false_link_rate_max: float = 0.05
    association_recall_given_candidate_min: float = 0.60
    visible_window_recall_min: float = 0.45
    candidate_annotation_coverage_min: float = 0.70
    candidate_recall_visible_min: float = 0.70
    processing_failure_rate_max: float = 0.05


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _score_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def _accepted_correctness(
    *,
    selected_track_is_target: bool | None,
    candidate_state: str | None,
    target_candidate_id: str | None,
    selected_candidate_id: str | None,
) -> bool | None:
    by_candidate = (
        selected_candidate_id == target_candidate_id
        if candidate_state == CANDIDATE_PRESENT
        and target_candidate_id is not None
        and selected_candidate_id is not None
        else None
    )
    if selected_track_is_target is not None and by_candidate is not None:
        if selected_track_is_target != by_candidate:
            raise ValueError(
                "annotation contradicts itself: selected_track_is_target does not "
                "match target_candidate_id"
            )
    return selected_track_is_target if selected_track_is_target is not None else by_candidate


def evaluate_reid_sequence(
    annotation: ReIDSequenceAnnotation,
    prediction: ReIDSequencePrediction,
    *,
    window_time_tolerance_sec: float = 0.05,
) -> dict[str, Any]:
    if annotation.video_id != prediction.video_id:
        raise ValueError(
            f"video_id mismatch: {annotation.video_id!r} != {prediction.video_id!r}"
        )
    if window_time_tolerance_sec < 0:
        raise ValueError("window_time_tolerance_sec must be >= 0")

    prediction_by_index = {window.window_index: window for window in prediction.windows}
    annotation_indices = {window.window_index for window in annotation.windows}
    count_names = (
        "annotation_windows",
        "scorable_windows",
        "unscored_uncertain_windows",
        "visible_windows",
        "not_visible_windows",
        "candidate_present_windows",
        "candidate_absent_windows",
        "candidate_unverifiable_windows",
        "candidate_scorable_visible_windows",
        "accepted_windows",
        "accepted_judged_windows",
        "accepted_unjudged_windows",
        "abstained_windows",
        "failed_windows",
        "true_accepts",
        "false_accepts",
        "false_links_visible_wrong_target",
        "false_accepts_not_visible",
        "correct_associations",
        "wrong_candidate_accepts",
        "missed_associations",
        "candidate_absent_abstentions",
        "candidate_unverifiable_abstentions",
        "correct_abstentions_not_visible",
        "prediction_windows_unscored",
    )
    counts: Counter[str] = Counter({name: 0 for name in count_names})
    reason_counts: Counter[str] = Counter()
    outcomes: list[dict[str, Any]] = []
    true_accept_scores: list[float] = []
    false_accept_scores: list[float] = []
    unjudged_accept_scores: list[float] = []
    abstain_scores: list[float] = []

    for truth in annotation.windows:
        counts["annotation_windows"] += 1
        predicted = prediction_by_index.get(truth.window_index)
        if predicted is None:
            decision = DECISION_FAILED
            selected_candidate_id = None
            best_candidate_id = None
            best_score = 0.0
            margin = 0.0
            reasons = ("MISSING_PREDICTION_WINDOW",)
        else:
            if abs(predicted.window_start - truth.window_start) > window_time_tolerance_sec:
                raise ValueError(
                    f"window {truth.window_index} start mismatch: "
                    f"{truth.window_start} != {predicted.window_start}"
                )
            if abs(predicted.window_end - truth.window_end) > window_time_tolerance_sec:
                raise ValueError(
                    f"window {truth.window_index} end mismatch: "
                    f"{truth.window_end} != {predicted.window_end}"
                )
            decision = predicted.decision
            selected_candidate_id = predicted.selected_candidate_id
            best_candidate_id = predicted.best_candidate_id
            best_score = predicted.best_score
            margin = predicted.margin
            reasons = predicted.reason_codes

        for reason in reasons:
            reason_counts[reason] += 1

        accepted_correct: bool | None = None
        if truth.target_visibility == VISIBILITY_UNCERTAIN:
            counts["unscored_uncertain_windows"] += 1
            outcome = "UNSCORED_UNCERTAIN"
        else:
            counts["scorable_windows"] += 1
            if truth.target_visibility == VISIBILITY_VISIBLE:
                counts["visible_windows"] += 1
                if truth.candidate_state == CANDIDATE_PRESENT:
                    counts["candidate_present_windows"] += 1
                    counts["candidate_scorable_visible_windows"] += 1
                elif truth.candidate_state == CANDIDATE_ABSENT:
                    counts["candidate_absent_windows"] += 1
                    counts["candidate_scorable_visible_windows"] += 1
                elif truth.candidate_state == CANDIDATE_UNVERIFIABLE:
                    counts["candidate_unverifiable_windows"] += 1
            else:
                counts["not_visible_windows"] += 1

            if decision == DECISION_ACCEPTED:
                counts["accepted_windows"] += 1
                accepted_correct = _accepted_correctness(
                    selected_track_is_target=truth.selected_track_is_target,
                    candidate_state=truth.candidate_state,
                    target_candidate_id=truth.target_candidate_id,
                    selected_candidate_id=selected_candidate_id,
                )
                if accepted_correct is True:
                    counts["accepted_judged_windows"] += 1
                    counts["true_accepts"] += 1
                    true_accept_scores.append(best_score)
                    outcome = "TRUE_ACCEPT"
                elif accepted_correct is False:
                    counts["accepted_judged_windows"] += 1
                    counts["false_accepts"] += 1
                    false_accept_scores.append(best_score)
                    if truth.target_visibility == VISIBILITY_NOT_VISIBLE:
                        counts["false_accepts_not_visible"] += 1
                        outcome = "FALSE_ACCEPT_NOT_VISIBLE"
                    else:
                        counts["false_links_visible_wrong_target"] += 1
                        outcome = "FALSE_LINK_VISIBLE_WRONG_TARGET"
                else:
                    counts["accepted_unjudged_windows"] += 1
                    unjudged_accept_scores.append(best_score)
                    outcome = "ACCEPT_UNJUDGED"

                if truth.candidate_state == CANDIDATE_PRESENT:
                    if selected_candidate_id == truth.target_candidate_id:
                        counts["correct_associations"] += 1
                    else:
                        counts["wrong_candidate_accepts"] += 1
            elif decision == DECISION_ABSTAINED:
                counts["abstained_windows"] += 1
                abstain_scores.append(best_score)
                if truth.target_visibility == VISIBILITY_NOT_VISIBLE:
                    counts["correct_abstentions_not_visible"] += 1
                    outcome = "CORRECT_ABSTENTION_NOT_VISIBLE"
                elif truth.candidate_state == CANDIDATE_PRESENT:
                    counts["missed_associations"] += 1
                    outcome = "MISSED_ASSOCIATION"
                elif truth.candidate_state == CANDIDATE_ABSENT:
                    counts["candidate_absent_abstentions"] += 1
                    outcome = "CANDIDATE_ABSENT_ABSTAINED"
                else:
                    counts["candidate_unverifiable_abstentions"] += 1
                    outcome = "CANDIDATE_UNVERIFIABLE_ABSTAINED"
            else:
                counts["failed_windows"] += 1
                if truth.candidate_state == CANDIDATE_PRESENT:
                    counts["missed_associations"] += 1
                outcome = (
                    "PROCESSING_FAILURE_VISIBLE"
                    if truth.target_visibility == VISIBILITY_VISIBLE
                    else "PROCESSING_FAILURE_NOT_VISIBLE"
                )

        outcomes.append(
            {
                "window_index": truth.window_index,
                "window_start": truth.window_start,
                "window_end": truth.window_end,
                "target_visibility": truth.target_visibility,
                "candidate_state": truth.candidate_state,
                "target_candidate_id": truth.target_candidate_id,
                "selected_track_is_target": truth.selected_track_is_target,
                "decision": decision,
                "selected_candidate_id": selected_candidate_id,
                "best_candidate_id": best_candidate_id,
                "best_score": round(best_score, 6),
                "margin": round(margin, 6),
                "accepted_correct": accepted_correct,
                "outcome": outcome,
                "reason_codes": list(reasons),
                "notes": truth.notes,
            }
        )

    counts["prediction_windows_unscored"] = sum(
        1 for window in prediction.windows if window.window_index not in annotation_indices
    )

    scorable = counts["scorable_windows"]
    accepted = counts["accepted_windows"]
    judged_accepts = counts["accepted_judged_windows"]
    abstained = counts["abstained_windows"]
    failed = counts["failed_windows"]
    visible = counts["visible_windows"]
    not_visible = counts["not_visible_windows"]
    candidate_present = counts["candidate_present_windows"]
    candidate_scorable = counts["candidate_scorable_visible_windows"]
    true_accepts = counts["true_accepts"]
    false_accepts = counts["false_accepts"]
    correct_associations = counts["correct_associations"]
    accepted_candidate_present = (
        counts["correct_associations"] + counts["wrong_candidate_accepts"]
    )
    correct_reid_decisions = (
        true_accepts
        + counts["correct_abstentions_not_visible"]
        + counts["candidate_absent_abstentions"]
    )
    end_to_end_successes = true_accepts + counts["correct_abstentions_not_visible"]

    metrics = {
        "annotation_coverage": round(
            _safe_divide(scorable, counts["annotation_windows"]), 6
        ),
        "decision_coverage": round(_safe_divide(accepted + abstained, scorable), 6),
        "accepted_judgement_coverage": round(
            _safe_divide(judged_accepts, accepted), 6
        ),
        "accepted_precision": round(_safe_divide(true_accepts, judged_accepts), 6),
        "false_link_rate": round(_safe_divide(false_accepts, judged_accepts), 6),
        "association_precision_given_candidate": round(
            _safe_divide(correct_associations, accepted_candidate_present), 6
        ),
        "association_recall_given_candidate": round(
            _safe_divide(correct_associations, candidate_present), 6
        ),
        "visible_window_recall": round(_safe_divide(true_accepts, visible), 6),
        "candidate_annotation_coverage": round(
            _safe_divide(candidate_scorable, visible), 6
        ),
        "candidate_recall_visible": round(
            _safe_divide(candidate_present, candidate_scorable), 6
        ),
        "nonvisible_abstention_rate": round(
            _safe_divide(counts["correct_abstentions_not_visible"], not_visible), 6
        ),
        "abstention_rate": round(_safe_divide(abstained, scorable), 6),
        "processing_failure_rate": round(_safe_divide(failed, scorable), 6),
        "reid_decision_accuracy": round(
            _safe_divide(correct_reid_decisions, scorable), 6
        ),
        "end_to_end_window_success_rate": round(
            _safe_divide(end_to_end_successes, scorable), 6
        ),
    }

    return {
        "schema_version": "reid-window-benchmark-report-v1",
        "video_id": annotation.video_id,
        "identity": annotation.identity,
        "parameters": {"window_time_tolerance_sec": window_time_tolerance_sec},
        "counts": dict(sorted(counts.items())),
        "metrics": metrics,
        "score_diagnostics": {
            "true_accepts": _score_summary(true_accept_scores),
            "false_accepts": _score_summary(false_accept_scores),
            "unjudged_accepts": _score_summary(unjudged_accept_scores),
            "abstentions": _score_summary(abstain_scores),
        },
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "windows": outcomes,
    }


def evaluate_reid_quality_gate(
    report: Mapping[str, Any], thresholds: ReIDGateThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or ReIDGateThresholds()
    metrics = report.get("metrics") or {}
    counts = report.get("counts") or {}
    definitions = [
        (
            "scorable_windows",
            ">=",
            thresholds.minimum_scorable_windows,
            float(counts.get("scorable_windows", 0)),
        ),
        (
            "accepted_judgement_coverage",
            ">=",
            thresholds.accepted_judgement_coverage_min,
            float(metrics.get("accepted_judgement_coverage", 0.0)),
        ),
        (
            "accepted_precision",
            ">=",
            thresholds.accepted_precision_min,
            float(metrics.get("accepted_precision", 0.0)),
        ),
        (
            "false_link_rate",
            "<=",
            thresholds.false_link_rate_max,
            float(metrics.get("false_link_rate", 0.0)),
        ),
        (
            "association_recall_given_candidate",
            ">=",
            thresholds.association_recall_given_candidate_min,
            float(metrics.get("association_recall_given_candidate", 0.0)),
        ),
        (
            "visible_window_recall",
            ">=",
            thresholds.visible_window_recall_min,
            float(metrics.get("visible_window_recall", 0.0)),
        ),
        (
            "candidate_annotation_coverage",
            ">=",
            thresholds.candidate_annotation_coverage_min,
            float(metrics.get("candidate_annotation_coverage", 0.0)),
        ),
        (
            "candidate_recall_visible",
            ">=",
            thresholds.candidate_recall_visible_min,
            float(metrics.get("candidate_recall_visible", 0.0)),
        ),
        (
            "processing_failure_rate",
            "<=",
            thresholds.processing_failure_rate_max,
            float(metrics.get("processing_failure_rate", 0.0)),
        ),
    ]
    checks: list[dict[str, Any]] = []
    for metric, comparator, threshold, actual in definitions:
        passed = actual >= threshold if comparator == ">=" else actual <= threshold
        checks.append(
            {
                "metric": metric,
                "comparator": comparator,
                "threshold": threshold,
                "actual": round(actual, 6),
                "passed": passed,
            }
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "thresholds": asdict(thresholds),
        "checks": checks,
        "note": (
            "This gate validates window-level ReID behavior only. It does not "
            "validate football scoring, physical metrics, or unseen-domain "
            "generalization."
        ),
    }
