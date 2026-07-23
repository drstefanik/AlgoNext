from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

EVALUATION_SCHEMA_VERSION = "evaluation-truth-v1"
TRACKING_QUALITY_VERSION = "tracking-quality-v1"

_TRACKING_SAMPLE_TARGET = 60.0
_LOW_COVERAGE_PCT = 45.0
_LOW_CONTINUITY_PCT = 65.0
_MIN_TRACKING_SAMPLES = 30

_UNVALIDATED_PHYSICAL_METRICS = {
    "distance_covered_m",
    "avg_speed_kmh",
    "top_speed_kmh",
    "top_speed_kmh_clamped",
    "sprints_count",
}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def _as_percent(value: Any) -> Optional[float]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if 0.0 <= parsed <= 1.0:
        parsed *= 100.0
    return _clamp(parsed)


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _collect_tracking_bboxes(tracking: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    segments = tracking.get("segments")
    if isinstance(segments, list):
        collected: List[Mapping[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            for bbox in segment.get("bboxes") or []:
                if isinstance(bbox, Mapping):
                    collected.append(bbox)
        return collected
    return [bbox for bbox in (tracking.get("bboxes") or []) if isinstance(bbox, Mapping)]


def _collect_lost_segments(tracking: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    segments = tracking.get("segments")
    if isinstance(segments, list):
        collected: List[Mapping[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            for lost in segment.get("lost_segments") or []:
                if isinstance(lost, Mapping):
                    collected.append(lost)
        return collected
    return [
        segment
        for segment in (tracking.get("lost_segments") or [])
        if isinstance(segment, Mapping)
    ]


def compute_image_motion_metrics(tracking: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Compute image-plane diagnostics without pretending pixels are metres.

    The coordinates are normalized to the video frame. Camera motion is not removed,
    so these values are useful only as tracking diagnostics, never as athletic data.
    """

    source = _as_mapping(tracking)
    points: List[tuple[float, float, float]] = []
    for bbox in _collect_tracking_bboxes(source):
        t = _safe_float(bbox.get("t"))
        x = _safe_float(bbox.get("x"))
        y = _safe_float(bbox.get("y"))
        w = _safe_float(bbox.get("w"))
        h = _safe_float(bbox.get("h"))
        if None in (t, x, y, w, h):
            continue
        points.append((float(t), float(x) + float(w) / 2.0, float(y) + float(h) / 2.0))

    points.sort(key=lambda item: item[0])
    path_length = 0.0
    speeds: List[float] = []
    motion_bursts = 0
    above_threshold = False
    burst_threshold = 0.02

    for previous, current in zip(points, points[1:]):
        t0, x0, y0 = previous
        t1, x1, y1 = current
        dt = t1 - t0
        if dt <= 0:
            continue
        distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        path_length += distance
        speed = distance / dt
        speeds.append(speed)
        is_above = speed >= burst_threshold
        if is_above and not above_threshold:
            motion_bursts += 1
        above_threshold = is_above

    tracked_span_sec = points[-1][0] - points[0][0] if len(points) >= 2 else 0.0
    average_speed = path_length / tracked_span_sec if tracked_span_sec > 0 else 0.0
    sorted_speeds = sorted(speeds)
    if sorted_speeds:
        p95_index = min(len(sorted_speeds) - 1, int(round(0.95 * (len(sorted_speeds) - 1))))
        p95_speed = sorted_speeds[p95_index]
    else:
        p95_speed = 0.0

    return {
        "metric_space": "image_plane_normalized",
        "camera_motion_compensated": False,
        "pitch_calibrated": False,
        "observed_samples": len(points),
        "tracked_span_sec": round(max(0.0, tracked_span_sec), 3),
        "normalized_path_length": round(path_length, 6),
        "avg_center_speed_norm_per_sec": round(average_speed, 6),
        "p95_center_speed_norm_per_sec": round(p95_speed, 6),
        "motion_bursts_proxy": motion_bursts,
    }


def sanitize_evidence_metrics(evidence_metrics: Mapping[str, Any] | None) -> Dict[str, Any]:
    source = _as_mapping(evidence_metrics)
    sanitized = {
        key: value
        for key, value in source.items()
        if key not in _UNVALIDATED_PHYSICAL_METRICS
    }
    removed = sorted(key for key in _UNVALIDATED_PHYSICAL_METRICS if key in source)
    if removed:
        sanitized["removed_unvalidated_metrics"] = removed
    return sanitized


def build_tracking_evaluation(
    *,
    candidate_metrics: Mapping[str, Any] | None = None,
    tracking: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    candidate = _as_mapping(candidate_metrics)
    tracking_source = _as_mapping(tracking)
    image_motion = compute_image_motion_metrics(tracking_source)

    coverage_pct = _as_percent(
        next(
            (
                value
                for value in (
                    tracking_source.get("coverage_pct_total"),
                    tracking_source.get("coverage_pct"),
                    candidate.get("coveragePct"),
                    candidate.get("coverage_pct"),
                )
                if value is not None
            ),
            None,
        )
    )
    coverage_pct = coverage_pct if coverage_pct is not None else 0.0

    continuity_pct = _as_percent(
        next(
            (
                value
                for value in (
                    candidate.get("stabilityScore"),
                    candidate.get("stability_score"),
                    tracking_source.get("stability_score"),
                )
                if value is not None
            ),
            None,
        )
    )
    lost_segments = _collect_lost_segments(tracking_source)
    if continuity_pct is None:
        continuity_pct = _clamp(100.0 - len(lost_segments) * 12.5)

    samples_used = int(
        max(
            0.0,
            _first_number(
                candidate.get("sampleFramesCount"),
                candidate.get("sample_frames_count"),
                tracking_source.get("bboxes_count"),
                image_motion.get("observed_samples"),
            )
            or 0.0,
        )
    )
    sample_sufficiency_pct = _clamp((samples_used / _TRACKING_SAMPLE_TARGET) * 100.0)

    segments_total = int(max(0.0, _safe_float(tracking_source.get("segments_total")) or 0.0))
    segments_with_player = int(
        max(0.0, _safe_float(tracking_source.get("segments_with_player")) or 0.0)
    )
    segments_with_player_pct = (
        _clamp((segments_with_player / float(segments_total)) * 100.0)
        if segments_total > 0
        else None
    )
    largest_gap_sec = _safe_float(tracking_source.get("largest_gap_sec"))

    tracking_quality_index = round(
        _clamp(
            coverage_pct * 0.50
            + continuity_pct * 0.30
            + sample_sufficiency_pct * 0.20
        ),
        1,
    )

    if coverage_pct >= 50.0 and continuity_pct >= 70.0 and samples_used >= 60:
        tracking_confidence = "medium"
    else:
        tracking_confidence = "low"

    reason_codes: List[str] = []
    if coverage_pct < _LOW_COVERAGE_PCT:
        reason_codes.append("LOW_TRACKING_COVERAGE")
    if continuity_pct < _LOW_CONTINUITY_PCT:
        reason_codes.append("LOW_TRACKLET_CONTINUITY")
    if samples_used < _MIN_TRACKING_SAMPLES:
        reason_codes.append("INSUFFICIENT_TRACKING_SAMPLES")
    if largest_gap_sec is not None and largest_gap_sec > 30.0:
        reason_codes.append("LONG_TRACKING_GAPS")

    reason_codes.extend(
        [
            "IDENTITY_NOT_VERIFIED_ACROSS_SHOTS",
            "CAMERA_MOTION_NOT_COMPENSATED",
            "PITCH_NOT_CALIBRATED",
            "BALL_AND_EVENTS_NOT_MODELLED",
            "PLAYER_SCORING_NOT_VALIDATED",
        ]
    )

    capabilities = {
        "person_detection": True,
        "short_term_tracking": True,
        "cross_shot_player_reidentification": False,
        "camera_motion_compensation": False,
        "pitch_calibration": False,
        "ball_tracking": False,
        "event_detection": False,
        "athletic_metrics": False,
        "technical_tactical_scoring": False,
    }

    limitations = [
        "L'identità del giocatore non è verificata tra cambi camera, replay e occlusioni.",
        "Il movimento è misurato nel piano immagine normalizzato, non in metri sul campo.",
        "Non sono disponibili tracking della palla o eventi tecnico-tattici validati.",
        "Il sistema non produce ancora un voto attendibile del calciatore.",
    ]

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "TRACKING_ONLY",
        "score_kind": "tracking_quality",
        "player_evaluation_available": False,
        "tracking_quality_index": tracking_quality_index,
        "tracking_confidence": tracking_confidence,
        "signals": {
            "coverage_pct": round(coverage_pct, 2),
            "tracklet_continuity_pct": round(continuity_pct, 2),
            "sample_sufficiency_pct": round(sample_sufficiency_pct, 2),
            "samples_used": samples_used,
            "segments_total": segments_total or None,
            "segments_with_player": segments_with_player or None,
            "segments_with_player_pct": (
                round(segments_with_player_pct, 2)
                if segments_with_player_pct is not None
                else None
            ),
            "largest_gap_sec": round(largest_gap_sec, 2) if largest_gap_sec is not None else None,
            "image_motion": image_motion,
        },
        "reason_codes": reason_codes,
        "capabilities": capabilities,
        "limitations": limitations,
        "provenance": {
            "kind": "tracking_quality",
            "version": TRACKING_QUALITY_VERSION,
            "validated_player_score": False,
            "metric_space": "image_plane_normalized",
        },
    }


def apply_evaluation_truth_gate(
    result: Mapping[str, Any] | None,
    *,
    candidate_metrics: Mapping[str, Any] | None = None,
    tracking: Mapping[str, Any] | None = None,
    evidence_metrics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    updated: Dict[str, Any] = dict(_as_mapping(result))
    sanitized_evidence = sanitize_evidence_metrics(evidence_metrics or updated.get("evidence_metrics"))
    if candidate_metrics:
        sanitized_evidence["candidate_metrics"] = dict(candidate_metrics)
    if tracking:
        sanitized_evidence["image_motion"] = compute_image_motion_metrics(tracking)

    evaluation = build_tracking_evaluation(
        candidate_metrics=candidate_metrics,
        tracking=tracking,
    )

    for key in (
        "match_rating_10",
        "matchRating10",
        "impact_100",
        "impact100",
        "impact_adj",
        "impact_components",
        "highlight_adj",
        "baseline_rating",
        "role_group",
        "overall_score",
        "overallScore",
        "role_score",
        "roleScore",
    ):
        updated[key] = None

    updated["radar"] = {}
    updated["breakdown"] = {}
    updated["skills_computed"] = {}
    updated["skills_missing"] = []
    updated.pop("report", None)
    updated.pop("player_runs", None)

    summary = dict(_as_mapping(updated.get("summary")))
    summary.update(
        {
            "evaluation_status": evaluation["status"],
            "player_evaluation_available": False,
            "tracking_quality_index": evaluation["tracking_quality_index"],
            "match_rating_10": None,
            "impact_100": None,
            "overall_score": None,
            "role_score": None,
        }
    )

    updated.update(
        {
            "schema_version": "2.0",
            "summary": summary,
            "evaluation_status": evaluation["status"],
            "score_kind": evaluation["score_kind"],
            "player_evaluation_available": False,
            "tracking_quality_index": evaluation["tracking_quality_index"],
            "tracking_quality": evaluation,
            "tracking_signals": evaluation["signals"],
            "score_provenance": evaluation["provenance"],
            "capabilities": evaluation["capabilities"],
            "limitations": evaluation["limitations"],
            "reason_codes": evaluation["reason_codes"],
            "evidence_metrics": sanitized_evidence,
            "explain": (
                "Diagnostica di computer vision: il numero mostrato misura la qualità "
                "dell'evidenza di tracking, non la qualità calcistica del giocatore. "
                "La valutazione del calciatore è sospesa finché ReID, calibrazione del "
                "campo ed eventi palla non saranno validati."
            ),
        }
    )
    return updated
