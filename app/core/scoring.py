from typing import Any, Dict, List, Optional, Tuple

ROLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Striker": {
        "Finishing": 0.24,
        "Positioning": 0.20,
        "Off-the-ball Movement": 0.20,
        "Composure": 0.16,
        "Shot Power": 0.12,
        "Heading": 0.08,
    },
    "Winger": {
        "Off-the-ball Movement": 0.22,
        "Positioning": 0.18,
        "Composure": 0.16,
        "Shot Power": 0.16,
        "Finishing": 0.18,
        "Heading": 0.10,
    },
    "Midfielder": {
        "Composure": 0.22,
        "Positioning": 0.22,
        "Off-the-ball Movement": 0.20,
        "Shot Power": 0.14,
        "Finishing": 0.14,
        "Heading": 0.08,
    },
    "Defender": {
        "Heading": 0.22,
        "Positioning": 0.22,
        "Composure": 0.18,
        "Off-the-ball Movement": 0.18,
        "Shot Power": 0.10,
        "Finishing": 0.10,
    },
    "Goalkeeper": {
        "Composure": 0.40,
        "Positioning": 0.30,
        "Off-the-ball Movement": 0.10,
        "Heading": 0.05,
        "Shot Power": 0.10,
        "Finishing": 0.05,
    },
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "Finishing": 1.0,
    "Heading": 1.0,
    "Positioning": 1.0,
    "Composure": 1.0,
    "Off-the-ball Movement": 1.0,
    "Shot Power": 1.0,
}


def weighted_overall_score(radar: Dict[str, float], role: Optional[str]) -> int:
    role = (role or "").strip()
    weights = ROLE_WEIGHTS.get(role, DEFAULT_WEIGHTS)

    pairs: List[Tuple[float, float]] = []
    for k, score in radar.items():
        if k not in weights:
            continue
        try:
            s = float(score)
        except Exception:
            continue
        s = max(0.0, min(100.0, s))
        pairs.append((s, float(weights[k])))

    if not pairs:
        return 0

    weighted_sum = sum(s * w for s, w in pairs)
    w_sum = sum(w for _, w in pairs) or 1.0
    value = weighted_sum / w_sum
    value = max(0.0, min(100.0, value))
    return int(round(value))


def keys_required_for_role(role: Optional[str]) -> List[str]:
    role = (role or "").strip()
    weights = ROLE_WEIGHTS.get(role, DEFAULT_WEIGHTS)
    return list(weights.keys())


def compute_scores(role: str, radar: Dict[str, Any]) -> Dict[str, Optional[float]]:
    expected_keys = keys_required_for_role(role)
    if not expected_keys:
        return {"overall_score": None, "role_score": None, "weight_sum": 0.0}

    weights = ROLE_WEIGHTS.get(role, DEFAULT_WEIGHTS)
    pairs: List[Tuple[float, float]] = []
    for key, value in radar.items():
        if key not in weights:
            continue
        try:
            score = float(value)
        except Exception:
            continue
        score = max(0.0, min(100.0, score))
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        pairs.append((score, weight))

    if not pairs:
        return {"overall_score": None, "role_score": None, "weight_sum": 0.0}

    weighted_sum = sum(score * weight for score, weight in pairs)
    weight_sum = sum(weight for _, weight in pairs)
    if weight_sum <= 0:
        return {"overall_score": None, "role_score": None, "weight_sum": 0.0}

    value = weighted_sum / weight_sum
    value = max(0.0, min(100.0, value))
    rounded = round(value, 1)
    return {"overall_score": rounded, "role_score": rounded, "weight_sum": weight_sum}


def _fallback_radar_from_tracking(
    tracking: Dict[str, Any] | None, evidence_metrics: Dict[str, Any] | None
) -> Dict[str, float]:
    def clamp(value: float) -> float:
        return max(0.0, min(100.0, value))

    def normalize(value: float | None, max_value: float) -> float | None:
        if value is None:
            return None
        if max_value <= 0:
            return None
        return clamp((float(value) / max_value) * 100.0)

    def has_metric(values: List[float | None]) -> bool:
        return any(value is not None and value > 0 for value in values)

    distance_m = None
    avg_speed_kmh = None
    top_speed_kmh = None
    if isinstance(evidence_metrics, dict):
        distance_m = evidence_metrics.get("distance_covered_m")
        avg_speed_kmh = evidence_metrics.get("avg_speed_kmh")
        top_speed_kmh = evidence_metrics.get("top_speed_kmh")

    coverage_pct = None
    stability_score = None
    lost_segments = []
    if isinstance(tracking, dict):
        coverage_pct = tracking.get("coverage_pct")
        stability_score = tracking.get("stability_score")
        lost_segments = tracking.get("lost_segments") or []

    visibility_score = None
    if coverage_pct is not None:
        try:
            visibility_score = clamp(float(coverage_pct))
        except (TypeError, ValueError):
            visibility_score = None

    stability_value = None
    if stability_score is not None:
        try:
            stability_value = clamp(float(stability_score) * 100.0)
        except (TypeError, ValueError):
            stability_value = None
    if stability_value is None and isinstance(lost_segments, list):
        stability_penalty = min(100.0, float(len(lost_segments)) * 10.0)
        stability_value = max(0.0, 100.0 - stability_penalty)

    distance_score = None
    avg_speed_score = None
    top_speed_score = None
    try:
        if distance_m is not None:
            distance_score = normalize(float(distance_m), 1500.0)
    except (TypeError, ValueError):
        distance_score = None
    try:
        if avg_speed_kmh is not None:
            avg_speed_score = normalize(float(avg_speed_kmh), 18.0)
    except (TypeError, ValueError):
        avg_speed_score = None
    try:
        if top_speed_kmh is not None:
            top_speed_score = normalize(float(top_speed_kmh), 28.0)
    except (TypeError, ValueError):
        top_speed_score = None

    if has_metric(
        [distance_score, avg_speed_score, top_speed_score, visibility_score, stability_value]
    ):
        radar: Dict[str, float] = {}
        if distance_score is not None:
            radar["distance_covered"] = distance_score
        if avg_speed_score is not None:
            radar["avg_speed"] = avg_speed_score
        if top_speed_score is not None:
            radar["top_speed"] = top_speed_score
        if visibility_score is not None:
            radar["visibility"] = visibility_score
        if stability_value is not None:
            radar["stability"] = clamp(stability_value)
        return radar

    if not isinstance(tracking, dict):
        return {
            "tracking_quality": 0.0,
            "activity_proxy": 0.0,
            "visibility": 0.0,
            "consistency": 0.0,
            "stability": 0.0,
        }

    coverage_pct = float(tracking.get("coverage_pct") or 0.0)
    lost_segments = tracking.get("lost_segments") or []
    stability_penalty = min(100.0, float(len(lost_segments)) * 10.0)
    stability_score = max(0.0, 100.0 - stability_penalty)

    bboxes = tracking.get("bboxes") or []
    centers = []
    for bbox in bboxes:
        try:
            cx = float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0)) / 2.0
            cy = float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0)) / 2.0
            t = float(bbox.get("t", 0.0))
        except (TypeError, ValueError):
            continue
        centers.append((t, cx, cy))

    centers.sort(key=lambda item: item[0])
    distances = []
    speeds = []
    for idx in range(1, len(centers)):
        t0, x0, y0 = centers[idx - 1]
        t1, x1, y1 = centers[idx]
        dt = max(0.0, t1 - t0)
        dx = x1 - x0
        dy = y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        distances.append(dist)
        if dt > 0:
            speeds.append(dist / dt)

    avg_speed = sum(speeds) / float(len(speeds)) if speeds else 0.0
    speed_score = clamp((avg_speed / 0.05) * 100.0)  # normalized motion proxy

    if distances:
        median_distance = sorted(distances)[len(distances) // 2]
        outliers = sum(1 for d in distances if d > max(0.01, median_distance * 3))
        consistency_score = clamp(100.0 - (outliers / float(len(distances))) * 100.0)
    else:
        consistency_score = 0.0

    visibility_score = clamp(coverage_pct)
    tracking_quality = clamp((visibility_score + stability_score) / 2.0)

    return {
        "tracking_quality": tracking_quality,
        "activity_proxy": speed_score,
        "visibility": visibility_score,
        "consistency": consistency_score,
        "stability": clamp(stability_score),
    }


def _average_radar_score(radar: Dict[str, float]) -> float:
    if not radar:
        return 0.0
    values = [float(value) for value in radar.values()]
    return sum(values) / float(len(values))


def compute_evaluation(
    role: str,
    radar: Dict[str, Any],
    tracking: Dict[str, Any] | None,
    evidence_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    sanitized_radar = {}
    for key, value in (radar or {}).items():
        if value is None:
            continue
        try:
            sanitized_radar[key] = float(value)
        except (TypeError, ValueError):
            continue
    if not sanitized_radar:
        sanitized_radar = _fallback_radar_from_tracking(tracking, evidence_metrics)
    score_payload = compute_scores(role, sanitized_radar)
    overall = score_payload.get("overall_score")
    role_score = score_payload.get("role_score")
    weight_sum = score_payload.get("weight_sum", 0.0) or 0.0
    if overall is None or role_score is None:
        average_score = _average_radar_score(sanitized_radar)
        overall = round(average_score, 1)
        role_score = round(average_score, 1)
    return {
        "radar": sanitized_radar,
        "overall_score": overall,
        "role_score": role_score,
        "weight_sum": weight_sum,
    }
