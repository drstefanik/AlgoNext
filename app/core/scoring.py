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
        return {"overall_score": None, "role_score": None}

    if not set(expected_keys).issubset(radar.keys()):
        return {"overall_score": None, "role_score": None}

    weights = ROLE_WEIGHTS.get(role, DEFAULT_WEIGHTS)
    pairs: List[Tuple[float, float]] = []
    for key in expected_keys:
        try:
            score = float(radar[key])
        except Exception:
            return {"overall_score": None, "role_score": None}
        score = max(0.0, min(100.0, score))
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        pairs.append((score, weight))

    if not pairs:
        return {"overall_score": None, "role_score": None}

    weighted_sum = sum(score * weight for score, weight in pairs)
    weight_sum = sum(weight for _, weight in pairs) or 1.0
    value = weighted_sum / weight_sum
    value = max(0.0, min(100.0, value))
    rounded = round(value, 1)
    return {"overall_score": rounded, "role_score": rounded}


def _fallback_radar_from_tracking(tracking: Dict[str, Any] | None) -> Dict[str, float]:
    if not isinstance(tracking, dict):
        return {}
    coverage_pct = float(tracking.get("coverage_pct") or 0.0)
    lost_segments = tracking.get("lost_segments") or []
    stability_penalty = min(100.0, float(len(lost_segments)) * 10.0)
    stability_score = max(0.0, 100.0 - stability_penalty)
    return {
        "coverage": max(0.0, min(100.0, coverage_pct)),
        "stability": max(0.0, min(100.0, stability_score)),
    }


def _average_radar_score(radar: Dict[str, float]) -> float:
    if not radar:
        return 0.0
    values = [float(value) for value in radar.values()]
    return sum(values) / float(len(values))


def compute_evaluation(
    role: str, radar: Dict[str, Any], tracking: Dict[str, Any] | None
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
        sanitized_radar = _fallback_radar_from_tracking(tracking)
    score_payload = compute_scores(role, sanitized_radar)
    overall = score_payload.get("overall_score")
    role_score = score_payload.get("role_score")
    if overall is None or role_score is None:
        average_score = _average_radar_score(sanitized_radar)
        overall = round(average_score, 1)
        role_score = round(average_score, 1)
    return {
        "radar": sanitized_radar,
        "overall_score": overall,
        "role_score": role_score,
    }
