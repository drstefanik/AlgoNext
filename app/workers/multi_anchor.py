"""Pure helpers for multi-anchor player tracking.

The module deliberately has no OpenCV/YOLO/database imports so that matching
behaviour can be unit-tested quickly and deterministically.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MAX_ANCHORS = 5


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _first_present(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _normalize_bbox(payload: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    nested = payload.get("bbox_xywh")
    if not isinstance(nested, Mapping):
        nested = payload.get("bbox")
    source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else payload

    values = {name: _as_float(source.get(name)) for name in ("x", "y", "w", "h")}
    if any(value is None for value in values.values()):
        return None

    x = float(values["x"])
    y = float(values["y"])
    w = float(values["w"])
    h = float(values["h"])
    epsilon = 1e-6
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    if x > 1 + epsilon or y > 1 + epsilon:
        return None
    if x + w > 1 + epsilon or y + h > 1 + epsilon:
        return None
    return {
        "x": max(0.0, min(1.0, x)),
        "y": max(0.0, min(1.0, y)),
        "w": max(0.0, min(1.0, w)),
        "h": max(0.0, min(1.0, h)),
    }


def normalize_anchor(payload: Any) -> Optional[Dict[str, Any]]:
    """Normalize one API/player-ref anchor into ``t + bbox`` form."""

    if not isinstance(payload, Mapping):
        return None
    timestamp = _as_float(
        _first_present(
            payload,
            ("frame_time_sec", "time_sec", "frameTimeSec", "timeSec", "t", "best_time_sec"),
        )
    )
    bbox = _normalize_bbox(payload)
    if timestamp is None or timestamp < 0 or bbox is None:
        return None

    frame_key = _first_present(payload, ("frame_key", "frameKey", "key", "best_preview_frame_key"))
    normalized: Dict[str, Any] = {"t": timestamp, **bbox}
    if isinstance(frame_key, str) and frame_key.strip():
        normalized["frame_key"] = frame_key.strip()
    return normalized


def normalize_anchors(payloads: Any, *, max_items: int = MAX_ANCHORS) -> List[Dict[str, Any]]:
    """Normalize, de-duplicate, sort and cap a collection of anchors."""

    if isinstance(payloads, Mapping):
        nested = payloads.get("selections")
        raw_items: Iterable[Any] = nested if isinstance(nested, list) else [payloads]
    elif isinstance(payloads, Iterable) and not isinstance(payloads, (str, bytes)):
        raw_items = payloads
    else:
        raw_items = []

    unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in raw_items:
        anchor = normalize_anchor(item)
        if anchor is None:
            continue
        identity = (
            anchor.get("frame_key") or "",
            round(float(anchor["t"]), 3),
            round(float(anchor["x"]), 4),
            round(float(anchor["y"]), 4),
            round(float(anchor["w"]), 4),
            round(float(anchor["h"]), 4),
        )
        unique.setdefault(identity, anchor)

    limit = max(0, int(max_items))
    return sorted(unique.values(), key=lambda item: float(item["t"]))[:limit]


def anchors_for_window(
    anchors: Sequence[Mapping[str, Any]],
    window_start: float,
    window_end: float,
    *,
    tolerance_sec: float = 1e-6,
) -> List[Dict[str, Any]]:
    start = float(window_start) - max(0.0, float(tolerance_sec))
    end = float(window_end) + max(0.0, float(tolerance_sec))
    return [dict(anchor) for anchor in anchors if start <= float(anchor.get("t", -1.0)) <= end]


def assign_anchors_to_windows(
    anchors: Sequence[Mapping[str, Any]],
    windows: Sequence[Tuple[float, float]],
    *,
    tolerance_sec: float = 1e-6,
) -> List[List[Dict[str, Any]]]:
    """Assign every anchor to one canonical window, even across overlaps."""

    buckets: List[List[Dict[str, Any]]] = [[] for _ in windows]
    tolerance = max(0.0, float(tolerance_sec))
    for anchor in anchors:
        timestamp = _as_float(anchor.get("t"))
        if timestamp is None:
            continue
        candidates = [
            index
            for index, (window_start, window_end) in enumerate(windows)
            if float(window_start) - tolerance
            <= timestamp
            <= float(window_end) + tolerance
        ]
        if not candidates:
            continue
        canonical_index = min(
            candidates,
            key=lambda index: (
                abs(
                    timestamp
                    - (
                        float(windows[index][0])
                        + float(windows[index][1])
                    )
                    * 0.5
                ),
                index,
            ),
        )
        buckets[canonical_index].append(dict(anchor))
    return buckets


def compute_tracking_window(
    anchor_times: Iterable[Any],
    video_duration_sec: Any,
    before_sec: Any,
    after_sec: Any,
) -> Tuple[float, float]:
    """Return a segment covering every valid anchor plus configured margins."""

    before = max(0.0, _as_float(before_sec) or 0.0)
    after = max(0.0, _as_float(after_sec) or 0.0)
    duration = max(0.0, _as_float(video_duration_sec) or 0.0)

    valid_times = [
        timestamp
        for timestamp in (_as_float(value) for value in anchor_times)
        if timestamp is not None and timestamp >= 0
    ]
    if duration > 0:
        valid_times = [min(duration, timestamp) for timestamp in valid_times]

    if valid_times:
        start = max(0.0, min(valid_times) - before)
        end = max(valid_times) + after
    else:
        start = 0.0
        end = before + after

    if duration > 0:
        start = min(start, duration)
        end = min(max(start, end), duration)
    return round(start, 3), round(max(0.0, end - start), 3)


def _bbox_iou(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax1, ay1 = float(a["x"]), float(a["y"])
    ax2, ay2 = ax1 + float(a["w"]), ay1 + float(a["h"])
    bx1, by1 = float(b["x"]), float(b["y"])
    bx2, by2 = bx1 + float(b["w"]), by1 + float(b["h"])
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax = float(a["x"]) + float(a["w"]) * 0.5
    ay = float(a["y"]) + float(a["h"]) * 0.5
    bx = float(b["x"]) + float(b["w"]) * 0.5
    by = float(b["y"]) + float(b["h"]) * 0.5
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _area_similarity_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    area_a = max(1e-9, float(a["w"]) * float(a["h"]))
    area_b = max(1e-9, float(b["w"]) * float(b["h"]))
    return min(area_a, area_b) / max(area_a, area_b)


def _aspect_similarity_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    aspect_a = float(a["w"]) / max(1e-9, float(a["h"]))
    aspect_b = float(b["w"]) / max(1e-9, float(b["h"]))
    return min(aspect_a, aspect_b) / max(aspect_a, aspect_b)


def _size_aspect_penalty(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    area_similarity = max(1e-9, _area_similarity_score(a, b))
    aspect_similarity = max(1e-9, _aspect_similarity_score(a, b))
    return (1.0 / area_similarity - 1.0) + (1.0 / aspect_similarity - 1.0)


def _detection_bbox(detection: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    bbox = detection.get("bbox")
    return _normalize_bbox(bbox if isinstance(bbox, Mapping) else detection)


def _empty_match() -> Dict[str, Any]:
    return {
        "track_id": None,
        "selected_track_ids": [],
        "source": "none",
        "score": 0.0,
        "anchor_time_sec": None,
        "anchor_frame_key": None,
        "anchor_matches": [],
        "metrics": {},
    }


def _select_anchor_match(
    track_map: Mapping[Any, Sequence[Mapping[str, Any]]],
    anchor: Mapping[str, Any],
    *,
    window_start: float,
    max_anchor_delta_sec: float = 1.5,
) -> Optional[Dict[str, Any]]:
    anchor_bbox = _normalize_bbox(anchor)
    if anchor_bbox is None:
        return None

    best_match: Optional[Dict[str, Any]] = None
    local_anchor_time = float(anchor["t"]) - float(window_start)
    for track_id, detections in track_map.items():
        if not detections:
            continue
        closest = min(
            detections,
            key=lambda detection: abs(
                float(detection.get("t", 0.0)) - local_anchor_time
            ),
        )
        delta = abs(float(closest.get("t", 0.0)) - local_anchor_time)
        if delta > max(0.0, float(max_anchor_delta_sec)):
            continue
        candidate_bbox = _detection_bbox(closest)
        if candidate_bbox is None:
            continue

        iou = _bbox_iou(anchor_bbox, candidate_bbox)
        center_distance = _center_distance(anchor_bbox, candidate_bbox)
        center_score = max(0.0, 1.0 - center_distance / 0.30)
        area_score = _area_similarity_score(anchor_bbox, candidate_bbox)
        aspect_score = _aspect_similarity_score(anchor_bbox, candidate_bbox)
        confidence = max(
            0.0, min(1.0, _as_float(closest.get("conf")) or 0.0)
        )
        score = (
            0.55 * iou
            + 0.20 * center_score
            + 0.15 * area_score
            + 0.05 * aspect_score
            + 0.05 * confidence
        )
        credible_shape = area_score >= 0.25 and aspect_score >= 0.45
        credible_position = iou >= 0.18 or (
            center_distance <= 0.12 and area_score >= 0.45
        )
        if not credible_shape or not credible_position or score < 0.28:
            continue

        candidate = {
            "track_id": int(track_id),
            "source": "anchor",
            "score": score,
            "anchor_time_sec": float(anchor["t"]),
            "anchor_frame_key": anchor.get("frame_key"),
            "metrics": {
                "iou": round(iou, 4),
                "center_distance": round(center_distance, 4),
                "area_similarity": round(area_score, 4),
                "aspect_similarity": round(aspect_score, 4),
                "time_delta_sec": round(delta, 4),
            },
        }
        if best_match is None or (
            candidate["score"],
            -delta,
        ) > (
            best_match["score"],
            -float(best_match["metrics"]["time_delta_sec"]),
        ):
            best_match = candidate
    return best_match


def _select_continuity_match(
    track_map: Mapping[Any, Sequence[Mapping[str, Any]]],
    previous_bbox: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    previous = (
        _normalize_bbox(previous_bbox)
        if isinstance(previous_bbox, Mapping)
        else None
    )
    if previous is None:
        return _empty_match()

    best_track_id: Optional[int] = None
    best_score = 0.0
    best_metrics: Dict[str, float] = {}
    for track_id, detections in track_map.items():
        if not detections:
            continue
        closest = min(detections, key=lambda detection: abs(float(detection.get("t", 0.0))))
        candidate_bbox = _detection_bbox(closest)
        if candidate_bbox is None:
            continue
        iou = _bbox_iou(previous, candidate_bbox)
        size_penalty = _size_aspect_penalty(previous, candidate_bbox)
        score = iou - 0.2 * size_penalty
        if score > best_score:
            best_score = score
            best_track_id = int(track_id)
            best_metrics = {
                "iou": round(iou, 4),
                "size_aspect_penalty": round(size_penalty, 4),
            }

    if best_track_id is None or best_score < 0.15:
        return _empty_match()
    return {
        "track_id": best_track_id,
        "selected_track_ids": [best_track_id],
        "source": "continuity",
        "score": round(best_score, 4),
        "anchor_time_sec": None,
        "anchor_frame_key": None,
        "anchor_matches": [],
        "metrics": best_metrics,
    }


def select_window_tracks(
    track_map: Mapping[Any, Sequence[Mapping[str, Any]]],
    anchors: Sequence[Mapping[str, Any]],
    *,
    window_start: float,
    window_end: float,
    previous_bbox: Optional[Mapping[str, Any]] = None,
    max_anchor_delta_sec: float = 1.5,
) -> Dict[str, Any]:
    """Select every credible anchor-local track in temporal anchor order.

    ``track_id`` remains the strongest individual match for callers that use
    the legacy singular field. ``selected_track_ids`` and ``anchor_matches``
    retain all matches so a camera cut or occlusion can change ByteTrack's
    local ID inside a single window without discarding later manual anchors.
    """

    if not track_map:
        return _empty_match()

    anchor_matches: List[Dict[str, Any]] = []
    best_anchor_match: Optional[Dict[str, Any]] = None
    for anchor in anchors_for_window(anchors, window_start, window_end):
        candidate = _select_anchor_match(
            track_map,
            anchor,
            window_start=window_start,
            max_anchor_delta_sec=max_anchor_delta_sec,
        )
        if candidate is None:
            continue
        anchor_matches.append(candidate)
        if best_anchor_match is None or (
            candidate["score"],
            -float(candidate["metrics"]["time_delta_sec"]),
        ) > (
            best_anchor_match["score"],
            -float(best_anchor_match["metrics"]["time_delta_sec"]),
        ):
            best_anchor_match = candidate

    if best_anchor_match is None:
        return _select_continuity_match(track_map, previous_bbox)

    normalized_matches = []
    for candidate in anchor_matches:
        normalized = dict(candidate)
        normalized["score"] = round(float(candidate["score"]), 4)
        normalized_matches.append(normalized)
    selected_track_ids = list(
        dict.fromkeys(int(match["track_id"]) for match in normalized_matches)
    )
    result = dict(best_anchor_match)
    result["score"] = round(float(best_anchor_match["score"]), 4)
    result["selected_track_ids"] = selected_track_ids
    result["anchor_matches"] = normalized_matches
    return result


def select_track_id_at_time(
    anchor_matches: Sequence[Mapping[str, Any]],
    timestamp: Any,
    *,
    fallback_track_id: Optional[int] = None,
) -> Optional[int]:
    """Resolve temporal anchor ownership using deterministic midpoint bounds."""

    target_time = _as_float(timestamp)
    ordered = [
        match
        for match in anchor_matches
        if _as_float(match.get("anchor_time_sec")) is not None
        and _as_float(match.get("track_id")) is not None
    ]
    ordered.sort(key=lambda match: float(match["anchor_time_sec"]))
    if target_time is None or not ordered:
        return fallback_track_id

    for index, match in enumerate(ordered[:-1]):
        next_match = ordered[index + 1]
        boundary = (
            float(match["anchor_time_sec"])
            + float(next_match["anchor_time_sec"])
        ) * 0.5
        if target_time < boundary:
            return int(match["track_id"])
    return int(ordered[-1]["track_id"])


def select_window_track(
    track_map: Mapping[Any, Sequence[Mapping[str, Any]]],
    anchors: Sequence[Mapping[str, Any]],
    *,
    window_start: float,
    window_end: float,
    previous_bbox: Optional[Mapping[str, Any]] = None,
    max_anchor_delta_sec: float = 1.5,
) -> Dict[str, Any]:
    """Backward-compatible singular entrypoint with multi-track diagnostics."""

    return select_window_tracks(
        track_map,
        anchors,
        window_start=window_start,
        window_end=window_end,
        previous_bbox=previous_bbox,
        max_anchor_delta_sec=max_anchor_delta_sec,
    )
