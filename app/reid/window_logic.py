from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

BBox = Mapping[str, Any]
Detection = Mapping[str, Any]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def bbox_iou(first: BBox, second: BBox) -> float:
    ax1 = _finite(first.get("x"))
    ay1 = _finite(first.get("y"))
    ax2 = ax1 + max(0.0, _finite(first.get("w")))
    ay2 = ay1 + max(0.0, _finite(first.get("h")))
    bx1 = _finite(second.get("x"))
    by1 = _finite(second.get("y"))
    bx2 = bx1 + max(0.0, _finite(second.get("w")))
    by2 = by1 + max(0.0, _finite(second.get("h")))
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(
        0.0, bx2 - bx1
    ) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(first: BBox, second: BBox) -> float:
    first_x = _finite(first.get("x")) + max(0.0, _finite(first.get("w"))) / 2.0
    first_y = _finite(first.get("y")) + max(0.0, _finite(first.get("h"))) / 2.0
    second_x = _finite(second.get("x")) + max(0.0, _finite(second.get("w"))) / 2.0
    second_y = _finite(second.get("y")) + max(0.0, _finite(second.get("h"))) / 2.0
    return math.hypot(first_x - second_x, first_y - second_y)


def geometry_similarity(first: BBox, second: BBox) -> float:
    first_area = max(1e-9, _finite(first.get("w")) * _finite(first.get("h")))
    second_area = max(1e-9, _finite(second.get("w")) * _finite(second.get("h")))
    area_ratio = min(first_area, second_area) / max(first_area, second_area)
    first_aspect = max(1e-9, _finite(first.get("w"))) / max(
        1e-9, _finite(first.get("h"))
    )
    second_aspect = max(1e-9, _finite(second.get("w"))) / max(
        1e-9, _finite(second.get("h"))
    )
    aspect_ratio = min(first_aspect, second_aspect) / max(first_aspect, second_aspect)
    distance_score = math.exp(-center_distance(first, second) / 0.18)
    return max(
        0.0,
        min(
            1.0,
            bbox_iou(first, second) * 0.45
            + distance_score * 0.30
            + area_ratio * 0.15
            + aspect_ratio * 0.10,
        ),
    )


def processing_order(
    windows: Sequence[tuple[float, float]], anchor_time: float
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if not windows:
        raise ValueError("windows must not be empty")
    anchor = float(anchor_time)
    containing = [
        index
        for index, (start, end) in enumerate(windows)
        if float(start) <= anchor <= float(end)
    ]
    if containing:
        anchor_index = min(
            containing,
            key=lambda index: abs(
                (float(windows[index][0]) + float(windows[index][1])) / 2.0 - anchor
            ),
        )
    else:
        anchor_index = min(
            range(len(windows)),
            key=lambda index: abs(
                (float(windows[index][0]) + float(windows[index][1])) / 2.0 - anchor
            ),
        )
    forward = tuple(range(anchor_index + 1, len(windows)))
    backward = tuple(range(anchor_index - 1, -1, -1))
    return anchor_index, forward, backward


def choose_descriptor_detections(
    detections: Sequence[Detection], max_samples: int
) -> list[Detection]:
    if max_samples <= 0 or not detections:
        return []
    ordered = sorted(
        detections,
        key=lambda detection: (
            _finite(detection.get("sample_index"), _finite(detection.get("t"))),
            _finite(detection.get("t")),
        ),
    )
    if len(ordered) <= max_samples:
        return list(ordered)
    quality_index = max(
        range(len(ordered)),
        key=lambda index: (
            _finite(ordered[index].get("conf"))
            * max(0.0, _finite(ordered[index].get("bbox", {}).get("w")))
            * max(0.0, _finite(ordered[index].get("bbox", {}).get("h")))
        ),
    )
    if max_samples == 1:
        return [ordered[quality_index]]
    indices = [
        int(round(position * (len(ordered) - 1) / float(max_samples - 1)))
        for position in range(max_samples)
    ]
    replace_at = min(
        range(len(indices)), key=lambda index: abs(indices[index] - quality_index)
    )
    indices[replace_at] = quality_index
    unique: list[int] = []
    for index in indices:
        if index not in unique:
            unique.append(index)
    for index in range(len(ordered)):
        if len(unique) >= max_samples:
            break
        if index not in unique:
            unique.append(index)
    return [ordered[index] for index in sorted(unique[:max_samples])]


def temporal_overlap_score(
    previous_bboxes: Sequence[BBox],
    detections: Sequence[Detection],
    *,
    time_offset: float,
    tolerance_sec: float,
) -> float | None:
    if not previous_bboxes or not detections:
        return None
    previous = [
        bbox for bbox in previous_bboxes if isinstance(bbox, Mapping) and "t" in bbox
    ]
    if not previous:
        return None
    overlaps: list[float] = []
    for detection in detections:
        bbox = detection.get("bbox")
        if not isinstance(bbox, Mapping):
            continue
        absolute_time = _finite(detection.get("t")) + float(time_offset)
        closest = min(
            previous,
            key=lambda item: abs(_finite(item.get("t")) - absolute_time),
        )
        if abs(_finite(closest.get("t")) - absolute_time) <= tolerance_sec:
            overlaps.append(bbox_iou(closest, bbox))
    if not overlaps:
        return None
    strongest = sorted(overlaps, reverse=True)[:3]
    return max(0.0, min(1.0, sum(strongest) / len(strongest)))


def candidate_rank(
    detections: Sequence[Detection],
    *,
    overlap_score: float | None,
    geometry_score: float | None,
) -> float:
    if not detections:
        return 0.0
    mean_confidence = sum(_finite(item.get("conf")) for item in detections) / len(
        detections
    )
    presence = min(1.0, len(detections) / 20.0)
    return (
        (overlap_score or 0.0) * 3.0
        + (geometry_score or 0.0) * 1.25
        + presence * 0.75
        + mean_confidence * 0.25
    )


def tracking_coverage_pct(
    segments: Sequence[Mapping[str, Any]], *, duration_sec: float, fps: float
) -> float:
    if duration_sec <= 0 or fps <= 0:
        return 0.0
    observed: set[int] = set()
    for segment in segments:
        for bbox in segment.get("bboxes") or []:
            if isinstance(bbox, Mapping) and bbox.get("t") is not None:
                observed.add(int(round(_finite(bbox.get("t")) * fps)))
    expected = max(1, int(round(duration_sec * fps)))
    return max(0.0, min(100.0, len(observed) / float(expected) * 100.0))


def largest_tracking_gap_sec(
    segments: Sequence[Mapping[str, Any]], *, duration_sec: float
) -> float:
    times = sorted(
        {
            _finite(bbox.get("t"))
            for segment in segments
            for bbox in (segment.get("bboxes") or [])
            if isinstance(bbox, Mapping) and bbox.get("t") is not None
        }
    )
    if not times:
        return max(0.0, float(duration_sec))
    gaps = [max(0.0, times[0])]
    gaps.extend(
        max(0.0, current - previous) for previous, current in zip(times, times[1:])
    )
    gaps.append(max(0.0, float(duration_sec) - times[-1]))
    return max(gaps, default=0.0)
