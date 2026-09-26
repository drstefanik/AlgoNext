"""Conservative continuity checks in camera-relative image coordinates.

These are identity guardrails, not calibrated athletic motion measurements.
"""

import math

from app.reid.window_logic import bbox_iou, center_distance
from app.vision.match_observations import estimate_camera_motion


def annotate_motion_context(samples, track_map):
    offsets = estimate_camera_motion(samples).get("_offsets") or []
    contexts = {}
    group = 0
    for index, (sample, offset) in enumerate(zip(samples, offsets)):
        if index and not offset.get("supported"):
            group += 1
        for detection in sample.get("detections") or []:
            box = detection.get("bbox") or {}
            # If two bodies overlap substantially, a raw tracker ID is not
            # sufficient evidence to propagate identity through the crossing.
            crowded = any(
                other.get("track_id") != detection.get("track_id")
                and bbox_iou(box, other.get("bbox") or {}) >= 0.20
                for other in sample.get("detections") or []
            )
            contexts[(index, detection.get("track_id"))] = {
                "group": group,
                "x": offset["x"],
                "y": offset["y"],
                "crowded": crowded,
            }
    for track_id, detections in track_map.items():
        for detection in detections:
            context = contexts.get((detection.get("sample_index"), int(track_id)))
            if context is not None:
                detection["_motion_context"] = context


def camera_relative_continuity(previous, current):
    """Return None for legacy data, otherwise attest a bounded local link."""
    a, b = previous.get("_motion_context"), current.get("_motion_context")
    if a is None and b is None:
        return None
    if not a or not b or a["crowded"] or b["crowded"]:
        return False
    first, second = dict(previous["bbox"]), dict(current["bbox"])
    if a["group"] == b["group"]:
        first["x"] -= a["x"]
        first["y"] -= a["y"]
        second["x"] -= b["x"]
        second["y"] -= b["y"]
    elif bbox_iou(first, second) < 0.30:
        # With no multi-player camera-motion consensus, allow only an
        # immediate overlapping observation; never bridge a pan by proximity.
        return False
    gap = abs(float(current["t"]) - float(previous["t"]))
    width = (float(first["w"]) + float(second["w"])) / 2
    height = (float(first["h"]) + float(second["h"])) / 2
    limit = max(0.006, 0.8 * width + 0.6 * height * gap)
    distance = center_distance(first, second)
    return math.isfinite(distance) and distance <= min(0.20, limit)
