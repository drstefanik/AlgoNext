from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

OBSERVABILITY_SCHEMA_VERSION = "match-observability-v1"
CAMERA_MOTION_METHOD = "multi-person-median-displacement-v1"
BALL_TRACKING_METHOD = "yolo-coco-sports-ball+bytetrack-v1"
EVENT_DETECTION_METHOD = "selected-player-ball-proximity-v1"


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bbox_center(value: Any, *, foot: bool = False) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    x = _finite(value.get("x"))
    y = _finite(value.get("y"))
    width = _finite(value.get("w"))
    height = _finite(value.get("h"))
    if None in (x, y, width, height) or width <= 0 or height <= 0:
        return None
    return (
        float(x) + float(width) * 0.5,
        float(y) + (float(height) if foot else float(height) * 0.5),
    )


def _track_id(value: Any) -> int | None:
    parsed = _finite(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _person_centers(sample: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    centers: dict[int, tuple[float, float]] = {}
    for detection in sample.get("detections") or []:
        if not isinstance(detection, Mapping):
            continue
        normalized_track_id = _track_id(detection.get("track_id"))
        if normalized_track_id is None:
            continue
        center = _bbox_center(detection.get("bbox"))
        if center is not None:
            centers[normalized_track_id] = center
    return centers


def estimate_camera_motion(
    samples: Sequence[Mapping[str, Any]],
    *,
    minimum_shared_tracks: int = 3,
    minimum_coverage_ratio: float = 0.35,
) -> dict[str, Any]:
    """Estimate broadcast-camera translation from the robust crowd motion.

    The median displacement of several independently tracked people is used as a
    conservative pan/tilt proxy. It is intentionally scoped to one continuous
    processing window and never crosses a camera cut or tracker reset.
    """

    ordered = sorted(
        (
            sample
            for sample in samples
            if isinstance(sample, Mapping) and _finite(sample.get("t")) is not None
        ),
        key=lambda item: float(item["t"]),
    )
    transitions_total = max(0, len(ordered) - 1)
    transitions_compensated = 0
    shared_tracks_total = 0
    cumulative_x = 0.0
    cumulative_y = 0.0
    offsets: list[dict[str, Any]] = []
    if ordered:
        offsets.append(
            {
                "t": round(float(ordered[0]["t"]), 6),
                "x": 0.0,
                "y": 0.0,
                "supported": False,
            }
        )

    for previous, current in zip(ordered, ordered[1:]):
        previous_centers = _person_centers(previous)
        current_centers = _person_centers(current)
        shared = sorted(set(previous_centers).intersection(current_centers))
        supported = False
        if len(shared) >= minimum_shared_tracks:
            displacements = [
                (
                    current_centers[track_id][0] - previous_centers[track_id][0],
                    current_centers[track_id][1] - previous_centers[track_id][1],
                )
                for track_id in shared
            ]
            median_x = median(item[0] for item in displacements)
            median_y = median(item[1] for item in displacements)
            residuals = [
                math.hypot(item[0] - median_x, item[1] - median_y)
                for item in displacements
            ]
            residual_median = median(residuals)
            inlier_limit = max(0.006, residual_median * 3.0)
            inliers = [
                displacement
                for displacement, residual in zip(displacements, residuals)
                if residual <= inlier_limit
            ]
            if len(inliers) >= minimum_shared_tracks:
                cumulative_x += median(item[0] for item in inliers)
                cumulative_y += median(item[1] for item in inliers)
                transitions_compensated += 1
                shared_tracks_total += len(inliers)
                supported = True
        offsets.append(
            {
                "t": round(float(current["t"]), 6),
                "x": round(cumulative_x, 8),
                "y": round(cumulative_y, 8),
                "supported": supported,
            }
        )

    coverage_ratio = (
        transitions_compensated / float(transitions_total)
        if transitions_total > 0
        else 0.0
    )
    available = bool(
        transitions_compensated >= 2 and coverage_ratio >= minimum_coverage_ratio
    )
    return {
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "status": "AVAILABLE" if available else "INSUFFICIENT_EVIDENCE",
        "available": available,
        "validated": False,
        "method": CAMERA_MOTION_METHOD,
        "transitions_total": transitions_total,
        "transitions_compensated": transitions_compensated,
        "coverage_ratio": round(coverage_ratio, 6),
        "mean_shared_tracks": round(
            shared_tracks_total / float(max(1, transitions_compensated)),
            3,
        ),
        "_offsets": offsets,
        "reason_codes": (
            ["EXPERIMENTAL_NOT_BENCHMARK_VALIDATED"]
            if available
            else ["INSUFFICIENT_MULTI_PERSON_MOTION_CONSENSUS"]
        ),
    }


def _nearest_offset(
    offsets: Sequence[Mapping[str, Any]],
    time_sec: float,
) -> Mapping[str, Any] | None:
    if not offsets:
        return None
    return min(
        offsets,
        key=lambda item: abs(float(item.get("t") or 0.0) - time_sec),
    )


def summarize_compensated_player_motion(
    player_bboxes: Sequence[Mapping[str, Any]],
    *,
    window_start: float,
    camera_motion: Mapping[str, Any],
) -> dict[str, Any]:
    offsets = [
        item
        for item in camera_motion.get("_offsets") or []
        if isinstance(item, Mapping)
    ]
    points: list[tuple[float, float, float, float, float]] = []
    for bbox in player_bboxes:
        if not isinstance(bbox, Mapping):
            continue
        timestamp = _finite(bbox.get("t"))
        center = _bbox_center(bbox, foot=True)
        if timestamp is None or center is None:
            continue
        local_time = float(timestamp) - float(window_start)
        offset = _nearest_offset(offsets, local_time)
        offset_x = _finite(offset.get("x")) if offset else 0.0
        offset_y = _finite(offset.get("y")) if offset else 0.0
        points.append(
            (
                float(timestamp),
                center[0],
                center[1],
                center[0] - float(offset_x or 0.0),
                center[1] - float(offset_y or 0.0),
            )
        )
    points.sort(key=lambda item: item[0])
    raw_path = 0.0
    compensated_path = 0.0
    for previous, current in zip(points, points[1:]):
        if current[0] <= previous[0]:
            continue
        raw_path += math.hypot(current[1] - previous[1], current[2] - previous[2])
        compensated_path += math.hypot(
            current[3] - previous[3],
            current[4] - previous[4],
        )
    available = bool(camera_motion.get("available") and len(points) >= 2)
    return {
        "status": "AVAILABLE" if available else "INSUFFICIENT_EVIDENCE",
        "available": available,
        "validated": False,
        "metric_space": "camera_compensated_image_plane_normalized",
        "observed_samples": len(points),
        "raw_path_length": round(raw_path, 6),
        "compensated_path_length": round(compensated_path, 6),
    }


def summarize_ball_tracking(
    samples: Sequence[Mapping[str, Any]],
    *,
    window_start: float,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    sampled_frames = 0
    for sample in sorted(
        (item for item in samples if isinstance(item, Mapping)),
        key=lambda item: float(item.get("t") or 0.0),
    ):
        sampled_frames += 1
        candidates = [
            item
            for item in sample.get("ball_detections") or []
            if isinstance(item, Mapping) and _bbox_center(item.get("bbox")) is not None
        ]
        if not candidates:
            continue
        selected = max(
            candidates,
            key=lambda item: float(_finite(item.get("conf")) or 0.0),
        )
        bbox = selected["bbox"]
        observations.append(
            {
                "t": round(
                    float(window_start) + float(_finite(sample.get("t")) or 0.0),
                    6,
                ),
                "x": round(float(bbox["x"]), 8),
                "y": round(float(bbox["y"]), 8),
                "w": round(float(bbox["w"]), 8),
                "h": round(float(bbox["h"]), 8),
                "conf": round(float(_finite(selected.get("conf")) or 0.0), 6),
                "track_id": _track_id(selected.get("track_id")),
            }
        )
    coverage_ratio = len(observations) / float(max(1, sampled_frames))
    available = len(observations) >= 2
    return {
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "status": "AVAILABLE" if available else "INSUFFICIENT_EVIDENCE",
        "available": available,
        "validated": False,
        "method": BALL_TRACKING_METHOD,
        "sampled_frames": sampled_frames,
        "observed_samples": len(observations),
        "sample_coverage_ratio": round(coverage_ratio, 6),
        "observations": observations,
        "reason_codes": (
            ["EXPERIMENTAL_NOT_BENCHMARK_VALIDATED"]
            if available
            else ["INSUFFICIENT_BALL_DETECTIONS"]
        ),
    }


def detect_selected_player_ball_events(
    player_bboxes: Sequence[Mapping[str, Any]],
    ball_tracking: Mapping[str, Any],
    *,
    fps: float,
) -> dict[str, Any]:
    players = [
        item
        for item in player_bboxes
        if isinstance(item, Mapping)
        and _finite(item.get("t")) is not None
        and _bbox_center(item, foot=True) is not None
    ]
    players.sort(key=lambda item: float(item["t"]))
    balls = [
        item
        for item in ball_tracking.get("observations") or []
        if isinstance(item, Mapping)
        and _finite(item.get("t")) is not None
        and _bbox_center(item) is not None
    ]
    tolerance = max(0.25, 0.75 / float(max(1.0, fps)))
    proximity_samples: list[dict[str, Any]] = []
    for ball in balls:
        if not players:
            break
        timestamp = float(ball["t"])
        player = min(players, key=lambda item: abs(float(item["t"]) - timestamp))
        if abs(float(player["t"]) - timestamp) > tolerance:
            continue
        player_point = _bbox_center(player, foot=True)
        ball_point = _bbox_center(ball)
        height = max(0.02, float(_finite(player.get("h")) or 0.0))
        distance = math.hypot(
            ball_point[0] - player_point[0],
            ball_point[1] - player_point[1],
        )
        normalized_distance = distance / height
        if normalized_distance <= 0.85:
            proximity_samples.append(
                {
                    "t": timestamp,
                    "distance_player_heights": normalized_distance,
                    "confidence": float(_finite(ball.get("conf")) or 0.0),
                }
            )

    events: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    maximum_gap = max(0.75, 2.0 / float(max(1.0, fps)))

    def flush() -> None:
        if not current:
            return
        if len(current) >= 2:
            events.append(
                {
                    "type": "BALL_PROXIMITY_SEQUENCE",
                    "start_sec": round(float(current[0]["t"]), 3),
                    "end_sec": round(float(current[-1]["t"]), 3),
                    "samples": len(current),
                    "minimum_distance_player_heights": round(
                        min(
                            float(item["distance_player_heights"])
                            for item in current
                        ),
                        4,
                    ),
                    "confidence": round(
                        sum(float(item["confidence"]) for item in current)
                        / float(len(current)),
                        4,
                    ),
                }
            )
        current.clear()

    for item in proximity_samples:
        if current and float(item["t"]) - float(current[-1]["t"]) > maximum_gap:
            flush()
        current.append(item)
    flush()

    available = bool(ball_tracking.get("available") and len(players) >= 2)
    return {
        "schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "status": "AVAILABLE" if available else "INSUFFICIENT_EVIDENCE",
        "available": available,
        "validated": False,
        "method": EVENT_DETECTION_METHOD,
        "events": events,
        "event_count": len(events),
        "reason_codes": (
            ["PROXIMITY_EVENTS_ONLY", "EXPERIMENTAL_NOT_BENCHMARK_VALIDATED"]
            if available
            else ["PLAYER_AND_BALL_OVERLAP_REQUIRED"]
        ),
    }


def build_segment_observability(
    samples: Sequence[Mapping[str, Any]],
    player_bboxes: Sequence[Mapping[str, Any]],
    *,
    window_start: float,
    fps: float,
) -> dict[str, Any]:
    camera = estimate_camera_motion(samples)
    player_motion = summarize_compensated_player_motion(
        player_bboxes,
        window_start=window_start,
        camera_motion=camera,
    )
    camera_public = {key: value for key, value in camera.items() if key != "_offsets"}
    camera_public["player_motion"] = player_motion
    ball = summarize_ball_tracking(samples, window_start=window_start)
    events = detect_selected_player_ball_events(player_bboxes, ball, fps=fps)
    return {
        "camera_motion": camera_public,
        "ball_tracking": ball,
        "event_detection": events,
    }


def aggregate_segment_observability(
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    camera_segments = [
        item.get("camera_motion")
        for item in segments
        if isinstance(item, Mapping) and isinstance(item.get("camera_motion"), Mapping)
    ]
    camera_total = sum(
        int(item.get("transitions_total") or 0) for item in camera_segments
    )
    camera_compensated = sum(
        int(item.get("transitions_compensated") or 0)
        for item in camera_segments
    )
    camera_available_segments = sum(
        1 for item in camera_segments if item.get("available") is True
    )
    camera_available = camera_available_segments > 0

    ball_by_time: dict[float, dict[str, Any]] = {}
    ball_sampled_frames = 0
    for segment in segments:
        ball = segment.get("ball_tracking") if isinstance(segment, Mapping) else None
        if not isinstance(ball, Mapping):
            continue
        ball_sampled_frames += int(ball.get("sampled_frames") or 0)
        for observation in ball.get("observations") or []:
            if not isinstance(observation, Mapping):
                continue
            timestamp = _finite(observation.get("t"))
            if timestamp is None:
                continue
            key = round(timestamp, 3)
            previous = ball_by_time.get(key)
            if previous is None or float(observation.get("conf") or 0.0) > float(
                previous.get("conf") or 0.0
            ):
                ball_by_time[key] = dict(observation)
    ball_observations = [ball_by_time[key] for key in sorted(ball_by_time)]
    ball_available = len(ball_observations) >= 2

    events_by_key: dict[tuple[str, float, float], dict[str, Any]] = {}
    event_detector_available = False
    for segment in segments:
        event_payload = (
            segment.get("event_detection") if isinstance(segment, Mapping) else None
        )
        if not isinstance(event_payload, Mapping):
            continue
        event_detector_available = bool(
            event_detector_available or event_payload.get("available") is True
        )
        for event in event_payload.get("events") or []:
            if not isinstance(event, Mapping):
                continue
            key = (
                str(event.get("type") or "UNKNOWN"),
                round(float(event.get("start_sec") or 0.0), 2),
                round(float(event.get("end_sec") or 0.0), 2),
            )
            events_by_key[key] = dict(event)
    events = [events_by_key[key] for key in sorted(events_by_key)]

    return {
        "observability_schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "camera_motion": {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "status": "AVAILABLE" if camera_available else "INSUFFICIENT_EVIDENCE",
            "available": camera_available,
            "validated": False,
            "method": CAMERA_MOTION_METHOD,
            "segments_total": len(camera_segments),
            "segments_available": camera_available_segments,
            "transitions_total": camera_total,
            "transitions_compensated": camera_compensated,
            "coverage_ratio": round(
                camera_compensated / float(max(1, camera_total)),
                6,
            ),
            "reason_codes": (
                ["EXPERIMENTAL_NOT_BENCHMARK_VALIDATED"]
                if camera_available
                else ["INSUFFICIENT_MULTI_PERSON_MOTION_CONSENSUS"]
            ),
        },
        "ball_tracking": {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "status": "AVAILABLE" if ball_available else "INSUFFICIENT_EVIDENCE",
            "available": ball_available,
            "validated": False,
            "method": BALL_TRACKING_METHOD,
            "sampled_frames": ball_sampled_frames,
            "observed_samples": len(ball_observations),
            "sample_coverage_ratio": round(
                len(ball_observations) / float(max(1, ball_sampled_frames)),
                6,
            ),
            "observations": ball_observations,
            "reason_codes": (
                ["EXPERIMENTAL_NOT_BENCHMARK_VALIDATED"]
                if ball_available
                else ["INSUFFICIENT_BALL_DETECTIONS"]
            ),
        },
        "event_detection": {
            "schema_version": OBSERVABILITY_SCHEMA_VERSION,
            "status": (
                "AVAILABLE" if event_detector_available else "INSUFFICIENT_EVIDENCE"
            ),
            "available": event_detector_available,
            "validated": False,
            "method": EVENT_DETECTION_METHOD,
            "events": events,
            "event_count": len(events),
            "reason_codes": (
                ["PROXIMITY_EVENTS_ONLY", "EXPERIMENTAL_NOT_BENCHMARK_VALIDATED"]
                if event_detector_available
                else ["PLAYER_AND_BALL_OVERLAP_REQUIRED"]
            ),
        },
    }
