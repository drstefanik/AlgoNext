from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

from app.benchmark.schema import SequencePrediction


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def _track_payload(
    bbox: Mapping[str, Any],
    *,
    track_id: str,
) -> dict[str, Any]:
    payload = {
        "track_id": track_id,
        "bbox": {
            "x": _finite_number(bbox.get("x"), "bbox.x"),
            "y": _finite_number(bbox.get("y"), "bbox.y"),
            "w": _finite_number(bbox.get("w"), "bbox.w"),
            "h": _finite_number(bbox.get("h"), "bbox.h"),
        },
    }
    confidence = bbox.get("conf")
    if confidence is not None:
        payload["confidence"] = _finite_number(confidence, "bbox.conf")
    return payload


def prediction_from_algonext_tracking(
    tracking: Mapping[str, Any],
    *,
    video_id: str,
    evaluation_fps: float | None = None,
) -> SequencePrediction:
    """Convert AlgoNext tracking.json into the benchmark prediction contract.

    Window-local ByteTrack identifiers are namespaced by segment. A cross-window
    identity is used only when an explicit ReID decision is ACCEPTED, so the
    benchmark can reward or penalize the association rather than granting it.
    """

    fps = (
        _finite_number(evaluation_fps, "evaluation_fps")
        if evaluation_fps is not None
        else _finite_number(tracking.get("fps", 5), "tracking.fps")
    )
    if fps <= 0:
        raise ValueError("evaluation_fps must be > 0")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must be a non-empty string")

    frame_tracks: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    frame_times: dict[int, float] = {}

    def add_bbox(bbox: Mapping[str, Any], track_id: str) -> None:
        time_sec = _finite_number(bbox.get("t"), "bbox.t")
        if time_sec < 0:
            raise ValueError("bbox.t must be >= 0")
        frame_index = int(round(time_sec * fps))
        track = _track_payload(bbox, track_id=track_id)
        existing = frame_tracks[frame_index].get(track_id)
        existing_confidence = (
            float(existing.get("confidence", 0.0)) if existing else -1.0
        )
        current_confidence = float(track.get("confidence", 0.0))
        if existing is None or current_confidence >= existing_confidence:
            frame_tracks[frame_index][track_id] = track
            frame_times[frame_index] = time_sec

    segments = tracking.get("segments")
    if isinstance(segments, list):
        for segment_index, segment in enumerate(segments, start=1):
            if not isinstance(segment, Mapping):
                continue
            selected_track_id = segment.get("selected_track_id")
            if selected_track_id is None:
                continue

            reid = segment.get("reid")
            if not isinstance(reid, Mapping):
                reid = {}
            identity_id = reid.get("identity_id") or segment.get("identity_id")
            identity_status = reid.get("status") or segment.get(
                "identity_status"
            )
            if (
                identity_status == "ACCEPTED"
                and isinstance(identity_id, str)
                and identity_id.strip()
            ):
                benchmark_track_id = f"identity/{identity_id.strip()}"
            else:
                benchmark_track_id = (
                    f"segment-{segment_index:04d}/track-{selected_track_id}"
                )

            for bbox in segment.get("bboxes") or []:
                if isinstance(bbox, Mapping):
                    add_bbox(bbox, benchmark_track_id)
    else:
        selected_track_id = tracking.get("track_id")
        if selected_track_id is not None:
            stable_track_id = f"track-{selected_track_id}"
            for bbox in tracking.get("bboxes") or []:
                if isinstance(bbox, Mapping):
                    add_bbox(bbox, stable_track_id)

    payload = {
        "schema_version": "tracking-prediction-v1",
        "video_id": video_id.strip(),
        "frames": [
            {
                "frame_index": frame_index,
                "time_sec": frame_times[frame_index],
                "tracks": [
                    frame_tracks[frame_index][track_id]
                    for track_id in sorted(frame_tracks[frame_index])
                ],
            }
            for frame_index in sorted(frame_tracks)
        ],
    }
    return SequencePrediction.from_payload(payload)
