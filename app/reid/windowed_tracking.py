from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import cv2
from ultralytics import YOLO

from app.core.tracking_outcome import StaleAnalysisAttemptError
from app.reid.appearance import (
    aggregate_appearance_descriptors,
    crop_from_normalized_bbox,
    extract_appearance_descriptor,
)
from app.reid.association import (
    ASSOCIATION_VERSION,
    DESCRIPTOR_VERSION,
    AppearanceDescriptor,
    AssociationThresholds,
    CandidateProfile,
    IdentityProfile,
    associate_identity,
    update_identity_profile,
)
from app.reid.full_match_runtime import persist_fail_closed_legacy_fallback
from app.reid.window_logic import (
    autonomous_tracking_evidence as _autonomous_tracking_evidence,
    bbox_iou,
    candidate_rank,
    center_distance,
    choose_descriptor_detections,
    geometry_similarity,
    largest_tracking_gap_sec,
    processing_order,
    temporal_overlap_score,
    tracking_coverage_pct,
)
from app.workers import tracking as legacy
from app.workers.multi_anchor import normalize_anchors

logger = logging.getLogger(__name__)
_SAFE_ATTEMPT_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _analysis_attempt_component(analysis_attempt_id: str | None) -> str:
    normalized = str(analysis_attempt_id or "").strip()
    if not normalized:
        return "legacy"
    if _SAFE_ATTEMPT_COMPONENT.fullmatch(normalized) is None:
        raise ValueError("analysis_attempt_id is not safe for artifact storage")
    return normalized


class ReIDUnavailable(RuntimeError):
    pass


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _association_thresholds() -> AssociationThresholds:
    return AssociationThresholds(
        min_combined_score=_env_float("PLAYER_REID_MIN_COMBINED_SCORE", 0.76, 0.0, 1.0),
        min_appearance_similarity=_env_float(
            "PLAYER_REID_MIN_APPEARANCE_SIMILARITY", 0.78, 0.0, 1.0
        ),
        strong_overlap_score=_env_float(
            "PLAYER_REID_STRONG_OVERLAP_SCORE", 0.65, 0.0, 1.0
        ),
        min_margin=_env_float("PLAYER_REID_MIN_MARGIN", 0.07, 0.0, 1.0),
        min_descriptor_quality=_env_float(
            "PLAYER_REID_MIN_DESCRIPTOR_QUALITY", 0.30, 0.0, 1.0
        ),
        min_descriptor_samples=_env_int("PLAYER_REID_MIN_DESCRIPTOR_SAMPLES", 2, 1, 20),
        # The current HSV descriptor can distinguish kit families, but it
        # cannot prove which teammate is present after an unrelated camera
        # shot. Only physical continuity through an overlapping window is
        # allowed to create an autonomous identity link. Later manual anchors
        # can still reseed the identity explicitly.
        require_strong_overlap=_env_bool("PLAYER_REID_REQUIRE_STRONG_OVERLAP", True),
    )


def _reset_tracker(model: YOLO) -> None:
    reset = getattr(model, "reset", None)
    if callable(reset):
        try:
            reset()
            return
        except Exception:
            logger.debug("YOLO reset() failed", exc_info=True)
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", None) or []:
        tracker_reset = getattr(tracker, "reset", None)
        if callable(tracker_reset):
            try:
                tracker_reset()
            except Exception:
                logger.debug("ByteTrack reset failed", exc_info=True)


def _descriptor_metadata(descriptor: AppearanceDescriptor | None) -> dict[str, Any]:
    if descriptor is None:
        return {
            "version": DESCRIPTOR_VERSION,
            "sample_count": 0,
            "quality": 0.0,
        }
    return {
        "version": descriptor.version,
        "sample_count": descriptor.sample_count,
        "quality": round(descriptor.quality, 6),
    }


def _extract_descriptors_for_tracks(
    segment_path: Path,
    track_map: Mapping[int, Sequence[Mapping[str, Any]]],
    track_ids: Sequence[int],
) -> dict[int, AppearanceDescriptor | None]:
    max_samples = _env_int("PLAYER_REID_SAMPLES_PER_CANDIDATE", 5, 2, 12)
    min_individual_quality = _env_float(
        "PLAYER_REID_MIN_INDIVIDUAL_CROP_QUALITY", 0.18, 0.0, 1.0
    )
    requests: list[tuple[float, int, Mapping[str, Any]]] = []
    for track_id in track_ids:
        for detection in choose_descriptor_detections(
            list(track_map.get(track_id) or []), max_samples
        ):
            requests.append((float(detection.get("t") or 0.0), track_id, detection))
    requests.sort(key=lambda item: item[0])

    descriptors: dict[int, list[AppearanceDescriptor]] = {
        track_id: [] for track_id in track_ids
    }
    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        return {track_id: None for track_id in track_ids}
    try:
        for time_sec, track_id, detection in requests:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            bbox = detection.get("bbox")
            if not isinstance(bbox, Mapping):
                continue
            crop = crop_from_normalized_bbox(frame, bbox)
            if crop is None:
                continue
            descriptor = extract_appearance_descriptor(crop)
            if descriptor is None or descriptor.quality < min_individual_quality:
                continue
            descriptors[track_id].append(descriptor)
    finally:
        cap.release()

    return {
        track_id: aggregate_appearance_descriptors(descriptors[track_id])
        for track_id in track_ids
    }


def _select_anchor_track(
    samples: Sequence[Mapping[str, Any]],
    track_map: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    anchor_time_local: float,
    anchor_bbox: Mapping[str, Any],
) -> int | None:
    nearby: list[tuple[int, Mapping[str, Any]]] = []
    for track_id, detections in track_map.items():
        for detection in detections:
            if abs(float(detection.get("t") or 0.0) - anchor_time_local) <= 2.0:
                nearby.append((int(track_id), detection))
    if not nearby and samples:
        sample = min(
            samples,
            key=lambda item: abs(float(item.get("t") or 0.0) - anchor_time_local),
        )
        for detection in sample.get("detections") or []:
            if isinstance(detection, Mapping) and detection.get("track_id") is not None:
                nearby.append((int(detection["track_id"]), detection))
    if not nearby:
        return None

    ranked: list[tuple[float, int]] = []
    for track_id, detection in nearby:
        bbox = detection.get("bbox")
        if not isinstance(bbox, Mapping):
            continue
        iou = legacy._bbox_iou(dict(anchor_bbox), dict(bbox))
        distance = center_distance(anchor_bbox, bbox)
        anchor_area = max(
            1e-9,
            float(anchor_bbox.get("w") or 0.0) * float(anchor_bbox.get("h") or 0.0),
        )
        candidate_area = max(
            1e-9,
            float(bbox.get("w") or 0.0) * float(bbox.get("h") or 0.0),
        )
        area_similarity = min(anchor_area, candidate_area) / max(
            anchor_area, candidate_area
        )
        anchor_aspect = max(1e-9, float(anchor_bbox.get("w") or 0.0)) / max(
            1e-9, float(anchor_bbox.get("h") or 0.0)
        )
        candidate_aspect = max(1e-9, float(bbox.get("w") or 0.0)) / max(
            1e-9, float(bbox.get("h") or 0.0)
        )
        aspect_similarity = min(anchor_aspect, candidate_aspect) / max(
            anchor_aspect, candidate_aspect
        )
        credible_shape = area_similarity >= 0.25 and aspect_similarity >= 0.45
        credible_position = iou >= 0.18 or (
            distance <= 0.12 and area_similarity >= 0.45
        )
        if not credible_shape or not credible_position:
            continue
        geometry = geometry_similarity(anchor_bbox, bbox)
        confidence = float(detection.get("conf") or 0.0)
        temporal = max(
            0.0,
            1.0 - abs(float(detection.get("t") or 0.0) - anchor_time_local) / 2.0,
        )
        ranked.append(
            (
                iou * 2.0 + geometry * 0.7 + temporal * 0.2 + confidence * 0.1,
                track_id,
            )
        )
    if not ranked:
        return None
    best_score, best_track_id = max(ranked)
    return best_track_id if best_score >= 0.35 else None


def _canonical_anchor_window(
    windows: Sequence[tuple[float, float]], timestamp: float
) -> int | None:
    """Assign an anchor to exactly one containing window.

    Full-match windows overlap. Without a canonical assignment, one manual
    reference can be counted and applied twice. Prefer the window whose centre
    is closest to the anchor and use the lower index as the deterministic
    tiebreaker.
    """

    containing = [
        index
        for index, (start, end) in enumerate(windows)
        if float(start) <= float(timestamp) <= float(end)
    ]
    if not containing:
        return None
    return min(
        containing,
        key=lambda index: (
            abs(
                (float(windows[index][0]) + float(windows[index][1])) * 0.5
                - float(timestamp)
            ),
            index,
        ),
    )


def _anchor_distance(anchor: Mapping[str, Any], player_ref: Mapping[str, Any]) -> float:
    """Return a deterministic distance used only to identify the primary ref."""

    distance = abs(float(anchor.get("t") or 0.0) - float(player_ref.get("t") or 0.0))
    for key in ("x", "y", "w", "h"):
        distance += abs(
            float(anchor.get(key) or 0.0) - float(player_ref.get(key) or 0.0)
        )
    return distance


def _shape_ratios(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[float, float]:
    first_area = max(
        1e-9,
        float(first.get("w") or 0.0) * float(first.get("h") or 0.0),
    )
    second_area = max(
        1e-9,
        float(second.get("w") or 0.0) * float(second.get("h") or 0.0),
    )
    first_aspect = max(1e-9, float(first.get("w") or 0.0)) / max(
        1e-9, float(first.get("h") or 0.0)
    )
    second_aspect = max(1e-9, float(second.get("w") or 0.0)) / max(
        1e-9, float(second.get("h") or 0.0)
    )
    return (
        min(first_area, second_area) / max(first_area, second_area),
        min(first_aspect, second_aspect) / max(first_aspect, second_aspect),
    )


def _motion_continuous(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    maximum_gap_sec: float,
    maximum_center_distance: float,
) -> bool:
    previous_sample = previous.get("sample_index")
    current_sample = current.get("sample_index")
    if (
        previous_sample is not None
        and current_sample is not None
        and abs(int(current_sample) - int(previous_sample)) != 1
    ):
        return False
    gap = abs(float(current.get("t") or 0.0) - float(previous.get("t") or 0.0))
    if gap > maximum_gap_sec:
        return False
    previous_bbox = previous.get("bbox")
    current_bbox = current.get("bbox")
    if not isinstance(previous_bbox, Mapping) or not isinstance(current_bbox, Mapping):
        return False
    area_similarity, aspect_similarity = _shape_ratios(previous_bbox, current_bbox)
    if area_similarity < 0.25 or aspect_similarity < 0.40:
        return False
    return (
        bbox_iou(previous_bbox, current_bbox) >= 0.02
        or center_distance(previous_bbox, current_bbox) <= maximum_center_distance
    )


def _anchor_tracklet_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    anchor_time_local: float,
    anchor_bbox: Mapping[str, Any],
    radius_sec: float,
) -> list[Mapping[str, Any]]:
    """Return the local, motion-continuous component owned by one anchor.

    ByteTrack IDs are window-local and can cross to another player after an
    occlusion or camera change. A manual bbox verifies only the component
    connected to its immediate detection, not every occurrence of that raw ID.
    """

    valid = [
        detection
        for detection in detections
        if detection.get("t") is not None and isinstance(detection.get("bbox"), Mapping)
    ]
    nearby = [
        detection
        for detection in valid
        if abs(float(detection.get("t") or 0.0) - float(anchor_time_local))
        <= radius_sec
    ]
    if not nearby:
        return []

    seed_candidates: list[tuple[float, float, float, Mapping[str, Any]]] = []
    for detection in nearby:
        bbox = detection["bbox"]
        area_similarity, aspect_similarity = _shape_ratios(anchor_bbox, bbox)
        iou = bbox_iou(anchor_bbox, bbox)
        distance = center_distance(anchor_bbox, bbox)
        if area_similarity < 0.25 or aspect_similarity < 0.45:
            continue
        if iou < 0.18 and not (distance <= 0.12 and area_similarity >= 0.45):
            continue
        seed_candidates.append(
            (
                abs(float(detection.get("t") or 0.0) - float(anchor_time_local)),
                -iou,
                -float(detection.get("conf") or 0.0),
                detection,
            )
        )
    if not seed_candidates:
        return []
    seed = min(seed_candidates, key=lambda item: item[:3])[3]
    ordered = sorted(
        valid,
        key=lambda item: (
            float(item.get("t") or 0.0),
            int(item.get("sample_index") or 0),
        ),
    )
    seed_index = next(
        index for index, detection in enumerate(ordered) if detection is seed
    )
    maximum_gap = _env_float("PLAYER_REID_MANUAL_ANCHOR_MAX_GAP_SEC", 1.0, 0.1, 3.0)
    maximum_center_distance = _env_float(
        "PLAYER_REID_MANUAL_ANCHOR_MAX_CENTER_DISTANCE", 0.25, 0.02, 0.5
    )

    selected = {seed_index}
    previous_detection = seed
    for index in range(seed_index - 1, -1, -1):
        detection = ordered[index]
        if not _motion_continuous(
            previous_detection,
            detection,
            maximum_gap_sec=maximum_gap,
            maximum_center_distance=maximum_center_distance,
        ):
            break
        selected.add(index)
        previous_detection = detection
    previous_detection = seed
    for index in range(seed_index + 1, len(ordered)):
        detection = ordered[index]
        if not _motion_continuous(
            previous_detection,
            detection,
            maximum_gap_sec=maximum_gap,
            maximum_center_distance=maximum_center_distance,
        ):
            break
        selected.add(index)
        previous_detection = detection
    return [ordered[index] for index in sorted(selected)]


def _anchor_descriptor_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    anchor_time_local: float,
    anchor_bbox: Mapping[str, Any],
    radius_sec: float,
) -> list[Mapping[str, Any]]:
    component = _anchor_tracklet_detections(
        detections,
        anchor_time_local=anchor_time_local,
        anchor_bbox=anchor_bbox,
        radius_sec=radius_sec,
    )
    return [
        detection
        for detection in component
        if abs(float(detection.get("t") or 0.0) - float(anchor_time_local))
        <= radius_sec
    ]


def _stitch_manual_anchor_bboxes(
    observations: Sequence[Mapping[str, Any]],
    samples: list[dict[str, Any]],
    *,
    fps: int,
    window_start: float,
    radius_sec: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Build one deterministic window track from one or more manual anchors.

    ByteTrack can assign a new local ID after a camera cut inside the same
    window. Each anchor owns the temporal region bounded by the midpoints to its
    neighbours. This makes every matched anchor operational instead of choosing
    only the highest-scoring one for the whole window.
    """

    ordered = sorted(
        observations,
        key=lambda item: (
            float(item["anchor"]["t"]),
            int(item["anchor"]["anchor_id"]),
        ),
    )
    if not ordered:
        return [], []

    track_bboxes: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for observation in ordered:
        track_id = int(observation["track_id"])
        anchor = observation["anchor"]
        anchor_id = int(anchor["anchor_id"])
        key = (track_id, anchor_id)
        if key in track_bboxes:
            continue
        anchor_time_local = max(0.0, float(anchor["t"]) - float(window_start))
        raw_detections: list[dict[str, Any]] = []
        for sample_index, sample in enumerate(samples):
            for detection in sample.get("detections") or []:
                if (
                    detection.get("track_id") is None
                    or int(detection["track_id"]) != track_id
                ):
                    continue
                raw_detections.append(
                    {
                        **dict(detection),
                        "t": float(sample.get("t") or 0.0),
                        "sample_index": sample_index,
                    }
                )
        anchor_tracklet = _anchor_tracklet_detections(
            raw_detections,
            anchor_time_local=anchor_time_local,
            anchor_bbox=anchor,
            radius_sec=radius_sec,
        )
        allowed_sample_indices = {
            int(item["sample_index"])
            for item in anchor_tracklet
            if item.get("sample_index") is not None
        }
        scoped_samples = [
            {
                **dict(sample),
                "detections": (
                    list(sample.get("detections") or [])
                    if sample_index in allowed_sample_indices
                    else []
                ),
            }
            for sample_index, sample in enumerate(samples)
        ]
        bboxes, _lost, _last = legacy._build_window_bboxes(
            scoped_samples,
            track_id,
            fps=fps,
            time_offset=window_start,
        )
        track_bboxes[key] = [dict(item) for item in bboxes]

    selected: dict[float, tuple[float, int, dict[str, Any]]] = {}
    for index, observation in enumerate(ordered):
        anchor = observation["anchor"]
        anchor_time = float(anchor["t"])
        lower = (
            float("-inf")
            if index == 0
            else (float(ordered[index - 1]["anchor"]["t"]) + anchor_time) * 0.5
        )
        upper = (
            float("inf")
            if index == len(ordered) - 1
            else (anchor_time + float(ordered[index + 1]["anchor"]["t"])) * 0.5
        )
        track_id = int(observation["track_id"])
        anchor_id = int(anchor["anchor_id"])
        candidates = track_bboxes.get((track_id, anchor_id)) or []
        owned = [
            bbox
            for bbox in candidates
            if lower <= float(bbox.get("t") or 0.0)
            and (float(bbox.get("t") or 0.0) < upper or index == len(ordered) - 1)
        ]
        if not owned and candidates:
            owned = [
                min(
                    candidates,
                    key=lambda bbox: abs(float(bbox.get("t") or 0.0) - anchor_time),
                )
            ]
        for bbox in owned:
            timestamp = float(bbox.get("t") or 0.0)
            time_key = round(timestamp, 6)
            proximity = abs(timestamp - anchor_time)
            previous = selected.get(time_key)
            candidate = (proximity, int(anchor["anchor_id"]), dict(bbox))
            if previous is None or candidate[:2] < previous[:2]:
                selected[time_key] = candidate

    stitched = [
        payload
        for _proximity, _anchor_id, payload in (
            selected[key] for key in sorted(selected)
        )
    ]
    track_ids = list(
        dict.fromkeys(int(observation["track_id"]) for observation in ordered)
    )
    return stitched, track_ids


def _boundary_bbox(
    bboxes: Sequence[Mapping[str, Any]], direction: str
) -> Mapping[str, Any] | None:
    if not bboxes:
        return None
    key = lambda bbox: float(bbox.get("t") or 0.0)
    return max(bboxes, key=key) if direction == "forward" else min(bboxes, key=key)


def _boundary_detection(
    detections: Sequence[Mapping[str, Any]], direction: str
) -> Mapping[str, Any] | None:
    if not detections:
        return None
    key = lambda detection: float(detection.get("t") or 0.0)
    return (
        min(detections, key=key) if direction == "forward" else max(detections, key=key)
    )


def _overlap_linked_detections(
    previous_bboxes: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    *,
    window_start: float,
    tolerance_sec: float,
) -> list[Mapping[str, Any]]:
    """Keep only detections physically linked through a window overlap.

    A raw ByteTrack ID can switch players elsewhere in the same minute. For a
    strong overlap association, the evidence and emitted boxes must therefore
    be limited to detections that coincide in both time and space with the
    already verified tracklet from the adjacent window.
    """

    previous = [
        bbox
        for bbox in previous_bboxes
        if isinstance(bbox, Mapping) and bbox.get("t") is not None
    ]
    if not previous:
        return []
    minimum_iou = _env_float("PLAYER_REID_OVERLAP_LINK_MIN_IOU", 0.35, 0.0, 1.0)
    linked_by_sample: dict[
        tuple[str, int | float], tuple[float, float, Mapping[str, Any]]
    ] = {}
    for detection in detections:
        bbox = detection.get("bbox")
        if not isinstance(bbox, Mapping) or detection.get("t") is None:
            continue
        absolute_time = float(window_start) + float(detection.get("t") or 0.0)
        closest = min(
            previous,
            key=lambda item: abs(float(item.get("t") or 0.0) - absolute_time),
        )
        time_delta = abs(float(closest.get("t") or 0.0) - absolute_time)
        overlap_iou = bbox_iou(closest, bbox)
        if time_delta > tolerance_sec or overlap_iou < minimum_iou:
            continue
        sample_index = detection.get("sample_index")
        key: tuple[str, int | float]
        if sample_index is not None:
            key = ("sample", int(sample_index))
        else:
            key = ("time", round(float(detection.get("t") or 0.0), 6))
        rank = (
            overlap_iou,
            float(detection.get("conf") or 0.0),
            detection,
        )
        previous_rank = linked_by_sample.get(key)
        if previous_rank is None or rank[:2] > previous_rank[:2]:
            linked_by_sample[key] = rank
    return [
        item[2]
        for _key, item in sorted(
            linked_by_sample.items(),
            key=lambda entry: (
                float(entry[1][2].get("t") or 0.0),
                str(entry[0]),
            ),
        )
    ]


def _detection_key(
    detection: Mapping[str, Any],
) -> tuple[str, int | float]:
    sample_index = detection.get("sample_index")
    if sample_index is not None:
        return ("sample", int(sample_index))
    return ("time", round(float(detection.get("t") or 0.0), 6))


def _tracklet_detections_from_overlap(
    detections: Sequence[Mapping[str, Any]],
    overlap_detections: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    fps: int,
) -> list[Mapping[str, Any]]:
    """Extend verified overlap seeds only through one continuous raw-ID component."""

    if direction not in {"forward", "backward"}:
        return []
    linked_by_key = {
        _detection_key(detection): detection
        for detection in overlap_detections
        if detection.get("t") is not None and isinstance(detection.get("bbox"), Mapping)
    }
    if not linked_by_key:
        return []

    grouped: dict[tuple[str, int | float], list[Mapping[str, Any]]] = {}
    for detection in detections:
        if detection.get("t") is None or not isinstance(detection.get("bbox"), Mapping):
            continue
        grouped.setdefault(_detection_key(detection), []).append(detection)

    ordered: list[tuple[tuple[str, int | float], Mapping[str, Any] | None]] = []
    for key, group in grouped.items():
        linked = linked_by_key.get(key)
        if linked is not None:
            chosen: Mapping[str, Any] | None = linked
        elif len(group) == 1:
            chosen = group[0]
        else:
            ranked = sorted(
                group,
                key=lambda item: float(item.get("conf") or 0.0),
                reverse=True,
            )
            best = ranked[0]
            best_bbox = best.get("bbox")
            equivalent = isinstance(best_bbox, Mapping) and all(
                isinstance(item.get("bbox"), Mapping)
                and bbox_iou(best_bbox, item["bbox"]) >= 0.90
                for item in ranked[1:]
            )
            chosen = best if equivalent else None
        ordered.append((key, chosen))
    ordered.sort(
        key=lambda item: (
            float(
                (item[1] or linked_by_key.get(item[0]) or grouped[item[0]][0]).get("t")
                or 0.0
            ),
            str(item[0]),
        )
    )
    index_by_key = {key: index for index, (key, _item) in enumerate(ordered)}
    if any(key not in index_by_key for key in linked_by_key):
        return []
    seed_indices = sorted(index_by_key[key] for key in linked_by_key)
    first_seed = seed_indices[0]
    last_seed = seed_indices[-1]
    seed_span = [ordered[index][1] for index in range(first_seed, last_seed + 1)]
    if any(item is None for item in seed_span):
        return []

    maximum_gap = _env_float(
        "PLAYER_REID_TRACKLET_MAX_GAP_SEC",
        max(1.25, 2.5 / max(1, fps)),
        0.1,
        5.0,
    )
    maximum_center_distance = _env_float(
        "PLAYER_REID_TRACKLET_MAX_CENTER_DISTANCE", 0.20, 0.02, 0.5
    )
    concrete_seed_span = [item for item in seed_span if item is not None]
    if any(
        not _motion_continuous(
            concrete_seed_span[index - 1],
            concrete_seed_span[index],
            maximum_gap_sec=maximum_gap,
            maximum_center_distance=maximum_center_distance,
        )
        for index in range(1, len(concrete_seed_span))
    ):
        return []

    selected = set(range(first_seed, last_seed + 1))
    if direction == "forward":
        previous = concrete_seed_span[-1]
        for index in range(last_seed + 1, len(ordered)):
            detection = ordered[index][1]
            if detection is None or not _motion_continuous(
                previous,
                detection,
                maximum_gap_sec=maximum_gap,
                maximum_center_distance=maximum_center_distance,
            ):
                break
            selected.add(index)
            previous = detection
    else:
        previous = concrete_seed_span[0]
        for index in range(first_seed - 1, -1, -1):
            detection = ordered[index][1]
            if detection is None or not _motion_continuous(
                previous,
                detection,
                maximum_gap_sec=maximum_gap,
                maximum_center_distance=maximum_center_distance,
            ):
                break
            selected.add(index)
            previous = detection
    return [
        ordered[index][1] for index in sorted(selected) if ordered[index][1] is not None
    ]


def _samples_for_tracklet(
    samples: Sequence[Mapping[str, Any]],
    *,
    track_id: int,
    tracklet_detections: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    exact_by_sample = {
        int(detection["sample_index"]): dict(detection)
        for detection in tracklet_detections
        if detection.get("sample_index") is not None
    }
    scoped: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        detection = exact_by_sample.get(sample_index)
        detections = (
            [{**detection, "track_id": int(track_id)}] if detection is not None else []
        )
        scoped.append({**dict(sample), "detections": detections})
    return scoped


def _build_candidate_profiles(
    segment_path: Path,
    track_map: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    previous_bboxes: Sequence[Mapping[str, Any]],
    window_start: float,
    direction: str,
    fps: int,
    strong_overlap_score: float | None = None,
) -> tuple[
    list[CandidateProfile],
    dict[str, int],
    dict[str, AppearanceDescriptor | None],
]:
    minimum_hits = _env_int("PLAYER_REID_MIN_TRACK_HITS", 3, 1, 1000)
    max_candidates = _env_int("PLAYER_REID_MAX_CANDIDATES", 6, 1, 20)
    tolerance = _env_float(
        "PLAYER_REID_OVERLAP_TOLERANCE_SEC",
        max(0.6, 2.0 / max(1, fps)),
        0.1,
        5.0,
    )
    strong_overlap_gate = (
        float(strong_overlap_score)
        if strong_overlap_score is not None
        else _env_float("PLAYER_REID_STRONG_OVERLAP_SCORE", 0.65, 0.0, 1.0)
    )
    minimum_overlap_samples = _env_int("PLAYER_REID_MIN_OVERLAP_LINK_SAMPLES", 2, 1, 20)
    overlap_uniqueness_margin = _env_float(
        "PLAYER_REID_OVERLAP_UNIQUENESS_MARGIN", 0.05, 0.0, 0.25
    )
    previous_boundary = _boundary_bbox(previous_bboxes, direction)
    ranked: list[tuple[float, int, float | None, float | None]] = []
    for raw_track_id, raw_detections in track_map.items():
        detections = [item for item in raw_detections if isinstance(item, Mapping)]
        if len(detections) < minimum_hits:
            continue
        overlap = temporal_overlap_score(
            previous_bboxes,
            detections,
            time_offset=window_start,
            tolerance_sec=tolerance,
        )
        candidate_boundary = _boundary_detection(detections, direction)
        geometry = None
        if previous_boundary is not None and candidate_boundary is not None:
            bbox = candidate_boundary.get("bbox")
            if isinstance(bbox, Mapping):
                geometry = geometry_similarity(previous_boundary, bbox)
        ranked.append(
            (
                candidate_rank(
                    detections,
                    overlap_score=overlap,
                    geometry_score=geometry,
                ),
                int(raw_track_id),
                overlap,
                geometry,
            )
        )
    ranked.sort(reverse=True)
    overlap_by_track = {
        track_id: overlap for _rank, track_id, overlap, _geometry in ranked
    }
    overlap_links_by_track: dict[int, list[Mapping[str, Any]]] = {}
    tracklet_by_track: dict[int, list[Mapping[str, Any]]] = {}
    plausible_overlap_gate = max(0.0, strong_overlap_gate - overlap_uniqueness_margin)
    for _rank, track_id, overlap, _geometry in ranked:
        if overlap is None or overlap < plausible_overlap_gate:
            continue
        raw_detections = [
            item
            for item in (track_map.get(track_id) or [])
            if isinstance(item, Mapping)
        ]
        overlap_links = _overlap_linked_detections(
            previous_bboxes,
            raw_detections,
            window_start=window_start,
            tolerance_sec=tolerance,
        )
        overlap_links_by_track[track_id] = overlap_links
        if len(overlap_links) >= minimum_overlap_samples:
            tracklet_by_track[track_id] = _tracklet_detections_from_overlap(
                raw_detections,
                overlap_links,
                direction=direction,
                fps=fps,
            )
    verified_strong_ids = {
        track_id
        for track_id, detections in overlap_links_by_track.items()
        if len(detections) >= minimum_overlap_samples
        and tracklet_by_track.get(track_id)
        and (overlap_by_track.get(track_id) or 0.0) >= strong_overlap_gate
    }
    plausible_overlap_ids = {
        track_id
        for track_id, detections in overlap_links_by_track.items()
        if len(detections) >= minimum_overlap_samples
        and tracklet_by_track.get(track_id)
    }
    unique_strong_track_id = (
        next(iter(verified_strong_ids))
        if len(verified_strong_ids) == 1 and len(plausible_overlap_ids) == 1
        else None
    )
    # The normal candidate cap is an efficiency limit, not an ambiguity
    # waiver. Always surface every physically plausible overlap candidate so
    # a near-threshold runner hidden below the cap cannot be ignored.
    selected = list(ranked[:max_candidates])
    selected_track_ids = {item[1] for item in selected}
    selected.extend(
        item
        for item in ranked
        if item[1] in plausible_overlap_ids and item[1] not in selected_track_ids
    )
    selected_ids = [item[1] for item in selected]
    scoped_track_map: dict[int, list[Mapping[str, Any]]] = {}
    scope_by_track: dict[int, str] = {}
    effective_overlap_by_track: dict[int, float | None] = {}
    for _rank, track_id, overlap, _geometry in selected:
        raw_detections = [
            item
            for item in (track_map.get(track_id) or [])
            if isinstance(item, Mapping)
        ]
        if track_id in plausible_overlap_ids:
            scoped_track_map[track_id] = tracklet_by_track[track_id]
            scope_by_track[track_id] = (
                "MOTION_CONTINUOUS_STRONG_OVERLAP"
                if track_id in verified_strong_ids
                else "MOTION_CONTINUOUS_PLAUSIBLE_OVERLAP"
            )
            effective_overlap_by_track[track_id] = overlap
        elif overlap is not None and overlap >= strong_overlap_gate:
            # A numerical overlap can be produced by a single accidental box
            # match or by disconnected raw-ID components. Never fall back to
            # the full ByteTrack ID.
            scoped_track_map[track_id] = []
            scope_by_track[track_id] = "STRONG_OVERLAP_UNRESOLVED"
            effective_overlap_by_track[track_id] = None
        else:
            scoped_track_map[track_id] = raw_detections
            scope_by_track[track_id] = "FULL_WINDOW"
            effective_overlap_by_track[track_id] = overlap
    descriptor_by_track = _extract_descriptors_for_tracks(
        segment_path, scoped_track_map, selected_ids
    )

    profiles: list[CandidateProfile] = []
    id_lookup: dict[str, int] = {}
    descriptor_lookup: dict[str, AppearanceDescriptor | None] = {}
    for _, track_id, overlap, geometry in selected:
        candidate_id = str(track_id)
        descriptor = descriptor_by_track.get(track_id)
        scoped_detections = scoped_track_map.get(track_id) or []
        effective_overlap = effective_overlap_by_track.get(track_id)
        id_lookup[candidate_id] = track_id
        descriptor_lookup[candidate_id] = descriptor
        profiles.append(
            CandidateProfile(
                candidate_id=candidate_id,
                descriptor=descriptor,
                overlap_score=effective_overlap,
                geometry_score=geometry,
                detection_count=len(scoped_detections),
                metadata={
                    "local_track_id": track_id,
                    "tracklet_scope": scope_by_track.get(track_id, "FULL_WINDOW"),
                    "tracklet_sample_indices": tuple(
                        int(item.get("sample_index"))
                        for item in scoped_detections
                        if item.get("sample_index") is not None
                    ),
                    "tracklet_detections": tuple(
                        dict(item) for item in scoped_detections
                    ),
                    "overlap_link_samples": len(
                        overlap_links_by_track.get(track_id) or []
                    ),
                    "strong_overlap_unique": (track_id == unique_strong_track_id),
                    "raw_overlap_score": overlap,
                },
            )
        )
    return profiles, id_lookup, descriptor_lookup


def _empty_segment(
    *,
    window_index: int,
    parent_window_index: int | None,
    window_start: float,
    window_end: float,
    direction: str,
    processing_direction: str,
    reason_code: str,
    identity_id: str,
) -> dict[str, Any]:
    return {
        "window_index": int(window_index),
        "parent_window_index": (
            int(parent_window_index) if parent_window_index is not None else None
        ),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "direction": direction,
        "processing_direction": processing_direction,
        "selected_track_id": None,
        "identity_id": None,
        "identity_status": "ABSTAINED",
        "reacquire_score": 0.0,
        "coverage_pct": 0.0,
        "lost_segments": [],
        "bboxes": [],
        "reid": {
            "version": ASSOCIATION_VERSION,
            "validated": False,
            "status": "ABSTAINED",
            "identity_id": identity_id,
            "selected_candidate_id": None,
            "best_score": 0.0,
            "margin": 0.0,
            "reason_codes": [reason_code],
            "candidates": [],
        },
    }


def _persist_tracking_output(
    job_id: str,
    output: dict[str, Any],
    *,
    analysis_attempt_id: str | None,
    endpoint_url: str,
    bucket: str,
    expires_seconds: int,
) -> dict[str, Any]:
    attempt_component = _analysis_attempt_component(analysis_attempt_id)
    payload = dict(output)
    payload["analysis_attempt_id"] = str(analysis_attempt_id or "").strip() or None
    tracking_dir = (
        Path("/tmp/fnh_jobs") / job_id / "attempts" / attempt_component / "tracking"
    )
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = tracking_dir / "tracking.json"
    with tracking_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    s3_internal = legacy._get_s3_client(endpoint_url)
    legacy._ensure_bucket_exists(s3_internal, bucket)
    tracking_key = f"jobs/{job_id}/attempts/{attempt_component}/tracking/tracking.json"
    legacy._upload_file(
        s3_internal,
        bucket,
        tracking_path,
        tracking_key,
        "application/json",
    )
    result = payload
    result["tracking_key"] = tracking_key
    result["tracking_url"] = legacy._presign_get_object(
        bucket, tracking_key, expires_seconds
    )
    return result


def _fallback(
    fallback: Callable[..., Dict[str, Any]] | None,
    reason_code: str,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    if fallback is None:
        raise ReIDUnavailable(reason_code)
    return persist_fail_closed_legacy_fallback(
        fallback(*args, **kwargs),
        reason_code=reason_code,
        job_id=args[0] if args else None,
        tracking_module=legacy,
        analysis_attempt_id=kwargs.get("analysis_attempt_id"),
    )


def _anchor_acquisition_profile(
    *,
    tracking_fps: int,
    tracking_detector_model: str,
) -> tuple[int, str]:
    """Return the deliberately higher-fidelity manual-anchor profile.

    The full-match CPU budget may reduce long videos to one frame per second
    and the nano detector. Manual references are the identity root, so the
    windows that contain them must not inherit that lossy profile.
    """

    anchor_fps = max(
        int(tracking_fps),
        _env_int("PLAYER_REID_ANCHOR_FPS", 5, 1, 15),
    )
    anchor_detector_model = (
        os.environ.get("PLAYER_REID_ANCHOR_DETECTOR_MODEL") or "yolo11s.pt"
    ).strip() or "yolo11s.pt"
    return anchor_fps, anchor_detector_model


def _anchor_failure_output(
    *,
    duration: float,
    fps: int,
    window_sec: float,
    overlap_sec: float,
    windows_total: int,
    windows_processed: int,
    player_ref: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    anchor_matches: Sequence[Mapping[str, Any]],
    anchor_fps: int,
    anchor_detector_model: str,
    status: str,
    reason_code: str,
    action_required: str,
) -> dict[str, Any]:
    """Build a terminal tracking diagnostic without fabricating match metrics."""

    anchors_matched = sum(
        1 for match in anchor_matches if match.get("status") == "MATCHED"
    )
    failure_phase = (
        "manual-anchor acquisition"
        if status.startswith("ANCHOR_")
        else "selected-player window processing"
    )
    return {
        "mode": "full_match_windowed",
        "identity_mode": "appearance_reid_v1",
        "method": "yolo+bytetrack+appearance_reid",
        "fps": fps,
        "window_sec": window_sec,
        "overlap_sec": overlap_sec,
        "segments": [],
        "segments_total": windows_total,
        "segments_with_player": 0,
        "autonomous_segments_with_player": 0,
        "autonomous_bboxes_count": 0,
        "tracking_scope_status": "EMPTY",
        "windows_processed": windows_processed,
        "coverage_pct_total": 0.0,
        "coverage_pct": 0.0,
        "largest_gap_sec": None,
        "anchors_total": len(anchors),
        "anchors_matched": anchors_matched,
        "anchor_matches": [dict(item) for item in anchor_matches],
        "anchor_reacquisitions": 0,
        "anchors_used": {
            "player_ref": dict(player_ref),
            "selections": [dict(item) for item in anchors],
        },
        "tracking_success": False,
        "partial": False,
        "partial_reason": None,
        "tracking_status": status,
        "action_required": action_required,
        "anchor_acquisition": {
            "fps": anchor_fps,
            "detector_model": anchor_detector_model,
            "windows_processed": windows_processed,
        },
        "reid_summary": {
            "status": status,
            "validated": False,
            "anchors_total": len(anchors),
            "anchors_matched": anchors_matched,
            "anchor_matches": [dict(item) for item in anchor_matches],
            "reason_codes": [reason_code],
        },
        "notes": (
            f"Full-match processing stopped during {failure_phase}. "
            "No player tracking or continuity metrics were produced."
        ),
    }


def track_player_windowed_reid(
    job_id: str,
    input_video_path: str,
    player_ref: dict,
    selections: list[dict[str, Any]],
    *,
    analysis_attempt_id: str | None = None,
    video_duration_sec: float,
    window_sec: float = 45.0,
    overlap_sec: float = 10.0,
    fps: int = 5,
    detector_model: str = "yolo11s.pt",
    tracker: str = "bytetrack.yaml",
    max_windows: int = 200,
    fallback: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    original_args = (job_id, input_video_path, player_ref, selections)
    original_kwargs = {
        "video_duration_sec": video_duration_sec,
        "analysis_attempt_id": analysis_attempt_id,
        "window_sec": window_sec,
        "overlap_sec": overlap_sec,
        "fps": fps,
        "detector_model": detector_model,
        "tracker": tracker,
        "max_windows": max_windows,
    }
    player_ref_norm = legacy._normalize_player_ref(player_ref)
    if player_ref_norm is None:
        return _fallback(
            fallback,
            "REID_PLAYER_REFERENCE_MISSING",
            *original_args,
            **original_kwargs,
        )

    endpoint_url = legacy.S3_ENDPOINT_URL
    public_endpoint = legacy.S3_PUBLIC_ENDPOINT_URL
    access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
    bucket = os.environ.get("S3_BUCKET", "").strip()
    expires_seconds = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))
    if (
        not endpoint_url
        or not public_endpoint
        or not access_key
        or not secret_key
        or not bucket
    ):
        return _fallback(
            fallback,
            "REID_STORAGE_CONFIGURATION_MISSING",
            *original_args,
            **original_kwargs,
        )

    duration = float(video_duration_sec or 0.0)
    if duration <= 0:
        cap = cv2.VideoCapture(str(input_video_path))
        if cap.isOpened():
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if source_fps > 0:
                duration = float(frame_count) / float(source_fps)
        cap.release()
    windows = legacy.iter_windows(
        duration,
        window=window_sec,
        overlap=overlap_sec,
        max_windows=max_windows,
    )
    if not windows:
        return _fallback(
            fallback,
            "REID_NO_WINDOWS_AVAILABLE",
            *original_args,
            **original_kwargs,
        )

    anchors = normalize_anchors(selections)
    if not anchors:
        anchors = normalize_anchors([player_ref_norm], max_items=1)
    if not anchors:
        return _fallback(
            fallback,
            "REID_ANCHOR_NORMALIZATION_FAILED",
            *original_args,
            **original_kwargs,
        )
    anchor_records: list[dict[str, Any]] = []
    anchors_by_window: dict[int, list[dict[str, Any]]] = {}
    anchor_matches_by_id: dict[int, dict[str, Any]] = {}
    for anchor_number, raw_anchor in enumerate(anchors, start=1):
        window_index = _canonical_anchor_window(windows, float(raw_anchor["t"]))
        anchor = {
            **dict(raw_anchor),
            "anchor_id": anchor_number,
            "window_index": window_index,
        }
        anchor_records.append(anchor)
        if window_index is None:
            anchor_matches_by_id[anchor_number] = {
                "anchor_id": anchor_number,
                "frame_key": anchor.get("frame_key"),
                "time_sec": float(anchor["t"]),
                "window_index": None,
                "status": "OUTSIDE_VIDEO_WINDOWS",
                "local_track_id": None,
                "source": "selection",
            }
        else:
            anchors_by_window.setdefault(window_index, []).append(anchor)

    primary_anchor = min(
        anchor_records,
        key=lambda item: _anchor_distance(item, player_ref_norm),
    )
    primary_anchor_id = int(primary_anchor["anchor_id"])
    thresholds = _association_thresholds()
    anchor_tracklet_radius = _env_float(
        "PLAYER_REID_MANUAL_ANCHOR_RADIUS_SEC",
        max(2.0, min(10.0, float(overlap_sec))),
        1.0,
        15.0,
    )
    anchor_descriptor_radius = _env_float(
        "PLAYER_REID_MANUAL_ANCHOR_DESCRIPTOR_RADIUS_SEC",
        1.0,
        0.25,
        3.0,
    )
    timeout_seconds = int(os.environ.get("TRACKING_TIMEOUT_SECONDS", "1200"))
    started_at = time.monotonic()
    model = YOLO(detector_model)
    anchor_fps, anchor_detector_model = _anchor_acquisition_profile(
        tracking_fps=fps,
        tracking_detector_model=detector_model,
    )
    anchor_model: YOLO | None = None
    attempt_component = _analysis_attempt_component(analysis_attempt_id)
    windows_dir = (
        Path("/tmp/fnh_jobs")
        / job_id
        / "attempts"
        / attempt_component
        / "tracking"
        / "windows"
    )
    windows_dir.mkdir(parents=True, exist_ok=True)
    identity_id = f"job-{job_id}-selected-player"
    window_cache: dict[
        int,
        tuple[
            Path,
            list[dict[str, Any]],
            dict[int, list[dict[str, Any]]],
            int,
        ],
    ] = {}

    def collect(
        index: int,
    ) -> tuple[
        Path,
        list[dict[str, Any]],
        dict[int, list[dict[str, Any]]],
        int,
    ]:
        nonlocal anchor_model
        cached = window_cache.get(index)
        if cached is not None:
            return cached
        window_start, window_end = windows[index]
        segment_path = windows_dir / f"window_{index + 1:04d}.mp4"
        is_anchor_window = index in anchors_by_window
        legacy._extract_segment(
            input_video_path,
            segment_path,
            window_start,
            max(0.0, window_end - window_start),
            accurate=is_anchor_window,
        )
        sample_fps = anchor_fps if is_anchor_window else fps
        selected_model = model
        if is_anchor_window and anchor_detector_model != detector_model:
            if anchor_model is None:
                anchor_model = YOLO(anchor_detector_model)
            selected_model = anchor_model
        try:
            samples, track_map = legacy._collect_window_samples(
                str(segment_path),
                fps=sample_fps,
                model=selected_model,
                tracker=tracker,
                job_id=job_id,
                analysis_attempt_id=analysis_attempt_id,
                tracking_started_at=started_at,
                tracking_timeout_seconds=timeout_seconds,
            )
        finally:
            _reset_tracker(selected_model)
        collected = (segment_path, samples, track_map, sample_fps)
        window_cache[index] = collected
        return collected

    def resolve_manual_anchors(
        index: int,
        segment_path: Path,
        samples: list[dict[str, Any]],
        track_map: dict[int, list[dict[str, Any]]],
        *,
        known: Mapping[int, tuple[int, AppearanceDescriptor]] | None = None,
    ) -> list[dict[str, Any]]:
        window_start, window_end = windows[index]
        known = known or {}
        pending: list[tuple[dict[str, Any], int]] = []
        observations: list[dict[str, Any]] = []
        for anchor in anchors_by_window.get(index, []):
            anchor_id = int(anchor["anchor_id"])
            source = (
                "primary_player_ref" if anchor_id == primary_anchor_id else "selection"
            )
            known_match = known.get(anchor_id)
            if known_match is not None:
                track_id, descriptor = known_match
                observations.append(
                    {
                        "anchor": anchor,
                        "track_id": int(track_id),
                        "descriptor": descriptor,
                    }
                )
                anchor_matches_by_id[anchor_id] = {
                    "anchor_id": anchor_id,
                    "frame_key": anchor.get("frame_key"),
                    "time_sec": float(anchor["t"]),
                    "window_index": index,
                    "window_start": float(window_start),
                    "window_end": float(window_end),
                    "status": "MATCHED",
                    "local_track_id": int(track_id),
                    "source": source,
                }
                continue
            track_id = _select_anchor_track(
                samples,
                track_map,
                anchor_time_local=max(0.0, float(anchor["t"]) - float(window_start)),
                anchor_bbox=anchor,
            )
            if track_id is None:
                anchor_matches_by_id[anchor_id] = {
                    "anchor_id": anchor_id,
                    "frame_key": anchor.get("frame_key"),
                    "time_sec": float(anchor["t"]),
                    "window_index": index,
                    "window_start": float(window_start),
                    "window_end": float(window_end),
                    "status": "TRACK_NOT_FOUND",
                    "local_track_id": None,
                    "source": source,
                }
                continue
            pending.append((anchor, int(track_id)))

        try:
            # Key descriptor requests by anchor_id, not raw track_id. Two
            # manual anchors can legitimately resolve to the same ByteTrack ID
            # at different times; merging their crops would contaminate and
            # double-weight the identity prototype.
            scoped_track_map: dict[int, Sequence[Mapping[str, Any]]] = {
                int(anchor["anchor_id"]): _anchor_descriptor_detections(
                    track_map.get(track_id) or [],
                    anchor_time_local=max(
                        0.0,
                        float(anchor["t"]) - float(window_start),
                    ),
                    anchor_bbox=anchor,
                    radius_sec=anchor_descriptor_radius,
                )
                for anchor, track_id in pending
            }
            descriptors = _extract_descriptors_for_tracks(
                segment_path,
                scoped_track_map,
                list(scoped_track_map),
            )
        except Exception:
            logger.exception(
                "Manual-anchor descriptor extraction failed "
                "job_id=%s window_index=%s",
                job_id,
                index,
            )
            for anchor, track_id in pending:
                anchor_id = int(anchor["anchor_id"])
                anchor_matches_by_id[anchor_id] = {
                    "anchor_id": anchor_id,
                    "frame_key": anchor.get("frame_key"),
                    "time_sec": float(anchor["t"]),
                    "window_index": index,
                    "window_start": float(window_start),
                    "window_end": float(window_end),
                    "status": "DESCRIPTOR_PROCESSING_FAILED",
                    "local_track_id": track_id,
                    "source": (
                        "primary_player_ref"
                        if anchor_id == primary_anchor_id
                        else "selection"
                    ),
                }
            return observations
        for anchor, track_id in pending:
            anchor_id = int(anchor["anchor_id"])
            source = (
                "primary_player_ref" if anchor_id == primary_anchor_id else "selection"
            )
            descriptor = descriptors.get(anchor_id)
            if descriptor is None:
                anchor_matches_by_id[anchor_id] = {
                    "anchor_id": anchor_id,
                    "frame_key": anchor.get("frame_key"),
                    "time_sec": float(anchor["t"]),
                    "window_index": index,
                    "window_start": float(window_start),
                    "window_end": float(window_end),
                    "status": "DESCRIPTOR_UNAVAILABLE",
                    "local_track_id": track_id,
                    "source": source,
                }
                continue
            observations.append(
                {
                    "anchor": anchor,
                    "track_id": track_id,
                    "descriptor": descriptor,
                }
            )
            anchor_matches_by_id[anchor_id] = {
                "anchor_id": anchor_id,
                "frame_key": anchor.get("frame_key"),
                "time_sec": float(anchor["t"]),
                "window_index": index,
                "window_start": float(window_start),
                "window_end": float(window_end),
                "status": "MATCHED",
                "local_track_id": track_id,
                "source": source,
            }
        return observations

    seed: (
        tuple[
            dict[str, Any],
            int,
            Path,
            list[dict[str, Any]],
            dict[int, list[dict[str, Any]]],
            int,
            int,
            AppearanceDescriptor,
        ]
        | None
    ) = None
    acquisition_errors = 0
    seed_candidates = [
        primary_anchor,
        *[
            anchor
            for anchor in anchor_records
            if int(anchor["anchor_id"]) != primary_anchor_id
        ],
    ]
    for seed_anchor in seed_candidates:
        seed_anchor_id = int(seed_anchor["anchor_id"])
        seed_window_index = seed_anchor.get("window_index")
        if seed_window_index is None:
            continue
        seed_window_index = int(seed_window_index)
        seed_start, seed_end = windows[seed_window_index]
        source = (
            "primary_player_ref" if seed_anchor_id == primary_anchor_id else "selection"
        )
        try:
            (
                seed_path,
                seed_samples,
                seed_track_map,
                seed_sample_fps,
            ) = collect(seed_window_index)
        except (legacy.TrackingTimeoutError, StaleAnalysisAttemptError):
            raise
        except Exception:
            acquisition_errors += 1
            logger.exception(
                "ReID anchor acquisition failed job_id=%s anchor_id=%s",
                job_id,
                seed_anchor_id,
            )
            anchor_matches_by_id[seed_anchor_id] = {
                "anchor_id": seed_anchor_id,
                "frame_key": seed_anchor.get("frame_key"),
                "time_sec": float(seed_anchor["t"]),
                "window_index": seed_window_index,
                "window_start": float(seed_start),
                "window_end": float(seed_end),
                "status": "WINDOW_PROCESSING_FAILED",
                "local_track_id": None,
                "source": source,
            }
            continue

        seed_track_id = _select_anchor_track(
            seed_samples,
            seed_track_map,
            anchor_time_local=max(0.0, float(seed_anchor["t"]) - float(seed_start)),
            anchor_bbox=seed_anchor,
        )
        if seed_track_id is None:
            anchor_matches_by_id[seed_anchor_id] = {
                "anchor_id": seed_anchor_id,
                "frame_key": seed_anchor.get("frame_key"),
                "time_sec": float(seed_anchor["t"]),
                "window_index": seed_window_index,
                "window_start": float(seed_start),
                "window_end": float(seed_end),
                "status": "TRACK_NOT_FOUND",
                "local_track_id": None,
                "source": source,
            }
            continue
        try:
            seed_local_time = max(0.0, float(seed_anchor["t"]) - float(seed_start))
            seed_tracklet_map = {
                seed_track_id: _anchor_descriptor_detections(
                    seed_track_map.get(seed_track_id) or [],
                    anchor_time_local=seed_local_time,
                    anchor_bbox=seed_anchor,
                    radius_sec=anchor_descriptor_radius,
                )
            }
            seed_descriptor = _extract_descriptors_for_tracks(
                seed_path, seed_tracklet_map, [seed_track_id]
            ).get(seed_track_id)
        except Exception:
            acquisition_errors += 1
            logger.exception(
                "ReID anchor descriptor acquisition failed " "job_id=%s anchor_id=%s",
                job_id,
                seed_anchor_id,
            )
            anchor_matches_by_id[seed_anchor_id] = {
                "anchor_id": seed_anchor_id,
                "frame_key": seed_anchor.get("frame_key"),
                "time_sec": float(seed_anchor["t"]),
                "window_index": seed_window_index,
                "window_start": float(seed_start),
                "window_end": float(seed_end),
                "status": "DESCRIPTOR_PROCESSING_FAILED",
                "local_track_id": seed_track_id,
                "source": source,
            }
            continue
        if seed_descriptor is None:
            anchor_matches_by_id[seed_anchor_id] = {
                "anchor_id": seed_anchor_id,
                "frame_key": seed_anchor.get("frame_key"),
                "time_sec": float(seed_anchor["t"]),
                "window_index": seed_window_index,
                "window_start": float(seed_start),
                "window_end": float(seed_end),
                "status": "DESCRIPTOR_UNAVAILABLE",
                "local_track_id": seed_track_id,
                "source": source,
            }
            continue
        seed = (
            seed_anchor,
            seed_window_index,
            seed_path,
            seed_samples,
            seed_track_map,
            seed_sample_fps,
            int(seed_track_id),
            seed_descriptor,
        )
        break

    if seed is None:
        for anchor in anchor_records:
            anchor_id = int(anchor["anchor_id"])
            anchor_matches_by_id.setdefault(
                anchor_id,
                {
                    "anchor_id": anchor_id,
                    "frame_key": anchor.get("frame_key"),
                    "time_sec": float(anchor["t"]),
                    "window_index": anchor.get("window_index"),
                    "status": "NOT_PROCESSED",
                    "local_track_id": None,
                    "source": (
                        "primary_player_ref"
                        if anchor_id == primary_anchor_id
                        else "selection"
                    ),
                },
            )
        anchor_matches = [
            anchor_matches_by_id[anchor_id]
            for anchor_id in sorted(anchor_matches_by_id)
        ]
        status = (
            "ANCHOR_ACQUISITION_ERROR" if acquisition_errors > 0 else "ANCHOR_NOT_FOUND"
        )
        reason_code = (
            "REID_ANCHOR_ACQUISITION_ERROR"
            if status == "ANCHOR_ACQUISITION_ERROR"
            else "REID_ANCHORS_NOT_FOUND"
        )
        action_required = (
            "RETRY_ANALYSIS"
            if status == "ANCHOR_ACQUISITION_ERROR"
            else "RESELECT_PLAYER"
        )
        output = _anchor_failure_output(
            duration=duration,
            fps=fps,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            windows_total=len(windows),
            windows_processed=len(window_cache),
            player_ref=player_ref_norm,
            anchors=anchor_records,
            anchor_matches=anchor_matches,
            anchor_fps=anchor_fps,
            anchor_detector_model=anchor_detector_model,
            status=status,
            reason_code=reason_code,
            action_required=action_required,
        )
        return _persist_tracking_output(
            job_id,
            output,
            analysis_attempt_id=analysis_attempt_id,
            endpoint_url=endpoint_url,
            bucket=bucket,
            expires_seconds=expires_seconds,
        )

    (
        seed_anchor,
        anchor_index,
        anchor_path,
        anchor_samples,
        anchor_track_map,
        anchor_sample_fps,
        anchor_track_id,
        anchor_descriptor,
    ) = seed
    anchor_time = float(seed_anchor["t"])
    anchor_start, anchor_end = windows[anchor_index]
    _anchor_index, forward_indices, backward_indices = processing_order(
        windows, anchor_time
    )
    if _anchor_index != anchor_index:
        raise RuntimeError("Canonical seed window does not match processing order")
    seed_anchor_id = int(seed_anchor["anchor_id"])

    anchor_observations = resolve_manual_anchors(
        anchor_index,
        anchor_path,
        anchor_samples,
        anchor_track_map,
        known={
            seed_anchor_id: (
                anchor_track_id,
                anchor_descriptor,
            )
        },
    )
    anchor_bboxes, anchor_track_ids = _stitch_manual_anchor_bboxes(
        anchor_observations,
        anchor_samples,
        fps=anchor_sample_fps,
        window_start=anchor_start,
        radius_sec=anchor_tracklet_radius,
    )
    if not anchor_bboxes:
        output = _anchor_failure_output(
            duration=duration,
            fps=fps,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            windows_total=len(windows),
            windows_processed=len(window_cache),
            player_ref=player_ref_norm,
            anchors=anchor_records,
            anchor_matches=[
                anchor_matches_by_id[anchor_id]
                for anchor_id in sorted(anchor_matches_by_id)
            ],
            anchor_fps=anchor_fps,
            anchor_detector_model=anchor_detector_model,
            status="ANCHOR_TRACK_EMPTY",
            reason_code="REID_ANCHOR_TRACK_EMPTY",
            action_required="RESELECT_PLAYER",
        )
        return _persist_tracking_output(
            job_id,
            output,
            analysis_attempt_id=analysis_attempt_id,
            endpoint_url=endpoint_url,
            bucket=bucket,
            expires_seconds=expires_seconds,
        )
    anchor_coverage = len(anchor_bboxes) / float(max(1, len(anchor_samples))) * 100.0
    anchor_segment = {
        "window_index": int(anchor_index),
        "parent_window_index": None,
        "window_start": float(anchor_start),
        "window_end": float(anchor_end),
        "direction": "anchor",
        "processing_direction": "anchor",
        "selected_track_id": anchor_track_id,
        "selected_track_ids": anchor_track_ids,
        "identity_id": identity_id,
        "identity_status": "ACCEPTED",
        "reacquire_score": 1.0,
        "coverage_pct": round(anchor_coverage, 2),
        "sample_fps": anchor_sample_fps,
        "lost_segments": [],
        "bboxes": anchor_bboxes,
        "reid": {
            "version": ASSOCIATION_VERSION,
            "validated": False,
            "status": "ACCEPTED",
            "identity_id": identity_id,
            "selected_candidate_id": str(anchor_track_id),
            "best_score": 1.0,
            "margin": 1.0,
            "reason_codes": [
                (
                    "MANUAL_MULTI_ANCHOR"
                    if len(anchor_observations) > 1
                    else "MANUAL_ANCHOR"
                )
            ],
            "descriptor": _descriptor_metadata(anchor_descriptor),
            "candidates": [],
        },
    }
    base_profile = IdentityProfile(
        identity_id=identity_id,
        descriptor=anchor_descriptor,
        source="manual_anchor_track",
    )
    for observation in anchor_observations:
        if int(observation["anchor"]["anchor_id"]) == seed_anchor_id:
            continue
        base_profile = update_identity_profile(base_profile, observation["descriptor"])
    segments_by_index: dict[int, dict[str, Any]] = {anchor_index: anchor_segment}
    accepted_associations = 0
    abstained_associations = 0
    processing_failures = 0
    anchor_reacquisitions = 0
    total_profile_samples = base_profile.descriptor.sample_count

    def process_direction(indices: Sequence[int], direction: str) -> None:
        nonlocal accepted_associations, abstained_associations
        nonlocal processing_failures, total_profile_samples
        nonlocal anchor_reacquisitions
        profile = base_profile
        previous_bboxes: Sequence[Mapping[str, Any]] = anchor_bboxes
        identity_available = True
        parent_window_index = anchor_index
        for index in indices:
            current_parent_window_index = parent_window_index
            parent_window_index = index
            window_start, window_end = windows[index]
            try:
                segment_path, samples, track_map, sample_fps = collect(index)
            except (legacy.TrackingTimeoutError, StaleAnalysisAttemptError):
                raise
            except Exception:
                logger.exception(
                    "ReID window failed job_id=%s index=%s direction=%s",
                    job_id,
                    index,
                    direction,
                )
                processing_failures += 1
                abstained_associations += 1
                for anchor in anchors_by_window.get(index, []):
                    anchor_id = int(anchor["anchor_id"])
                    anchor_matches_by_id[anchor_id] = {
                        "anchor_id": anchor_id,
                        "frame_key": anchor.get("frame_key"),
                        "time_sec": float(anchor["t"]),
                        "window_index": index,
                        "window_start": float(window_start),
                        "window_end": float(window_end),
                        "status": "WINDOW_PROCESSING_FAILED",
                        "local_track_id": None,
                        "source": (
                            "primary_player_ref"
                            if anchor_id == primary_anchor_id
                            else "selection"
                        ),
                    }
                segments_by_index[index] = _empty_segment(
                    window_index=index,
                    parent_window_index=current_parent_window_index,
                    window_start=window_start,
                    window_end=window_end,
                    direction=direction,
                    processing_direction=direction,
                    reason_code="WINDOW_PROCESSING_FAILED",
                    identity_id=identity_id,
                )
                identity_available = False
                continue

            manual_anchors = anchors_by_window.get(index, [])
            if manual_anchors:
                manual_observations = resolve_manual_anchors(
                    index,
                    segment_path,
                    samples,
                    track_map,
                )
                if not manual_observations:
                    abstained_associations += 1
                    segments_by_index[index] = _empty_segment(
                        window_index=index,
                        parent_window_index=current_parent_window_index,
                        window_start=window_start,
                        window_end=window_end,
                        direction="anchor",
                        processing_direction=direction,
                        reason_code="MANUAL_ANCHOR_SET_INCOMPLETE",
                        identity_id=identity_id,
                    )
                    identity_available = False
                    continue

                manual_bboxes, manual_track_ids = _stitch_manual_anchor_bboxes(
                    manual_observations,
                    samples,
                    fps=sample_fps,
                    window_start=window_start,
                    radius_sec=anchor_tracklet_radius,
                )
                if not manual_bboxes or not manual_track_ids:
                    for observation in manual_observations:
                        anchor_id = int(observation["anchor"]["anchor_id"])
                        match = anchor_matches_by_id.get(anchor_id)
                        if match:
                            match["status"] = "TRACK_EMPTY"
                    abstained_associations += 1
                    segments_by_index[index] = _empty_segment(
                        window_index=index,
                        parent_window_index=current_parent_window_index,
                        window_start=window_start,
                        window_end=window_end,
                        direction="anchor",
                        processing_direction=direction,
                        reason_code="MANUAL_ANCHOR_TRACK_EMPTY",
                        identity_id=identity_id,
                    )
                    identity_available = False
                    continue

                if identity_available:
                    for observation in manual_observations:
                        profile = update_identity_profile(
                            profile, observation["descriptor"]
                        )
                else:
                    first_descriptor = manual_observations[0]["descriptor"]
                    profile = IdentityProfile(
                        identity_id=identity_id,
                        descriptor=first_descriptor,
                        source="manual_anchor_reseed",
                    )
                    for observation in manual_observations[1:]:
                        profile = update_identity_profile(
                            profile, observation["descriptor"]
                        )
                    anchor_reacquisitions += 1

                total_profile_samples = max(
                    total_profile_samples, profile.descriptor.sample_count
                )
                previous_bboxes = manual_bboxes
                identity_available = True
                coverage = len(manual_bboxes) / float(max(1, len(samples))) * 100.0
                primary_track_id = manual_track_ids[0]
                segments_by_index[index] = {
                    "window_index": int(index),
                    "parent_window_index": int(current_parent_window_index),
                    "window_start": float(window_start),
                    "window_end": float(window_end),
                    "direction": "anchor",
                    "processing_direction": direction,
                    "selected_track_id": primary_track_id,
                    "selected_track_ids": manual_track_ids,
                    "identity_id": identity_id,
                    "identity_status": "ACCEPTED",
                    "reacquire_score": 1.0,
                    "coverage_pct": round(float(coverage), 2),
                    "sample_fps": sample_fps,
                    "lost_segments": [],
                    "bboxes": manual_bboxes,
                    "reid": {
                        "version": ASSOCIATION_VERSION,
                        "validated": False,
                        "status": "ACCEPTED",
                        "identity_id": identity_id,
                        "selected_candidate_id": str(primary_track_id),
                        "best_score": 1.0,
                        "margin": 1.0,
                        "reason_codes": [
                            (
                                "MANUAL_MULTI_ANCHOR"
                                if len(manual_observations) > 1
                                else "MANUAL_ANCHOR_RESEED"
                            )
                        ],
                        "descriptor": _descriptor_metadata(profile.descriptor),
                        "candidates": [],
                    },
                }
                continue

            candidates, id_lookup, descriptor_lookup = _build_candidate_profiles(
                segment_path,
                track_map,
                previous_bboxes=previous_bboxes,
                window_start=window_start,
                direction=direction,
                fps=sample_fps,
                strong_overlap_score=thresholds.strong_overlap_score,
            )
            decision = associate_identity(
                profile,
                candidates,
                thresholds=thresholds,
            )
            selected_profile = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.candidate_id == (decision.selected_candidate_id or "")
                ),
                None,
            )
            selected_track_id = (
                id_lookup.get(decision.selected_candidate_id or "")
                if decision.accepted
                else None
            )
            descriptor = (
                descriptor_lookup.get(decision.selected_candidate_id or "")
                if decision.accepted
                else None
            )
            selected_metadata = (
                dict(selected_profile.metadata or {})
                if selected_profile is not None
                else {}
            )
            tracklet_sample_indices = tuple(
                int(value)
                for value in (selected_metadata.get("tracklet_sample_indices") or ())
            )
            tracklet_detections = tuple(
                item
                for item in (selected_metadata.get("tracklet_detections") or ())
                if isinstance(item, Mapping)
            )
            if selected_track_id is not None and tracklet_detections:
                scoped_samples = _samples_for_tracklet(
                    samples,
                    track_id=selected_track_id,
                    tracklet_detections=tracklet_detections,
                )
                bboxes, lost_segments, _ = legacy._build_window_bboxes(
                    scoped_samples,
                    selected_track_id,
                    fps=sample_fps,
                    time_offset=window_start,
                )
            else:
                bboxes, lost_segments = [], []
            if (
                decision.accepted
                and selected_track_id is not None
                and descriptor is not None
                and bboxes
            ):
                accepted_associations += 1
                # Autonomous links remain experimental until the post-hoc kit
                # guard has run. Keep the identity prototype manual-anchor-only
                # so a later-rejected window cannot contaminate propagation.
                previous_bboxes = bboxes
                identity_status = "ACCEPTED"
                segment_identity_id: str | None = identity_id
                identity_available = True
            else:
                abstained_associations += 1
                selected_track_id = None
                bboxes = []
                lost_segments = []
                identity_status = "ABSTAINED"
                segment_identity_id = None
                identity_available = False
            coverage = len(bboxes) / float(max(1, len(samples))) * 100.0
            reid_payload = decision.to_payload()
            reid_payload.update(
                {
                    "identity_id": identity_id,
                    "descriptor": _descriptor_metadata(descriptor),
                    "tracklet_scope": selected_metadata.get("tracklet_scope"),
                    "tracklet_detection_count": len(tracklet_sample_indices),
                }
            )
            segments_by_index[index] = {
                "window_index": int(index),
                "parent_window_index": int(current_parent_window_index),
                "window_start": float(window_start),
                "window_end": float(window_end),
                "direction": direction,
                "processing_direction": direction,
                "selected_track_id": selected_track_id,
                "identity_id": segment_identity_id,
                "identity_status": identity_status,
                "reacquire_score": round(float(decision.best_score), 4),
                "coverage_pct": round(float(coverage), 2),
                "sample_fps": sample_fps,
                "lost_segments": lost_segments,
                "bboxes": bboxes,
                "reid": reid_payload,
            }
            processed = len(segments_by_index)
            if processed % 5 == 0 or processed == len(windows):
                pct = 10 + int((processed / float(len(windows))) * 30)
                legacy._update_tracking_progress(
                    job_id,
                    pct,
                    "Tracking player with experimental ReID",
                    analysis_attempt_id=analysis_attempt_id,
                )

    try:
        process_direction(forward_indices, "forward")
        process_direction(backward_indices, "backward")
    except (legacy.TrackingTimeoutError, StaleAnalysisAttemptError):
        raise
    except Exception:
        logger.exception(
            "ReID selected-player window processing failed job_id=%s",
            job_id,
        )
        for anchor in anchor_records:
            anchor_id = int(anchor["anchor_id"])
            anchor_matches_by_id.setdefault(
                anchor_id,
                {
                    "anchor_id": anchor_id,
                    "frame_key": anchor.get("frame_key"),
                    "time_sec": float(anchor["t"]),
                    "window_index": anchor.get("window_index"),
                    "status": "NOT_PROCESSED",
                    "local_track_id": None,
                    "source": (
                        "primary_player_ref"
                        if anchor_id == primary_anchor_id
                        else "selection"
                    ),
                },
            )
        output = _anchor_failure_output(
            duration=duration,
            fps=fps,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            windows_total=len(windows),
            windows_processed=len(window_cache),
            player_ref=player_ref_norm,
            anchors=anchor_records,
            anchor_matches=[
                anchor_matches_by_id[anchor_id]
                for anchor_id in sorted(anchor_matches_by_id)
            ],
            anchor_fps=anchor_fps,
            anchor_detector_model=anchor_detector_model,
            status="WINDOW_PROCESSING_ERROR",
            reason_code="REID_WINDOW_PROCESSING_ERROR",
            action_required="RETRY_ANALYSIS",
        )
        return _persist_tracking_output(
            job_id,
            output,
            analysis_attempt_id=analysis_attempt_id,
            endpoint_url=endpoint_url,
            bucket=bucket,
            expires_seconds=expires_seconds,
        )
    segments = [segments_by_index[index] for index in range(len(windows))]
    segments_with_player = sum(1 for segment in segments if segment.get("bboxes"))
    autonomous_evidence = _autonomous_tracking_evidence(
        segments,
        fps=float(max(1, fps)),
    )
    autonomous_segments_with_player = int(autonomous_evidence["segments_with_player"])
    autonomous_bboxes_count = int(autonomous_evidence["bboxes_count"])
    for segment_index, count in autonomous_evidence["segment_counts"].items():
        reid_payload = dict(segments[segment_index].get("reid") or {})
        reid_payload["autonomous_bboxes_count"] = int(count)
        segments[segment_index]["reid"] = reid_payload
    coverage_pct = tracking_coverage_pct(
        segments,
        duration_sec=duration,
        fps=float(max(1, fps)),
    )
    largest_gap = largest_tracking_gap_sec(
        segments,
        duration_sec=duration,
    )
    attempted_associations = accepted_associations + abstained_associations
    accepted_ratio = (
        accepted_associations / float(attempted_associations)
        if attempted_associations > 0
        else 0.0
    )
    for anchor in anchor_records:
        anchor_id = int(anchor["anchor_id"])
        anchor_matches_by_id.setdefault(
            anchor_id,
            {
                "anchor_id": anchor_id,
                "frame_key": anchor.get("frame_key"),
                "time_sec": float(anchor["t"]),
                "window_index": anchor.get("window_index"),
                "status": "NOT_PROCESSED",
                "local_track_id": None,
                "source": (
                    "primary_player_ref"
                    if anchor_id == primary_anchor_id
                    else "selection"
                ),
            },
        )
    anchor_matches = [
        anchor_matches_by_id[anchor_id] for anchor_id in sorted(anchor_matches_by_id)
    ]
    anchors_matched = sum(
        1 for match in anchor_matches if match.get("status") == "MATCHED"
    )
    autonomous_identity_proven = bool(autonomous_evidence["proven"])
    tracking_success = bool(
        segments_with_player and anchors_matched and autonomous_identity_proven
    )
    tracking_scope_status = (
        "CROSS_WINDOW_EVIDENCE"
        if autonomous_identity_proven
        else "ANCHOR_ONLY" if segments_with_player and anchors_matched else "EMPTY"
    )
    output: dict[str, Any] = {
        "mode": "full_match_windowed",
        "identity_mode": "appearance_reid_v1",
        "method": "yolo+bytetrack+appearance_reid",
        "fps": fps,
        "window_sec": window_sec,
        "overlap_sec": overlap_sec,
        "segments": segments,
        "segments_total": len(segments),
        "segments_with_player": segments_with_player,
        "autonomous_segments_with_player": autonomous_segments_with_player,
        "autonomous_bboxes_count": autonomous_bboxes_count,
        "tracking_scope_status": tracking_scope_status,
        "windows_processed": len(window_cache),
        "coverage_pct_total": round(coverage_pct, 2),
        "largest_gap_sec": round(largest_gap, 2),
        "coverage_pct": round(coverage_pct, 2),
        "anchors_total": len(anchor_records),
        "anchors_matched": anchors_matched,
        "anchor_matches": anchor_matches,
        "anchor_reacquisitions": anchor_reacquisitions,
        "tracking_success": tracking_success,
        "partial": bool(tracking_success and coverage_pct < 5.0),
        "partial_reason": (
            "SPARSE_CROSS_WINDOW_EVIDENCE"
            if tracking_success and coverage_pct < 5.0
            else None
        ),
        "tracking_status": (
            ("SPARSE_CROSS_WINDOW_EVIDENCE" if coverage_pct < 5.0 else "SUCCEEDED")
            if tracking_success
            else (
                "ANCHOR_ONLY"
                if segments_with_player and anchors_matched
                else "NO_PLAYER_TRACK"
            )
        ),
        "action_required": (None if tracking_success else "RESELECT_PLAYER"),
        "anchor_acquisition": {
            "fps": anchor_fps,
            "detector_model": anchor_detector_model,
            "windows_processed": sum(
                1 for index in window_cache if index in anchors_by_window
            ),
            "seed_anchor_id": seed_anchor_id,
            "seed_window_index": anchor_index,
            "seed_anchor": {
                key: value
                for key, value in seed_anchor.items()
                if key
                in {
                    "t",
                    "x",
                    "y",
                    "w",
                    "h",
                    "frame_key",
                    "anchor_id",
                    "window_index",
                }
            },
        },
        "anchors_used": {
            "player_ref": player_ref_norm,
            "selections": anchors,
        },
        "reid_summary": {
            "status": "EXPERIMENTAL",
            "validated": False,
            "identity_id": identity_id,
            "descriptor_version": DESCRIPTOR_VERSION,
            "association_version": ASSOCIATION_VERSION,
            "anchor_window_index": anchor_index,
            "anchor_local_track_id": anchor_track_id,
            "anchor_descriptor": _descriptor_metadata(anchor_descriptor),
            "accepted_associations": accepted_associations,
            "abstained_associations": abstained_associations,
            "autonomous_segments_with_player": (autonomous_segments_with_player),
            "autonomous_bboxes_count": autonomous_bboxes_count,
            "autonomous_minimum_samples": int(autonomous_evidence["minimum_samples"]),
            "autonomous_boundary_tolerance_sec": float(
                autonomous_evidence["boundary_tolerance_sec"]
            ),
            "tracking_scope_status": tracking_scope_status,
            "processing_failures": processing_failures,
            "accepted_ratio": round(accepted_ratio, 6),
            "profile_samples": total_profile_samples,
            "anchors_total": len(anchor_records),
            "anchors_matched": anchors_matched,
            "anchor_matches": anchor_matches,
            "anchor_reacquisitions": anchor_reacquisitions,
            "reason_codes": [
                "EXPERIMENTAL_NOT_VALIDATED",
                "BENCHMARK_REQUIRED_BEFORE_PLAYER_SCORING",
                *([] if autonomous_identity_proven else ["AUTONOMOUS_REID_NOT_PROVEN"]),
            ],
        },
        "notes": (
            "Cross-window identity associations use an experimental appearance "
            "descriptor, temporal overlap, and geometry. Ambiguous windows are "
            "omitted instead of being assigned to the selected player."
        ),
    }
    return _persist_tracking_output(
        job_id,
        output,
        analysis_attempt_id=analysis_attempt_id,
        endpoint_url=endpoint_url,
        bucket=bucket,
        expires_seconds=expires_seconds,
    )
