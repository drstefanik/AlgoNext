from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import cv2
from ultralytics import YOLO

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
from app.reid.window_logic import (
    candidate_rank,
    choose_descriptor_detections,
    geometry_similarity,
    largest_tracking_gap_sec,
    processing_order,
    temporal_overlap_score,
    tracking_coverage_pct,
)
from app.workers import tracking as legacy

logger = logging.getLogger(__name__)


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


def _association_thresholds() -> AssociationThresholds:
    return AssociationThresholds(
        min_combined_score=_env_float(
            "PLAYER_REID_MIN_COMBINED_SCORE", 0.76, 0.0, 1.0
        ),
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
        min_descriptor_samples=_env_int(
            "PLAYER_REID_MIN_DESCRIPTOR_SAMPLES", 2, 1, 20
        ),
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
        geometry = geometry_similarity(anchor_bbox, bbox)
        confidence = float(detection.get("conf") or 0.0)
        temporal = max(
            0.0,
            1.0
            - abs(float(detection.get("t") or 0.0) - anchor_time_local) / 2.0,
        )
        ranked.append(
            (
                iou * 2.0
                + geometry * 0.7
                + temporal * 0.2
                + confidence * 0.1,
                track_id,
            )
        )
    if not ranked:
        return None
    best_score, best_track_id = max(ranked)
    return best_track_id if best_score >= 0.35 else None


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
    return min(detections, key=key) if direction == "forward" else max(
        detections, key=key
    )


def _build_candidate_profiles(
    segment_path: Path,
    track_map: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    previous_bboxes: Sequence[Mapping[str, Any]],
    window_start: float,
    direction: str,
    fps: int,
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
    selected = ranked[:max_candidates]
    selected_ids = [item[1] for item in selected]
    descriptor_by_track = _extract_descriptors_for_tracks(
        segment_path, track_map, selected_ids
    )

    profiles: list[CandidateProfile] = []
    id_lookup: dict[str, int] = {}
    descriptor_lookup: dict[str, AppearanceDescriptor | None] = {}
    for _, track_id, overlap, geometry in selected:
        candidate_id = str(track_id)
        descriptor = descriptor_by_track.get(track_id)
        id_lookup[candidate_id] = track_id
        descriptor_lookup[candidate_id] = descriptor
        profiles.append(
            CandidateProfile(
                candidate_id=candidate_id,
                descriptor=descriptor,
                overlap_score=overlap,
                geometry_score=geometry,
                detection_count=len(track_map.get(track_id) or []),
                metadata={"local_track_id": track_id},
            )
        )
    return profiles, id_lookup, descriptor_lookup


def _empty_segment(
    *,
    window_start: float,
    window_end: float,
    direction: str,
    reason_code: str,
    identity_id: str,
) -> dict[str, Any]:
    return {
        "window_start": float(window_start),
        "window_end": float(window_end),
        "direction": direction,
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
    endpoint_url: str,
    bucket: str,
    expires_seconds: int,
) -> dict[str, Any]:
    tracking_dir = Path("/tmp/fnh_jobs") / job_id / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = tracking_dir / "tracking.json"
    with tracking_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    s3_internal = legacy._get_s3_client(endpoint_url)
    legacy._ensure_bucket_exists(s3_internal, bucket)
    tracking_key = f"jobs/{job_id}/tracking/tracking.json"
    legacy._upload_file(
        s3_internal,
        bucket,
        tracking_path,
        tracking_key,
        "application/json",
    )
    result = dict(output)
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
    output = fallback(*args, **kwargs)
    output = dict(output)
    output["reid_summary"] = {
        "status": "FALLBACK_LEGACY",
        "validated": False,
        "reason_codes": [reason_code],
    }
    return output


def track_player_windowed_reid(
    job_id: str,
    input_video_path: str,
    player_ref: dict,
    selections: list[dict[str, Any]],
    *,
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

    anchor_time = float(player_ref_norm.get("t") or 0.0)
    anchor_index, forward_indices, backward_indices = processing_order(
        windows, anchor_time
    )
    thresholds = _association_thresholds()
    timeout_seconds = int(os.environ.get("TRACKING_TIMEOUT_SECONDS", "1200"))
    started_at = time.monotonic()
    model = YOLO(detector_model)
    windows_dir = Path("/tmp/fnh_jobs") / job_id / "tracking" / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    identity_id = f"job-{job_id}-selected-player"

    def collect(
        index: int,
    ) -> tuple[Path, list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
        window_start, window_end = windows[index]
        segment_path = windows_dir / f"window_{index + 1:04d}.mp4"
        legacy._extract_segment(
            input_video_path,
            segment_path,
            window_start,
            max(0.0, window_end - window_start),
        )
        samples, track_map = legacy._collect_window_samples(
            str(segment_path),
            fps=fps,
            model=model,
            tracker=tracker,
            job_id=job_id,
            tracking_started_at=started_at,
            tracking_timeout_seconds=timeout_seconds,
        )
        _reset_tracker(model)
        return segment_path, samples, track_map

    anchor_start, anchor_end = windows[anchor_index]
    try:
        anchor_path, anchor_samples, anchor_track_map = collect(anchor_index)
    except legacy.TrackingTimeoutError:
        raise
    except Exception:
        logger.exception("ReID anchor window failed job_id=%s", job_id)
        return _fallback(
            fallback,
            "REID_ANCHOR_WINDOW_FAILED",
            *original_args,
            **original_kwargs,
        )

    anchor_bbox = {
        "x": float(player_ref_norm.get("x") or 0.0),
        "y": float(player_ref_norm.get("y") or 0.0),
        "w": float(player_ref_norm.get("w") or 0.0),
        "h": float(player_ref_norm.get("h") or 0.0),
    }
    anchor_track_id = _select_anchor_track(
        anchor_samples,
        anchor_track_map,
        anchor_time_local=max(0.0, anchor_time - anchor_start),
        anchor_bbox=anchor_bbox,
    )
    if anchor_track_id is None:
        return _fallback(
            fallback,
            "REID_ANCHOR_TRACK_NOT_FOUND",
            *original_args,
            **original_kwargs,
        )
    anchor_descriptor = _extract_descriptors_for_tracks(
        anchor_path, anchor_track_map, [anchor_track_id]
    ).get(anchor_track_id)
    if anchor_descriptor is None:
        return _fallback(
            fallback,
            "REID_ANCHOR_DESCRIPTOR_UNAVAILABLE",
            *original_args,
            **original_kwargs,
        )

    anchor_bboxes, anchor_lost, _ = legacy._build_window_bboxes(
        anchor_samples,
        anchor_track_id,
        fps=fps,
        time_offset=anchor_start,
    )
    if not anchor_bboxes:
        return _fallback(
            fallback,
            "REID_ANCHOR_TRACK_EMPTY",
            *original_args,
            **original_kwargs,
        )
    anchor_coverage = (
        len(anchor_bboxes) / float(max(1, len(anchor_samples))) * 100.0
    )
    anchor_segment = {
        "window_start": float(anchor_start),
        "window_end": float(anchor_end),
        "direction": "anchor",
        "selected_track_id": anchor_track_id,
        "identity_id": identity_id,
        "identity_status": "ACCEPTED",
        "reacquire_score": 1.0,
        "coverage_pct": round(anchor_coverage, 2),
        "lost_segments": anchor_lost,
        "bboxes": anchor_bboxes,
        "reid": {
            "version": ASSOCIATION_VERSION,
            "validated": False,
            "status": "ACCEPTED",
            "identity_id": identity_id,
            "selected_candidate_id": str(anchor_track_id),
            "best_score": 1.0,
            "margin": 1.0,
            "reason_codes": ["MANUAL_ANCHOR"],
            "descriptor": _descriptor_metadata(anchor_descriptor),
            "candidates": [],
        },
    }
    base_profile = IdentityProfile(
        identity_id=identity_id,
        descriptor=anchor_descriptor,
        source="manual_anchor_track",
    )
    segments_by_index: dict[int, dict[str, Any]] = {
        anchor_index: anchor_segment
    }
    accepted_associations = 0
    abstained_associations = 0
    processing_failures = 0
    total_profile_samples = anchor_descriptor.sample_count

    def process_direction(indices: Sequence[int], direction: str) -> None:
        nonlocal accepted_associations, abstained_associations
        nonlocal processing_failures, total_profile_samples
        profile = base_profile
        previous_bboxes: Sequence[Mapping[str, Any]] = anchor_bboxes
        for index in indices:
            window_start, window_end = windows[index]
            try:
                segment_path, samples, track_map = collect(index)
            except legacy.TrackingTimeoutError:
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
                segments_by_index[index] = _empty_segment(
                    window_start=window_start,
                    window_end=window_end,
                    direction=direction,
                    reason_code="WINDOW_PROCESSING_FAILED",
                    identity_id=identity_id,
                )
                continue

            candidates, id_lookup, descriptor_lookup = _build_candidate_profiles(
                segment_path,
                track_map,
                previous_bboxes=previous_bboxes,
                window_start=window_start,
                direction=direction,
                fps=fps,
            )
            decision = associate_identity(
                profile,
                candidates,
                thresholds=thresholds,
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
            if selected_track_id is not None:
                bboxes, lost_segments, _ = legacy._build_window_bboxes(
                    samples,
                    selected_track_id,
                    fps=fps,
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
                profile = update_identity_profile(profile, descriptor)
                total_profile_samples = max(
                    total_profile_samples, profile.descriptor.sample_count
                )
                previous_bboxes = bboxes
                identity_status = "ACCEPTED"
                segment_identity_id: str | None = identity_id
            else:
                abstained_associations += 1
                selected_track_id = None
                bboxes = []
                lost_segments = []
                identity_status = "ABSTAINED"
                segment_identity_id = None
            coverage = len(bboxes) / float(max(1, len(samples))) * 100.0
            reid_payload = decision.to_payload()
            reid_payload.update(
                {
                    "identity_id": identity_id,
                    "descriptor": _descriptor_metadata(descriptor),
                }
            )
            segments_by_index[index] = {
                "window_start": float(window_start),
                "window_end": float(window_end),
                "direction": direction,
                "selected_track_id": selected_track_id,
                "identity_id": segment_identity_id,
                "identity_status": identity_status,
                "reacquire_score": round(float(decision.best_score), 4),
                "coverage_pct": round(float(coverage), 2),
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
                )

    process_direction(forward_indices, "forward")
    process_direction(backward_indices, "backward")
    segments = [segments_by_index[index] for index in range(len(windows))]
    segments_with_player = sum(
        1 for segment in segments if segment.get("bboxes")
    )
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
        "coverage_pct_total": round(coverage_pct, 2),
        "largest_gap_sec": round(largest_gap, 2),
        "coverage_pct": round(coverage_pct, 2),
        "anchors_used": {
            "player_ref": player_ref_norm,
            "selections": selections,
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
            "processing_failures": processing_failures,
            "accepted_ratio": round(accepted_ratio, 6),
            "profile_samples": total_profile_samples,
            "reason_codes": [
                "EXPERIMENTAL_NOT_VALIDATED",
                "BENCHMARK_REQUIRED_BEFORE_PLAYER_SCORING",
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
        endpoint_url=endpoint_url,
        bucket=bucket,
        expires_seconds=expires_seconds,
    )
