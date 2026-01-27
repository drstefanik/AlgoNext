import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np
from botocore.client import Config
from botocore.exceptions import ClientError
from ultralytics import YOLO

try:
    import lapx as lap  # noqa: F401
except ModuleNotFoundError:
    import lap  # noqa: F401

from app.core.db import SessionLocal
from app.core.models import AnalysisJob
from app.core.normalizers import normalize_failure_reason

logger = logging.getLogger(__name__)

CANDIDATE_MIN_HITS = int(os.environ.get("CANDIDATE_MIN_HITS", "2"))
CANDIDATE_MIN_SECONDS = float(os.environ.get("CANDIDATE_MIN_SECONDS", "0") or 0)
CANDIDATE_TOP_N = int(os.environ.get("CANDIDATE_TOP_N", "5"))
PRIMARY_COVERAGE_THRESHOLD = 0.05

class TrackingTimeoutError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_player_ref(player_ref: Dict[str, Any]) -> Dict[str, float] | None:
    if not player_ref or isinstance(player_ref, str):
        return None
    if {"t", "x", "y", "w", "h"}.issubset(player_ref.keys()):
        return {
            "t": float(player_ref.get("t", 0.0)),
            "x": float(player_ref.get("x", 0.0)),
            "y": float(player_ref.get("y", 0.0)),
            "w": float(player_ref.get("w", 0.0)),
            "h": float(player_ref.get("h", 0.0)),
        }
    bbox = player_ref.get("bbox") or {}
    if not bbox:
        return None
    return {
        "t": float(player_ref.get("time_sec", 0.0)),
        "x": float(bbox.get("x", 0.0)),
        "y": float(bbox.get("y", 0.0)),
        "w": float(bbox.get("w", 0.0)),
        "h": float(bbox.get("h", 0.0)),
    }


def _update_tracking_progress(job_id: str, pct: int, message: str) -> None:
    pct = max(0, min(100, int(pct)))
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        if not job:
            return
        progress = job.progress or {}
        current_pct = progress.get("pct") or 0
        try:
            current_pct = int(current_pct)
        except (TypeError, ValueError):
            current_pct = 0
        job.progress = {
            **progress,
            "step": "TRACKING",
            "pct": max(current_pct, pct),
            "message": message,
            "updated_at": _utc_now_iso(),
        }
        db.commit()
    finally:
        db.close()


def _update_candidates_frames_processed(job_id: str, frames_processed: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        if not job:
            return
        result = dict(job.result or {})
        candidates_payload = result.get("candidates")
        if not isinstance(candidates_payload, dict):
            candidates_payload = {"candidates": list(candidates_payload or [])}
        candidates_payload["framesProcessed"] = int(frames_processed)
        candidates_payload.pop("frames_processed", None)
        result["candidates"] = candidates_payload
        result["framesProcessed"] = int(frames_processed)
        job.result = result
        job.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


def _mark_tracking_timeout(job_id: str, timeout_seconds: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        if not job:
            return
        warnings = list(job.warnings or [])
        if "TRACKING_TIMEOUT" not in warnings:
            warnings.append("TRACKING_TIMEOUT")
        job.warnings = warnings
        job.status = "FAILED"
        job.failure_reason = normalize_failure_reason("TRACKING_TIMEOUT")
        job.error = f"Tracking exceeded timeout of {timeout_seconds} seconds"
        job.progress = {
            **(job.progress or {}),
            "step": "FAILED",
            "pct": 100,
            "message": "Tracking timeout",
            "updated_at": _utc_now_iso(),
        }
        db.commit()
    finally:
        db.close()


def _get_s3_client(endpoint_url: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )


def _ensure_bucket_exists(s3_client, bucket: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise
    s3_client.create_bucket(Bucket=bucket)


def _upload_file(
    s3_internal, bucket: str, local_path: Path, key: str, content_type: str
) -> None:
    try:
        s3_internal.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("NoSuchBucket", "404", "NotFound"):
            _ensure_bucket_exists(s3_internal, bucket)
            s3_internal.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=key,
                ExtraArgs={"ContentType": content_type},
            )
        else:
            raise


def _presign_get_url(
    s3_public,
    bucket: str,
    key: str,
    expires_seconds: int,
) -> str:
    return s3_public.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


def _bbox_xyxy_to_xywh_norm(
    xyxy: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    x = x1 / float(width)
    y = y1 / float(height)
    return (
        max(0.0, min(1.0, x)),
        max(0.0, min(1.0, y)),
        max(0.0, min(1.0, w / float(width))),
        max(0.0, min(1.0, h / float(height))),
    )


def _bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax1 = a["x"]
    ay1 = a["y"]
    ax2 = a["x"] + a["w"]
    ay2 = a["y"] + a["h"]
    bx1 = b["x"]
    by1 = b["y"]
    bx2 = b["x"] + b["w"]
    by2 = b["y"] + b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def _center_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax = a["x"] + a["w"] * 0.5
    ay = a["y"] + a["h"] * 0.5
    bx = b["x"] + b["w"] * 0.5
    by = b["y"] + b["h"] * 0.5
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _area_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    area_a = max(1e-6, a["w"] * a["h"])
    area_b = max(1e-6, b["w"] * b["h"])
    ratio = area_a / area_b
    if ratio < 1.0:
        ratio = 1.0 / ratio
    return ratio


def _smooth_bbox(
    prev: Optional[Dict[str, float]],
    current: Dict[str, float],
    alpha: float = 0.2,
) -> Dict[str, float]:
    if prev is None:
        return dict(current)
    cx = current["x"] + current["w"] * 0.5
    cy = current["y"] + current["h"] * 0.5
    px = prev["x"] + prev["w"] * 0.5
    py = prev["y"] + prev["h"] * 0.5
    cw = current["w"]
    ch = current["h"]
    pw = prev["w"]
    ph = prev["h"]
    smoothed_cx = alpha * cx + (1 - alpha) * px
    smoothed_cy = alpha * cy + (1 - alpha) * py
    smoothed_w = alpha * cw + (1 - alpha) * pw
    smoothed_h = alpha * ch + (1 - alpha) * ph
    x = smoothed_cx - smoothed_w * 0.5
    y = smoothed_cy - smoothed_h * 0.5
    return {
        "x": max(0.0, min(1.0, x)),
        "y": max(0.0, min(1.0, y)),
        "w": max(0.0, min(1.0, smoothed_w)),
        "h": max(0.0, min(1.0, smoothed_h)),
    }


def track_player(
    job_id: str,
    input_video_path: str,
    player_ref: dict,
    selections: List[Dict[str, Any]],
    *,
    fps: int = 5,
    detector_model: str = "yolo11s.pt",
    tracker: str = "bytetrack.yaml",
) -> Dict[str, Any]:
    """
    Returns dict:
    {
      "method": "yolo+bytetrack",
      "fps": 5,
      "track_id": int,
      "coverage_pct": float,
      "bboxes": [{"t":float,"x":float,"y":float,"w":float,"h":float,"conf":float}, ...],
      "lost_segments": [{"start":float,"end":float}, ...],
      "anchors_used": {"player_ref": {...}, "selections": [...]},
      "notes": str
    }
    """
    player_ref_norm = _normalize_player_ref(player_ref)
    if player_ref_norm is None:
        raise RuntimeError("Missing player_ref for tracking")

    s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL", "").strip()
    s3_access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    s3_secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
    s3_bucket = os.environ.get("S3_BUCKET", "").strip()
    expires_seconds = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))
    s3_public_endpoint_url = os.environ.get("S3_PUBLIC_ENDPOINT_URL", "").strip()

    if (
        not s3_endpoint_url
        or not s3_public_endpoint_url
        or not s3_access_key
        or not s3_secret_key
        or not s3_bucket
    ):
        raise RuntimeError(
            "Missing S3 env vars: S3_ENDPOINT_URL, S3_PUBLIC_ENDPOINT_URL, "
            "S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET"
        )

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for tracking")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if orig_fps <= 0:
        orig_fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = max(1, int(round(orig_fps / float(fps))))
    total_samples = max(1, int(np.ceil(total_frames / frame_interval)))

    model = YOLO(detector_model)
    samples: List[Dict[str, Any]] = []

    last_progress_time = time.monotonic()
    processed_samples = 0
    start_pct = 10
    end_pct = 40
    tracking_timeout_seconds = int(os.environ.get("TRACKING_TIMEOUT_SECONDS", "1200"))
    tracking_started_at = time.monotonic()

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            timestamp = frame_index / float(orig_fps)
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                conf=0.25,
                iou=0.5,
                classes=[0],
                verbose=False,
            )
            detections: List[Dict[str, Any]] = []
            if results:
                boxes = results[0].boxes
                if boxes is not None and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                    ids = boxes.id.cpu().numpy() if boxes.id is not None else []
                    for idx in range(len(xyxy)):
                        if idx >= len(ids):
                            continue
                        track_id = ids[idx]
                        if track_id is None:
                            continue
                        bbox_norm = _bbox_xyxy_to_xywh_norm(
                            tuple(xyxy[idx]), width, height
                        )
                        detections.append(
                            {
                                "track_id": int(track_id),
                                "bbox": {
                                    "x": bbox_norm[0],
                                    "y": bbox_norm[1],
                                    "w": bbox_norm[2],
                                    "h": bbox_norm[3],
                                },
                                "conf": float(confs[idx])
                                if idx < len(confs)
                                else 0.0,
                            }
                        )

            samples.append({"t": float(timestamp), "detections": detections})
            processed_samples += 1
            now = time.monotonic()
            if now - tracking_started_at > tracking_timeout_seconds:
                _mark_tracking_timeout(job_id, tracking_timeout_seconds)
                raise TrackingTimeoutError("Tracking timeout exceeded")
            if now - last_progress_time >= 10:
                pct = start_pct + int(
                    (processed_samples / float(total_samples)) * (end_pct - start_pct)
                )
                _update_tracking_progress(job_id, pct, "Tracking player")
                last_progress_time = now

            frame_index += 1
    finally:
        cap.release()

    _update_tracking_progress(job_id, end_pct, "Tracking player")

    if not samples:
        raise RuntimeError("No frames sampled for tracking")

    ref_norm = {
        "x": float(player_ref_norm.get("x", 0.0)),
        "y": float(player_ref_norm.get("y", 0.0)),
        "w": float(player_ref_norm.get("w", 0.0)),
        "h": float(player_ref_norm.get("h", 0.0)),
    }
    ref_time = float(player_ref_norm.get("t", 0.0))
    ref_index = min(
        range(len(samples)),
        key=lambda i: abs(samples[i]["t"] - ref_time),
    )
    selected_track_id: Optional[int] = None
    best_iou = 0.0
    best_dist = 1e9
    for det in samples[ref_index]["detections"]:
        iou = _bbox_iou(ref_norm, det["bbox"])
        dist = _center_distance(ref_norm, det["bbox"])
        if iou > best_iou or (iou == best_iou and dist < best_dist):
            best_iou = iou
            best_dist = dist
            selected_track_id = det["track_id"]

    notes: List[str] = []
    if selected_track_id is None:
        notes.append("No track matched player_ref; falling back to selections")

    selection_map: Dict[int, Dict[str, float]] = {}
    for selection in selections:
        if "frame_time_sec" in selection:
            sel_time = float(selection.get("frame_time_sec", 0.0))
            sel_bbox = selection
        else:
            sel_time = float(selection.get("time_sec", 0.0))
            sel_bbox = selection.get("bbox") or {}
        index = min(
            range(len(samples)),
            key=lambda i: abs(samples[i]["t"] - sel_time),
        )
        selection_map[index] = {
            "x": float(sel_bbox.get("x", 0.0)),
            "y": float(sel_bbox.get("y", 0.0)),
            "w": float(sel_bbox.get("w", 0.0)),
            "h": float(sel_bbox.get("h", 0.0)),
        }

    if selected_track_id is None:
        best_candidate = None
        best_score = 0.0
        for idx, anchor_bbox in selection_map.items():
            for det in samples[idx]["detections"]:
                iou = _bbox_iou(anchor_bbox, det["bbox"])
                if iou > best_score:
                    best_score = iou
                    best_candidate = det
        if best_candidate is not None and best_score > 0.2:
            selected_track_id = best_candidate["track_id"]
            notes.append(
                f"Selected track_id {selected_track_id} from selection anchors"
            )

    bboxes: List[Dict[str, float]] = []
    lost_segments: List[Dict[str, float]] = []
    missing_count = 0
    missing_threshold = max(1, int(round(2 * fps)))
    first_missing_time = None
    last_missing_time = None
    smoothed: Optional[Dict[str, float]] = None

    for index, sample in enumerate(samples):
        timestamp = sample["t"]
        detections = sample["detections"]
        selected_detection = None
        if selected_track_id is not None:
            for det in detections:
                if det["track_id"] == selected_track_id:
                    selected_detection = det
                    break

        if selected_detection is None and index in selection_map:
            anchor_bbox = selection_map[index]
            best_anchor = None
            best_anchor_iou = 0.0
            for det in detections:
                iou = _bbox_iou(anchor_bbox, det["bbox"])
                size_ratio = _area_similarity(anchor_bbox, det["bbox"])
                if iou > best_anchor_iou and iou >= 0.3 and size_ratio <= 2.0:
                    best_anchor_iou = iou
                    best_anchor = det
            if best_anchor is not None:
                if selected_track_id != best_anchor["track_id"]:
                    notes.append(
                        f"Reacquired track_id {best_anchor['track_id']} at t={timestamp:.2f}s"
                    )
                selected_track_id = best_anchor["track_id"]
                selected_detection = best_anchor

        if selected_detection is None:
            missing_count += 1
            if missing_count == 1:
                first_missing_time = timestamp
            last_missing_time = timestamp
            if missing_count == missing_threshold:
                notes.append(
                    f"Lost track starting at t={first_missing_time:.2f}s"
                )
        else:
            if missing_count >= missing_threshold and first_missing_time is not None:
                lost_segments.append(
                    {
                        "start": float(first_missing_time),
                        "end": float(last_missing_time or timestamp),
                    }
                )
            missing_count = 0
            first_missing_time = None
            last_missing_time = None

            bbox = dict(selected_detection["bbox"])
            bbox = _smooth_bbox(smoothed, bbox)
            smoothed = bbox
            bboxes.append(
                {
                    "t": float(timestamp),
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "w": bbox["w"],
                    "h": bbox["h"],
                    "conf": float(selected_detection.get("conf", 0.0)),
                }
            )

    if missing_count >= missing_threshold and first_missing_time is not None:
        lost_segments.append(
            {
                "start": float(first_missing_time),
                "end": float(last_missing_time or samples[-1]["t"]),
            }
        )

    coverage_pct = (len(bboxes) / float(len(samples))) * 100.0
    tracking_output: Dict[str, Any] = {
        "method": "yolo+bytetrack",
        "fps": fps,
        "track_id": selected_track_id,
        "coverage_pct": round(coverage_pct, 2),
        "bboxes": bboxes,
        "lost_segments": lost_segments,
        "anchors_used": {"player_ref": player_ref_norm, "selections": selections},
        "notes": "; ".join(notes) if notes else "",
    }

    tracking_dir = Path("/tmp/fnh_jobs") / job_id / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = tracking_dir / "tracking.json"
    with tracking_path.open("w", encoding="utf-8") as handle:
        json.dump(tracking_output, handle, ensure_ascii=False, indent=2)

    s3_internal = _get_s3_client(s3_endpoint_url)
    s3_public = _get_s3_client(s3_public_endpoint_url)
    _ensure_bucket_exists(s3_internal, s3_bucket)
    tracking_key = f"jobs/{job_id}/tracking/tracking.json"
    _upload_file(s3_internal, s3_bucket, tracking_path, tracking_key, "application/json")
    tracking_url = _presign_get_url(s3_public, s3_bucket, tracking_key, expires_seconds)

    tracking_output["tracking_key"] = tracking_key
    tracking_output["tracking_url"] = tracking_url

    return tracking_output


def _select_candidate_samples(
    detections_sorted: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not detections_sorted:
        return []
    count = len(detections_sorted)
    third_points = [
        int(round((count - 1) * (1 / 3))),
        int(round((count - 1) * (2 / 3))),
        count - 1,
    ]
    areas = [det["bbox"]["w"] * det["bbox"]["h"] for det in detections_sorted]
    best_index = int(np.argmax(areas)) if areas else 0
    indices = []
    for idx in [third_points[0], third_points[1], third_points[2], best_index]:
        if idx not in indices and 0 <= idx < count:
            indices.append(idx)
    if len(indices) < 4:
        for idx in range(count):
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= 4:
                break
    return [detections_sorted[idx] for idx in indices[:4]]


def _build_candidate_tracks(
    track_map: Dict[int, List[Dict[str, Any]]],
    sample_count: int,
    *,
    min_hits: int,
    min_seconds: float,
    top_n: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    candidates: List[Dict[str, Any]] = []
    if sample_count <= 0:
        return candidates, 0, len(track_map)
    track_records: List[Dict[str, Any]] = []
    for idx, (track_id, detections) in enumerate(track_map.items()):
        detections_sorted = sorted(detections, key=lambda d: d["sample_index"])
        unique_hits = len({det["sample_index"] for det in detections_sorted})
        hit_count = unique_hits
        if hit_count == 0:
            continue
        coverage_pct = hit_count / float(sample_count)
        avg_box_area = float(
            np.mean([det["bbox"]["w"] * det["bbox"]["h"] for det in detections_sorted])
        )
        avg_conf = float(
            np.mean([det.get("conf", 0.0) for det in detections_sorted])
        )
        segments = 1
        last_index = detections_sorted[0]["sample_index"]
        for det in detections_sorted[1:]:
            if det["sample_index"] - last_index > 1:
                segments += 1
            last_index = det["sample_index"]
        id_switches = max(0, segments - 1)
        normalized_switches = id_switches / float(max(1, hit_count - 1))
        stability_score = max(0.0, 1.0 - normalized_switches)
        selected_samples = _select_candidate_samples(detections_sorted)
        sample_frames = [
            {
                "time_sec": float(sample["t"]),
                "bbox": sample["bbox"],
            }
            for sample in selected_samples
        ]
        duration = float(
            detections_sorted[-1]["t"] - detections_sorted[0]["t"]
        )
        required_primary_hits = max(2, int(np.ceil(sample_count * 0.10)))
        is_eligible = hit_count >= min_hits and (
            min_seconds <= 0 or duration >= min_seconds
        )
        if is_eligible and hit_count >= required_primary_hits:
            tier = "PRIMARY"
        elif hit_count >= 1:
            tier = "SECONDARY"
        else:
            tier = "LOW_COVERAGE"
        payload = {
            "track_id": int(track_id),
            "coverage_pct": round(coverage_pct, 4),
            "stability_score": round(float(stability_score), 3),
            "avg_box_area": round(float(avg_box_area), 6),
            "id_switches": int(id_switches),
            "tier": tier,
            "sample_frames": sample_frames,
        }
        if tier == "LOW_COVERAGE":
            payload["quality"] = "LOW_COVERAGE"
            payload["lowCoverage"] = True
        if idx < 5:
            logger.info(
                "tiering track=%s detections=%d samples=%d coverage=%.4f tier=%s",
                track_id,
                hit_count,
                sample_count,
                coverage_pct,
                tier,
            )
        track_records.append(
            {
                "tier": tier,
                "hit_count": hit_count,
                "avg_box_area": avg_box_area,
                "avg_conf": avg_conf,
                "stability_score": stability_score,
                "coverage_pct": coverage_pct,
                "payload": payload,
            }
        )

    primary = [item["payload"] for item in track_records if item["tier"] == "PRIMARY"]
    secondary = [
        item["payload"] for item in track_records if item["tier"] == "SECONDARY"
    ]
    low_coverage = [
        item for item in track_records if item["tier"] == "LOW_COVERAGE"
    ]

    primary.sort(
        key=lambda item: (
            item.get("stability_score", 0),
            item.get("coverage_pct", 0),
            item.get("avg_box_area", 0),
        ),
        reverse=True,
    )
    secondary.sort(
        key=lambda item: (
            item.get("stability_score", 0),
            item.get("coverage_pct", 0),
            item.get("avg_box_area", 0),
        ),
        reverse=True,
    )
    low_coverage.sort(
        key=lambda item: (
            item["hit_count"],
            item["avg_box_area"],
            item["avg_conf"],
        ),
        reverse=True,
    )

    candidates = primary + secondary
    if top_n > 0 and low_coverage:
        candidates.extend(
            [item["payload"] for item in low_coverage[:top_n]]
        )
    if not candidates and track_records:
        fallback = sorted(
            track_records,
            key=lambda item: (
                item["stability_score"],
                item["coverage_pct"],
                item["avg_box_area"],
            ),
            reverse=True,
        )[: max(1, top_n)]
        for item in fallback:
            payload = dict(item["payload"])
            payload["tier"] = "SECONDARY"
            payload["quality"] = "LOW_COVERAGE"
            payload["lowCoverage"] = True
            candidates.append(payload)
    total_tracks = len(track_records)
    raw_tracks = len(track_map)
    return candidates, total_tracks, raw_tracks


def track_all_players(
    job_id: str,
    input_video_path: str,
    *,
    fps: int = 5,
    detector_model: str = "yolo11s.pt",
    tracker: str = "bytetrack.yaml",
) -> Dict[str, Any]:
    s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL", "").strip()
    s3_access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    s3_secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
    s3_bucket = os.environ.get("S3_BUCKET", "").strip()
    expires_seconds = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))
    s3_public_endpoint_url = os.environ.get("S3_PUBLIC_ENDPOINT_URL", "").strip()

    if (
        not s3_endpoint_url
        or not s3_public_endpoint_url
        or not s3_access_key
        or not s3_secret_key
        or not s3_bucket
    ):
        raise RuntimeError(
            "Missing S3 env vars: S3_ENDPOINT_URL, S3_PUBLIC_ENDPOINT_URL, "
            "S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET"
        )

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for tracking")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if orig_fps <= 0:
        orig_fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = max(1, int(round(orig_fps / float(fps))))
    total_samples = max(1, int(np.ceil(total_frames / frame_interval)))

    model = YOLO(detector_model)
    samples: List[Dict[str, Any]] = []
    track_map: Dict[int, List[Dict[str, Any]]] = {}

    last_progress_time = time.monotonic()
    processed_samples = 0
    start_pct = 10
    end_pct = 40
    tracking_timeout_seconds = int(os.environ.get("TRACKING_TIMEOUT_SECONDS", "1200"))
    tracking_started_at = time.monotonic()

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            timestamp = frame_index / float(orig_fps)
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker,
                conf=0.25,
                iou=0.5,
                classes=[0],
                verbose=False,
            )
            detections: List[Dict[str, Any]] = []
            if results:
                boxes = results[0].boxes
                if boxes is not None and boxes.xyxy is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                    ids = boxes.id.cpu().numpy() if boxes.id is not None else []
                    for idx in range(len(xyxy)):
                        if idx >= len(ids):
                            continue
                        track_id = ids[idx]
                        if track_id is None:
                            continue
                        bbox_norm = _bbox_xyxy_to_xywh_norm(
                            tuple(xyxy[idx]), width, height
                        )
                        detection = {
                            "track_id": int(track_id),
                            "bbox": {
                                "x": bbox_norm[0],
                                "y": bbox_norm[1],
                                "w": bbox_norm[2],
                                "h": bbox_norm[3],
                            },
                            "conf": float(confs[idx]) if idx < len(confs) else 0.0,
                        }
                        detections.append(detection)
                        track_map.setdefault(int(track_id), []).append(
                            {
                                "t": float(timestamp),
                                "bbox": detection["bbox"],
                                "conf": detection["conf"],
                                "sample_index": processed_samples,
                            }
                        )

            samples.append({"t": float(timestamp), "detections": detections})
            processed_samples += 1
            if processed_samples % 2 == 0 or processed_samples == total_samples:
                _update_candidates_frames_processed(job_id, processed_samples)
            now = time.monotonic()
            if now - tracking_started_at > tracking_timeout_seconds:
                _mark_tracking_timeout(job_id, tracking_timeout_seconds)
                raise TrackingTimeoutError("Tracking timeout exceeded")
            if now - last_progress_time >= 10:
                pct = start_pct + int(
                    (processed_samples / float(total_samples)) * (end_pct - start_pct)
                )
                _update_tracking_progress(job_id, pct, "Tracking all players")
                last_progress_time = now

            frame_index += 1
    finally:
        cap.release()

    _update_tracking_progress(job_id, end_pct, "Tracking all players")

    if not samples:
        raise RuntimeError("No frames sampled for tracking")

    frames_processed = len(samples)
    candidates, total_tracks, raw_tracks = _build_candidate_tracks(
        track_map,
        frames_processed,
        min_hits=CANDIDATE_MIN_HITS,
        min_seconds=CANDIDATE_MIN_SECONDS,
        top_n=CANDIDATE_TOP_N,
    )

    if not candidates:
        return {
            "method": "yolo+bytetrack",
            "fps": fps,
            "generated_at": _utc_now_iso(),
            "candidates": [],
            "framesProcessed": frames_processed,
            "totalTracks": total_tracks,
            "rawTracks": raw_tracks,
            "primaryCount": 0,
            "secondaryCount": 0,
            "autodetection": {
                "totalTracks": total_tracks,
                "rawTracks": raw_tracks,
                "primaryCount": 0,
                "secondaryCount": 0,
                "thresholdPct": PRIMARY_COVERAGE_THRESHOLD,
                "minHits": CANDIDATE_MIN_HITS,
                "minSeconds": CANDIDATE_MIN_SECONDS,
                "topN": CANDIDATE_TOP_N,
            },
            "autodetection_status": "LOW_COVERAGE",
        }

    s3_internal = _get_s3_client(s3_endpoint_url)
    s3_public = _get_s3_client(s3_public_endpoint_url)
    _ensure_bucket_exists(s3_internal, s3_bucket)

    candidates_dir = Path("/tmp/fnh_jobs") / job_id / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video_path))
    try:
        for candidate in candidates:
            sample_frames = candidate.get("sample_frames") or []
            uploaded_frames: List[Dict[str, Any]] = []
            for idx, sample in enumerate(sample_frames, start=1):
                timestamp = float(sample["time_sec"])
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_name = f"track_{candidate['track_id']}_{idx:02d}.jpg"
                frame_path = candidates_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                frame_key = f"jobs/{job_id}/candidates/{frame_name}"
                _upload_file(
                    s3_internal,
                    s3_bucket,
                    frame_path,
                    frame_key,
                    "image/jpeg",
                )
                uploaded_frames.append(
                    {
                        "time_sec": timestamp,
                        "bbox": sample["bbox"],
                        "bucket": s3_bucket,
                        "key": frame_key,
                    }
                )
            candidate["sample_frames"] = uploaded_frames
    finally:
        cap.release()

    candidates = [c for c in candidates if len(c.get("sample_frames") or []) >= 1]
    primary_count = len([c for c in candidates if c.get("tier") == "PRIMARY"])
    secondary_count = len([c for c in candidates if c.get("tier") == "SECONDARY"])
    autodetection_status = "LOW_COVERAGE" if primary_count == 0 else "OK"

    return {
        "method": "yolo+bytetrack",
        "fps": fps,
        "generated_at": _utc_now_iso(),
        "expires_in": expires_seconds,
        "candidates": candidates,
        "framesProcessed": frames_processed,
        "totalTracks": total_tracks,
        "rawTracks": raw_tracks,
        "primaryCount": primary_count,
        "secondaryCount": secondary_count,
        "autodetection": {
            "totalTracks": total_tracks,
            "rawTracks": raw_tracks,
            "primaryCount": primary_count,
            "secondaryCount": secondary_count,
            "thresholdPct": PRIMARY_COVERAGE_THRESHOLD,
            "minHits": CANDIDATE_MIN_HITS,
            "minSeconds": CANDIDATE_MIN_SECONDS,
            "topN": CANDIDATE_TOP_N,
        },
        "autodetection_status": autodetection_status,
        "bucket": s3_bucket,
        "assets_prefix": f"jobs/{job_id}/candidates/",
    }


def track_all_players_from_frames(
    job_id: str,
    frames: List[Dict[str, Any]],
    *,
    detector_model: str = "yolo11s.pt",
    tracker: str = "bytetrack.yaml",
) -> Dict[str, Any]:
    s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL", "").strip()
    s3_access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    s3_secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
    s3_bucket = os.environ.get("S3_BUCKET", "").strip()
    expires_seconds = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))
    s3_public_endpoint_url = os.environ.get("S3_PUBLIC_ENDPOINT_URL", "").strip()

    if (
        not s3_endpoint_url
        or not s3_public_endpoint_url
        or not s3_access_key
        or not s3_secret_key
        or not s3_bucket
    ):
        raise RuntimeError(
            "Missing S3 env vars: S3_ENDPOINT_URL, S3_PUBLIC_ENDPOINT_URL, "
            "S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET"
        )

    if not frames:
        raise RuntimeError("No preview frames available for tracking")

    model = YOLO(detector_model)
    track_map: Dict[int, List[Dict[str, Any]]] = {}
    total_samples = len(frames)

    last_progress_time = time.monotonic()
    processed_samples = 0
    start_pct = 10
    end_pct = 40
    tracking_timeout_seconds = int(os.environ.get("TRACKING_TIMEOUT_SECONDS", "1200"))
    tracking_started_at = time.monotonic()

    for frame_index, frame_info in enumerate(frames):
        frame_path = frame_info["path"]
        timestamp = float(frame_info.get("time_sec") or 0.0)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            processed_samples += 1
            if processed_samples % 2 == 0 or processed_samples == total_samples:
                _update_candidates_frames_processed(job_id, processed_samples)
            continue
        height, width = frame.shape[:2]
        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker,
            conf=0.25,
            iou=0.5,
            classes=[0],
            verbose=False,
        )
        detections: List[Dict[str, Any]] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
                ids = boxes.id.cpu().numpy() if boxes.id is not None else []
                for idx in range(len(xyxy)):
                    if idx >= len(ids):
                        continue
                    track_id = ids[idx]
                    if track_id is None:
                        continue
                    bbox_norm = _bbox_xyxy_to_xywh_norm(
                        tuple(xyxy[idx]), width, height
                    )
                    detection = {
                        "track_id": int(track_id),
                        "bbox": {
                            "x": bbox_norm[0],
                            "y": bbox_norm[1],
                            "w": bbox_norm[2],
                            "h": bbox_norm[3],
                        },
                        "conf": float(confs[idx]) if idx < len(confs) else 0.0,
                    }
                    detections.append(detection)
                    track_map.setdefault(int(track_id), []).append(
                        {
                            "t": float(timestamp),
                            "bbox": detection["bbox"],
                            "conf": detection["conf"],
                            "sample_index": frame_index,
                        }
                    )

        processed_samples += 1
        if processed_samples % 2 == 0 or processed_samples == total_samples:
            _update_candidates_frames_processed(job_id, processed_samples)
        now = time.monotonic()
        if now - tracking_started_at > tracking_timeout_seconds:
            _mark_tracking_timeout(job_id, tracking_timeout_seconds)
            raise TrackingTimeoutError("Tracking timeout exceeded")
        if now - last_progress_time >= 10:
            pct = start_pct + int(
                (processed_samples / float(total_samples)) * (end_pct - start_pct)
            )
            _update_tracking_progress(job_id, pct, "Tracking all players")
            last_progress_time = now

    _update_tracking_progress(job_id, end_pct, "Tracking all players")

    frames_processed = total_samples
    candidates, total_tracks, raw_tracks = _build_candidate_tracks(
        track_map,
        total_samples,
        min_hits=CANDIDATE_MIN_HITS,
        min_seconds=CANDIDATE_MIN_SECONDS,
        top_n=CANDIDATE_TOP_N,
    )

    if not candidates:
        return {
            "method": "yolo+bytetrack",
            "fps": len(frames),
            "generated_at": _utc_now_iso(),
            "candidates": [],
            "framesProcessed": frames_processed,
            "totalTracks": total_tracks,
            "rawTracks": raw_tracks,
            "primaryCount": 0,
            "secondaryCount": 0,
            "autodetection": {
                "totalTracks": total_tracks,
                "rawTracks": raw_tracks,
                "primaryCount": 0,
                "secondaryCount": 0,
                "thresholdPct": PRIMARY_COVERAGE_THRESHOLD,
                "minHits": CANDIDATE_MIN_HITS,
                "minSeconds": CANDIDATE_MIN_SECONDS,
                "topN": CANDIDATE_TOP_N,
            },
            "autodetection_status": "LOW_COVERAGE",
        }

    s3_internal = _get_s3_client(s3_endpoint_url)
    s3_public = _get_s3_client(s3_public_endpoint_url)
    _ensure_bucket_exists(s3_internal, s3_bucket)

    candidates_dir = Path("/tmp/fnh_jobs") / job_id / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    frame_lookup: List[Tuple[float, Path]] = [
        (float(item.get("time_sec") or 0.0), Path(item["path"])) for item in frames
    ]

    def _find_frame_path(timestamp: float) -> Optional[Path]:
        if not frame_lookup:
            return None
        return min(frame_lookup, key=lambda item: abs(item[0] - timestamp))[1]

    for candidate in candidates:
        sample_frames = candidate.get("sample_frames") or []
        uploaded_frames: List[Dict[str, Any]] = []
        for idx, sample in enumerate(sample_frames, start=1):
            timestamp = float(sample["time_sec"])
            frame_path = _find_frame_path(timestamp)
            if frame_path is None or not frame_path.exists():
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            frame_name = f"track_{candidate['track_id']}_{idx:02d}.jpg"
            output_path = candidates_dir / frame_name
            cv2.imwrite(str(output_path), frame)
            frame_key = f"jobs/{job_id}/candidates/{frame_name}"
            _upload_file(
                s3_internal,
                s3_bucket,
                output_path,
                frame_key,
                "image/jpeg",
            )
            uploaded_frames.append(
                {
                    "time_sec": timestamp,
                    "bbox": sample["bbox"],
                    "bucket": s3_bucket,
                    "key": frame_key,
                }
            )
        candidate["sample_frames"] = uploaded_frames

    candidates = [c for c in candidates if len(c.get("sample_frames") or []) >= 1]
    primary_count = len([c for c in candidates if c.get("tier") == "PRIMARY"])
    secondary_count = len([c for c in candidates if c.get("tier") == "SECONDARY"])
    autodetection_status = "LOW_COVERAGE" if primary_count == 0 else "OK"

    return {
        "method": "yolo+bytetrack",
        "fps": len(frames),
        "generated_at": _utc_now_iso(),
        "expires_in": expires_seconds,
        "candidates": candidates,
        "framesProcessed": frames_processed,
        "totalTracks": total_tracks,
        "rawTracks": raw_tracks,
        "primaryCount": primary_count,
        "secondaryCount": secondary_count,
        "autodetection": {
            "totalTracks": total_tracks,
            "rawTracks": raw_tracks,
            "primaryCount": primary_count,
            "secondaryCount": secondary_count,
            "thresholdPct": PRIMARY_COVERAGE_THRESHOLD,
            "minHits": CANDIDATE_MIN_HITS,
            "minSeconds": CANDIDATE_MIN_SECONDS,
            "topN": CANDIDATE_TOP_N,
        },
        "autodetection_status": autodetection_status,
        "bucket": s3_bucket,
        "assets_prefix": f"jobs/{job_id}/candidates/",
    }
