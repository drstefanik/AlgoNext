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

from app.core.db import SessionLocal
from app.core.models import AnalysisJob

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if not player_ref or "bbox" not in player_ref:
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

    ref_bbox = player_ref.get("bbox") or {}
    ref_norm = {
        "x": float(ref_bbox.get("x", 0.0)),
        "y": float(ref_bbox.get("y", 0.0)),
        "w": float(ref_bbox.get("w", 0.0)),
        "h": float(ref_bbox.get("h", 0.0)),
    }
    ref_time = float(player_ref.get("time_sec", 0.0))
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
        "anchors_used": {"player_ref": player_ref, "selections": selections},
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
