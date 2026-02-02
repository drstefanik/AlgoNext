import os
import traceback
import time
import json
import shutil
import subprocess
import logging
import socket
from functools import lru_cache
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from sqlalchemy.orm import Session

from app.workers.celery_app import celery
from app.core.db import SessionLocal
from app.core.models import AnalysisJob
from app.core.normalizers import normalize_failure_reason
from app.core.scoring import compute_evaluation, keys_required_for_role
from app.workers.tracking import (
    TrackingTimeoutError,
    track_all_players,
    track_all_players_from_frames,
    track_player,
)

logger = logging.getLogger(__name__)


class AnalysisError(RuntimeError):
    pass


def _keep_workdir() -> bool:
    return os.environ.get("KEEP_WORKDIR", "0") == "1"


def _cleanup_workdir(base_dir: Optional[Path]) -> None:
    if base_dir is None or not base_dir.exists():
        return
    if _keep_workdir():
        logger.info("KEEP_WORKDIR=1 leaving workdir %s", base_dir)
        return
    shutil.rmtree(base_dir, ignore_errors=True)


def _preview_frame_count() -> int:
    try:
        preview_count = int(os.environ.get("PREVIEW_FRAME_COUNT", "16"))
    except ValueError:
        preview_count = 16
    return max(1, preview_count)


def _build_preview_timestamps(
    duration: float | None,
    preview_count: int,
    anchor_time: float | None = None,
) -> List[float]:
    if preview_count <= 0:
        return []
    if not duration or duration <= 0:
        return [i * 10 for i in range(preview_count)]

    if anchor_time is None:
        step = duration / (preview_count + 1)
        return [round(step * (i + 1), 3) for i in range(preview_count)]

    anchor = max(0.0, min(duration, float(anchor_time)))
    window = min(120.0, duration / 2)
    start = max(0.0, anchor - window)
    end = min(duration, anchor + window)

    if preview_count == 1:
        return [round(anchor, 3)]

    step = (end - start) / float(max(1, preview_count - 1))
    timestamps = [round(start + step * i, 3) for i in range(preview_count)]
    closest_index = min(
        range(len(timestamps)), key=lambda i: abs(timestamps[i] - anchor)
    )
    timestamps[closest_index] = round(anchor, 3)
    return sorted(timestamps)


# ----------------------------
# Helpers: time / db commit / progress
# ----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_commit(db: Session) -> None:
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def reload_job(db: Session, job_id: str) -> Optional[AnalysisJob]:
    return db.get(AnalysisJob, job_id)


def update_job(db: Session, job_id: str, updater: Callable[[AnalysisJob], None]) -> bool:
    job = reload_job(db, job_id)
    if not job:
        return False
    updater(job)
    safe_commit(db)
    return True


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


def set_progress(job: AnalysisJob, step: str, pct: int, message: str = "") -> None:
    pct = max(0, min(100, int(pct)))
    job.progress = {
        "step": step,
        "pct": pct,
        "message": message,
        "updated_at": utc_now_iso(),
    }




# ----------------------------
# Helpers: ffmpeg / download / clips
# ----------------------------
def _run(cmd: List[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")
    return (res.stdout or "").strip()


def _run_json(cmd: List[str]) -> Dict[str, Any]:
    return json.loads(_run(cmd))


def ffmpeg_extract_segment(
    input_path: Path, output_path: Path, start: float, duration: float
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(output_path),
        ]
    )


def ensure_ffmpeg_available() -> None:
    _run(["ffmpeg", "-version"])
    _run(["ffprobe", "-version"])


def _is_http_url(value: str) -> bool:
    return urlsplit(value).scheme in ("http", "https")


def _is_shared_object_url(value: str) -> bool:
    return urlsplit(value).path.startswith("/api/v1/download-shared-object/")


def _is_private_ip(ip_value: str) -> bool:
    try:
        ip = ip_address(ip_value)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
    )


def _ensure_public_url(url: str) -> None:
    parsed = urlsplit(url)
    host = parsed.hostname
    if not host:
        raise RuntimeError("Invalid URL host for video download.")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443)
    except socket.gaierror as exc:
        raise RuntimeError("Unable to resolve URL host for video download.") from exc
    for _, _, _, _, sockaddr in infos:
        ip_value = sockaddr[0]
        if _is_private_ip(ip_value):
            raise RuntimeError("URL host is not allowed for video download.")


def download_video(
    source: str,
    dst_path: Path,
    s3_client,
    bucket: str,
    progress_callback: Optional[Callable[[], None]] = None,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    last_progress_tick = time.monotonic()
    progress_tick_seconds = 5

    if not source:
        raise RuntimeError("Missing video source.")

    if _is_http_url(source):
        _ensure_public_url(source)
        if _is_shared_object_url(source):
            raise RuntimeError("Shared object URLs are not supported for video download.")

        log_every_bytes = 100 * 1024 * 1024
        bytes_downloaded = 0
        next_log_bytes = log_every_bytes

        with requests.get(
            source,
            stream=True,
            timeout=(10, 1800),
            headers={"User-Agent": "AlgoNextWorker/1.0"},
        ) as r:
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

                        now = time.monotonic()
                        if (
                            progress_callback is not None
                            and now - last_progress_tick >= progress_tick_seconds
                        ):
                            last_progress_tick = now
                            progress_callback()

                        if bytes_downloaded >= next_log_bytes:
                            mb_downloaded = bytes_downloaded / (1024 * 1024)
                            logger.info(
                                "Downloaded %.1f MB from %s", mb_downloaded, source
                            )
                            next_log_bytes += log_every_bytes
        return

    object_key = source.lstrip("/")
    if not object_key:
        raise RuntimeError("Missing MinIO object key for video download.")
    if s3_client is None:
        raise RuntimeError("Missing S3 client for video download.")

    def _tick(_: int) -> None:
        nonlocal last_progress_tick
        if progress_callback is None:
            return
        now = time.monotonic()
        if now - last_progress_tick >= progress_tick_seconds:
            last_progress_tick = now
            progress_callback()

    callback = _tick if progress_callback is not None else None
    s3_client.download_file(
        Bucket=bucket,
        Key=object_key,
        Filename=str(dst_path),
        Callback=callback,
    )


def resolve_job_video_source(job: AnalysisJob, fallback_bucket: str) -> Tuple[str, str]:
    if job.video_key:
        bucket = job.video_bucket or fallback_bucket
        if not bucket:
            raise RuntimeError("Missing video bucket for download.")
        return job.video_key, bucket
    if job.video_url:
        return job.video_url, fallback_bucket
    raise RuntimeError("Missing video source.")


def probe_video_meta(path: Path) -> Dict:
    try:
        out = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
        return json.loads(out)
    except Exception:
        return {}


def probe_image_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        out = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(path),
            ]
        )
        data = json.loads(out)
        streams = data.get("streams") or []
        for stream in streams:
            width = stream.get("width")
            height = stream.get("height")
            if width and height:
                return int(width), int(height)
    except Exception:
        return None, None
    return None, None


def get_duration_seconds(meta: Dict) -> Optional[float]:
    try:
        duration = meta.get("format", {}).get("duration")
        if duration is None:
            return None
        return float(duration)
    except (TypeError, ValueError):
        return None


def probe_frame_count(path: Path) -> int:
    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-print_format",
            "json",
            str(path),
        ]
    )
    streams = data.get("streams") or []
    if not streams:
        raise AnalysisError("Unable to read video stream for frame count")
    frames = streams[0].get("nb_read_frames") or streams[0].get("nb_frames")
    if frames is None:
        raise AnalysisError("Unable to count frames in video stream")
    return int(frames)


def probe_scene_change_count(path: Path, threshold: float = 0.3) -> int:
    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"movie={path.as_posix()},select='gt(scene,{threshold})'",
            "-show_entries",
            "frame=pkt_pts_time",
            "-print_format",
            "json",
        ]
    )
    frames = data.get("frames")
    if frames is None:
        raise AnalysisError("Unable to detect scene changes in video")
    return len(frames)


def extract_video_features(path: Path, meta: Dict) -> Dict[str, Any]:
    duration = get_duration_seconds(meta)
    if not duration or duration <= 0:
        raise AnalysisError("Insufficient video signal to compute score")

    frame_count = probe_frame_count(path)
    if frame_count <= 0:
        raise AnalysisError("Insufficient video signal to compute score")

    scene_change_count = probe_scene_change_count(path)
    scene_change_rate = scene_change_count / max(duration, 1.0)
    fps = frame_count / max(duration, 1.0)

    return {
        "duration_seconds": duration,
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "scene_change_count": scene_change_count,
        "scene_change_rate": round(scene_change_rate, 4),
    }


def _score_value(base: float, activity_score: float, scene_bonus: float) -> int:
    raw = base + (activity_score * 45.0) + scene_bonus
    return int(round(max(0.0, min(100.0, raw))))


def compute_skill_scores(
    features: Dict[str, Any],
) -> Tuple[Dict[str, Optional[int]], List[str]]:
    scene_change_count = features.get("scene_change_count")
    scene_change_rate = features.get("scene_change_rate")
    if scene_change_count is None or scene_change_rate is None:
        raise AnalysisError("Insufficient video signal to compute score")

    if scene_change_count < 1:
        raise AnalysisError("Insufficient video signal to compute score")

    activity_score = min(1.0, float(scene_change_rate) / 0.25)
    scene_bonus = min(float(scene_change_count), 10.0) * 2.0

    skills: Dict[str, Optional[int]] = {}
    missing: List[str] = []

    def add_skill(name: str, condition: bool, base: float) -> None:
        if not condition:
            skills[name] = None
            missing.append(name)
            return
        skills[name] = _score_value(base, activity_score, scene_bonus)

    add_skill("Finishing", scene_change_count >= 2, 34.0)
    add_skill("Shot Power", scene_change_count >= 2, 32.0)
    add_skill("Heading", scene_change_count >= 3, 30.0)
    add_skill("Positioning", scene_change_rate >= 0.03, 38.0)
    add_skill("Off-the-ball Movement", scene_change_rate >= 0.04, 36.0)
    add_skill("Composure", scene_change_rate >= 0.015, 40.0)

    if all(value is None for value in skills.values()):
        raise AnalysisError("Insufficient video signal to compute score")

    return skills, missing


def extract_clips(input_path: Path, out_dir: Path) -> Tuple[List[Dict], Optional[str]]:
    """
    Clip demo: 2 segmenti (5s ciascuno).
    Se input è troppo corto, ffmpeg può fallire: in quel caso generiamo 1 clip breve.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = [
        {"name": "clip_001.mp4", "start": 2, "duration": 5},
        {"name": "clip_002.mp4", "start": 12, "duration": 5},
    ]

    created: List[Dict] = []
    error: Optional[str] = None
    for c in clips:
        out_path = out_dir / c["name"]
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(c["start"]),
            "-i",
            str(input_path),
            "-t",
            str(c["duration"]),
            "-c",
            "copy",
            str(out_path),
        ]
        try:
            _run(cmd)
            created.append(
                {
                    "file": out_path,
                    "start": c["start"],
                    "end": c["start"] + c["duration"],
                }
            )
        except Exception as exc:
            error = str(exc)
            if not created:
                fallback = out_dir / "clip_001.mp4"
                try:
                    _run(
                        [
                            "ffmpeg",
                            "-y",
                            "-ss",
                            "0",
                            "-i",
                            str(input_path),
                            "-t",
                            "3",
                            "-c",
                            "copy",
                            str(fallback),
                        ]
                    )
                    created.append({"file": fallback, "start": 0, "end": 3})
                except Exception as fallback_exc:
                    error = f"{error}\n{fallback_exc}"
            break

    return created, error


def build_motion_segments(
    bboxes: List[Dict[str, Any]], speed_threshold: float = 0.02
) -> List[Dict[str, float]]:
    if len(bboxes) < 2:
        return []
    segments: List[Dict[str, float]] = []
    active_start = None
    last_time = None
    last_center = None
    for bbox in bboxes:
        t = float(bbox.get("t", 0.0))
        center = (
            float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0)) * 0.5,
            float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0)) * 0.5,
        )
        if last_time is None or last_center is None:
            last_time = t
            last_center = center
            continue
        dt = t - last_time
        if dt <= 0:
            last_time = t
            last_center = center
            continue
        dist = ((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2) ** 0.5
        speed = dist / dt
        if speed >= speed_threshold:
            if active_start is None:
                active_start = last_time
        else:
            if active_start is not None:
                segments.append({"start": float(active_start), "end": float(t)})
                active_start = None
        last_time = t
        last_center = center
    if active_start is not None and last_time is not None:
        segments.append({"start": float(active_start), "end": float(last_time)})
    return segments


# ----------------------------
# Helpers: S3/MinIO (upload + signed urls)
# ----------------------------
REGION = os.getenv("AWS_REGION") or os.getenv("S3_REGION", "us-east-1")
S3_ENDPOINT_URL = (
    os.getenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000").strip()
)
S3_PUBLIC_ENDPOINT_URL = (
    os.getenv("S3_PUBLIC_ENDPOINT_URL", "https://s3.nextgroupintl.com").strip()
)


def make_s3(endpoint_url: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=REGION,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def get_s3_client(endpoint_url: str):
    return make_s3(endpoint_url)


def get_public_s3_client():
    endpoint_url = (S3_PUBLIC_ENDPOINT_URL or "").strip()
    if not endpoint_url:
        raise RuntimeError("Missing S3_PUBLIC_ENDPOINT_URL")
    return make_s3(endpoint_url)


@lru_cache(maxsize=1)
def get_presign_s3_client():
    return get_public_s3_client()


def ensure_public_s3_client(s3_client):
    public_endpoint = S3_PUBLIC_ENDPOINT_URL
    endpoint_url = getattr(getattr(s3_client, "meta", None), "endpoint_url", "") or ""
    if not public_endpoint or endpoint_url.rstrip("/") != public_endpoint.rstrip("/"):
        return get_public_s3_client()
    return s3_client


def ensure_bucket_exists(s3_client, bucket: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise
    s3_client.create_bucket(Bucket=bucket)


def upload_file(
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
            ensure_bucket_exists(s3_internal, bucket)
            s3_internal.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=key,
                ExtraArgs={"ContentType": content_type},
            )
        else:
            raise


def presign_get_object(
    bucket: str,
    key: str,
    expires_seconds: int,
) -> str:
    s3_public = get_presign_s3_client()
    return s3_public.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


# ----------------------------
# Celery task
# ----------------------------
@celery.task(name="app.workers.pipeline.extract_preview_frames", bind=True)
def extract_preview_frames(self, job_id: str) -> None:
    db: Session = SessionLocal()
    base_dir: Optional[Path] = None

    s3_endpoint_url = S3_ENDPOINT_URL
    s3_access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    s3_secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
    s3_bucket = os.environ.get("S3_BUCKET", "").strip()

    max_retries = int(os.environ.get("PREVIEW_TASK_RETRIES", "2"))
    try:
        job = reload_job(db, job_id)
        if not job:
            return {}

        if job.preview_frames:
            return

        if not s3_endpoint_url or not s3_bucket or not s3_access_key or not s3_secret_key:
            raise RuntimeError(
                "Missing S3 env vars: S3_ENDPOINT_URL, S3_ACCESS_KEY, "
                "S3_SECRET_KEY, S3_BUCKET"
            )

        update_job(
            db,
            job_id,
            lambda job: set_progress(
                job, "EXTRACTING_PREVIEWS", 15, "Extracting preview frames"
            ),
        )

        ensure_ffmpeg_available()

        base_dir = Path("/tmp/fnh_jobs_previews") / job_id
        if base_dir.exists():
            _cleanup_workdir(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        input_path = base_dir / "input.mp4"
        frames_dir = base_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        s3_internal = get_s3_client(s3_endpoint_url)
        ensure_bucket_exists(s3_internal, s3_bucket)

        video_source, source_bucket = resolve_job_video_source(job, s3_bucket)
        try:
            download_video(video_source, input_path, s3_internal, source_bucket)
        except Exception as exc:
            logger.exception(
                "Preview download failed for job %s (source=%s, bucket=%s)",
                job_id,
                video_source,
                source_bucket,
            )
            raise RuntimeError(f"Download failed: {exc}") from exc

        video_meta = probe_video_meta(input_path) or {}
        duration = get_duration_seconds(video_meta)

        preview_count = _preview_frame_count()
        player_ref = _normalize_player_ref(job.player_ref or {})
        anchor_time = player_ref.get("t") if player_ref else None
        timestamps = _build_preview_timestamps(duration, preview_count, anchor_time)

        preview_frames: List[Dict[str, Any]] = []
        for index, timestamp in enumerate(timestamps, start=1):
            frame_name = f"frame_{index:04d}.jpg"
            frame_path = frames_dir / frame_name

            try:
                _run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(timestamp),
                        "-i",
                        str(input_path),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(frame_path),
                    ]
                )
            except Exception:
                break

            width, height = probe_image_dimensions(frame_path)
            frame_key = f"jobs/{job_id}/frames/{frame_name}"
            upload_file(s3_internal, s3_bucket, frame_path, frame_key, "image/jpeg")
            preview_frames.append(
                {
                    "time_sec": timestamp,
                    "bucket": s3_bucket,
                    "key": frame_key,
                    "width": width,
                    "height": height,
                    "tracks": [],
                }
            )

        if preview_frames:
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "preview_frames", preview_frames),
                    set_progress(job, "PREVIEWS_READY", 20, "Preview frames ready"),
                ),
            )
            logger.info("Preview frames generated for job %s", job_id)
            try:
                extract_candidates.delay(job_id)
            except Exception:
                logger.exception(
                    "Failed to enqueue candidates extraction after previews job_id=%s",
                    job_id,
                )
    except Exception as exc:
        logger.exception("Failed to generate preview frames for job %s", job_id)
        if self.request.retries < max_retries:
            raise self.retry(exc=exc, countdown=5 * (2 ** self.request.retries))
        try:
            error_message = f"PREVIEW_FRAMES_FAILED: {exc}"
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "status", "FAILED"),
                    setattr(job, "error", error_message),
                    setattr(
                        job,
                        "failure_reason",
                        normalize_failure_reason("preview_generation_failed"),
                    ),
                    set_progress(job, "PREVIEWS_FAILED", 20, "Preview generation failed"),
                ),
            )
        except Exception:
            db.rollback()
    finally:
        _cleanup_workdir(base_dir)
        db.close()


@celery.task(name="app.workers.pipeline.kickoff_job", bind=True)
def kickoff_job(self, job_id: str) -> None:
    try:
        extract_preview_frames.delay(job_id)
    except Exception:
        logger.exception("kickoff_job failed to enqueue tasks job_id=%s", job_id)


def _truncate_stack(stack: str, limit: int = 2000) -> str:
    if not stack:
        return ""
    if len(stack) <= limit:
        return stack
    return stack[: limit - 3] + "..."


def _build_candidates_error_detail(exc: Exception) -> Dict[str, Any]:
    message = str(exc) if str(exc) else exc.__class__.__name__
    stack = _truncate_stack(traceback.format_exc())
    code = "CANDIDATES_ERROR"
    lower_message = message.lower()
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        code = "YOLO_IMPORT_ERROR"
    elif "ultralytics" in lower_message or "yolo" in lower_message:
        code = "YOLO_IMPORT_ERROR"
    elif "model" in lower_message and "not found" in lower_message:
        code = "MODEL_NOT_FOUND"
    elif "cuda" in lower_message:
        code = "CUDA"
    return {"message": message, "stack": stack, "code": code}


def _normalize_bbox_xywh(bbox: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(bbox, dict):
        return {}
    if {"x", "y", "w", "h"}.issubset(bbox):
        try:
            return {
                "x": float(bbox["x"]),
                "y": float(bbox["y"]),
                "w": float(bbox["w"]),
                "h": float(bbox["h"]),
            }
        except (TypeError, ValueError):
            return {}
    if {"x1", "y1", "x2", "y2"}.issubset(bbox):
        try:
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
        except (TypeError, ValueError):
            return {}
        return {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
        }
    return {}


@celery.task(name="app.workers.pipeline.extract_candidates", bind=True)
def extract_candidates(self, job_id: str) -> Dict[str, Any]:
    db: Session = SessionLocal()
    base_dir: Optional[Path] = None

    s3_endpoint_url = S3_ENDPOINT_URL
    s3_public_endpoint_url = S3_PUBLIC_ENDPOINT_URL
    s3_access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    s3_secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
    s3_bucket = os.environ.get("S3_BUCKET", "").strip()

    max_retries = int(os.environ.get("CANDIDATES_TASK_RETRIES", "2"))
    try:
        job = reload_job(db, job_id)
        if not job:
            return {}

        existing_candidates = (job.result or {}).get("candidates")
        if existing_candidates:
            return {}

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

        base_dir = Path("/tmp/fnh_jobs_candidates") / job_id
        if base_dir.exists():
            _cleanup_workdir(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        s3_internal = get_s3_client(s3_endpoint_url)
        ensure_bucket_exists(s3_internal, s3_bucket)

        preview_frames = list(job.preview_frames or [])
        if not preview_frames:
            if self.request.retries < max_retries:
                raise self.retry(countdown=5)
            logger.warning(
                "extract_candidates exiting: preview_frames_missing job_id=%s",
                job_id,
            )
            return {}
        preview_time_index: List[Tuple[float, Dict[str, Any]]] = []
        for frame in preview_frames:
            if not isinstance(frame, dict):
                continue
            frame_key = frame.get("key") or frame.get("s3_key")
            if isinstance(frame_key, str) and frame_key:
                frame["key"] = frame_key
            if "tracks" not in frame or not isinstance(frame.get("tracks"), list):
                frame["tracks"] = []
            time_sec = frame.get("time_sec")
            if time_sec is not None:
                try:
                    preview_time_index.append((float(time_sec), frame))
                except (TypeError, ValueError):
                    continue

        update_job(
            db,
            job_id,
            lambda job: set_progress(
                job, "TRACKING_CANDIDATES", 18, "Tracking all players"
            ),
        )

        preview_inputs: List[Dict[str, Any]] = []
        if preview_frames:
            frames_dir = base_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            valid_keys = sum(1 for frame in preview_frames if (frame or {}).get("key"))
            logger.info(
                "extract_candidates: preview_frames=%d valid_keys=%d",
                len(preview_frames),
                valid_keys,
            )
            for index, frame in enumerate(preview_frames, start=1):
                frame_key = (frame or {}).get("key")
                if not frame_key:
                    continue
                frame_path = frames_dir / f"preview_{index:04d}.jpg"
                s3_internal.download_file(s3_bucket, frame_key, str(frame_path))
                preview_inputs.append(
                    {
                        "path": frame_path,
                        "time_sec": float((frame or {}).get("time_sec") or 0.0),
                        "key": frame_key,
                    }
                )
            candidates_output = track_all_players_from_frames(job_id, preview_inputs)
            logger.info(
                "extract_candidates: preview_inputs=%d", len(preview_inputs)
            )
            logger.info(
                "extract_candidates: candidates_output type=%s",
                type(candidates_output),
            )
            try:
                logger.info(
                    "extract_candidates: candidates_output keys=%s",
                    list(candidates_output.keys()),
                )
            except Exception:
                logger.info(
                    "extract_candidates: candidates_output is not dict-like"
                )
            logger.info(
                "extract_candidates: candidates_output preview=%s",
                str(candidates_output)[:1200],
            )
        else:
            input_path = base_dir / "input.mp4"
            video_source, source_bucket = resolve_job_video_source(job, s3_bucket)
            download_video(video_source, input_path, s3_internal, source_bucket)
            candidates_output = track_all_players(job_id, str(input_path))
        if isinstance(candidates_output, dict):
            candidates_list = list(candidates_output.get("candidates") or [])
            frames_processed = (
                candidates_output.get("framesProcessed")
                or candidates_output.get("frames_processed")
            )
            if frames_processed is None:
                frames_processed = len(preview_inputs)
            total_tracks = candidates_output.get("totalTracks")
            if total_tracks is None:
                total_tracks = (
                    (candidates_output.get("autodetection") or {}).get("totalTracks")
                    or (len(candidates_list) if candidates_list else 0)
                )
            raw_tracks = candidates_output.get("rawTracks")
            if raw_tracks is None:
                raw_tracks = (
                    (candidates_output.get("autodetection") or {}).get("rawTracks")
                    or total_tracks
                )
            primary_count = candidates_output.get("primaryCount")
            if primary_count is None:
                primary_count = (
                    (candidates_output.get("autodetection") or {}).get("primaryCount")
                    or 0
                )
            secondary_count = candidates_output.get("secondaryCount")
            if secondary_count is None:
                secondary_count = (
                    (candidates_output.get("autodetection") or {}).get("secondaryCount")
                    or 0
                )
        else:
            candidates_list = list(candidates_output or [])
            frames_processed = len(preview_inputs)
            total_tracks = len(candidates_list)
            raw_tracks = total_tracks
            primary_count = 0
            secondary_count = 0

        def _closest_preview_frame_key(
            time_sec: Optional[float], frames: List[Dict[str, Any]]
        ) -> Optional[str]:
            if time_sec is None or not frames:
                return None
            closest = min(
                frames,
                key=lambda item: abs(float(item.get("time_sec") or 0.0) - float(time_sec)),
            )
            return closest.get("key") or closest.get("s3_key")

        def _preview_time_epsilon() -> float:
            if len(preview_time_index) < 2:
                return 0.5
            sorted_times = sorted(time for time, _ in preview_time_index)
            diffs = [
                next_time - prev_time
                for prev_time, next_time in zip(sorted_times, sorted_times[1:])
                if next_time > prev_time
            ]
            if not diffs:
                return 0.5
            return max(0.25, min(diffs) / 2.0)

        def _preview_frame_for_time(time_sec: Optional[float]) -> Optional[Dict[str, Any]]:
            if time_sec is None or not preview_time_index:
                return None
            try:
                target = float(time_sec)
            except (TypeError, ValueError):
                return None
            closest_time, closest_frame = min(
                preview_time_index, key=lambda item: abs(item[0] - target)
            )
            if abs(closest_time - target) <= _preview_time_epsilon():
                return closest_frame
            return None

        if preview_frames and candidates_list:
            track_tiers = {
                candidate.get("track_id"): candidate.get("tier")
                for candidate in candidates_list
                if isinstance(candidate, dict)
            }
            for candidate in candidates_list:
                if not isinstance(candidate, dict):
                    continue
                sample_frames = candidate.get("sample_frames") or []
                if not sample_frames:
                    continue
                best_sample = max(
                    sample_frames,
                    key=lambda item: (item.get("bbox") or {}).get("w", 0.0)
                    * (item.get("bbox") or {}).get("h", 0.0),
                )
                best_time = best_sample.get("time_sec")
                best_key = _closest_preview_frame_key(best_time, preview_frames)
                if best_key:
                    candidate["best_preview_frame_key"] = best_key
                if best_time is not None:
                    candidate["best_time_sec"] = float(best_time)

            frame_tracks = []
            if isinstance(candidates_output, dict):
                frame_tracks = candidates_output.get("frame_tracks") or []
            tracks_by_key: Dict[str, Dict[str, Any]] = {}
            for frame_track in frame_tracks:
                if not isinstance(frame_track, dict):
                    continue
                key = frame_track.get("frame_key") or frame_track.get("key")
                if isinstance(key, str) and key:
                    tracks_by_key[key] = frame_track
            for frame in preview_frames:
                if not isinstance(frame, dict):
                    continue
                frame_key = frame.get("key") or frame.get("s3_key")
                frame_track = tracks_by_key.get(frame_key)
                if frame_track is None and frame_key:
                    frame_track = tracks_by_key.get(frame_key.split("/")[-1])
                if frame_track is None:
                    continue
                tracks_payload = []
                for track in frame_track.get("tracks") or []:
                    if not isinstance(track, dict):
                        continue
                    track_id = track.get("track_id")
                    tracks_payload.append(
                        {
                            "track_id": track_id,
                            "bbox": _normalize_bbox_xywh(track.get("bbox") or {}),
                            "tier": track_tiers.get(track_id),
                            "score_hint": track.get("score_hint"),
                        }
                    )
                frame_tracks_list = frame.get("tracks")
                if not isinstance(frame_tracks_list, list):
                    frame_tracks_list = []
                    frame["tracks"] = frame_tracks_list
                if tracks_payload:
                    frame_tracks_list.extend(tracks_payload)

            def _track_id_variants(value: Any) -> set[Any]:
                if value is None or isinstance(value, bool):
                    return set()
                if isinstance(value, int):
                    return {value, str(value)}
                if isinstance(value, str):
                    trimmed = value.strip()
                    variants = {trimmed}
                    if trimmed.isdigit():
                        variants.add(int(trimmed))
                    return variants
                try:
                    return {str(value)}
                except Exception:
                    return set()

            def _frame_has_track(tracks: list, track_id: Any) -> bool:
                if not tracks:
                    return False
                track_variants = _track_id_variants(track_id)
                for item in tracks:
                    if not isinstance(item, dict):
                        continue
                    if _track_id_variants(item.get("track_id")) & track_variants:
                        return True
                return False

            for candidate in candidates_list:
                if not isinstance(candidate, dict):
                    continue
                track_id = candidate.get("track_id")
                tier = candidate.get("tier")
                score_hint = candidate.get("score_hint")
                for sample in candidate.get("sample_frames") or []:
                    if not isinstance(sample, dict):
                        continue
                    frame = _preview_frame_for_time(sample.get("time_sec"))
                    if frame is None:
                        continue
                    frame_tracks_list = frame.get("tracks")
                    if not isinstance(frame_tracks_list, list):
                        frame_tracks_list = []
                        frame["tracks"] = frame_tracks_list
                    if _frame_has_track(frame_tracks_list, track_id):
                        continue
                    frame_tracks_list.append(
                        {
                            "track_id": track_id,
                            "tier": tier,
                            "score_hint": score_hint or sample.get("score_hint"),
                            "bbox": _normalize_bbox_xywh(sample.get("bbox") or {}),
                        }
                    )
        if preview_frames:
            update_job(
                db,
                job_id,
                lambda job: setattr(job, "preview_frames", preview_frames),
            )
        error_detail = None
        if len(preview_inputs) == 0:
            error_detail = "NO_VALID_PREVIEW_KEYS"
        elif total_tracks == 0:
            error_detail = "NO_TRACKS_RETURNED"
        autodetection_status = (
            "READY" if primary_count > 0 and not error_detail else "LOW_COVERAGE"
        )
        candidates_payload = {
            "candidates": candidates_list,
            "framesProcessed": frames_processed,
            "autodetection": {
                "totalTracks": total_tracks,
                "rawTracks": raw_tracks,
                "primaryCount": primary_count,
                "secondaryCount": secondary_count,
                "thresholdPct": 0.05,
                "minHits": int(os.environ.get("CANDIDATE_MIN_HITS", "2")),
                "minSeconds": float(os.environ.get("CANDIDATE_MIN_SECONDS", "0") or 0),
                "topN": int(os.environ.get("CANDIDATE_TOP_N", "5")),
            },
            "autodetection_status": autodetection_status,
            "error_detail": error_detail,
        }
        update_job(
            db,
            job_id,
            lambda job: (
                setattr(job, "status", "WAITING_FOR_SELECTION"),
                setattr(
                    job,
                    "result",
                    {
                        **(job.result or {}),
                        "candidates": candidates_payload,
                        "framesProcessed": frames_processed,
                        "totalTracks": total_tracks,
                        "rawTracks": raw_tracks,
                        "primaryCount": primary_count,
                        "secondaryCount": secondary_count,
                    },
                ),
                set_progress(
                    job, "CANDIDATES_READY", 22, "Candidate tracks ready"
                ),
            ),
        )
        return {
            "framesProcessed": frames_processed,
            "totalTracks": total_tracks,
            "primaryCount": primary_count,
        }
    except Exception as exc:
        logger.exception("extract_candidates failed job_id=%s", job_id)
        if max_retries > 0:
            try:
                raise self.retry(exc=exc, countdown=15, max_retries=max_retries)
            except Exception:
                pass
        try:
            job = reload_job(db, job_id)
            preview_frames = list(job.preview_frames or []) if job else []
        except Exception:
            preview_frames = []
        error_detail = _build_candidates_error_detail(exc)
        def _update_failed(job: AnalysisJob) -> None:
            warnings = list({*(job.warnings or []), "CANDIDATES_FAILED"})
            setattr(job, "warnings", warnings)
            candidates_payload = {
                "candidates": [],
                "framesProcessed": 0,
                "autodetection": {
                    "totalTracks": 0,
                    "rawTracks": 0,
                    "primaryCount": 0,
                    "secondaryCount": 0,
                    "thresholdPct": 0.05,
                    "minHits": int(os.environ.get("CANDIDATE_MIN_HITS", "2")),
                    "minSeconds": float(os.environ.get("CANDIDATE_MIN_SECONDS", "0") or 0),
                    "topN": int(os.environ.get("CANDIDATE_TOP_N", "5")),
                },
                "autodetection_status": "FAILED",
                "error_detail": error_detail,
            }
            setattr(
                job,
                "result",
                {
                    **(job.result or {}),
                    "candidates": candidates_payload,
                    "framesProcessed": 0,
                    "totalTracks": 0,
                    "rawTracks": 0,
                    "primaryCount": 0,
                    "secondaryCount": 0,
                },
            )
            if preview_frames:
                set_progress(
                    job,
                    "WAITING_FOR_SELECTION",
                    20,
                    "Waiting for player selection",
                )
        update_job(db, job_id, lambda job: _update_failed(job))
        return {
            "framesProcessed": 0,
            "totalTracks": 0,
            "primaryCount": 0,
        }
    finally:
        _cleanup_workdir(base_dir)
        db.close()


@celery.task(name="app.workers.pipeline.run_analysis", bind=True)
def run_analysis(self, job_id: str):
    db: Session = SessionLocal()
    base_dir: Optional[Path] = None
    video_features: Optional[Dict[str, Any]] = None
    skills_computed: Dict[str, Optional[int]] = {}
    skills_missing: List[str] = []

    # Env vars (required)
    s3_endpoint_url = S3_ENDPOINT_URL
    s3_access_key = os.environ.get("S3_ACCESS_KEY", "").strip()
    s3_secret_key = os.environ.get("S3_SECRET_KEY", "").strip()
    s3_bucket = os.environ.get("S3_BUCKET", "").strip()

    max_retries = int(os.environ.get("ANALYSIS_TASK_RETRIES", "2"))
    try:
        job = reload_job(db, job_id)
        if not job:
            logger.warning("run_analysis early-exit: job_not_found job_id=%s", job_id)
            return

        logger.info(
            "run_analysis loaded job_id=%s status=%s role=%s category=%s",
            job_id,
            getattr(job, "status", None),
            getattr(job, "role", None),
            getattr(job, "category", None),
        )

        role = job.role
        category = job.category

        target = job.target or {}
        selections = target.get("selections") or []
        logger.info(
            "run_analysis target job_id=%s selections_len=%s target_keys=%s",
            job_id,
            len(selections),
            sorted(list(target.keys()))[:50],
        )

        if len(selections) < 1:
            logger.warning(
                "run_analysis early-exit: waiting_for_selection job_id=%s", job_id
            )
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "status", "WAITING_FOR_SELECTION"),
                    setattr(job, "error", None),
                    set_progress(
                        job,
                        "WAITING_FOR_SELECTION",
                        5,
                        "Waiting for player selection",
                    ),
                ),
            )
            try:
                extract_preview_frames.delay(job_id)
            except Exception:
                logger.exception(
                    "run_analysis failed to enqueue preview job_id=%s",
                    job_id,
                )
            return

        # RUNNING
        update_job(
            db,
            job_id,
            lambda job: (
                setattr(job, "status", "RUNNING"),
                setattr(job, "error", None),
                setattr(job, "failure_reason", normalize_failure_reason(None)),
                setattr(job, "warnings", []),
                set_progress(job, "STARTING", 1, "Job started"),
            ),
        )
        logger.info("run_analysis status set RUNNING job_id=%s", job_id)

        # Preconditions
        if not s3_endpoint_url or not s3_bucket or not s3_access_key or not s3_secret_key:
            raise RuntimeError(
                "Missing S3 env vars: S3_ENDPOINT_URL, S3_ACCESS_KEY, "
                "S3_SECRET_KEY, S3_BUCKET"
            )

        logger.info(
            "run_analysis preconditions job_id=%s s3_endpoint=%s bucket=%s",
            job_id,
            s3_endpoint_url,
            s3_bucket,
        )
        ensure_ffmpeg_available()
        logger.info("run_analysis ffmpeg ok job_id=%s", job_id)

        # Prepare workspace
        base_dir = Path("/tmp/fnh_jobs") / job_id
        if base_dir.exists():
            _cleanup_workdir(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        input_path = base_dir / "input.mp4"
        clips_dir = base_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "run_analysis workspace ready job_id=%s base_dir=%s input_path=%s",
            job_id,
            str(base_dir),
            str(input_path),
        )

        # S3 clients
        logger.info("run_analysis s3 client init job_id=%s", job_id)
        s3_internal = get_s3_client(s3_endpoint_url)

        # Ensure bucket exists
        ensure_bucket_exists(s3_internal, s3_bucket)
        logger.info("run_analysis bucket ok job_id=%s bucket=%s", job_id, s3_bucket)

        # Download
        update_job(
            db,
            job_id,
            lambda job: set_progress(job, "DOWNLOADING", 10, "Downloading video"),
        )
        download_pct = 10

        def download_progress_tick() -> None:
            nonlocal download_pct
            if download_pct >= 18:
                return
            download_pct = min(18, download_pct + 5)
            update_job(
                db,
                job_id,
                lambda job: set_progress(
                    job, "DOWNLOADING", download_pct, "Downloading video"
                ),
            )

        video_source, source_bucket = resolve_job_video_source(job, s3_bucket)
        download_video(
            video_source,
            input_path,
            s3_internal,
            source_bucket,
            progress_callback=download_progress_tick,
        )

        # Probe meta
        update_job(
            db,
            job_id,
            lambda job: set_progress(job, "PROBING", 20, "Probing video metadata"),
        )
        video_meta = probe_video_meta(input_path) or {}
        if not video_meta:
            raise AnalysisError("Insufficient video signal to compute score")
        update_job(db, job_id, lambda job: setattr(job, "video_meta", video_meta))

        # Upload input (so UI can always access it, even if we pause)
        update_job(
            db,
            job_id,
            lambda job: set_progress(
                job, "UPLOADING_INPUT", 30, "Uploading input video to storage"
            ),
        )
        input_key = f"jobs/{job_id}/input.mp4"
        upload_file(s3_internal, s3_bucket, input_path, input_key, "video/mp4")
        # Persist input asset into result early (so frontend can show it immediately)
        def store_input_asset(job: AnalysisJob) -> None:
            existing = job.result or {}
            assets = (existing.get("assets") or {}) if isinstance(existing, dict) else {}
            assets = dict(assets)
            assets.pop("input_video_url", None)
            assets["input_video"] = {
                "bucket": s3_bucket,
                "key": input_key,
            }
            job.result = {**existing, "assets": assets}

        update_job(db, job_id, store_input_asset)

        # Extract preview frames for UI selection
        update_job(
            db,
            job_id,
            lambda job: set_progress(
                job, "EXTRACTING_FRAMES", 25, "Extracting preview frames"
            ),
        )
        preview_count = _preview_frame_count()
        duration = get_duration_seconds(video_meta)
        player_ref = _normalize_player_ref(job.player_ref or {})
        anchor_time = player_ref.get("t") if player_ref else None
        timestamps = _build_preview_timestamps(duration, preview_count, anchor_time)

        frames_dir = base_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        preview_frames: List[Dict[str, Any]] = []
        for index, timestamp in enumerate(timestamps, start=1):
            frame_name = f"frame_{index:04d}.jpg"
            frame_path = frames_dir / frame_name

            try:
                _run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(timestamp),
                        "-i",
                        str(input_path),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(frame_path),
                    ]
                )
            except Exception:
                break

            width, height = probe_image_dimensions(frame_path)
            frame_key = f"jobs/{job_id}/frames/{frame_name}"
            upload_file(s3_internal, s3_bucket, frame_path, frame_key, "image/jpeg")
            preview_frames.append(
                {
                    "time_sec": timestamp,
                    "bucket": s3_bucket,
                    "key": frame_key,
                    "width": width,
                    "height": height,
                }
            )

        if preview_frames:
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "preview_frames", preview_frames),
                    set_progress(job, "PREVIEWS_READY", 30, "Preview frames ready"),
                ),
            )

        tracking_input_path = input_path
        tracking_time_offset = 0.0
        tracking_window_before = float(
            os.environ.get("TRACKING_WINDOW_BEFORE_SEC", "60") or 60
        )
        tracking_window_after = float(
            os.environ.get("TRACKING_WINDOW_AFTER_SEC", "60") or 60
        )
        tracking_anchor = (target.get("selection") or {}) if isinstance(target, dict) else {}
        t0_value = None
        if isinstance(tracking_anchor, dict):
            t0_value = tracking_anchor.get("time_sec")
        if t0_value is None and isinstance(job.player_ref, dict):
            t0_value = job.player_ref.get("best_time_sec")
        try:
            t0 = float(t0_value) if t0_value is not None else 0.0
        except (TypeError, ValueError):
            t0 = 0.0

        window_start = max(0.0, t0 - tracking_window_before)
        window_duration = tracking_window_before + tracking_window_after
        if duration and duration > 0:
            if window_start >= duration:
                window_start = max(0.0, duration - window_duration)
            max_duration = max(0.0, duration - window_start)
            window_duration = min(window_duration, max_duration)

        if window_duration > 0:
            segment_path = base_dir / "segment.mp4"
            try:
                ffmpeg_extract_segment(
                    input_path,
                    segment_path,
                    window_start,
                    window_duration,
                )
                tracking_input_path = segment_path
                tracking_time_offset = window_start
                logger.info(
                    "run_analysis tracking window job_id=%s start=%.2fs duration=%.2fs t0=%.2fs",
                    job_id,
                    window_start,
                    window_duration,
                    t0,
                )
            except Exception:
                logger.exception(
                    "run_analysis failed to extract tracking segment job_id=%s",
                    job_id,
                )

        job = reload_job(db, job_id)
        if not job:
            return
        existing_candidates = (job.result or {}).get("candidates")
        if not existing_candidates:
            update_job(
                db,
                job_id,
                lambda job: set_progress(
                    job, "TRACKING_CANDIDATES", 32, "Tracking all players"
                ),
            )
            candidates_output = track_all_players(
                job_id,
                str(tracking_input_path),
            )
            update_job(
                db,
                job_id,
                lambda job: setattr(
                    job,
                    "result",
                    {
                        **(job.result or {}),
                        "candidates": candidates_output,
                    },
                ),
            )

        # Block analysis until player_ref is present (but keep assets/frames stored)
        job = reload_job(db, job_id)
        if not job:
            return
        player_ref = _normalize_player_ref(job.player_ref or job.anchor or {})
        if player_ref is None:
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "status", "WAITING_FOR_PLAYER"),
                    setattr(job, "error", None),
                    set_progress(
                        job,
                        "WAITING_FOR_PLAYER",
                        30,
                        "Waiting for player reference",
                    ),
                ),
            )
            return

        # Tracking
        update_job(
            db,
            job_id,
            lambda job: set_progress(job, "TRACKING", 35, "Tracking selected player"),
        )
        tracking_player_ref = dict(player_ref)
        tracking_selections = [dict(sel) for sel in selections]
        if tracking_time_offset:
            tracking_player_ref["t"] = max(
                0.0, float(tracking_player_ref.get("t", 0.0)) - tracking_time_offset
            )
            adjusted_selections = []
            for selection in tracking_selections:
                if "frame_time_sec" in selection:
                    selection_time = float(selection.get("frame_time_sec", 0.0))
                    adjusted_time = max(0.0, selection_time - tracking_time_offset)
                    selection["frame_time_sec"] = adjusted_time
                    selection["time_sec"] = adjusted_time
                else:
                    selection_time = float(selection.get("time_sec", 0.0))
                    selection["time_sec"] = max(
                        0.0, selection_time - tracking_time_offset
                    )
                adjusted_selections.append(selection)
            tracking_selections = adjusted_selections
        tracking_output = track_player(
            job_id,
            str(tracking_input_path),
            tracking_player_ref,
            tracking_selections,
        )
        tracking_asset = {
            "bucket": s3_bucket,
            "key": tracking_output.get("tracking_key"),
            "url": tracking_output.get("tracking_url"),
        }
        motion_segments = build_motion_segments(tracking_output.get("bboxes") or [])
        update_job(
            db,
            job_id,
            lambda job: setattr(
                job,
                "result",
                {
                    **(job.result or {}),
                    "tracking": {
                        "method": tracking_output.get("method"),
                        "fps": tracking_output.get("fps"),
                        "coverage_pct": tracking_output.get("coverage_pct"),
                        "bboxes_count": len(tracking_output.get("bboxes") or []),
                        "track_id": tracking_output.get("track_id"),
                        "lost_segments": tracking_output.get("lost_segments"),
                        "motion_segments": motion_segments,
                        "notes": tracking_output.get("notes"),
                        "asset": tracking_asset,
                    },
                },
            ),
        )

        # Extract visual features for scoring
        update_job(
            db,
            job_id,
            lambda job: set_progress(
                job, "EXTRACTING_FEATURES", 50, "Extracting visual features"
            ),
        )
        video_features = extract_video_features(input_path, video_meta)
        logger.info(
            "VIDEO_FEATURES_USED",
            extra={
                "frames_analyzed": video_features.get("frame_count"),
                "features": video_features,
            },
        )
        update_job(
            db,
            job_id,
            lambda job: setattr(
                job,
                "result",
                {
                    **(job.result or {}),
                    "raw_video_features": video_features,
                },
            ),
        )

        # Extract clips
        update_job(
            db,
            job_id,
            lambda job: set_progress(job, "EXTRACTING", 55, "Extracting clips"),
        )
        extracted, clip_extraction_error = extract_clips(input_path, clips_dir)

        # Upload clips + signed
        update_job(
            db,
            job_id,
            lambda job: set_progress(
                job, "UPLOADING_CLIPS", 75, "Uploading clips to storage"
            ),
        )

        clips_out: List[Dict[str, Any]] = []
        for i, c in enumerate(extracted, start=1):
            clip_path: Path = c["file"]
            clip_key = f"jobs/{job_id}/clips/{clip_path.name}"
            upload_file(s3_internal, s3_bucket, clip_path, clip_key, "video/mp4")
            clip_start = c["start"]
            clip_end = c["end"]
            clips_out.append(
                {
                    "index": i,
                    "label": f"{clip_start}s-{clip_end}s",
                    "start_sec": clip_start,
                    "end_sec": clip_end,
                    "bucket": s3_bucket,
                    "key": clip_key,
                }
            )

        # Analysis output from visual features
        update_job(
            db,
            job_id,
            lambda job: set_progress(job, "ANALYZING", 85, "Running analysis"),
        )
        if video_features is None:
            raise AnalysisError("Insufficient video signal to compute score")

        logger.info(
            "SCORE_INPUTS",
            extra={
                "role": role,
                "category": category,
                "video_features_present": True,
                "video_features": video_features,
            },
        )

        skills_computed, skills_missing = compute_skill_scores(video_features)
        radar = {k: v for k, v in skills_computed.items() if v is not None}
        evaluation = compute_evaluation(role, radar, tracking_output)
        overall = evaluation.get("overall_score")
        role_score = evaluation.get("role_score")
        radar = evaluation.get("radar") or {}
        if overall is None:
            overall = 0.0
        if role_score is None:
            role_score = float(overall)
        if not radar:
            radar = {"tracking_quality": 0.0, "activity_proxy": 0.0, "visibility": 0.0, "consistency": 0.0}

        # Final result
        update_job(
            db,
            job_id,
            lambda job: set_progress(job, "FINALIZING", 95, "Saving results"),
        )

        def finalize_job(job: AnalysisJob) -> None:
            existing_result = job.result or {}

            existing_assets = (
                (existing_result.get("assets") or {})
                if isinstance(existing_result, dict)
                else {}
            )
            sanitized_assets = {
                key: value
                for key, value in existing_assets.items()
                if key not in ("input_video_url",)
            }
            input_video = {
                "bucket": s3_bucket,
                "key": input_key,
            }
            clip_assets = [
                {
                    "label": clip["label"],
                    "start_sec": clip["start_sec"],
                    "end_sec": clip["end_sec"],
                    "bucket": clip["bucket"],
                    "key": clip["key"],
                }
                for clip in clips_out
            ]

            warnings: List[str] = list(job.warnings or [])
            def add_warning(code: str) -> None:
                if code not in warnings:
                    warnings.append(code)

            expected_radar_keys = keys_required_for_role(role)
            if not set(expected_radar_keys).issubset(radar.keys()):
                add_warning("INCOMPLETE_RADAR")
            if overall is None:
                add_warning("MISSING_OVERALL_SCORE")
            if role_score is None:
                add_warning("MISSING_ROLE_SCORE")
            if not clip_assets:
                add_warning("MISSING_CLIPS")
            if clip_extraction_error:
                add_warning("CLIP_EXTRACTION_FAILED")
            if not input_key:
                add_warning("MISSING_INPUT_VIDEO")

            run_status = "COMPLETED" if not warnings else "PARTIAL"
            player_runs = (
                existing_result.get("player_runs")
                if isinstance(existing_result, dict)
                else None
            )
            if not isinstance(player_runs, dict):
                player_runs = {}
            player_track_id = None
            if isinstance(job.player_ref, dict):
                player_track_id = job.player_ref.get("track_id")
            if player_track_id is not None:
                player_runs[str(player_track_id)] = {
                    "track_id": player_track_id,
                    "status": run_status,
                    "result": {
                        "overallScore": overall,
                        "overall_score": overall,
                        "roleScore": role_score,
                        "role_score": role_score,
                        "radar": radar,
                    },
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }

            # keep existing assets/preview_frames/tracking if already present
            job.result = {
                **existing_result,
                "schema_version": "1.3",
                "summary": {
                    "player_role": role,
                    "overall_score": overall,
                    "role_score": role_score,
                },
                "overall_score": overall,
                "overallScore": overall,
                "role_score": role_score,
                "roleScore": role_score,
                "radar": radar,
                "raw_video_features": video_features,
                "skills_computed": skills_computed,
                "skills_missing": skills_missing,
                "assets": {
                    **sanitized_assets,
                    "input_video": input_video,
                    "clips": clip_assets,
                },
                "clips": clip_assets,
                "clip_extraction_error": clip_extraction_error,
                "player_runs": player_runs,
            }

            job.warnings = warnings
            job.status = run_status
            job.failure_reason = normalize_failure_reason(None)
            set_progress(job, "DONE", 100, "Completed")

        update_job(db, job_id, finalize_job)

    except TrackingTimeoutError:
        try:
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "status", "FAILED"),
                    setattr(job, "error", "Tracking timeout exceeded"),
                    setattr(
                        job,
                        "failure_reason",
                        normalize_failure_reason("TRACKING_TIMEOUT"),
                    ),
                    setattr(job, "warnings", list({*(job.warnings or []), "TRACKING_TIMEOUT"})),
                    set_progress(job, "FAILED", 100, "Tracking timeout"),
                ),
            )
        except Exception:
            db.rollback()
        return
    except AnalysisError as e:
        try:
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "status", "FAILED"),
                    setattr(job, "error", str(e)),
                    setattr(
                        job,
                        "failure_reason",
                        normalize_failure_reason("insufficient_visual_signal"),
                    ),
                    set_progress(job, "FAILED", 100, f"Failed: {str(e)}"),
                ),
            )
        except Exception:
            db.rollback()
        return
    except Exception as e:
        try:
            if self.request.retries < max_retries:
                raise self.retry(exc=e, countdown=10 * (2 ** self.request.retries))
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "status", "FAILED"),
                    setattr(job, "error", str(e)),
                    set_progress(job, "FAILED", 100, f"Failed: {str(e)}"),
                ),
            )
        except Exception:
            db.rollback()
        return
    finally:
        _cleanup_workdir(base_dir)
        db.close()
