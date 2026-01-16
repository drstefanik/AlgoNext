import os
import time
import json
import shutil
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from sqlalchemy.orm import Session

from app.workers.celery_app import celery
from app.core.env import is_production_env
from app.core.db import SessionLocal
from app.core.models import AnalysisJob

logger = logging.getLogger(__name__)


class AnalysisError(RuntimeError):
    pass


# ----------------------------
# Role weights (v1)
# ----------------------------
ROLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Striker": {
        "Finishing": 0.24,
        "Positioning": 0.20,
        "Off-the-ball Movement": 0.20,
        "Composure": 0.16,
        "Shot Power": 0.12,
        "Heading": 0.08,
    },
    "Winger": {
        "Off-the-ball Movement": 0.22,
        "Positioning": 0.18,
        "Composure": 0.16,
        "Shot Power": 0.16,
        "Finishing": 0.18,
        "Heading": 0.10,
    },
    "Midfielder": {
        "Composure": 0.22,
        "Positioning": 0.22,
        "Off-the-ball Movement": 0.20,
        "Shot Power": 0.14,
        "Finishing": 0.14,
        "Heading": 0.08,
    },
    "Defender": {
        "Heading": 0.22,
        "Positioning": 0.22,
        "Composure": 0.18,
        "Off-the-ball Movement": 0.18,
        "Shot Power": 0.10,
        "Finishing": 0.10,
    },
    "Goalkeeper": {
        "Composure": 0.40,
        "Positioning": 0.30,
        "Off-the-ball Movement": 0.10,
        "Heading": 0.05,
        "Shot Power": 0.10,
        "Finishing": 0.05,
    },
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "Finishing": 1.0,
    "Heading": 1.0,
    "Positioning": 1.0,
    "Composure": 1.0,
    "Off-the-ball Movement": 1.0,
    "Shot Power": 1.0,
}


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


def set_progress(job: AnalysisJob, step: str, pct: int, message: str = "") -> None:
    pct = max(0, min(100, int(pct)))
    job.progress = {
        "step": step,
        "pct": pct,
        "message": message,
        "updated_at": utc_now_iso(),
    }


def weighted_overall_score(radar: Dict[str, float], role: Optional[str]) -> int:
    role = (role or "").strip()
    weights = ROLE_WEIGHTS.get(role, DEFAULT_WEIGHTS)

    pairs: List[Tuple[float, float]] = []
    for k, score in radar.items():
        if k not in weights:
            continue
        try:
            s = float(score)
        except Exception:
            continue
        s = max(0.0, min(100.0, s))
        pairs.append((s, float(weights[k])))

    if not pairs:
        return 0

    weighted_sum = sum(s * w for s, w in pairs)
    w_sum = sum(w for _, w in pairs) or 1.0
    value = weighted_sum / w_sum
    value = max(0.0, min(100.0, value))
    return int(round(value))


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


def ensure_ffmpeg_available() -> None:
    _run(["ffmpeg", "-version"])
    _run(["ffprobe", "-version"])


def download_video(
    url: str, dst_path: Path, progress_callback: Optional[Callable[[], None]] = None
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    log_every_bytes = 100 * 1024 * 1024
    bytes_downloaded = 0
    next_log_bytes = log_every_bytes
    last_progress_tick = time.monotonic()
    progress_tick_seconds = 5

    with requests.get(
        url,
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
                        logger.info("Downloaded %.1f MB from %s", mb_downloaded, url)
                        next_log_bytes += log_every_bytes


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


def extract_clips(input_path: Path, out_dir: Path) -> List[Dict]:
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
        except Exception:
            if not created:
                fallback = out_dir / "clip_001.mp4"
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
            break

    return created


# ----------------------------
# Helpers: S3/MinIO (upload + signed urls)
# ----------------------------
def get_s3_client(endpoint_url: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        region_name=os.environ.get("MINIO_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )


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


def _normalize_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/")


def resolve_public_endpoint(internal_endpoint: str, public_endpoint: str) -> str:
    resolved = public_endpoint.strip()
    if not resolved:
        raise RuntimeError("Missing MINIO_PUBLIC_ENDPOINT.")

    internal_normalized = _normalize_endpoint(internal_endpoint)
    public_normalized = _normalize_endpoint(resolved)
    if internal_normalized and internal_normalized == public_normalized:
        raise RuntimeError(
            "MINIO_PUBLIC_ENDPOINT must differ from MINIO_INTERNAL_ENDPOINT."
        )

    return resolved


def rewrite_presigned_to_public(url: str) -> str:
    public_endpoint = (os.environ.get("MINIO_PUBLIC_ENDPOINT") or "").strip()
    if not public_endpoint:
        raise RuntimeError("Missing MINIO_PUBLIC_ENDPOINT.")
    parsed_url = urlsplit(url)
    parsed_public = urlsplit(public_endpoint)
    if not parsed_public.scheme or not parsed_public.netloc:
        raise RuntimeError("MINIO_PUBLIC_ENDPOINT must include scheme and host.")
    if not is_production_env():
        logger.info(
            "MINIO_PRESIGN_REWRITE",
            extra={
                "minio_public_endpoint": public_endpoint,
                "before_host": parsed_url.netloc,
                "after_host": parsed_public.netloc,
            },
        )
    return urlunsplit(
        (
            parsed_public.scheme,
            parsed_public.netloc,
            parsed_url.path,
            parsed_url.query,
            parsed_url.fragment,
        )
    )


def presign_get_url(
    s3_internal,
    bucket: str,
    key: str,
    expires_seconds: int,
) -> str:
    signed_url = s3_internal.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )
    return rewrite_presigned_to_public(signed_url)


# ----------------------------
# Celery task
# ----------------------------
@celery.task(name="app.workers.pipeline.run_analysis")
def run_analysis(job_id: str):
    db: Session = SessionLocal()
    base_dir: Optional[Path] = None
    video_features: Optional[Dict[str, Any]] = None
    skills_computed: Dict[str, Optional[int]] = {}
    skills_missing: List[str] = []

    # Env vars (required)
    minio_internal_endpoint = os.environ.get("MINIO_INTERNAL_ENDPOINT", "").strip()
    minio_access_key = os.environ.get("MINIO_ACCESS_KEY", "").strip()
    minio_secret_key = os.environ.get("MINIO_SECRET_KEY", "").strip()
    minio_bucket = (
        os.environ.get("MINIO_BUCKET") or os.environ.get("S3_BUCKET") or ""
    ).strip()

    try:
        job = reload_job(db, job_id)
        if not job:
            return

        video_url = job.video_url
        role = job.role
        category = job.category

        selections = (job.target or {}).get("selections") or []
        if len(selections) < 1:
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
            return

        # RUNNING
        update_job(
            db,
            job_id,
            lambda job: (
                setattr(job, "status", "RUNNING"),
                setattr(job, "error", None),
                setattr(job, "failure_reason", None),
                set_progress(job, "STARTING", 1, "Job started"),
            ),
        )

        # Preconditions
        if (
            not minio_internal_endpoint
            or not minio_bucket
            or not minio_access_key
            or not minio_secret_key
        ):
            raise RuntimeError(
                "Missing MinIO env vars: MINIO_INTERNAL_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET"
            )

        ensure_ffmpeg_available()

        # Prepare workspace
        base_dir = Path("/tmp/fnh_jobs") / job_id
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)
        base_dir.mkdir(parents=True, exist_ok=True)

        input_path = base_dir / "input.mp4"
        clips_dir = base_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        # S3 clients
        s3_internal = get_s3_client(minio_internal_endpoint)

        # Ensure bucket exists
        ensure_bucket_exists(s3_internal, minio_bucket)

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

        download_video(video_url, input_path, progress_callback=download_progress_tick)

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
        upload_file(s3_internal, minio_bucket, input_path, input_key, "video/mp4")
        # Persist input asset into result early (so frontend can show it immediately)
        def store_input_asset(job: AnalysisJob) -> None:
            existing = job.result or {}
            assets = (existing.get("assets") or {}) if isinstance(existing, dict) else {}
            assets = dict(assets)
            assets.pop("input_video_url", None)
            assets["input_video"] = {
                "bucket": minio_bucket,
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
        preview_count = int(os.environ.get("PREVIEW_FRAME_COUNT", "8"))
        preview_count = max(1, min(12, preview_count))

        duration = get_duration_seconds(video_meta)
        if duration and duration > 0:
            step = duration / (preview_count + 1)
            timestamps = [round(step * (i + 1), 3) for i in range(preview_count)]
        else:
            timestamps = [i * 10 for i in range(preview_count)]

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
            upload_file(s3_internal, minio_bucket, frame_path, frame_key, "image/jpeg")
            preview_frames.append(
                {
                    "time_sec": timestamp,
                    "bucket": minio_bucket,
                    "key": frame_key,
                    "width": width,
                    "height": height,
                }
            )

        if preview_frames:
            update_job(
                db,
                job_id,
                lambda job: setattr(
                    job,
                    "result",
                    {
                        **(job.result or {}),
                        "preview_frames": preview_frames,
                    },
                ),
            )

        # Block analysis until player_ref is present (but keep assets/frames stored)
        job = reload_job(db, job_id)
        if not job:
            return
        if not (job.player_ref or "").strip():
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

        # Tracking (stub)
        update_job(
            db,
            job_id,
            lambda job: (
                set_progress(job, "TRACKING", 40, "Tracking selected player"),
                setattr(
                    job,
                    "result",
                    {
                        **(job.result or {}),
                        "tracking": {
                            "method": "stub",
                            "selected_boxes": len(selections),
                            "notes": "tracking not implemented yet",
                        },
                    },
                ),
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
        extracted = extract_clips(input_path, clips_dir)

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
            upload_file(s3_internal, minio_bucket, clip_path, clip_key, "video/mp4")
            clips_out.append(
                {
                    "index": i,
                    "start": c["start"],
                    "end": c["end"],
                    "bucket": minio_bucket,
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
        overall = weighted_overall_score(radar, role)

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
                "bucket": minio_bucket,
                "key": input_key,
            }
            clip_assets = [
                {
                    "start": clip["start"],
                    "end": clip["end"],
                    "bucket": clip["bucket"],
                    "key": clip["key"],
                }
                for clip in clips_out
            ]

            # keep existing assets/preview_frames/tracking if already present
            job.result = {
                **existing_result,
                "schema_version": "1.2",
                "summary": {
                    "player_role": role,
                    "overall_score": overall,
                },
                "radar": radar,
                "raw_video_features": video_features,
                "skills_computed": skills_computed,
                "skills_missing": skills_missing,
                "assets": {
                    **sanitized_assets,
                    "input_video": input_video,
                    "clips": clip_assets,
                },
                "clips": clips_out,
            }

            job.status = "COMPLETED"
            job.failure_reason = None
            set_progress(job, "DONE", 100, "Completed")

        update_job(db, job_id, finalize_job)

    except AnalysisError as e:
        try:
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "status", "FAILED"),
                    setattr(job, "error", str(e)),
                    setattr(job, "failure_reason", "insufficient_visual_signal"),
                    set_progress(job, "FAILED", 100, f"Failed: {str(e)}"),
                ),
            )
        except Exception:
            db.rollback()
        return
    except Exception as e:
        try:
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
        if base_dir is not None and base_dir.exists():
            try:
                shutil.rmtree(base_dir)
            except Exception:
                pass
        db.close()
