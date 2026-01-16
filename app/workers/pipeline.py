import os
import time
import json
import shutil
import subprocess
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from sqlalchemy.orm import Session

from app.workers.celery_app import celery
from app.core.db import SessionLocal
from app.core.models import AnalysisJob

logger = logging.getLogger(__name__)


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
    # Minimal ffprobe json (safe)
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
            # Se fallisce, stop e proviamo a fare almeno una clip "safe" di 3s da 0
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
    # signature v4 ok per MinIO + AWS
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket_exists(s3_client, bucket: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code not in ("404", "NoSuchBucket", "NotFound"):
            # Se è un errore diverso (permission ecc.), non mascherarlo
            raise
    # MinIO: create senza LocationConstraint
    s3_client.create_bucket(Bucket=bucket)


def upload_file(s3_internal, bucket: str, local_path: Path, key: str, content_type: str) -> None:
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


def presign_get_url(s3_public, bucket: str, key: str, expires_seconds: int) -> str:
    return s3_public.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


# ----------------------------
# Celery task
# ----------------------------
@celery.task(name="app.workers.pipeline.run_analysis")
def run_analysis(job_id: str):
    db: Session = SessionLocal()
    base_dir: Optional[Path] = None

    # Env vars (required)
    S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "").strip()
    S3_PUBLIC_ENDPOINT_URL = os.environ.get("S3_PUBLIC_ENDPOINT_URL", "").strip()
    S3_BUCKET = os.environ.get("S3_BUCKET", "").strip()
    expires_seconds = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))

    try:
        job = reload_job(db, job_id)
        if not job:
            return
        video_url = job.video_url
        role = job.role

        selections = (job.target or {}).get("selections") or []
        if len(selections) < 2:
            update_job(
                db,
                job_id,
                lambda job: (
                    setattr(job, "status", "FAILED"),
                    setattr(job, "error", "PLAYER_SELECTION_REQUIRED"),
                    set_progress(
                        job,
                        "FAILED",
                        100,
                        "Failed: PLAYER_SELECTION_REQUIRED",
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
                set_progress(job, "STARTING", 1, "Job started"),
            ),
        )

        # Preconditions
        if not S3_ENDPOINT_URL or not S3_PUBLIC_ENDPOINT_URL or not S3_BUCKET:
            raise RuntimeError(
                "Missing S3 env vars: S3_ENDPOINT_URL, S3_PUBLIC_ENDPOINT_URL, S3_BUCKET"
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
        s3_internal = get_s3_client(S3_ENDPOINT_URL)
        s3_public = get_s3_client(S3_PUBLIC_ENDPOINT_URL)

        # ✅ Ensure bucket exists (kills NoSuchBucket forever)
        ensure_bucket_exists(s3_internal, S3_BUCKET)

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
        update_job(db, job_id, lambda job: setattr(job, "video_meta", video_meta))

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

        # Upload input
        update_job(
            db,
            job_id,
            lambda job: set_progress(
                job, "UPLOADING_INPUT", 30, "Uploading input video to storage"
            ),
        )
        input_key = f"jobs/{job_id}/input.mp4"
        upload_file(s3_internal, S3_BUCKET, input_path, input_key, "video/mp4")

        input_signed = presign_get_url(s3_public, S3_BUCKET, input_key, expires_seconds)

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

        clips_out: List[Dict] = []
        for i, c in enumerate(extracted, start=1):
            clip_path: Path = c["file"]
            clip_key = f"jobs/{job_id}/clips/{clip_path.name}"
            upload_file(s3_internal, S3_BUCKET, clip_path, clip_key, "video/mp4")
            clip_signed = presign_get_url(s3_public, S3_BUCKET, clip_key, expires_seconds)
            clips_out.append(
                {
                    "index": i,
                    "start": c["start"],
                    "end": c["end"],
                    "s3_key": clip_key,
                    "signed_url": clip_signed,
                    "expires_in": expires_seconds,
                }
            )

        # Mock analysis output (replace with real pipeline)
        update_job(
            db,
            job_id,
            lambda job: set_progress(job, "ANALYZING", 85, "Running analysis"),
        )
        time.sleep(0.2)

        radar = {
            "Finishing": 72,
            "Heading": 65,
            "Positioning": 78,
            "Composure": 70,
            "Off-the-ball Movement": 80,
            "Shot Power": 74,
        }
        overall = weighted_overall_score(radar, role)

        # Final result (SIGNED URLs)
        update_job(
            db,
            job_id,
            lambda job: set_progress(job, "FINALIZING", 95, "Saving results"),
        )

        def finalize_job(job: AnalysisJob) -> None:
            job.result = {
                "schema_version": "1.1",
                "summary": {
                    "player_role": role,
                    "overall_score": overall,
                },
                "radar": radar,
                "assets": {
                    "input_video": {
                        "s3_key": input_key,
                        "signed_url": input_signed,
                        "expires_in": expires_seconds,
                    }
                },
                "clips": clips_out,
            }
            job.status = "COMPLETED"
            set_progress(job, "DONE", 100, "Completed")

        update_job(db, job_id, finalize_job)

    except Exception as e:
        # Mark FAILED consistently
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
