import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import requests

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.core.deps import get_db
from app.core.models import AnalysisJob
from app.schemas import JobCreate, JobOut, PlayerRefPayload, SelectionPayload
from app.workers.pipeline import run_analysis
from app.workers.pipeline import (
    ensure_bucket_exists,
    get_s3_client,
    presign_get_url,
    resolve_public_endpoint,
    upload_file,
)

router = APIRouter()

POLLING_SAFE_STATUSES = {
    "WAITING_FOR_SELECTION",
    "WAITING_FOR_PLAYER",
    "CREATED",
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
}


def normalize_status(status: str) -> str:
    if status in POLLING_SAFE_STATUSES:
        return status
    if status == "WAITING_FOR_ANCHOR":
        return "WAITING_FOR_SELECTION"
    return "QUEUED"


def normalize_payload(payload: object) -> dict:
    if payload is None:
        return {}
    encoded = jsonable_encoder(payload)
    if isinstance(encoded, dict):
        return encoded
    return {"value": encoded}


def load_s3_context() -> Dict[str, Any]:
    s3_endpoint_url = (os.environ.get("S3_ENDPOINT_URL") or "").strip()
    s3_public_endpoint_url = (os.environ.get("S3_PUBLIC_ENDPOINT_URL") or "").strip()
    s3_access_key = (os.environ.get("S3_ACCESS_KEY") or "").strip()
    s3_secret_key = (os.environ.get("S3_SECRET_KEY") or "").strip()
    s3_bucket = (os.environ.get("S3_BUCKET") or "").strip()
    expires_seconds = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))

    if (
        not s3_endpoint_url
        or not s3_public_endpoint_url
        or not s3_access_key
        or not s3_secret_key
        or not s3_bucket
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing S3 env vars: S3_ENDPOINT_URL, S3_PUBLIC_ENDPOINT_URL, "
                "S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET"
            ),
        )

    try:
        resolve_public_endpoint(s3_endpoint_url, s3_public_endpoint_url)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "s3_internal": get_s3_client(s3_endpoint_url),
        "bucket": s3_bucket,
        "expires_seconds": expires_seconds,
    }


def attach_presigned_urls(result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    s3_internal = context["s3_internal"]
    bucket_default = context["bucket"]
    expires_seconds = context["expires_seconds"]
    def presign(bucket: str, key: str) -> str:
        return presign_get_url(
            s3_internal,
            bucket,
            key,
            expires_seconds,
        )

    def normalize_asset(asset: Dict[str, Any], include_url: bool) -> Dict[str, Any]:
        bucket = asset.get("bucket") or bucket_default
        key = asset.get("key") or asset.get("s3_key")
        normalized = {**asset}
        if bucket:
            normalized["bucket"] = bucket
        if key:
            normalized["key"] = key
        if bucket and key:
            signed_url = presign(bucket, key)
            normalized["signed_url"] = signed_url
            if include_url:
                normalized["url"] = signed_url
        return normalized

    hydrated = {**result}

    assets = hydrated.get("assets")
    if isinstance(assets, dict):
        assets_copy = {**assets}
        input_video = assets_copy.get("input_video")
        if isinstance(input_video, dict):
            input_video = normalize_asset(input_video, include_url=True)
            assets_copy["input_video"] = input_video
            if "input_video_url" not in assets_copy:
                assets_copy["input_video_url"] = input_video.get("signed_url")

        clips = assets_copy.get("clips")
        if isinstance(clips, list):
            assets_copy["clips"] = [
                normalize_asset(clip, include_url=True) if isinstance(clip, dict) else clip
                for clip in clips
            ]

        hydrated["assets"] = assets_copy

    preview_frames = hydrated.get("preview_frames")
    if isinstance(preview_frames, list):
        hydrated["preview_frames"] = [
            normalize_asset(frame, include_url=False)
            if isinstance(frame, dict)
            else frame
            for frame in preview_frames
        ]

    clips_root = hydrated.get("clips")
    if isinstance(clips_root, list):
        hydrated["clips"] = [
            normalize_asset(clip, include_url=True) if isinstance(clip, dict) else clip
            for clip in clips_root
        ]

    return hydrated


def _run_command(cmd: List[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")
    return (res.stdout or "").strip()


def ensure_ffmpeg_available() -> None:
    _run_command(["ffmpeg", "-version"])
    _run_command(["ffprobe", "-version"])


def download_video(url: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(
        url,
        stream=True,
        timeout=(10, 1800),
        headers={"User-Agent": "AlgoNextAPI/1.0"},
    ) as response:
        response.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def probe_video_duration(path: Path) -> Optional[float]:
    try:
        out = _run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ]
        )
        data = json.loads(out)
        duration = data.get("format", {}).get("duration")
        if duration is None:
            return None
        return float(duration)
    except Exception:
        return None


def probe_image_dimensions(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        out = _run_command(
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


@router.post("/jobs", response_model=JobOut)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    job_id = str(uuid4())

    target = {
        "player": {
            "team_name": payload.team_name,
            "player_name": payload.player_name,
            "shirt_number": payload.shirt_number,
        },
        "selections": [],
        "tracking": {"status": "PENDING"},
    }

    job = AnalysisJob(
        id=job_id,
        status="WAITING_FOR_SELECTION",
        category=payload.category,
        role=payload.role,
        video_url=payload.video_url,
        target=target,
        video_meta={},
        anchor={},
        player_ref=None,
        progress={"step": "CREATED", "pct": 0, "message": "Job created"},
        result={},
        error=None,
        failure_reason=None,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    return JobOut(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    db: Session = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        result_payload = normalize_payload(job.result)
        if result_payload:
            context = load_s3_context()
            result_payload = attach_presigned_urls(result_payload, context)

        return {
            "id": job.id,
            "status": normalize_status(job.status),
            "category": job.category,
            "role": job.role,
            "video_url": job.video_url,
            "progress": normalize_payload(job.progress),
            "result": result_payload,
            "error": job.error,
            "failure_reason": job.failure_reason,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        }
    finally:
        db.close()


# ✅ endpoint leggero per polling
@router.get("/jobs/{job_id}/status")
def job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "status": normalize_status(job.status),
        "progress": normalize_payload(job.progress),
        "error": job.error,
        "failure_reason": job.failure_reason,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


@router.get("/jobs/{job_id}/poll")
def job_poll(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result_payload = job.result if job.status == "COMPLETED" else None
    if result_payload:
        context = load_s3_context()
        result_payload = attach_presigned_urls(result_payload, context)

    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "failure_reason": job.failure_reason,
        "result": result_payload,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


# ✅ result “pulito” (solo quando pronto)
@router.get("/jobs/{job_id}/result")
def job_result(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "FAILED":
        raise HTTPException(status_code=409, detail=job.error or "Job failed")

    if job.status != "COMPLETED":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    result_payload = normalize_payload(job.result)
    if result_payload:
        context = load_s3_context()
        result_payload = attach_presigned_urls(result_payload, context)
    return result_payload


@router.post("/jobs/{job_id}/selection", response_model=JobOut)
def save_selection(
    job_id: str, payload: SelectionPayload, db: Session = Depends(get_db)
):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    current_target = job.target or {}
    player = current_target.get("player") or {}
    tracking = current_target.get("tracking") or {"status": "PENDING"}
    job.target = {
        "player": player,
        "selections": [selection.dict() for selection in payload.selections],
        "tracking": tracking,
    }

    if job.status == "WAITING_FOR_SELECTION":
        job.status = "CREATED"

    db.commit()
    db.refresh(job)
    return JobOut(job_id=job.id, status=job.status)


@router.post("/jobs/{job_id}/player-ref")
def save_player_ref(
    job_id: str, payload: PlayerRefPayload, db: Session = Depends(get_db)
):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    player_ref = payload.dict(exclude_none=True)
    bbox = player_ref.get("bbox") or {}
    if bbox.get("w", 0) <= 0 or bbox.get("h", 0) <= 0:
        raise HTTPException(status_code=400, detail="Invalid bbox dimensions")

    job.player_ref = payload.frame_key

    progress = job.progress or {}
    current_pct = progress.get("pct") or 0
    try:
        current_pct = int(current_pct)
    except (TypeError, ValueError):
        current_pct = 0
    job.progress = {
        **progress,
        "step": "PLAYER_SELECTED",
        "pct": max(current_pct, 12),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    db.commit()
    db.refresh(job)
    return {"status": "ok", "player_ref": job.player_ref}


@router.get("/jobs/{job_id}/frames")
def get_frames(job_id: str, count: int = 8, db: Session = Depends(get_db)):
    if count < 1:
        raise HTTPException(status_code=400, detail="Count must be >= 1")

    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    context = load_s3_context()
    s3_internal = context["s3_internal"]
    minio_bucket = context["bucket"]
    expires_seconds = context["expires_seconds"]

    ensure_ffmpeg_available()

    base_dir = Path("/tmp/fnh_jobs") / job_id
    input_path = base_dir / "input.mp4"
    frames_dir = base_dir / "frames"

    if not input_path.exists():
        download_video(job.video_url, input_path)

    duration = probe_video_duration(input_path)
    if duration and duration > 0:
        step = duration / (count + 1)
        timestamps = [round(step * (i + 1), 3) for i in range(count)]
    else:
        timestamps = [i * 10 for i in range(count)]

    frames_dir.mkdir(parents=True, exist_ok=True)

    ensure_bucket_exists(s3_internal, minio_bucket)

    frames: List[Dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps, start=1):
        frame_name = f"frame_{index:03d}.jpg"
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            try:
                _run_command(
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
            except RuntimeError:
                break

        width, height = probe_image_dimensions(frame_path)
        frame_key = f"jobs/{job_id}/frames/{frame_name}"
        upload_file(s3_internal, minio_bucket, frame_path, frame_key, "image/jpeg")
        signed_url = presign_get_url(
            s3_internal,
            minio_bucket,
            frame_key,
            expires_seconds,
        )
        frames.append(
            {
                "index": index,
                "timestamp": timestamp,
                "s3_key": frame_key,
                "signed_url": signed_url,
                "expires_in": expires_seconds,
                "width": width,
                "height": height,
            }
        )

    if not frames:
        raise HTTPException(status_code=500, detail="No frames extracted")

    return {"count": len(frames), "frames": frames}


@router.post("/jobs/{job_id}/enqueue", response_model=JobOut)
def enqueue_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # idempotente: se già avanzato, non reinvio
    if job.status in ["QUEUED", "RUNNING", "COMPLETED", "FAILED"]:
        return JobOut(job_id=job.id, status=job.status)

    if job.status not in ["WAITING_FOR_SELECTION", "CREATED", "WAITING_FOR_PLAYER"]:
        raise HTTPException(status_code=400, detail="Job not enqueueable")

    selections = (job.target or {}).get("selections") or []
    if len(selections) < 2:
        raise HTTPException(
            status_code=400,
            detail={"error": "PLAYER_SELECTION_REQUIRED"},
        )

    job.status = "QUEUED"
    job.error = None
    db.commit()
    db.refresh(job)

    run_analysis.delay(job.id)

    return JobOut(job_id=job.id, status=job.status)
