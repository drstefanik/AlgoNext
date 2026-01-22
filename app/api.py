import json
import logging
import os
import socket
import subprocess
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.core.deps import get_db
from app.core.models import AnalysisJob
from app.core.normalizers import normalize_failure_reason
from app.schemas import JobCreate, PlayerRefPayload, SelectionPayload
from app.workers.pipeline import extract_preview_frames, run_analysis
from app.workers.pipeline import (
    ensure_bucket_exists,
    get_s3_client,
    upload_file,
)

logger = logging.getLogger(__name__)

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

S3_ENDPOINT_URL = (os.environ.get("S3_ENDPOINT_URL") or "").strip()
S3_PUBLIC_ENDPOINT_URL = (os.environ.get("S3_PUBLIC_ENDPOINT_URL") or "").strip()
S3_ACCESS_KEY = (os.environ.get("S3_ACCESS_KEY") or "").strip()
S3_SECRET_KEY = (os.environ.get("S3_SECRET_KEY") or "").strip()
S3_BUCKET = (os.environ.get("S3_BUCKET") or "").strip()
SIGNED_URL_EXPIRES_SECONDS = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))


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


def build_meta(request: Request | None = None) -> dict:
    request_id = getattr(request.state, "request_id", None) if request else None
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def ok_response(data: dict, request: Request | None = None) -> dict:
    return {"ok": True, "data": data, "meta": build_meta(request)}


def error_detail(code: str, message: str, details: dict | None = None) -> dict:
    payload = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return {"error": payload}


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
        raise HTTPException(
            status_code=400,
            detail=error_detail("INVALID_URL", "Invalid URL host"),
        )
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443)
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=400,
            detail=error_detail("INVALID_URL", "Unable to resolve URL host"),
        ) from exc
    for _, _, _, _, sockaddr in infos:
        ip_value = sockaddr[0]
        if _is_private_ip(ip_value):
            raise HTTPException(
                status_code=400,
                detail=error_detail("FORBIDDEN_URL", "URL host is not allowed"),
            )


def _is_shared_object_url(url: str) -> bool:
    return urlsplit(url).path.startswith("/api/v1/download-shared-object/")


def resolve_job_video_source(job: AnalysisJob, fallback_bucket: str) -> Tuple[str, str]:
    if job.video_key:
        bucket = job.video_bucket or fallback_bucket
        if not bucket:
            raise HTTPException(
                status_code=400,
                detail=error_detail("VIDEO_BUCKET_MISSING", "Missing video bucket."),
            )
        return job.video_key, bucket
    if job.video_url:
        parsed = urlsplit(job.video_url)
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(
                status_code=400,
                detail=error_detail(
                    "INVALID_URL",
                    "video_url must be an http(s) URL.",
                ),
            )
        if _is_shared_object_url(job.video_url):
            raise HTTPException(
                status_code=400,
                detail=error_detail(
                    "SHARED_OBJECT_UNSUPPORTED",
                    "Shared object URLs are not supported for video download.",
                ),
            )
        return job.video_url, fallback_bucket
    raise HTTPException(
        status_code=400,
        detail=error_detail("VIDEO_SOURCE_MISSING", "Missing video source."),
    )


def load_s3_context() -> Dict[str, Any]:
    if (
        not S3_ENDPOINT_URL
        or not S3_PUBLIC_ENDPOINT_URL
        or not S3_ACCESS_KEY
        or not S3_SECRET_KEY
        or not S3_BUCKET
    ):
        raise HTTPException(
            status_code=500,
            detail=error_detail(
                "S3_CONFIG_MISSING",
                "Missing S3 env vars: S3_ENDPOINT_URL, S3_PUBLIC_ENDPOINT_URL, "
                "S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET",
            ),
        )

    s3_internal = get_s3_client(S3_ENDPOINT_URL)
    s3_public = get_s3_client(S3_PUBLIC_ENDPOINT_URL)

    return {
        "s3_internal": s3_internal,
        "s3_public": s3_public,
        "bucket": S3_BUCKET,
        "expires_seconds": SIGNED_URL_EXPIRES_SECONDS,
    }


def presign_get_url(
    s3_public,
    bucket: str,
    key: str,
    expires_seconds: int,
) -> str:
    return s3_public.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


def attach_presigned_urls(result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    s3_public = context["s3_public"]
    bucket_default = context["bucket"]
    expires_seconds = context["expires_seconds"]

    def presign(bucket: str, key: str) -> str:
        return presign_get_url(s3_public, bucket, key, expires_seconds)

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

    tracking = hydrated.get("tracking")
    if isinstance(tracking, dict):
        tracking_copy = {**tracking}
        asset = tracking_copy.get("asset")
        if isinstance(asset, dict):
            tracking_copy["asset"] = normalize_asset(asset, include_url=True)
        hydrated["tracking"] = tracking_copy

    return hydrated


def _run_command(cmd: List[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{res.stderr}")
    return (res.stdout or "").strip()


def ensure_ffmpeg_available() -> None:
    _run_command(["ffmpeg", "-version"])
    _run_command(["ffprobe", "-version"])


def download_video(url: str, dst_path: Path, s3_client, bucket: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not url:
        raise HTTPException(
            status_code=400,
            detail=error_detail("VIDEO_SOURCE_MISSING", "Missing video source."),
        )
    parsed = urlsplit(url)
    if parsed.scheme in ("http", "https"):
        _ensure_public_url(url)
        if _is_shared_object_url(url):
            raise HTTPException(
                status_code=400,
                detail=error_detail(
                    "SHARED_OBJECT_UNSUPPORTED",
                    "Shared object URLs are not supported for video download.",
                ),
            )
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
        return

    object_key = url.lstrip("/")
    if not object_key:
        raise HTTPException(
            status_code=400,
            detail=error_detail("VIDEO_KEY_MISSING", "Missing MinIO object key."),
        )
    s3_client.download_file(
        Bucket=bucket,
        Key=object_key,
        Filename=str(dst_path),
    )


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


@router.post("/jobs")
def create_job(payload: JobCreate, request: Request, db: Session = Depends(get_db)):
    job_id = str(uuid4())
    video_url = payload.video_url
    video_bucket = None
    video_key = None
    if payload.video_key:
        context = load_s3_context()
        s3_public = context["s3_public"]
        bucket = payload.video_bucket or context["bucket"]
        if not bucket:
            raise HTTPException(
                status_code=400,
                detail=error_detail(
                    "VIDEO_BUCKET_MISSING",
                    "video_bucket is required when using video_key.",
                ),
            )
        expires_seconds = context["expires_seconds"]
        video_url = s3_public.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": payload.video_key},
            ExpiresIn=expires_seconds,
        )
        video_bucket = bucket
        video_key = payload.video_key
    elif video_url:
        if video_url.lower().startswith(("http://", "https://")):
            parsed = urlsplit(video_url)
            if parsed.scheme not in ("http", "https"):
                raise HTTPException(
                    status_code=400,
                    detail=error_detail(
                        "INVALID_URL",
                        "video_url must be an http(s) URL.",
                    ),
                )
            if _is_shared_object_url(video_url):
                raise HTTPException(
                    status_code=400,
                    detail=error_detail(
                        "SHARED_OBJECT_UNSUPPORTED",
                        "Shared object URLs are not supported for video download.",
                    ),
                )
        else:
            context = load_s3_context()
            s3_public = context["s3_public"]
            bucket = payload.video_bucket or context["bucket"]
            if not bucket:
                raise HTTPException(
                    status_code=400,
                    detail=error_detail(
                        "VIDEO_BUCKET_MISSING",
                        "video_bucket is required when using video_key.",
                    ),
                )
            expires_seconds = context["expires_seconds"]
            video_url = s3_public.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": video_url},
                ExpiresIn=expires_seconds,
            )
            video_bucket = bucket
            video_key = payload.video_url

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
        video_url=video_url,
        video_bucket=video_bucket,
        video_key=video_key,
        target=target,
        video_meta={},
        anchor={},
        player_ref=None,
        progress={"step": "CREATED", "pct": 0, "message": "Job created"},
        result={},
        error=None,
        failure_reason=normalize_failure_reason(None),
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    extract_preview_frames.delay(job.id)

    return ok_response({"job_id": job.id, "id": job.id, "status": job.status}, request)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    db: Session = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail=error_detail("JOB_NOT_FOUND", "Job not found"),
            )

        result_payload = normalize_payload(job.result)
        preview_frames = job.preview_frames or []
        context = None
        if result_payload or preview_frames:
            context = load_s3_context()
        if result_payload:
            result_payload = attach_presigned_urls(result_payload, context)
        if preview_frames:
            preview_frames = attach_presigned_urls(
                {"preview_frames": preview_frames}, context
            ).get("preview_frames", preview_frames)

        return ok_response(
            {
                "job_id": job.id,
                "id": job.id,
                "status": normalize_status(job.status),
                "category": job.category,
                "role": job.role,
                "video_url": job.video_url,
                "preview_frames": preview_frames,
                "progress": normalize_payload(job.progress),
                "result": result_payload,
                "error": job.error,
                "failure_reason": job.failure_reason,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            },
            request,
        )
    finally:
        db.close()


# ✅ endpoint leggero per polling
@router.get("/jobs/{job_id}/status")
def job_status(job_id: str, request: Request, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    return ok_response(
        {
            "job_id": job.id,
            "id": job.id,
            "status": normalize_status(job.status),
            "progress": normalize_payload(job.progress),
            "error": job.error,
            "failure_reason": job.failure_reason,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        },
        request,
    )


@router.get("/jobs/{job_id}/poll")
def job_poll(job_id: str, request: Request, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    result_payload = job.result if job.status == "COMPLETED" else None
    if result_payload:
        context = load_s3_context()
        result_payload = attach_presigned_urls(result_payload, context)

    return ok_response(
        {
            "job_id": job.id,
            "id": job.id,
            "status": normalize_status(job.status),
            "progress": normalize_payload(job.progress),
            "error": job.error,
            "failure_reason": job.failure_reason,
            "result": normalize_payload(result_payload),
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        },
        request,
    )


# ✅ result “pulito” (solo quando pronto)
@router.get("/jobs/{job_id}/result")
def job_result(job_id: str, request: Request, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    if job.status == "FAILED":
        raise HTTPException(
            status_code=409,
            detail=error_detail("JOB_FAILED", job.error or "Job failed"),
        )

    if job.status != "COMPLETED":
        raise HTTPException(
            status_code=409,
            detail=error_detail("JOB_NOT_COMPLETED", "Job not completed yet"),
        )

    result_payload = normalize_payload(job.result)
    if result_payload:
        context = load_s3_context()
        result_payload = attach_presigned_urls(result_payload, context)
    return ok_response({"job_id": job.id, "id": job.id, "result": result_payload}, request)


def _build_target_from_selections(payload: SelectionPayload) -> Dict[str, Any]:
    selections = []
    for selection in payload.selections:
        selection_data = selection.dict()
        selections.append(
            {
                "time_sec": selection_data["frame_time_sec"],
                "bbox": {
                    "x": selection_data["x"],
                    "y": selection_data["y"],
                    "w": selection_data["w"],
                    "h": selection_data["h"],
                },
            }
        )
    return {"selections": selections, "tracking": {"status": "PENDING"}}


def _save_target_selection(
    job_id: str, payload: SelectionPayload, request: Request, db: Session
):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )
    job.target = _build_target_from_selections(payload)

    if job.status == "WAITING_FOR_SELECTION":
        job.status = "CREATED"

    db.commit()
    db.refresh(job)
    return ok_response({"job_id": job.id, "id": job.id, "status": job.status}, request)


@router.post("/jobs/{job_id}/selection")
def save_selection(
    job_id: str, payload: SelectionPayload, request: Request, db: Session = Depends(get_db)
):
    return _save_target_selection(job_id, payload, request, db)


@router.post("/jobs/{job_id}/target")
def save_target(
    job_id: str, payload: SelectionPayload, request: Request, db: Session = Depends(get_db)
):
    return _save_target_selection(job_id, payload, request, db)


@router.post("/jobs/{job_id}/player-ref")
def save_player_ref(
    job_id: str, payload: PlayerRefPayload, request: Request, db: Session = Depends(get_db)
):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    if payload.w <= 0 or payload.h <= 0:
        raise HTTPException(
            status_code=400,
            detail=error_detail("INVALID_BBOX", "Invalid bbox dimensions"),
        )

    job.anchor = {
        "time_sec": payload.t,
        "bbox": {"x": payload.x, "y": payload.y, "w": payload.w, "h": payload.h},
    }
    job.player_ref = job.anchor["time_sec"]

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
    return ok_response(
        {"job_id": job.id, "id": job.id, "player_ref": job.player_ref}, request
    )


@router.get("/jobs/{job_id}/frames")
def get_frames(
    job_id: str, request: Request, count: int = 8, db: Session = Depends(get_db)
):
    if count < 1:
        raise HTTPException(
            status_code=400,
            detail=error_detail("INVALID_COUNT", "Count must be >= 1"),
        )

    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )
    context = load_s3_context()
    s3_internal = context["s3_internal"]
    s3_public = context["s3_public"]
    minio_bucket = context["bucket"]
    expires_seconds = context["expires_seconds"]

    ensure_ffmpeg_available()

    base_dir = Path("/tmp/fnh_jobs") / job_id
    input_path = base_dir / "input.mp4"
    frames_dir = base_dir / "frames"

    if not input_path.exists():
        video_source, source_bucket = resolve_job_video_source(job, minio_bucket)
        download_video(video_source, input_path, s3_internal, source_bucket)

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
            s3_public,
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
        raise HTTPException(
            status_code=500,
            detail=error_detail("FRAMES_EXTRACTION_FAILED", "No frames extracted"),
        )

    return ok_response({"count": len(frames), "frames": frames}, request)


@router.get("/jobs/{job_id}/frames/list")
def list_frames(job_id: str, request: Request, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    context = load_s3_context()
    s3_internal = context["s3_internal"]
    s3_public = context["s3_public"]
    bucket = context["bucket"]
    expires_seconds = context["expires_seconds"]

    prefix = f"jobs/{job_id}/frames/"
    paginator = s3_internal.get_paginator("list_objects_v2")

    items: List[Dict[str, Any]] = []
    image_suffixes = (".jpg", ".jpeg", ".png", ".webp")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if not key or key.endswith("/"):
                continue
            name = key.split("/")[-1]
            if not name.lower().endswith(image_suffixes):
                continue
            signed_url = presign_get_url(
                s3_public,
                bucket,
                key,
                expires_seconds,
            )
            items.append(
                {
                    "name": name,
                    "url": signed_url,
                    "key": key,
                    "width": None,
                    "height": None,
                    "time_sec": None,
                }
            )

    items.sort(key=lambda item: item["key"])
    logger.info(
        "frames/list bucket=%s prefix=%s found=%s keys=%s",
        bucket,
        prefix,
        len(items),
        [item["key"] for item in items[:3]],
    )
    return ok_response({"items": items}, request)


@router.post("/jobs/{job_id}/enqueue")
def enqueue_job(job_id: str, request: Request, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    # idempotente: se già avanzato, non reinvio
    if job.status in ["QUEUED", "RUNNING", "COMPLETED", "FAILED"]:
        return ok_response({"job_id": job.id, "id": job.id, "status": job.status}, request)

    if job.status not in ["WAITING_FOR_SELECTION", "CREATED", "WAITING_FOR_PLAYER"]:
        raise HTTPException(
            status_code=400,
            detail=error_detail("JOB_NOT_ENQUEUEABLE", "Job not enqueueable"),
        )

    selections = (job.target or {}).get("selections") or []
    player_ref = job.anchor or {}
    if len(selections) < 1 or not (
        player_ref.get("bbox") and player_ref.get("time_sec") is not None
    ):
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "PLAYER_SELECTION_REQUIRED", "Player selection is required"
            ),
        )

    job.status = "QUEUED"
    job.error = None
    db.commit()
    db.refresh(job)

    run_analysis.delay(job.id)

    return ok_response({"job_id": job.id, "id": job.id, "status": job.status}, request)
