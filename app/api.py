import json
import logging
import math
import os
import re
import socket
import subprocess
from functools import lru_cache
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from uuid import uuid4

import requests
from botocore.exceptions import ClientError

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.core.deps import get_db
from app.core.models import AnalysisJob
from app.core.normalizers import normalize_failure_reason
from app.schemas import (
    JobCreate,
    PlayerRefPayload,
    PickPlayerPayload,
    SelectionPayload,
    TargetSelectionPayload,
    TrackSelectionPayload,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MIN_FRAMES_FOR_EVAL = int(os.environ.get("MIN_FRAMES_FOR_EVAL", "30"))

POLLING_SAFE_STATUSES = {
    "WAITING_FOR_SELECTION",
    "WAITING_FOR_PLAYER",
    "WAITING_FOR_TARGET",
    "READY_TO_ENQUEUE",
    "CREATED",
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "PARTIAL",
    "FAILED",
}

S3_ENDPOINT_URL = (os.environ.get("S3_ENDPOINT_URL") or "").strip()
S3_PUBLIC_ENDPOINT_URL = (os.environ.get("S3_PUBLIC_ENDPOINT_URL") or "").strip()
S3_ACCESS_KEY = (os.environ.get("S3_ACCESS_KEY") or "").strip()
S3_SECRET_KEY = (os.environ.get("S3_SECRET_KEY") or "").strip()
S3_BUCKET = (os.environ.get("S3_BUCKET") or "").strip()
SIGNED_URL_EXPIRES_SECONDS = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))
FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


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


def _normalize_clip_asset(clip: Dict[str, Any]) -> Dict[str, Any]:
    start = clip.get("start_sec", clip.get("start"))
    end = clip.get("end_sec", clip.get("end"))
    url = clip.get("url") or clip.get("signed_url")
    label = clip.get("label")
    if label is None and start is not None and end is not None:
        label = f"{start}s-{end}s"
    return {
        "label": label,
        "url": url,
        "startSec": start,
        "endSec": end,
    }


def _build_result_assets(result_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    radar = result_payload.get("radar") or {}
    overall = (
        result_payload.get("overall_score")
        or result_payload.get("overallScore")
        or (result_payload.get("summary") or {}).get("overall_score")
    )
    role_score = (
        result_payload.get("role_score")
        or result_payload.get("roleScore")
        or (result_payload.get("summary") or {}).get("role_score")
    )
    result = {
        "overallScore": overall,
        "roleScore": role_score,
        "radar": radar,
    }

    assets_payload = result_payload.get("assets") or {}
    input_video_url = (
        assets_payload.get("input_video_url")
        or (assets_payload.get("input_video") or {}).get("url")
        or assets_payload.get("inputVideoUrl")
    )
    clips_raw = assets_payload.get("clips") or result_payload.get("clips") or []
    clips: List[Dict[str, Any]] = []
    for clip in clips_raw:
        if isinstance(clip, dict):
            clips.append(_normalize_clip_asset(clip))

    assets = {"inputVideoUrl": input_video_url, "clips": clips}
    return result, assets


def _build_candidate_payload(
    candidates_payload: Dict[str, Any], context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    candidates = candidates_payload.get("candidates") or []
    if not isinstance(candidates, list):
        return []
    s3_public = context["s3_public"]
    bucket_default = context["bucket"]
    expires_seconds = context["expires_seconds"]
    items: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        sample_frames = candidate.get("sample_frames") or []
        normalized_frames: List[Dict[str, Any]] = []
        for sample in sample_frames:
            if not isinstance(sample, dict):
                continue
            key = sample.get("key")
            if not key:
                continue
            bucket = sample.get("bucket") or bucket_default
            signed_url = presign_get_url(
                s3_public,
                bucket,
                key,
                expires_seconds,
            )
            frame_time_sec = float(
                sample.get("time_sec", sample.get("frame_time_sec", 0.0))
            )
            normalized_frames.append(
                {
                    "frame_time_sec": frame_time_sec,
                    "time_sec": frame_time_sec,
                    "image_url": signed_url,
                    "bbox": sample.get("bbox") or {},
                }
            )
        items.append(
            {
                "trackId": candidate.get("track_id"),
                "coveragePct": candidate.get("coverage_pct"),
                "stabilityScore": candidate.get("stability_score"),
                "avgBoxArea": candidate.get("avg_box_area"),
                "tier": candidate.get("tier"),
                "quality": candidate.get("quality"),
                "lowCoverage": candidate.get("lowCoverage"),
                "bestPreviewFrameKey": candidate.get("best_preview_frame_key"),
                "sampleFrames": normalized_frames,
            }
        )
    return items


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
    from app.workers.pipeline import get_s3_client

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


def _validate_filename(filename: str) -> None:
    if not filename or not FILENAME_PATTERN.match(filename):
        raise HTTPException(
            status_code=400,
            detail=error_detail("INVALID_FILENAME", "Invalid filename"),
        )


def _stream_s3_image(key: str) -> StreamingResponse:
    context = load_s3_context()
    s3_internal = context["s3_internal"]
    bucket = context["bucket"]
    try:
        response = s3_internal.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = (exc.response.get("Error") or {}).get("Code")
        if error_code in {"NoSuchKey", "404", "NotFound"}:
            raise HTTPException(
                status_code=404,
                detail=error_detail("FILE_NOT_FOUND", "File not found"),
            ) from exc
        raise

    headers = {
        "Cache-Control": "public, max-age=3600",
        "Content-Type": "image/jpeg",
    }
    return StreamingResponse(response["Body"], media_type="image/jpeg", headers=headers)


def _get_public_s3_client():
    from app.workers.pipeline import get_s3_client

    if not S3_PUBLIC_ENDPOINT_URL:
        raise HTTPException(
            status_code=500,
            detail=error_detail(
                "S3_CONFIG_MISSING",
                "Missing S3 env vars: S3_PUBLIC_ENDPOINT_URL",
            ),
        )
    return get_s3_client(S3_PUBLIC_ENDPOINT_URL)


@lru_cache(maxsize=1)
def _get_presign_s3_client():
    return _get_public_s3_client()


def _ensure_public_s3_client(s3_client):
    endpoint_url = getattr(getattr(s3_client, "meta", None), "endpoint_url", "") or ""
    if endpoint_url.rstrip("/") != S3_PUBLIC_ENDPOINT_URL.rstrip("/"):
        return _get_public_s3_client()
    return s3_client


def presign_get_url(
    s3_public,
    bucket: str,
    key: str,
    expires_seconds: int,
) -> str:
    presign_client = _get_presign_s3_client()
    return presign_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


def enqueue_job(job_id: str) -> None:
    from app.workers.pipeline import kickoff_job

    try:
        kickoff_job.delay(job_id)
    except Exception:
        logger.exception("create_job enqueue failed job_id=%s", job_id)


def build_public_video_url(
    job: AnalysisJob,
    context: Dict[str, Any] | None,
) -> str | None:
    if not job.video_url and not job.video_key:
        return None
    if context is None:
        return job.video_url
    bucket = job.video_bucket or context["bucket"]
    s3_public = context["s3_public"]
    expires_seconds = context["expires_seconds"]
    if job.video_key and bucket:
        return presign_get_url(s3_public, bucket, job.video_key, expires_seconds)
    if job.video_url and not job.video_url.lower().startswith(("http://", "https://")):
        if bucket:
            return presign_get_url(s3_public, bucket, job.video_url, expires_seconds)
    return job.video_url


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


def _hydrate_player_ref_sample_frames(
    player_ref: Dict[str, Any], context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    if not isinstance(player_ref, dict):
        return {}
    sample_frames = player_ref.get("sample_frames")
    if not isinstance(sample_frames, list):
        return player_ref

    if context is None:
        if not (
            S3_ENDPOINT_URL
            and S3_PUBLIC_ENDPOINT_URL
            and S3_ACCESS_KEY
            and S3_SECRET_KEY
            and S3_BUCKET
        ):
            return player_ref
        context = load_s3_context()

    s3_public = context["s3_public"]
    bucket_default = context["bucket"]
    expires_seconds = context["expires_seconds"]
    hydrated_frames: List[Dict[str, Any]] = []
    for frame in sample_frames:
        if not isinstance(frame, dict):
            hydrated_frames.append(frame)
            continue
        key = frame.get("key") or frame.get("s3_key")
        bucket = frame.get("bucket") or bucket_default
        if key and bucket:
            signed_url = presign_get_url(s3_public, bucket, key, expires_seconds)
            hydrated_frame = {
                **frame,
                "bucket": bucket,
                "key": key,
                "signed_url": signed_url,
            }
            hydrated_frame.setdefault("image_url", signed_url)
            hydrated_frames.append(hydrated_frame)
        else:
            hydrated_frames.append(frame)

    return {**player_ref, "sample_frames": hydrated_frames}


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
def create_job(
    payload: JobCreate,
    request: Request,
    db: Session = Depends(get_db),
):
    payload_dict = normalize_payload(payload)
    request_id = getattr(request.state, "request_id", None)
    logger.info(
        "JOBS_PAYLOAD_KEYS",
        extra={"payload_keys": list(payload_dict.keys()), "request_id": request_id},
    )
    job_id = str(uuid4())
    try:
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
            video_url = presign_get_url(
                s3_public, bucket, payload.video_key, expires_seconds
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
                video_url = presign_get_url(s3_public, bucket, video_url, expires_seconds)
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
            status="CREATED",
            category=payload.category,
            role=payload.role,
            video_url=video_url,
            video_bucket=video_bucket,
            video_key=video_key,
            target=target,
            video_meta={},
            anchor={},
            player_ref={},
            progress={"step": "CREATED", "pct": 0, "message": "Job created"},
            result={},
            warnings=[],
            error=None,
            failure_reason=normalize_failure_reason(None),
        )

        db.add(job)
        db.commit()
        db.refresh(job)
        from app.workers.pipeline import kickoff_job

        kickoff_job.delay(str(job.id))
        logger.info("Enqueued kickoff_job", extra={"job_id": str(job.id)})

        return ok_response(
            {"job_id": job.id, "id": job.id, "status": job.status}, request
        )
    except HTTPException:
        logger.exception("create_job failed job_id=%s", job_id)
        raise
    except Exception:
        logger.exception("create_job failed job_id=%s", job_id)
        raise


@router.get("/jobs")
def jobs_root(request: Request):
    return ok_response({"status": "ready"}, request)


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

        selections = (job.target or {}).get("selections") or []
        normalized_selections = [_normalize_selection(sel) for sel in selections]
        raw_player_ref = normalize_payload(job.player_ref) if job.player_ref else None
        if raw_player_ref:
            if context is None and raw_player_ref.get("sample_frames"):
                context = load_s3_context()
            player_ref = _hydrate_player_ref_sample_frames(raw_player_ref, context)
        else:
            player_ref = _normalize_player_ref(job.anchor or {})
        player_saved = _has_player_ref(job.player_ref)
        target_confirmed = bool((job.target or {}).get("confirmed"))

        needs_video_presign = bool(
            job.video_key
            or (
                job.video_url
                and not job.video_url.startswith(("http://", "https://"))
            )
        )
        if needs_video_presign and context is None:
            context = load_s3_context()
        public_video_url = build_public_video_url(job, context)

        result, assets = _build_result_assets(result_payload or {})
        if not assets.get("inputVideoUrl") and public_video_url:
            assets = {**assets, "inputVideoUrl": public_video_url}
        if preview_frames:
            preview_frames = [
                {**frame, "key": frame.get("key") or frame.get("s3_key")}
                if isinstance(frame, dict)
                else frame
                for frame in preview_frames
            ]
            preview_frames = _normalize_preview_frames(preview_frames)

        return ok_response(
            {
                "job_id": job.id,
                "id": job.id,
                "status": normalize_status(job.status),
                "category": job.category,
                "role": job.role,
                "video_url": public_video_url or job.video_url,
                "preview_frames": preview_frames,
                "progress": normalize_payload(job.progress),
                "warnings": list(job.warnings or []),
                "player_ref": player_ref,
                "playerSaved": player_saved,
                "target": {
                    **(job.target or {}),
                    "selections": normalized_selections,
                },
                "targetSaved": target_confirmed,
                "result": result,
                "assets": assets,
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
            "warnings": list(job.warnings or []),
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

    result_payload = normalize_payload(job.result)
    if result_payload:
        context = load_s3_context()
        result_payload = attach_presigned_urls(result_payload, context)
    result, assets = _build_result_assets(result_payload or {})

    return ok_response(
        {
            "job_id": job.id,
            "id": job.id,
            "status": normalize_status(job.status),
            "progress": normalize_payload(job.progress),
            "warnings": list(job.warnings or []),
            "error": job.error,
            "failure_reason": job.failure_reason,
            "result": result,
            "assets": assets,
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

    if job.status not in ("COMPLETED", "PARTIAL"):
        raise HTTPException(
            status_code=409,
            detail=error_detail("JOB_NOT_COMPLETED", "Job not completed yet"),
        )

    result_payload = normalize_payload(job.result)
    if result_payload:
        context = load_s3_context()
        result_payload = attach_presigned_urls(result_payload, context)
    result, assets = _build_result_assets(result_payload or {})
    return ok_response(
        {
            "job_id": job.id,
            "id": job.id,
            "result": result,
            "assets": assets,
            "warnings": list(job.warnings or []),
        },
        request,
    )


@router.get("/jobs/{job_id}/candidates")
def job_candidates(job_id: str, request: Request, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    result_payload = normalize_payload(job.result)
    candidates_payload_raw = result_payload.get("candidates") or {}
    if isinstance(candidates_payload_raw, list):
        candidates_payload = {"candidates": candidates_payload_raw}
    elif isinstance(candidates_payload_raw, dict):
        candidates_payload = candidates_payload_raw
    else:
        candidates_payload = {}
    if not candidates_payload:
        progress_step = (job.progress or {}).get("step")
        if progress_step == "TRACKING_CANDIDATES":
            status = "PROCESSING"
        elif progress_step in ("EXTRACTING_PREVIEWS", "PREVIEWS_READY"):
            status = "PENDING_QUEUE"
        else:
            status = "NOT_STARTED"
        return ok_response(
            {
                "status": status,
                "framesProcessed": 0,
                "candidates": [],
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
                "autodetection_status": "PROCESSING",
                "error_detail": None,
            },
            request,
        )

    context = load_s3_context()
    frames_processed = int(
        candidates_payload.get("framesProcessed")
        or candidates_payload.get("frames_processed")
        or result_payload.get("framesProcessed")
        or 0
    )
    autodetection = (
        candidates_payload.get("autodetection")
        or result_payload.get("autodetection")
        or {
            "totalTracks": 0,
            "rawTracks": 0,
            "primaryCount": 0,
            "secondaryCount": 0,
            "thresholdPct": 0.05,
            "minHits": int(os.environ.get("CANDIDATE_MIN_HITS", "2")),
            "minSeconds": float(os.environ.get("CANDIDATE_MIN_SECONDS", "0") or 0),
            "topN": int(os.environ.get("CANDIDATE_TOP_N", "5")),
        }
    )
    if "thresholdPct" not in autodetection:
        autodetection["thresholdPct"] = 0.05
    autodetection.setdefault(
        "minHits", int(os.environ.get("CANDIDATE_MIN_HITS", "2"))
    )
    autodetection.setdefault(
        "minSeconds", float(os.environ.get("CANDIDATE_MIN_SECONDS", "0") or 0)
    )
    autodetection.setdefault(
        "topN", int(os.environ.get("CANDIDATE_TOP_N", "5"))
    )
    if "totalTracks" not in autodetection:
        autodetection["totalTracks"] = (
            result_payload.get("totalTracks") or len(candidates_payload.get("candidates") or [])
        )
    autodetection.setdefault(
        "rawTracks", result_payload.get("rawTracks") or autodetection["totalTracks"]
    )
    if "primaryCount" not in autodetection:
        autodetection["primaryCount"] = result_payload.get("primaryCount") or 0
    if "secondaryCount" not in autodetection:
        autodetection["secondaryCount"] = result_payload.get("secondaryCount") or 0
    autodetection_status = candidates_payload.get("autodetection_status") or "PROCESSING"
    error_detail_payload = candidates_payload.get("error_detail")
    candidates = _build_candidate_payload(candidates_payload, context)
    if autodetection_status == "FAILED":
        status = "FAILED"
        candidates = []
    elif (
        autodetection_status in {"PROCESSING", "PENDING_QUEUE"}
        and not candidates
    ):
        status = "PROCESSING"
        candidates = []
    elif frames_processed < MIN_FRAMES_FOR_EVAL and not candidates:
        status = "PROCESSING"
        candidates = []
        autodetection_status = "PROCESSING"
    elif autodetection_status == "LOW_COVERAGE":
        status = "LOW_COVERAGE"
    else:
        status = "READY"
    return ok_response(
        {
            "status": status,
            "framesProcessed": frames_processed,
            "candidates": candidates,
            "autodetection": autodetection,
            "autodetection_status": autodetection_status,
            "error_detail": error_detail_payload,
        },
        request,
    )


@router.get("/jobs/{job_id}/candidates/{filename}")
def get_candidate_preview(job_id: str, filename: str) -> StreamingResponse:
    _validate_filename(filename)
    key = f"jobs/{job_id}/candidates/{filename}"
    return _stream_s3_image(key)


def _parse_track_selection_payload(payload: Any) -> TrackSelectionPayload:
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_PAYLOAD",
                "Invalid select-track payload",
                {"errors": [{"field": "payload", "message": "Payload must be an object"}]},
            ),
        )
    try:
        return TrackSelectionPayload.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_PAYLOAD",
                "Invalid select-track payload",
                {"errors": exc.errors()},
            ),
        ) from exc


def _parse_pick_player_payload(payload: Any) -> PickPlayerPayload:
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_PAYLOAD",
                "Invalid pick-player payload",
                {"errors": [{"field": "payload", "message": "Payload must be an object"}]},
            ),
        )
    try:
        return PickPlayerPayload.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_PAYLOAD",
                "Invalid pick-player payload",
                {"errors": exc.errors()},
            ),
        ) from exc


def _track_id_variants(value: Any) -> set[Any]:
    if value is None or isinstance(value, bool):
        return set()
    variants: set[Any] = set()
    if isinstance(value, int):
        variants.add(value)
        variants.add(str(value))
        return variants
    if isinstance(value, str):
        trimmed = value.strip()
        variants.add(trimmed)
        if trimmed.isdigit():
            variants.add(int(trimmed))
        return variants
    try:
        variants.add(str(value))
    except Exception:
        pass
    return variants


def _build_preview_lookup(preview_frames: list) -> Dict[str, Dict[str, Any]]:
    preview_lookup: Dict[str, Dict[str, Any]] = {}
    if isinstance(preview_frames, list):
        for frame in preview_frames:
            if not isinstance(frame, dict):
                continue
            key = frame.get("key") or frame.get("s3_key")
            if isinstance(key, str) and key:
                preview_lookup[key] = frame
                preview_lookup.setdefault(key.split("/")[-1], frame)
    return preview_lookup


def _find_preview_frame(preview_lookup: Dict[str, Dict[str, Any]], frame_key: str) -> Dict[str, Any] | None:
    if not frame_key:
        return None
    return preview_lookup.get(frame_key)


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


def _normalize_preview_frames(preview_frames: list) -> list:
    normalized_frames = []
    for frame in preview_frames or []:
        if not isinstance(frame, dict):
            normalized_frames.append(frame)
            continue
        tracks = frame.get("tracks")
        if not isinstance(tracks, list):
            tracks = []
        normalized_tracks = []
        for track in tracks:
            if not isinstance(track, dict):
                continue
            normalized_tracks.append(
                {
                    **track,
                    "bbox": _normalize_bbox_xywh(track.get("bbox") or {}),
                }
            )
        normalized_frames.append(
            {
                **frame,
                "tracks": normalized_tracks,
            }
        )
    return normalized_frames

def _bbox_iou(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    try:
        ax1 = float(box_a.get("x", 0.0))
        ay1 = float(box_a.get("y", 0.0))
        ax2 = ax1 + float(box_a.get("w", 0.0))
        ay2 = ay1 + float(box_a.get("h", 0.0))
        bx1 = float(box_b.get("x", 0.0))
        by1 = float(box_b.get("y", 0.0))
        bx2 = bx1 + float(box_b.get("w", 0.0))
        by2 = by1 + float(box_b.get("h", 0.0))
    except (TypeError, ValueError):
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom

@router.post("/jobs/{job_id}/select-track")
def select_track(
    job_id: str,
    request: Request,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    normalized_payload = _parse_track_selection_payload(payload)
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    result_payload = normalize_payload(job.result)
    candidates_payload_raw = result_payload.get("candidates") or {}
    if isinstance(candidates_payload_raw, list):
        candidates = candidates_payload_raw
    elif isinstance(candidates_payload_raw, dict):
        candidates = candidates_payload_raw.get("candidates") or []
    else:
        candidates = []

    selected_ids = _track_id_variants(normalized_payload.track_id)
    candidate = next(
        (
            item
            for item in candidates
            if isinstance(item, dict)
            and _track_id_variants(item.get("track_id")) & selected_ids
        ),
        None,
    )
    if not candidate:
        raise HTTPException(
            status_code=404,
            detail=error_detail("TRACK_NOT_FOUND", "Track not found"),
        )

    player_ref_payload: Dict[str, Any] = {
        "track_id": candidate.get("track_id", normalized_payload.track_id),
        "tier": candidate.get("tier") or "UNKNOWN",
    }
    for key, value in candidate.items():
        if key == "track_id":
            continue
        player_ref_payload[key] = value
    if normalized_payload.selection:
        player_ref_payload.update(
            {
                "t": float(normalized_payload.selection.frame_time_sec),
                "x": float(normalized_payload.selection.x),
                "y": float(normalized_payload.selection.y),
                "w": float(normalized_payload.selection.w),
                "h": float(normalized_payload.selection.h),
            }
        )
    elif candidate.get("sample_frames"):
        sample_frames = candidate.get("sample_frames")
        if isinstance(sample_frames, list) and sample_frames:
            sample = sample_frames[0]
            if isinstance(sample, dict):
                bbox = sample.get("bbox") or {}
                if {"x", "y", "w", "h"}.issubset(bbox):
                    time_sec = sample.get("time_sec") or sample.get("frame_time_sec")
                    if time_sec is not None:
                        player_ref_payload.update(
                            {
                                "t": float(time_sec),
                                "x": float(bbox["x"]),
                                "y": float(bbox["y"]),
                                "w": float(bbox["w"]),
                                "h": float(bbox["h"]),
                            }
                        )
    job.player_ref = player_ref_payload
    job.status = "WAITING_FOR_TARGET"
    job.updated_at = datetime.now(timezone.utc)

    progress = job.progress or {}
    current_pct = progress.get("pct") or 0
    try:
        current_pct = int(current_pct)
    except (TypeError, ValueError):
        current_pct = 0
    job.progress = {
        **progress,
        "step": "WAITING_FOR_TARGET",
        "pct": max(current_pct, 14),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    db.commit()
    db.refresh(job)
    target_suggestion = None
    if normalized_payload.selection:
        selection = normalized_payload.selection
        target_suggestion = {
            "time_sec": selection.frame_time_sec,
            "bbox": {
                "x": selection.x,
                "y": selection.y,
                "w": selection.w,
                "h": selection.h,
            },
        }
    return ok_response(
        {
            "job_id": job.id,
            "id": job.id,
            "status": job.status,
            "playerSaved": True,
            "playerRef": job.player_ref,
            "player_ref": job.player_ref,
            "target": None,
            "target_suggestion": target_suggestion,
            "progress": normalize_payload(job.progress),
        },
        request,
    )


@router.post("/jobs/{job_id}/pick-player")
def pick_player(
    job_id: str,
    request: Request,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    normalized_payload = _parse_pick_player_payload(payload)
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    preview_lookup = _build_preview_lookup(job.preview_frames or [])
    selected_frame = _find_preview_frame(preview_lookup, normalized_payload.frame_key)
    if not selected_frame:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_SELECTION",
                "frame_key not present in preview frames",
                {"frame_key": normalized_payload.frame_key},
            ),
        )

    tracks = selected_frame.get("tracks") or []
    selected_ids = _track_id_variants(normalized_payload.track_id)
    selected_track = next(
        (
            track
            for track in tracks
            if isinstance(track, dict)
            and _track_id_variants(track.get("track_id")) & selected_ids
        ),
        None,
    )
    if not selected_track:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_SELECTION",
                "track_id not present in frame",
                {
                    "frame_key": normalized_payload.frame_key,
                    "track_id": normalized_payload.track_id,
                },
            ),
        )

    frame_key = selected_frame.get("key") or selected_frame.get("s3_key")
    time_sec = selected_frame.get("time_sec")
    bbox = selected_track.get("bbox") or {}
    player_ref_payload = {
        "track_id": selected_track.get("track_id", normalized_payload.track_id),
        "tier": selected_track.get("tier") or "UNKNOWN",
        "best_preview_frame_key": frame_key,
        "best_time_sec": float(time_sec) if time_sec is not None else None,
        "bbox": bbox,
        "selection_source": "preview_frame_pick",
    }
    if time_sec is not None and {"x", "y", "w", "h"}.issubset(bbox):
        player_ref_payload.update(
            {
                "t": float(time_sec),
                "x": float(bbox["x"]),
                "y": float(bbox["y"]),
                "w": float(bbox["w"]),
                "h": float(bbox["h"]),
            }
        )
    job.player_ref = player_ref_payload

    target_payload = {**(job.target or {})}
    target_payload["confirmed"] = False
    target_payload["selection"] = {
        "frame_key": frame_key,
        "time_sec": float(time_sec) if time_sec is not None else None,
        "bbox": bbox,
    }
    job.target = target_payload
    job.status = "WAITING_FOR_TARGET"
    job.updated_at = datetime.now(timezone.utc)

    progress = job.progress or {}
    current_pct = progress.get("pct") or 0
    try:
        current_pct = int(current_pct)
    except (TypeError, ValueError):
        current_pct = 0
    job.progress = {
        **progress,
        "step": "WAITING_FOR_TARGET_CONFIRMATION",
        "pct": max(current_pct, 14),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    db.commit()
    db.refresh(job)

    return ok_response(
        {
            "job_id": job.id,
            "id": job.id,
            "status": job.status,
            "player_ref": job.player_ref,
            "target": job.target,
            "progress": normalize_payload(job.progress),
        },
        request,
    )


@router.post("/jobs/{job_id}/analyze-player")
def analyze_player(
    job_id: str,
    request: Request,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    normalized_payload = _parse_pick_player_payload(payload)
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    preview_lookup = _build_preview_lookup(job.preview_frames or [])
    selected_frame = _find_preview_frame(preview_lookup, normalized_payload.frame_key)
    if not selected_frame:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_SELECTION",
                "frame_key not present in preview frames",
                {"frame_key": normalized_payload.frame_key},
            ),
        )

    tracks = selected_frame.get("tracks") or []
    selected_ids = _track_id_variants(normalized_payload.track_id)
    selected_track = next(
        (
            track
            for track in tracks
            if isinstance(track, dict)
            and _track_id_variants(track.get("track_id")) & selected_ids
        ),
        None,
    )
    if not selected_track:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_SELECTION",
                "track_id not present in frame",
                {
                    "frame_key": normalized_payload.frame_key,
                    "track_id": normalized_payload.track_id,
                },
            ),
        )

    track_id = selected_track.get("track_id", normalized_payload.track_id)
    track_id_key = str(track_id)
    existing_result = job.result or {}
    player_runs = (
        existing_result.get("player_runs")
        if isinstance(existing_result, dict)
        else None
    )
    if isinstance(player_runs, dict):
        existing_run = player_runs.get(track_id_key)
        if isinstance(existing_run, dict) and existing_run.get("status") in {
            "COMPLETED",
            "PARTIAL",
        }:
            return ok_response(
                {
                    "job_id": job.id,
                    "id": job.id,
                    "status": existing_run.get("status"),
                    "progress": normalize_payload(job.progress),
                    "result": existing_run.get("result"),
                },
                request,
            )

    current_track = (
        (job.player_ref or {}).get("track_id") if isinstance(job.player_ref, dict) else None
    )
    if (
        job.status in {"COMPLETED", "PARTIAL"}
        and _track_id_variants(current_track) & _track_id_variants(track_id)
    ):
        return ok_response(
            {
                "job_id": job.id,
                "id": job.id,
                "status": job.status,
                "progress": normalize_payload(job.progress),
                "result": normalize_payload(job.result),
            },
            request,
        )
    if (
        job.status in {"QUEUED", "RUNNING"}
        and _track_id_variants(current_track) & _track_id_variants(track_id)
    ):
        return ok_response(
            {
                "job_id": job.id,
                "id": job.id,
                "status": job.status,
                "progress": normalize_payload(job.progress),
            },
            request,
        )

    frame_key = selected_frame.get("key") or selected_frame.get("s3_key")
    time_sec = selected_frame.get("time_sec")
    bbox = _normalize_bbox_xywh(selected_track.get("bbox") or {})
    player_ref_payload = {
        "track_id": track_id,
        "tier": selected_track.get("tier") or "UNKNOWN",
        "best_preview_frame_key": frame_key,
        "best_time_sec": float(time_sec) if time_sec is not None else None,
        "bbox": bbox,
        "selection_source": "preview_frame_click",
    }
    if time_sec is not None and {"x", "y", "w", "h"}.issubset(bbox):
        player_ref_payload.update(
            {
                "t": float(time_sec),
                "x": float(bbox["x"]),
                "y": float(bbox["y"]),
                "w": float(bbox["w"]),
                "h": float(bbox["h"]),
            }
        )
    job.player_ref = player_ref_payload

    target_payload = {
        "confirmed": True,
        "selection": {
            "frame_key": frame_key,
            "time_sec": float(time_sec) if time_sec is not None else None,
            "bbox": bbox,
        },
        "selections": [
            {
                "frame_key": frame_key,
                "frame_time_sec": float(time_sec) if time_sec is not None else None,
                "x": float(bbox.get("x", 0.0)),
                "y": float(bbox.get("y", 0.0)),
                "w": float(bbox.get("w", 0.0)),
                "h": float(bbox.get("h", 0.0)),
            }
        ],
        "tracking": {"status": "PENDING"},
    }
    job.target = target_payload

    job.status = "QUEUED"
    job.error = None
    progress = job.progress or {}
    current_pct = progress.get("pct") or 0
    try:
        current_pct = int(current_pct)
    except (TypeError, ValueError):
        current_pct = 0
    job.progress = {
        **progress,
        "step": "QUEUED",
        "pct": max(current_pct, 20),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    player_runs = player_runs if isinstance(player_runs, dict) else {}
    player_runs[track_id_key] = {
        "track_id": track_id,
        "status": "QUEUED",
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    job.result = {
        **existing_result,
        "player_runs": player_runs,
    }
    job.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(job)

    from app.workers.pipeline import run_analysis

    run_analysis.delay(job.id)

    return ok_response(
        {
            "job_id": job.id,
            "id": job.id,
            "status": job.status,
            "progress": normalize_payload(job.progress),
        },
        request,
    )


@router.post("/jobs/{job_id}/select-target")
def select_target(
    job_id: str,
    request: Request,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
):
    return _confirm_target_selection(job_id, payload, request, db)


def _build_target_from_selections(payload: SelectionPayload) -> Dict[str, Any]:
    selections = []
    selection_payload = None
    for selection in payload.selections:
        selection_data = selection.dict()
        selection_entry = {
            "frame_time_sec": selection_data["frame_time_sec"],
            "x": selection_data["x"],
            "y": selection_data["y"],
            "w": selection_data["w"],
            "h": selection_data["h"],
        }
        if selection_data.get("frame_key"):
            selection_entry["frame_key"] = selection_data["frame_key"]
        selections.append(selection_entry)
        if selection_payload is None:
            selection_payload = {
                "frame_key": selection_data.get("frame_key"),
                "time_sec": selection_data["frame_time_sec"],
                "bbox": {
                    "x": selection_data["x"],
                    "y": selection_data["y"],
                    "w": selection_data["w"],
                    "h": selection_data["h"],
                },
            }
    return {
        "confirmed": True,
        "selection": selection_payload,
        "selections": selections,
        "tracking": {"status": "PENDING"},
    }


class _InvalidTargetPayload(Exception):
    def __init__(self, missing: List[str]) -> None:
        super().__init__("Invalid target payload")
        self.missing = missing


def _preview_time_epsilon(frames: List[Dict[str, Any]]) -> float:
    if len(frames) < 2:
        return 0.5
    times = sorted(
        float(frame.get("time_sec") or 0.0)
        for frame in frames
        if isinstance(frame, dict)
    )
    diffs = [
        next_time - prev_time
        for prev_time, next_time in zip(times, times[1:])
        if next_time > prev_time
    ]
    if not diffs:
        return 0.5
    return max(0.25, min(diffs) / 2.0)


def _normalize_target_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, TargetSelectionPayload):
        return {
            "frame_key": payload.frame_key,
            "time_sec": payload.time_sec,
            "bbox": payload.bbox,
            "track_id": payload.track_id,
            "force": payload.force,
        }
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        raise _InvalidTargetPayload(["payload"])

    frame_key = payload.get("frame_key") or payload.get("frameKey") or payload.get("key")
    time_sec = (
        payload.get("time_sec")
        or payload.get("timeSec")
        or payload.get("frame_time_sec")
        or payload.get("frameTimeSec")
    )
    track_id = payload.get("track_id") or payload.get("trackId")
    force = bool(payload.get("force")) if "force" in payload else False

    bbox = payload.get("bbox")
    if not isinstance(bbox, dict):
        bbox = {
            "x": payload.get("x"),
            "y": payload.get("y"),
            "w": payload.get("w"),
            "h": payload.get("h"),
        }

    missing: List[str] = []
    if frame_key is None and time_sec is None:
        missing.extend(["frame_key", "time_sec"])
    if track_id is None:
        missing.append("track_id")
    if not isinstance(bbox, dict):
        missing.append("bbox")
    else:
        for key in ("x", "y", "w", "h"):
            if bbox.get(key) is None:
                missing.append(f"bbox.{key}")

    if missing:
        raise _InvalidTargetPayload(sorted(set(missing)))

    try:
        time_value = float(time_sec) if time_sec is not None else None
    except (TypeError, ValueError):
        raise _InvalidTargetPayload(["time_sec"])

    bbox = PlayerRefPayload._validate_bbox_xywh(
        {
            "x": float(bbox["x"]),
            "y": float(bbox["y"]),
            "w": float(bbox["w"]),
            "h": float(bbox["h"]),
        }
    )

    return {
        "frame_key": frame_key,
        "time_sec": time_value,
        "bbox": bbox,
        "track_id": track_id,
        "force": force,
    }


def _error_response(
    code: str,
    message: str,
    request: Request,
    status_code: int,
    details: dict | None = None,
) -> JSONResponse:
    meta = build_meta(request)
    error_payload: Dict[str, Any] = {"code": code, "message": message}
    if details:
        error_payload["details"] = details
    payload = {
        "ok": False,
        "error": error_payload,
        "meta": meta,
        "request_id": meta.get("request_id"),
    }
    return JSONResponse(status_code=status_code, content=payload)


def _confirm_target_selection(
    job_id: str,
    payload: Any,
    request: Request,
    db: Session,
):
    request_id = getattr(request.state, "request_id", None)
    try:
        job = db.get(AnalysisJob, job_id)
        if not job:
            return _error_response(
                "JOB_NOT_FOUND",
                "Job not found",
                request,
                status_code=404,
            )

        normalized_payload = _normalize_target_payload(payload)
        logger.info(
            "confirm-target payload job_id=%s request_id=%s payload=%s",
            job_id,
            request_id,
            normalized_payload,
        )

        preview_frames = job.preview_frames or []
        preview_lookup = _build_preview_lookup(preview_frames)

        frame_key = normalized_payload["frame_key"]
        time_sec = normalized_payload["time_sec"]
        bbox = normalized_payload["bbox"]
        track_id = normalized_payload["track_id"]
        force = normalized_payload["force"]

        logger.info(
            "confirm-target frame_key_received job_id=%s frame_key=%s time_sec=%s",
            job_id,
            frame_key,
            time_sec,
        )

        selected_frame = None
        if frame_key:
            selected_frame = preview_lookup.get(frame_key)
            if selected_frame is None and preview_lookup:
                logger.info(
                    "confirm-target frame_not_found job_id=%s frame_key=%s",
                    job_id,
                    frame_key,
                )
                return _error_response(
                    "INVALID_FRAME_KEY",
                    "Frame key not found",
                    request,
                    status_code=400,
                    details={"frame_key": frame_key},
                )
            if selected_frame:
                logger.info(
                    "confirm-target frame_found job_id=%s frame_key=%s",
                    job_id,
                    selected_frame.get("key"),
                )
                frame_key = selected_frame.get("key") or frame_key
                if selected_frame.get("time_sec") is not None:
                    time_sec = float(selected_frame.get("time_sec"))
        elif preview_lookup and time_sec is not None:
            selected_frame = min(
                preview_lookup.values(),
                key=lambda item: abs(float(item.get("time_sec") or 0.0) - float(time_sec)),
            )
            closest_time = float(selected_frame.get("time_sec") or 0.0)
            if abs(closest_time - float(time_sec)) > _preview_time_epsilon(preview_frames):
                logger.info(
                    "confirm-target frame_time_mismatch job_id=%s time_sec=%s",
                    job_id,
                    time_sec,
                )
                return _error_response(
                    "INVALID_FRAME_KEY",
                    "Frame time does not match preview frames",
                    request,
                    status_code=400,
                    details={"time_sec": time_sec},
                )
            frame_key = selected_frame.get("key") or frame_key
            time_sec = float(selected_frame.get("time_sec") or time_sec)
        elif time_sec is not None and not preview_lookup:
            return _error_response(
                "INVALID_PAYLOAD",
                "frame_key is required when preview frames are unavailable",
                request,
                status_code=400,
                details={"missing": ["frame_key"]},
            )

        if time_sec is None:
            return _error_response(
                "INVALID_PAYLOAD",
                "Missing time_sec for target selection",
                request,
                status_code=400,
                details={"missing": ["time_sec"]},
            )

        logger.info(
            "confirm-target player_ref job_id=%s present=%s track_id=%s",
            job_id,
            bool(job.player_ref),
            track_id,
        )

        if selected_frame:
            tracks = selected_frame.get("tracks") or []
            if not tracks:
                logger.info(
                    "confirm-target no_tracks_in_frame job_id=%s frame_key=%s",
                    job_id,
                    frame_key,
                )
                if not force:
                    return _error_response(
                        "TRACK_NOT_IN_FRAME",
                        "Selected track not present in frame",
                        request,
                        status_code=409,
                        details={
                            "track_id": track_id,
                            "frame_key": frame_key,
                        },
                    )

            track = next(
                (
                    track
                    for track in tracks
                    if isinstance(track, dict)
                    and _track_id_variants(track.get("track_id"))
                    & _track_id_variants(track_id)
                ),
                None,
            )
            if track is None:
                logger.info(
                    "confirm-target track_not_in_frame job_id=%s frame_key=%s track_id=%s",
                    job_id,
                    frame_key,
                    track_id,
                )
                if not force:
                    return _error_response(
                        "TRACK_NOT_IN_FRAME",
                        "Selected track not present in frame",
                        request,
                        status_code=409,
                        details={
                            "track_id": track_id,
                            "frame_key": frame_key,
                        },
                    )
            else:
                track_bbox = track.get("bbox") or {}
                logger.info(
                    "confirm-target track_bbox_found job_id=%s frame_key=%s",
                    job_id,
                    frame_key,
                )
                iou = _bbox_iou(bbox, track_bbox)
                logger.info(
                    "confirm-target iou_calculated job_id=%s iou=%s",
                    job_id,
                    iou,
                )
                if iou < 0.2 and not force:
                    return _error_response(
                        "TARGET_MISMATCH",
                        "Target box does not match selected player in this frame.",
                        request,
                        status_code=409,
                        details={
                            "track_id": track_id,
                            "iou": iou,
                        },
                    )

        selection_payload = {
            "frame_key": frame_key,
            "time_sec": float(time_sec),
            "bbox": bbox,
        }

        existing_selection = (job.target or {}).get("selection") or {}
        if (job.target or {}).get("confirmed") and _selection_matches(
            existing_selection, selection_payload
        ):
            logger.info(
                "confirm-target idempotent job_id=%s frame_key=%s",
                job_id,
                frame_key,
            )
            return ok_response(
                {
                    "job_id": job.id,
                    "id": job.id,
                    "status": job.status,
                    "target": job.target,
                    "targetRef": job.target,
                    "warnings": list(job.warnings or []),
                    "progress": normalize_payload(job.progress),
                },
                request,
            )

        selections_payload = [
            {
                "frame_key": frame_key,
                "frame_time_sec": float(time_sec),
                "x": bbox["x"],
                "y": bbox["y"],
                "w": bbox["w"],
                "h": bbox["h"],
            }
        ]
        target_payload = {**(job.target or {})}
        target_payload["confirmed"] = True
        target_payload["selection"] = selection_payload
        target_payload["selections"] = selections_payload
        target_payload.setdefault("tracking", {"status": "PENDING"})
        job.target = target_payload
        job.status = "READY_TO_ENQUEUE"
        job.updated_at = datetime.now(timezone.utc)

        warnings: List[str] = list(job.warnings or [])
        if "TARGET_MISMATCH" in warnings and force:
            warnings = [warning for warning in warnings if warning != "TARGET_MISMATCH"]
        job.warnings = warnings

        progress = job.progress or {}
        current_pct = progress.get("pct") or 0
        try:
            current_pct = int(current_pct)
        except (TypeError, ValueError):
            current_pct = 0
        job.progress = {
            **progress,
            "step": "READY_TO_ENQUEUE",
            "pct": max(current_pct, 16),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        db.commit()
        db.refresh(job)

        return ok_response(
            {
                "job_id": job.id,
                "id": job.id,
                "status": job.status,
                "target": job.target,
                "targetRef": job.target,
                "warnings": list(job.warnings or []),
                "progress": normalize_payload(job.progress),
            },
            request,
        )
    except _InvalidTargetPayload as exc:
        logger.info(
            "confirm-target invalid_payload job_id=%s payload=%s missing=%s",
            job_id,
            payload,
            exc.missing,
        )
        return _error_response(
            "INVALID_PAYLOAD",
            "Missing required fields",
            request,
            status_code=400,
            details={"missing": exc.missing},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "confirm-target failed job_id=%s request_id=%s payload=%s",
            job_id,
            request_id,
            payload,
        )
        return _error_response(
            "INTERNAL_ERROR",
            str(exc),
            request,
            status_code=500,
            details={"hint": "Unexpected error while confirming target."},
        )


def _normalize_selection(selection: Dict[str, Any]) -> Dict[str, float]:
    if "frame_time_sec" in selection or "frame_key" in selection:
        frame_time_sec = float(selection.get("frame_time_sec", 0.0))
        normalized = {
            "frame_time_sec": frame_time_sec,
            "time_sec": frame_time_sec,
            "x": float(selection.get("x", 0.0)),
            "y": float(selection.get("y", 0.0)),
            "w": float(selection.get("w", 0.0)),
            "h": float(selection.get("h", 0.0)),
        }
        if selection.get("frame_key"):
            normalized["frame_key"] = selection.get("frame_key")
        return normalized
    bbox = selection.get("bbox") or {}
    frame_time_sec = float(selection.get("time_sec", 0.0))
    normalized = {
        "frame_time_sec": frame_time_sec,
        "time_sec": frame_time_sec,
        "x": float(bbox.get("x", 0.0)),
        "y": float(bbox.get("y", 0.0)),
        "w": float(bbox.get("w", 0.0)),
        "h": float(bbox.get("h", 0.0)),
    }
    if selection.get("frame_key"):
        normalized["frame_key"] = selection.get("frame_key")
    return normalized


def _selection_matches(existing: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
    if not isinstance(existing, dict) or not isinstance(incoming, dict):
        return False
    if existing.get("frame_key") != incoming.get("frame_key"):
        return False
    existing_time = existing.get("time_sec")
    incoming_time = incoming.get("time_sec")
    if existing_time is not None and incoming_time is not None:
        if not math.isclose(float(existing_time), float(incoming_time), rel_tol=1e-3):
            return False
    existing_bbox = existing.get("bbox") or {}
    incoming_bbox = incoming.get("bbox") or {}
    for key in ("x", "y", "w", "h"):
        if key not in existing_bbox or key not in incoming_bbox:
            return False
        if not math.isclose(
            float(existing_bbox[key]), float(incoming_bbox[key]), rel_tol=1e-3
        ):
            return False
    return True


def _normalize_player_ref(anchor: Dict[str, Any]) -> Dict[str, float] | None:
    if not anchor or isinstance(anchor, str):
        return None

    def f(v: Any) -> float:
        try:
            if v is None:
                return 0.0
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    if {"t", "x", "y", "w", "h"}.issubset(anchor.keys()):
        return {
            "t": f(anchor.get("t")),
            "x": f(anchor.get("x")),
            "y": f(anchor.get("y")),
            "w": f(anchor.get("w")),
            "h": f(anchor.get("h")),
        }
    bbox = anchor.get("bbox") or {}
    if not bbox:
        return None
    return {
        "t": f(anchor.get("time_sec")),
        "x": f(bbox.get("x")),
        "y": f(bbox.get("y")),
        "w": f(bbox.get("w")),
        "h": f(bbox.get("h")),
    }


def _has_player_ref(player_ref: Any) -> bool:
    if not isinstance(player_ref, dict) or not player_ref:
        return False
    if "track_id" in player_ref:
        return True
    if {"t", "x", "y", "w", "h"}.issubset(player_ref.keys()):
        return True
    bbox = player_ref.get("bbox")
    if isinstance(bbox, dict) and {"x", "y", "w", "h"}.issubset(bbox):
        return True
    return False


def _save_target_selection(
    job_id: str, payload: SelectionPayload, request: Request, db: Session
):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )
    target_payload = _build_target_from_selections(payload)
    preview_frames = job.preview_frames or []
    preview_lookup: Dict[str, Dict[str, Any]] = {}
    if isinstance(preview_frames, list):
        for frame in preview_frames:
            if not isinstance(frame, dict):
                continue
            key = frame.get("key") or frame.get("s3_key")
            if isinstance(key, str) and key:
                preview_lookup[key] = frame
                preview_lookup.setdefault(key.split("/")[-1], frame)
    if preview_lookup:
        for selection in target_payload.get("selections", []):
            if not isinstance(selection, dict):
                continue
            if selection.get("frame_key"):
                continue
            time_sec = selection.get("frame_time_sec")
            if time_sec is None:
                continue
            selected_frame = min(
                preview_lookup.values(),
                key=lambda item: abs(
                    float(item.get("time_sec") or 0.0) - float(time_sec)
                ),
            )
            if selected_frame.get("key"):
                selection["frame_key"] = selected_frame.get("key")
        selection_payload = target_payload.get("selection")
        if isinstance(selection_payload, dict) and not selection_payload.get(
            "frame_key"
        ):
            time_sec = selection_payload.get("time_sec")
            if time_sec is not None:
                selected_frame = min(
                    preview_lookup.values(),
                    key=lambda item: abs(
                        float(item.get("time_sec") or 0.0) - float(time_sec)
                    ),
                )
                if selected_frame.get("key"):
                    selection_payload["frame_key"] = selected_frame.get("key")
    job.target = target_payload

    player_ref = job.player_ref or job.anchor or {}
    has_player_ref = _has_player_ref(player_ref)
    if has_player_ref:
        job.status = "READY_TO_ENQUEUE"
    else:
        job.status = "WAITING_FOR_PLAYER"

    progress = job.progress or {}
    current_pct = progress.get("pct") or 0
    try:
        current_pct = int(current_pct)
    except (TypeError, ValueError):
        current_pct = 0
    if job.status == "READY_TO_ENQUEUE":
        progress_step = "READY_TO_ENQUEUE"
        progress_pct = max(current_pct, 16)
    else:
        progress_step = "WAITING_FOR_PLAYER"
        progress_pct = max(current_pct, 10)
    job.progress = {
        **progress,
        "step": progress_step,
        "pct": progress_pct,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

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
    job_id: str, request: Request, payload: dict = Body(...), db: Session = Depends(get_db)
):
    return _confirm_target_selection(job_id, payload, request, db)


@router.post("/jobs/{job_id}/player-ref")
async def save_player_ref(
    job_id: str,
    request: Request,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
) -> dict:

    raw_body = await request.body()
    if raw_body:
        logger.info("player-ref body=%s", raw_body.decode("utf-8", errors="replace"))
    else:
        logger.info("player-ref body=<empty>")
    logger.info("player-ref raw_payload=%s", payload)
    try:
        # --- compat: accept flat payload from UI ---
        if isinstance(payload, dict):
            # if payload already has bbox:{x,y,w,h} normalize to bbox_xywh
            if (
                "bbox" in payload
                and "bbox_xywh" not in payload
                and isinstance(payload.get("bbox"), dict)
            ):
                b = payload.get("bbox") or {}
                if all(k in b for k in ("x", "y", "w", "h")):
                    payload["bbox_xywh"] = b

            # accept camelCase flat payload too
            if "frameTimeSec" in payload and all(
                k in payload for k in ("x", "y", "w", "h")
            ):
                payload = {
                    "frame_time_sec": payload.get("frameTimeSec"),
                    "bbox_xywh": {
                        "x": payload.get("x"),
                        "y": payload.get("y"),
                        "w": payload.get("w"),
                        "h": payload.get("h"),
                    },
                }

            # accept snake_case flat payload too
            elif "frame_time_sec" in payload and all(
                k in payload for k in ("x", "y", "w", "h")
            ):
                payload = {
                    "frame_time_sec": payload.get("frame_time_sec"),
                    "bbox_xywh": {
                        "x": payload.get("x"),
                        "y": payload.get("y"),
                        "w": payload.get("w"),
                        "h": payload.get("h"),
                    },
                }
        # --- end compat ---
        normalized_payload = PlayerRefPayload.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    logger.info("player-ref normalized_payload=%s", normalized_payload.model_dump())
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    try:
        bbox_xywh = normalized_payload.bbox_xywh or None
        bbox_xyxy = normalized_payload.bbox_xyxy or None

        # Se arriva solo bbox_xyxy, converti in xywh
        if (not bbox_xywh) and bbox_xyxy:
            x1 = float(bbox_xyxy.get("x1", 0.0))
            y1 = float(bbox_xyxy.get("y1", 0.0))
            x2 = float(bbox_xyxy.get("x2", 0.0))
            y2 = float(bbox_xyxy.get("y2", 0.0))
            bbox_xywh = {
                "x": x1,
                "y": y1,
                "w": max(0.0, x2 - x1),
                "h": max(0.0, y2 - y1),
            }

        # Se ancora niente bbox valida -> 422 (non 500)
        if not bbox_xywh or not all(k in bbox_xywh for k in ("x", "y", "w", "h")):
            raise HTTPException(
                status_code=422,
                detail=error_detail("INVALID_BBOX", "Missing bbox_xywh (or bbox_xyxy)"),
            )

        t0 = normalized_payload.frame_time_sec or 0.0
        player_ref_payload = {
            "t": float(t0),
            "x": bbox_xywh["x"],
            "y": bbox_xywh["y"],
            "w": bbox_xywh["w"],
            "h": bbox_xywh["h"],
        }
        job.player_ref = player_ref_payload
        job.anchor = {
            "time_sec": float(t0),
            "bbox": bbox_xywh,
            "bbox_xyxy": bbox_xyxy,
        }

        target_payload = job.target or {}
        selections = target_payload.get("selections") or []
        target_confirmed = bool(target_payload.get("confirmed"))
        if selections and target_confirmed:
            job.status = "READY_TO_ENQUEUE"
        elif selections:
            job.status = "WAITING_FOR_TARGET"
        else:
            job.status = "WAITING_FOR_TARGET"

        progress = job.progress or {}
        current_pct = progress.get("pct") or 0
        try:
            current_pct = int(current_pct)
        except (TypeError, ValueError):
            current_pct = 0
        if job.status == "READY_TO_ENQUEUE":
            progress_step = "READY_TO_ENQUEUE"
            progress_pct = max(current_pct, 16)
        elif job.status == "WAITING_FOR_TARGET":
            progress_step = "WAITING_FOR_TARGET"
            progress_pct = max(current_pct, 14)
        else:
            progress_step = "PLAYER_SELECTED"
            progress_pct = max(current_pct, 12)
        job.progress = {
            **progress,
            "step": progress_step,
            "pct": progress_pct,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        db.commit()
        db.refresh(job)
        normalized_player_ref = _normalize_player_ref(job.player_ref or {})
        if normalized_player_ref is None:
            raise HTTPException(
                status_code=500,
                detail=error_detail(
                    "PLAYER_REF_SAVE_FAILED",
                    "player_ref missing after save",
                ),
            )
        return ok_response(
            {
                "job_id": job.id,
                "id": job.id,
                "player_ref": normalized_player_ref,
            },
            request,
        )
    except Exception:
        logger.exception("player-ref failed job_id=%s payload=%s", job_id, payload)
        raise


@router.post("/jobs/{job_id}/confirm-selection")
def confirm_selection(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    selections = (job.target or {}).get("selections") or []
    if not selections:
        raise HTTPException(
            status_code=422,
            detail=error_detail("MISSING_SELECTION", "Target selection is missing"),
        )

    player_ref = job.player_ref or job.anchor or {}
    if _normalize_player_ref(player_ref) is None:
        raise HTTPException(
            status_code=422,
            detail=error_detail("MISSING_PLAYER_REF", "Player reference is missing"),
        )

    job.status = "CREATED"

    progress = job.progress or {}
    current_pct = progress.get("pct") or 0
    try:
        current_pct = int(current_pct)
    except (TypeError, ValueError):
        current_pct = 0

    job.progress = {
        **progress,
        "step": "SELECTION_CONFIRMED",
        "pct": max(current_pct, 15),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    db.commit()
    db.refresh(job)
    return ok_response({"job_id": job.id, "id": job.id, "status": job.status}, request)


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
    preview_frames = job.preview_frames or []
    if not preview_frames:
        raise HTTPException(
            status_code=409,
            detail=error_detail("FRAMES_NOT_READY", "Preview frames not ready"),
        )

    context = load_s3_context()
    preview_frames = attach_presigned_urls(
        {"preview_frames": preview_frames}, context
    ).get("preview_frames", preview_frames)
    preview_frames = [
        {**frame, "key": frame.get("key") or frame.get("s3_key")}
        if isinstance(frame, dict)
        else frame
        for frame in preview_frames
    ]
    preview_frames = _normalize_preview_frames(preview_frames)

    items: List[Dict[str, Any]] = []
    for frame in preview_frames[:count]:
        if not isinstance(frame, dict):
            continue
        items.append(
            {
                "time_sec": frame.get("time_sec"),
                "signed_url": frame.get("signed_url"),
                "width": frame.get("width"),
                "height": frame.get("height"),
                "key": frame.get("key") or frame.get("s3_key"),
            }
        )

    return ok_response({"items": items}, request)


@router.get("/jobs/{job_id}/frames/{filename}")
def get_frame_file(job_id: str, filename: str) -> StreamingResponse:
    _validate_filename(filename)
    key = f"jobs/{job_id}/frames/{filename}"
    return _stream_s3_image(key)


@router.get("/jobs/{job_id}/frames/list")
def list_frames(
    job_id: str,
    request: Request,
    count: int = 8,
    db: Session = Depends(get_db),
):
    raise HTTPException(
        status_code=410,
        detail=error_detail(
            "DEPRECATED",
            "Deprecated. Use /jobs/{id}/frames?count=8",
        ),
    )


@router.get("/jobs/{job_id}/frames/overlay")
def overlay_frames(job_id: str, request: Request, db: Session = Depends(get_db)):
    raise HTTPException(
        status_code=410,
        detail=error_detail(
            "DEPRECATED",
            "Deprecated. Use /jobs/{id}/frames?count=8",
        ),
    )


@router.post("/jobs/{job_id}/enqueue")
def enqueue_job(
    job_id: str,
    request: Request,
    payload: dict | None = Body(default=None),
    db: Session = Depends(get_db),
):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=error_detail("JOB_NOT_FOUND", "Job not found"),
        )

    # idempotente: se già avanzato, non reinvio
    if job.status in ["QUEUED", "RUNNING", "COMPLETED", "FAILED"]:
        return ok_response({"job_id": job.id, "id": job.id, "status": job.status}, request)

    target_confirmed = bool((job.target or {}).get("confirmed"))
    missing: List[str] = []
    if not _has_player_ref(job.player_ref):
        missing.append("player_ref")
    if not target_confirmed:
        missing.append("target")
    if missing:
        meta = build_meta(request)
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": {
                    "code": "NOT_READY",
                    "missing": sorted(set(missing)),
                    "message": "Missing required selections for enqueue",
                },
                "request_id": meta.get("request_id"),
                "meta": meta,
            },
        )

    job.status = "QUEUED"
    job.error = None
    progress = job.progress or {}
    current_pct = progress.get("pct") or 0
    try:
        current_pct = int(current_pct)
    except (TypeError, ValueError):
        current_pct = 0
    job.progress = {
        **progress,
        "step": "QUEUED",
        "pct": max(current_pct, 20),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    db.commit()
    db.refresh(job)

    from app.workers.pipeline import run_analysis

    run_analysis.delay(job.id)

    return ok_response({"job_id": job.id, "id": job.id, "status": job.status}, request)


@router.post("/jobs/{job_id}/confirm-selection")
def confirm_selection(job_id: str, request: Request, db: Session = Depends(get_db)):
    return enqueue_job(job_id, request, db)
