from __future__ import annotations

import copy
import logging
from importlib import import_module
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SELECTION_FRAME_MARKER = "/frames/frame_"
_TRACKING_FRAME_MARKER = "/tracking_frames/tracking_frame_"
_NOT_FOUND_CODES = {
    "404",
    "NoSuchBucket",
    "NoSuchKey",
    "NotFound",
    "NoSuchObject",
}


def _asset_keys(frames: Any) -> tuple[str, ...]:
    if not isinstance(frames, list):
        return ()
    keys: list[str] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        key = frame.get("key") or frame.get("s3_key")
        if isinstance(key, str) and key:
            keys.append(key)
    return tuple(keys)


def _frame_signature(frames: Any) -> tuple[tuple[str, float], ...]:
    signature: list[tuple[str, float]] = []
    if not isinstance(frames, list):
        return ()
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        key = frame.get("key") or frame.get("s3_key")
        if not isinstance(key, str) or not key:
            continue
        try:
            time_sec = float(frame.get("time_sec") or 0.0)
        except (TypeError, ValueError):
            time_sec = 0.0
        signature.append((key, time_sec))
    return tuple(signature)


def _is_selection_frame_key(key: Any) -> bool:
    return isinstance(key, str) and _SELECTION_FRAME_MARKER in key


def _all_selection_frames(frames: Any) -> bool:
    keys = _asset_keys(frames)
    return bool(keys) and all(_is_selection_frame_key(key) for key in keys)


def _object_exists(s3_client: Any, bucket: str, key: str) -> bool:
    head_object = getattr(s3_client, "head_object", None)
    if not callable(head_object):
        return False
    try:
        head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        response = getattr(exc, "response", None)
        error = response.get("Error", {}) if isinstance(response, dict) else {}
        code = str(error.get("Code", ""))
        status = str(
            (response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode", "")
        ) if isinstance(response, dict) else ""
        if code in _NOT_FOUND_CODES or status == "404":
            return False
        raise


def _tracking_asset_payload(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "time_sec",
        "bucket",
        "key",
        "width",
        "height",
        "has_player",
        "is_target",
        "tracks",
        "asset_kind",
    }
    return [
        {key: copy.deepcopy(value) for key, value in frame.items() if key in allowed}
        for frame in frames
        if isinstance(frame, dict)
    ]


def _merge_selection_tracks(
    immutable_frames: list[dict[str, Any]],
    updated_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    updated_by_key = {
        str(frame.get("key") or frame.get("s3_key")): frame
        for frame in updated_frames
        if isinstance(frame, dict) and (frame.get("key") or frame.get("s3_key"))
    }
    merged = copy.deepcopy(immutable_frames)
    for frame in merged:
        if not isinstance(frame, dict):
            continue
        key = str(frame.get("key") or frame.get("s3_key") or "")
        updated = updated_by_key.get(key)
        if isinstance(updated, dict) and isinstance(updated.get("tracks"), list):
            frame["tracks"] = copy.deepcopy(updated.get("tracks") or [])
    return merged


def install_preview_asset_policy(pipeline_module: Any) -> bool:
    """Keep selection frames immutable and store derived tracking frames separately.

    Selection frames are first-write assets. Later analysis stages may enrich their
    track metadata, but may not replace their key-to-timestamp mapping or overwrite
    the S3 object. Tracking-derived review frames use an independent namespace.
    """

    current_generator = getattr(
        pipeline_module, "_generate_tracking_preview_frames", None
    )
    current_update_job = getattr(pipeline_module, "update_job", None)
    current_upload_file = getattr(pipeline_module, "upload_file", None)
    if not callable(current_generator):
        raise RuntimeError(
            "pipeline module does not expose _generate_tracking_preview_frames"
        )
    if not callable(current_update_job):
        raise RuntimeError("pipeline module does not expose update_job")
    if not callable(current_upload_file):
        raise RuntimeError("pipeline module does not expose upload_file")
    if getattr(current_generator, "__algonext_preview_asset_policy__", False):
        return False

    pending_tracking_frames: dict[str, list[dict[str, Any]]] = {}
    pending_lock = RLock()

    def upload_immutable_asset(
        s3_internal: Any,
        bucket: str,
        local_path: Any,
        key: str,
        content_type: str,
    ) -> None:
        if _is_selection_frame_key(key) and _object_exists(s3_internal, bucket, key):
            logger.warning(
                "Skipped overwrite of immutable selection frame bucket=%s key=%s",
                bucket,
                key,
            )
            return
        current_upload_file(s3_internal, bucket, local_path, key, content_type)

    def generate_tracking_review_frames(
        *,
        job_id: str,
        input_path: Any,
        frames_dir: Any,
        s3_internal: Any,
        s3_bucket: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        frames_dir.mkdir(parents=True, exist_ok=True)
        review_frames: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            time_sec = candidate.get("time_sec")
            if time_sec is None:
                continue
            try:
                normalized_time = float(time_sec)
            except (TypeError, ValueError):
                continue

            frame_name = f"tracking_frame_{index:04d}.jpg"
            frame_path = frames_dir / frame_name
            try:
                pipeline_module._run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(normalized_time),
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
                logger.exception(
                    "Failed to extract tracking review frame job_id=%s time_sec=%s",
                    job_id,
                    normalized_time,
                )
                continue

            width, height = pipeline_module.probe_image_dimensions(frame_path)
            frame_key = f"jobs/{job_id}/tracking_frames/{frame_name}"
            pipeline_module.upload_file(
                s3_internal,
                s3_bucket,
                frame_path,
                frame_key,
                "image/jpeg",
            )
            review_frames.append(
                {
                    "time_sec": normalized_time,
                    "bucket": s3_bucket,
                    "key": frame_key,
                    "width": width,
                    "height": height,
                    "has_player": bool(candidate.get("has_player")),
                    "is_target": bool(candidate.get("is_target")),
                    "tracks": copy.deepcopy(candidate.get("tracks") or []),
                    "asset_kind": "tracking_review_frame",
                }
            )

        if review_frames:
            with pending_lock:
                pending_tracking_frames[job_id] = copy.deepcopy(review_frames)
        return review_frames

    def preserve_selection_frames(db: Any, job_id: str, updater: Callable[[Any], None]):
        before = pipeline_module.reload_job(db, job_id)
        immutable_selection = copy.deepcopy(
            list(getattr(before, "preview_frames", None) or []) if before else []
        )

        outcome = current_update_job(db, job_id, updater)
        after = pipeline_module.reload_job(db, job_id)
        if after is None:
            return outcome

        current_frames = copy.deepcopy(list(getattr(after, "preview_frames", None) or []))
        with pending_lock:
            pending = copy.deepcopy(pending_tracking_frames.get(job_id) or [])

        result = getattr(after, "result", None)
        updated_result = dict(result) if isinstance(result, dict) else {}
        changed = False

        if pending and _asset_keys(current_frames) == _asset_keys(pending):
            tracking_assets = _tracking_asset_payload(pending)
            updated_result["tracking_review_frames"] = tracking_assets
            assets = dict(updated_result.get("assets") or {})
            assets["tracking_review_frames"] = tracking_assets
            updated_result["assets"] = assets
            after.preview_frames = immutable_selection
            changed = True
            with pending_lock:
                pending_tracking_frames.pop(job_id, None)
            logger.info(
                "Preserved %d selection frame(s) and stored %d tracking review frame(s) job_id=%s",
                len(immutable_selection),
                len(tracking_assets),
                job_id,
            )
        elif (
            immutable_selection
            and _all_selection_frames(immutable_selection)
            and _all_selection_frames(current_frames)
        ):
            before_signature = _frame_signature(immutable_selection)
            after_signature = _frame_signature(current_frames)
            if before_signature != after_signature:
                after.preview_frames = immutable_selection
                changed = True
                logger.warning(
                    "Suppressed replacement of immutable selection frames job_id=%s",
                    job_id,
                )
            else:
                merged_frames = _merge_selection_tracks(
                    immutable_selection,
                    current_frames,
                )
                if merged_frames != current_frames:
                    after.preview_frames = merged_frames
                    changed = True

        if changed:
            integrity = dict(updated_result.get("preview_asset_integrity") or {})
            integrity.update(
                {
                    "selection_frames_immutable": True,
                    "selection_namespace": f"jobs/{job_id}/frames/",
                    "tracking_review_namespace": f"jobs/{job_id}/tracking_frames/",
                }
            )
            if (
                immutable_selection
                and _frame_signature(immutable_selection)
                != _frame_signature(current_frames)
                and _all_selection_frames(current_frames)
            ):
                integrity["selection_refresh_suppressed"] = True
            updated_result["preview_asset_integrity"] = integrity
            after.result = updated_result
            pipeline_module.safe_commit(db)

        return outcome

    setattr(upload_immutable_asset, "__algonext_preview_asset_policy__", True)
    setattr(upload_immutable_asset, "__algonext_original__", current_upload_file)
    setattr(
        generate_tracking_review_frames,
        "__algonext_preview_asset_policy__",
        True,
    )
    setattr(
        generate_tracking_review_frames,
        "__algonext_original__",
        current_generator,
    )
    setattr(preserve_selection_frames, "__algonext_preview_asset_policy__", True)
    setattr(preserve_selection_frames, "__algonext_original__", current_update_job)

    pipeline_module.upload_file = upload_immutable_asset
    pipeline_module._generate_tracking_preview_frames = generate_tracking_review_frames
    pipeline_module.update_job = preserve_selection_frames
    logger.info(
        "Installed immutable selection-preview policy with separate tracking assets"
    )
    return True


def install_worker_preview_asset_policy(
    module_loader: Callable[[str], Any] = import_module,
) -> bool:
    pipeline_module = module_loader("app.workers.pipeline")
    return install_preview_asset_policy(pipeline_module)
