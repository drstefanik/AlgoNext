from __future__ import annotations

import copy
import logging
from importlib import import_module
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


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


def install_preview_asset_policy(pipeline_module: Any) -> bool:
    """Keep selection frames immutable and store derived tracking frames separately.

    The legacy analysis task generated review frames after tracking under the same
    ``jobs/<job_id>/frames/frame_XXXX.jpg`` keys used during player selection. It
    then replaced ``job.preview_frames``. That invalidated the saved timestamp and
    bbox of the selected player because the object behind the original key changed.

    This policy moves tracking-derived images to ``tracking_frames/`` and intercepts
    only the update that would replace the immutable selection-frame collection.
    """

    current_generator = getattr(
        pipeline_module, "_generate_tracking_preview_frames", None
    )
    current_update_job = getattr(pipeline_module, "update_job", None)
    if not callable(current_generator):
        raise RuntimeError(
            "pipeline module does not expose _generate_tracking_preview_frames"
        )
    if not callable(current_update_job):
        raise RuntimeError("pipeline module does not expose update_job")
    if getattr(current_generator, "__algonext_preview_asset_policy__", False):
        return False

    pending_tracking_frames: dict[str, list[dict[str, Any]]] = {}
    pending_lock = RLock()

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
        selection_frames = copy.deepcopy(
            list(getattr(before, "preview_frames", None) or []) if before else []
        )

        outcome = current_update_job(db, job_id, updater)

        with pending_lock:
            pending = copy.deepcopy(pending_tracking_frames.get(job_id) or [])
        if not pending:
            return outcome

        after = pipeline_module.reload_job(db, job_id)
        if after is None:
            return outcome

        current_frames = list(getattr(after, "preview_frames", None) or [])
        if _asset_keys(current_frames) != _asset_keys(pending):
            return outcome

        result = getattr(after, "result", None)
        updated_result = dict(result) if isinstance(result, dict) else {}
        tracking_assets = _tracking_asset_payload(pending)
        updated_result["tracking_review_frames"] = tracking_assets

        assets = dict(updated_result.get("assets") or {})
        assets["tracking_review_frames"] = tracking_assets
        updated_result["assets"] = assets
        updated_result["preview_asset_integrity"] = {
            "selection_frames_immutable": True,
            "tracking_review_namespace": f"jobs/{job_id}/tracking_frames/",
        }

        after.preview_frames = selection_frames
        after.result = updated_result
        pipeline_module.safe_commit(db)
        with pending_lock:
            pending_tracking_frames.pop(job_id, None)

        logger.info(
            "Preserved %d selection frame(s) and stored %d tracking review frame(s) job_id=%s",
            len(selection_frames),
            len(tracking_assets),
            job_id,
        )
        return outcome

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
