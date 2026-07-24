from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.benchmark.reid_adapters import prediction_from_algonext_reid


def _sample_evenly(
    items: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indices = [
        int(round(position * (len(items) - 1) / float(count - 1)))
        for position in range(count)
    ]
    return [items[index] for index in sorted(set(indices))]


def _bbox_payload(value: Mapping[str, Any]) -> dict[str, float] | None:
    try:
        x = float(value.get("x"))
        y = float(value.get("y"))
        w = float(value.get("w"))
        h = float(value.get("h"))
    except (TypeError, ValueError):
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _evidence_for_segment(
    segment: Mapping[str, Any], *, fps: float, samples_per_window: int
) -> list[dict[str, Any]]:
    bboxes = [
        item for item in (segment.get("bboxes") or []) if isinstance(item, Mapping)
    ]
    evidence: list[dict[str, Any]] = []
    if bboxes:
        for bbox in _sample_evenly(bboxes, samples_per_window):
            try:
                time_sec = float(bbox.get("t"))
            except (TypeError, ValueError):
                continue
            item: dict[str, Any] = {
                "time_sec": round(max(0.0, time_sec), 6),
                "frame_index": int(round(max(0.0, time_sec) * fps)),
                "note": "predicted selected track",
            }
            normalized_bbox = _bbox_payload(bbox)
            if normalized_bbox is not None:
                item["bbox"] = normalized_bbox
            evidence.append(item)
        if evidence:
            return evidence

    start = float(segment.get("window_start") or 0.0)
    end = float(segment.get("window_end") or start)
    duration = max(0.0, end - start)
    count = max(1, samples_per_window)
    for index in range(count):
        fraction = (index + 1) / float(count + 1)
        time_sec = start + duration * fraction
        evidence.append(
            {
                "time_sec": round(time_sec, 6),
                "frame_index": int(round(time_sec * fps)),
                "note": "uniform window sample",
            }
        )
    return evidence


def _candidate_review_payload(candidate: Mapping[str, Any], fps: float) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id"))
    evidence: list[dict[str, Any]] = []
    for raw in candidate.get("evidence") or []:
        if not isinstance(raw, Mapping):
            continue
        try:
            time_sec = float(raw.get("time_sec"))
        except (TypeError, ValueError):
            continue
        item: dict[str, Any] = {
            "time_sec": round(max(0.0, time_sec), 6),
            "frame_index": int(
                raw.get("frame_index")
                if isinstance(raw.get("frame_index"), int)
                else round(max(0.0, time_sec) * fps)
            ),
        }
        bbox = raw.get("bbox")
        if isinstance(bbox, Mapping):
            normalized_bbox = _bbox_payload(bbox)
            if normalized_bbox is not None:
                item["bbox"] = normalized_bbox
        confidence = raw.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            item["confidence"] = round(float(confidence), 6)
        evidence.append(item)
    return {
        "candidate_id": candidate_id,
        "combined_score": candidate.get("combined_score"),
        "appearance_similarity": candidate.get("appearance_similarity"),
        "overlap_score": candidate.get("overlap_score"),
        "geometry_score": candidate.get("geometry_score"),
        "descriptor_quality": candidate.get("descriptor_quality"),
        "descriptor_samples": candidate.get("descriptor_samples"),
        "reason_codes": candidate.get("reason_codes") or [],
        "evidence": evidence,
    }


def build_reid_annotation_template(
    tracking: Mapping[str, Any],
    *,
    video_id: str,
    identity: str,
    samples_per_window: int = 3,
) -> dict[str, Any]:
    if samples_per_window < 1:
        raise ValueError("samples_per_window must be >= 1")
    prediction = prediction_from_algonext_reid(tracking, video_id=video_id)
    segments = tracking.get("segments")
    if not isinstance(segments, list):
        raise ValueError("tracking.segments must be an array")
    try:
        fps = float(tracking.get("fps") or 1.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("tracking.fps must be a number") from exc
    if fps <= 0:
        raise ValueError("tracking.fps must be > 0")

    prediction_by_index = {
        window.window_index: window for window in prediction.windows
    }
    windows: list[dict[str, Any]] = []
    for window_index, raw_segment in enumerate(segments):
        segment = raw_segment if isinstance(raw_segment, Mapping) else {}
        predicted = prediction_by_index[window_index]
        reid = segment.get("reid")
        reid = reid if isinstance(reid, Mapping) else {}
        candidates = [
            _candidate_review_payload(item, fps)
            for item in (reid.get("candidates") or [])
            if isinstance(item, Mapping) and item.get("candidate_id") is not None
        ]
        windows.append(
            {
                "window_index": window_index,
                "window_start": predicted.window_start,
                "window_end": predicted.window_end,
                "target_visibility": "UNCERTAIN",
                "candidate_state": None,
                "target_candidate_id": None,
                "selected_track_is_target": None,
                "evidence_frames": _evidence_for_segment(
                    segment, fps=fps, samples_per_window=samples_per_window
                ),
                "notes": None,
                "review_context": {
                    "direction": segment.get("direction"),
                    "decision": predicted.decision,
                    "selected_candidate_id": predicted.selected_candidate_id,
                    "best_candidate_id": predicted.best_candidate_id,
                    "best_score": predicted.best_score,
                    "margin": predicted.margin,
                    "reason_codes": list(predicted.reason_codes),
                    "coverage_pct": segment.get("coverage_pct"),
                    "candidates": candidates,
                },
            }
        )
    return {
        "schema_version": "reid-window-annotation-v1",
        "video_id": video_id,
        "identity": identity,
        "fps": fps,
        "windows": windows,
        "review_instructions": {
            "target_visibility": {
                "VISIBLE": "The target identity is human-verifiable in the window.",
                "NOT_VISIBLE": "The target identity is not visible in the window.",
                "UNCERTAIN": "Identity cannot be verified; the window is excluded.",
            },
            "candidate_state": {
                "PRESENT": "The correct local candidate ID is verifiable.",
                "ABSENT": "The target is visible but absent from the candidate set.",
                "UNVERIFIABLE": (
                    "The target is visible, but persisted evidence is insufficient to "
                    "map it to a candidate ID."
                ),
            },
            "selected_track_is_target": (
                "For ACCEPTED windows, judge whether the selected track is the target."
            ),
        },
    }


def _collect_evidence_records(template: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for window in template.get("windows") or []:
        if not isinstance(window, dict):
            continue
        for evidence in window.get("evidence_frames") or []:
            if isinstance(evidence, dict):
                records.append(evidence)
        context = window.get("review_context")
        if not isinstance(context, dict):
            continue
        for candidate in context.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            for evidence in candidate.get("evidence") or []:
                if isinstance(evidence, dict):
                    records.append(evidence)
    return records


def _extract_frame(
    video_path: Path,
    output_path: Path,
    *,
    time_sec: float,
    ffmpeg_binary: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, time_sec):.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not output_path.exists():
        detail = (result.stderr or result.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(
            f"frame extraction failed at {time_sec:.3f}s: {detail[-1000:]}"
        )


def _asset_filename(frame_index: int, time_sec: float) -> str:
    milliseconds = int(round(time_sec * 1000.0))
    return f"frame_{frame_index:08d}_{milliseconds:010d}.jpg"


def materialize_reid_review_pack(
    tracking: Mapping[str, Any],
    *,
    video_path: str | Path,
    output_dir: str | Path,
    video_id: str,
    identity: str,
    samples_per_window: int = 3,
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    video = Path(video_path)
    if not video.is_file():
        raise ValueError(f"video file not found: {video}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    template = build_reid_annotation_template(
        tracking,
        video_id=video_id,
        identity=identity,
        samples_per_window=samples_per_window,
    )
    extracted: dict[int, tuple[str, float]] = {}
    for evidence in _collect_evidence_records(template):
        try:
            time_sec = float(evidence.get("time_sec"))
        except (TypeError, ValueError):
            continue
        frame_index = evidence.get("frame_index")
        if not isinstance(frame_index, int):
            frame_index = int(round(time_sec * float(template["fps"])))
            evidence["frame_index"] = frame_index
        cached = extracted.get(frame_index)
        if cached is None:
            filename = _asset_filename(frame_index, time_sec)
            relative_path = f"frames/{filename}"
            _extract_frame(
                video,
                destination / relative_path,
                time_sec=time_sec,
                ffmpeg_binary=ffmpeg_binary,
            )
            extracted[frame_index] = (relative_path, time_sec)
        else:
            relative_path, time_sec = cached
        evidence["time_sec"] = round(time_sec, 6)
        evidence["image_path"] = relative_path

    template_path = destination / "reid-window-annotation.template.json"
    template_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_path = destination / "index.html"
    html_path.write_text(render_reid_review_html(template), encoding="utf-8")
    return {
        "template": template,
        "template_path": str(template_path),
        "html_path": str(html_path),
        "frames_extracted": len(extracted),
    }


def _review_asset(name: str) -> str:
    asset_path = Path(__file__).with_name("reid_review_assets") / name
    try:
        return asset_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"unable to load ReID review asset: {asset_path}") from exc


def render_reid_review_html(template: Mapping[str, Any]) -> str:
    encoded = json.dumps(template, ensure_ascii=False).replace("</", "<\\/")
    title_video_id = html.escape(str(template.get("video_id", "")), quote=True)
    return (
        _review_asset("review.html")
        .replace("__ALGONEXT_TITLE__", title_video_id)
        .replace("__ALGONEXT_STYLE__", _review_asset("review.css").rstrip())
        .replace("__ALGONEXT_DATA__", encoded)
        .replace("__ALGONEXT_SCRIPT__", _review_asset("review.js").rstrip())
    )
