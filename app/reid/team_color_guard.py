from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GUARD_VERSION = "kit-color-guard-v1"
COLOR_FAMILIES = (
    "RED_WARM",
    "YELLOW",
    "GREEN",
    "CYAN_BLUE",
    "PURPLE",
    "WHITE",
    "BLACK",
    "NEUTRAL",
)


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def team_color_guard_enabled() -> bool:
    value = (os.environ.get("PLAYER_REID_TEAM_COLOR_GUARD_ENABLED") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class KitColorSignature:
    distribution: tuple[float, ...]
    dominant_family: str
    confidence: float
    quality: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": GUARD_VERSION,
            "dominant_family": self.dominant_family,
            "confidence": round(self.confidence, 6),
            "quality": round(self.quality, 6),
            "distribution": {
                family: round(value, 6)
                for family, value in zip(COLOR_FAMILIES, self.distribution)
            },
        }


def _normalized_bbox(value: Mapping[str, Any]) -> dict[str, float] | None:
    source = value.get("bbox") if isinstance(value.get("bbox"), Mapping) else value
    try:
        x = float(source.get("x"))
        y = float(source.get("y"))
        w = float(source.get("w"))
        h = float(source.get("h"))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _crop(frame: np.ndarray, bbox: Mapping[str, Any]) -> np.ndarray | None:
    normalized = _normalized_bbox(bbox)
    if normalized is None or frame is None or frame.ndim != 3:
        return None
    height, width = frame.shape[:2]
    x1 = int(round(normalized["x"] * width))
    y1 = int(round(normalized["y"] * height))
    x2 = int(round((normalized["x"] + normalized["w"]) * width))
    y2 = int(round((normalized["y"] + normalized["h"]) * height))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)].copy()


def extract_kit_color_signature(crop: np.ndarray) -> KitColorSignature | None:
    if crop is None or crop.ndim != 3:
        return None
    height, width = crop.shape[:2]
    if width < 6 or height < 12:
        return None

    normalized = cv2.resize(crop, (48, 96), interpolation=cv2.INTER_AREA)
    # The central upper body is the most stable team-colour cue. Avoid the box edges,
    # legs, grass and neighbouring players as much as possible.
    torso = normalized[10:58, 8:40]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    rows, cols = hue.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    spatial_weight = np.exp(
        -(((xx - center_x) / max(1.0, cols * 0.42)) ** 2)
        -(((yy - center_y) / max(1.0, rows * 0.55)) ** 2)
    ).astype(np.float32)
    chroma_weight = 0.55 + 0.45 * (saturation / 255.0)
    weights = spatial_weight * chroma_weight

    family_index = np.full(hue.shape, COLOR_FAMILIES.index("NEUTRAL"), dtype=np.int16)
    black = value < 62
    white = (saturation < 46) & (value >= 155)
    neutral = (saturation < 58) & ~black & ~white
    chromatic = ~(black | white | neutral)

    family_index[black] = COLOR_FAMILIES.index("BLACK")
    family_index[white] = COLOR_FAMILIES.index("WHITE")
    family_index[neutral] = COLOR_FAMILIES.index("NEUTRAL")
    family_index[chromatic & ((hue < 27) | (hue >= 168))] = COLOR_FAMILIES.index("RED_WARM")
    family_index[chromatic & (hue >= 27) & (hue < 39)] = COLOR_FAMILIES.index("YELLOW")
    family_index[chromatic & (hue >= 39) & (hue < 86)] = COLOR_FAMILIES.index("GREEN")
    family_index[chromatic & (hue >= 86) & (hue < 136)] = COLOR_FAMILIES.index("CYAN_BLUE")
    family_index[chromatic & (hue >= 136) & (hue < 168)] = COLOR_FAMILIES.index("PURPLE")

    totals = np.array(
        [float(weights[family_index == index].sum()) for index in range(len(COLOR_FAMILIES))],
        dtype=np.float64,
    )
    total = float(totals.sum())
    if total <= 1e-9:
        return None
    distribution = totals / total
    dominant_index = int(np.argmax(distribution))
    dominant_share = float(distribution[dominant_index])
    entropy = -float(np.sum(distribution * np.log(distribution + 1e-12))) / math.log(len(COLOR_FAMILIES))
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    size_score = min(1.0, (width * height) / float(24 * 48))
    sharpness_score = min(1.0, sharpness / 160.0)
    confidence = max(0.0, min(1.0, dominant_share * (1.0 - 0.35 * entropy)))
    quality = max(
        0.0,
        min(1.0, 0.45 * confidence + 0.35 * size_score + 0.20 * sharpness_score),
    )
    return KitColorSignature(
        distribution=tuple(float(value) for value in distribution),
        dominant_family=COLOR_FAMILIES[dominant_index],
        confidence=confidence,
        quality=quality,
    )


def signature_similarity(first: KitColorSignature, second: KitColorSignature) -> float:
    # Bhattacharyya coefficient: 1 means identical categorical colour evidence.
    return max(
        0.0,
        min(
            1.0,
            sum(
                math.sqrt(max(0.0, left) * max(0.0, right))
                for left, right in zip(first.distribution, second.distribution)
            ),
        ),
    )


def signatures_compatible(
    anchor: KitColorSignature,
    observed: KitColorSignature,
    *,
    minimum_confidence: float | None = None,
    minimum_similarity: float | None = None,
) -> bool | None:
    confidence_gate = (
        minimum_confidence
        if minimum_confidence is not None
        else _env_float("PLAYER_REID_TEAM_COLOR_MIN_CONFIDENCE", 0.42, 0.0, 1.0)
    )
    similarity_gate = (
        minimum_similarity
        if minimum_similarity is not None
        else _env_float("PLAYER_REID_TEAM_COLOR_MIN_SIMILARITY", 0.60, 0.0, 1.0)
    )
    if anchor.confidence < confidence_gate or observed.confidence < confidence_gate:
        return None
    if anchor.dominant_family != observed.dominant_family:
        return False
    return signature_similarity(anchor, observed) >= similarity_gate


class _VideoReader:
    def __init__(self, path: str | Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"unable to open video for kit-colour guard: {path}")

    def read(self, time_sec: float) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


def _sample_evenly(items: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if not items or count <= 0:
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


def _signature_at(
    read_frame: Callable[[float], np.ndarray | None],
    time_sec: float,
    bbox: Mapping[str, Any],
) -> KitColorSignature | None:
    frame = read_frame(time_sec)
    if frame is None:
        return None
    crop = _crop(frame, bbox)
    return extract_kit_color_signature(crop) if crop is not None else None


def _anchor_signature(
    read_frame: Callable[[float], np.ndarray | None],
    player_ref: Mapping[str, Any],
) -> KitColorSignature | None:
    try:
        time_sec = float(player_ref.get("t", player_ref.get("best_time_sec")))
    except (TypeError, ValueError):
        return None
    bbox = _normalized_bbox(player_ref)
    if bbox is None:
        return None
    return _signature_at(read_frame, time_sec, bbox)


def _bbox_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    left = _normalized_bbox(first)
    right = _normalized_bbox(second)
    if left is None or right is None:
        return 0.0
    x1 = max(left["x"], right["x"])
    y1 = max(left["y"], right["y"])
    x2 = min(left["x"] + left["w"], right["x"] + right["w"])
    y2 = min(left["y"] + left["h"], right["y"] + right["h"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left["w"] * left["h"] + right["w"] * right["h"] - intersection
    return intersection / union if union > 0.0 else 0.0


def _anchor_geometry_evidence(
    segments: Sequence[Mapping[str, Any]],
    anchor_indices: Sequence[int],
    player_ref: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        anchor_time = float(player_ref.get("t", player_ref.get("best_time_sec")))
    except (TypeError, ValueError):
        anchor_time = -1.0
    anchor_bbox = _normalized_bbox(player_ref)
    if anchor_time < 0 or anchor_bbox is None or not anchor_indices:
        return {
            "passed": False,
            "reason_codes": ["ANCHOR_GEOMETRY_UNAVAILABLE"],
            "nearest_time_sec": None,
            "time_delta_sec": None,
            "iou": 0.0,
        }
    candidates: list[Mapping[str, Any]] = []
    for index in anchor_indices:
        candidates.extend(
            item
            for item in (segments[index].get("bboxes") or [])
            if isinstance(item, Mapping) and item.get("t") is not None
        )
    if not candidates:
        return {
            "passed": False,
            "reason_codes": ["ANCHOR_TRACK_EMPTY"],
            "nearest_time_sec": None,
            "time_delta_sec": None,
            "iou": 0.0,
        }
    nearest = min(candidates, key=lambda item: abs(float(item.get("t")) - anchor_time))
    nearest_time = float(nearest.get("t"))
    time_delta = abs(nearest_time - anchor_time)
    overlap = _bbox_iou(anchor_bbox, nearest)
    maximum_time_delta = _env_float(
        "PLAYER_REID_ANCHOR_MAX_TIME_DELTA_SEC", 1.25, 0.0, 5.0
    )
    minimum_iou = _env_float("PLAYER_REID_ANCHOR_MIN_IOU", 0.08, 0.0, 1.0)
    reason_codes: list[str] = []
    if time_delta > maximum_time_delta:
        reason_codes.append("ANCHOR_SAMPLE_TOO_FAR")
    if overlap < minimum_iou:
        reason_codes.append("ANCHOR_BBOX_MISMATCH")
    return {
        "passed": not reason_codes,
        "reason_codes": reason_codes,
        "nearest_time_sec": round(nearest_time, 6),
        "time_delta_sec": round(time_delta, 6),
        "iou": round(overlap, 6),
        "minimum_iou": minimum_iou,
        "maximum_time_delta_sec": maximum_time_delta,
    }


def _segment_color_evidence(
    read_frame: Callable[[float], np.ndarray | None],
    segment: Mapping[str, Any],
    anchor: KitColorSignature,
) -> dict[str, Any]:
    max_samples = _env_int("PLAYER_REID_TEAM_COLOR_SAMPLES_PER_SEGMENT", 5, 1, 12)
    bboxes = [item for item in (segment.get("bboxes") or []) if isinstance(item, Mapping)]
    evidence: list[dict[str, Any]] = []
    compatible_count = 0
    incompatible_count = 0
    unknown_count = 0
    for bbox in _sample_evenly(bboxes, max_samples):
        try:
            time_sec = float(bbox.get("t"))
        except (TypeError, ValueError):
            unknown_count += 1
            continue
        signature = _signature_at(read_frame, time_sec, bbox)
        if signature is None:
            unknown_count += 1
            evidence.append({"time_sec": round(time_sec, 6), "status": "UNKNOWN"})
            continue
        compatible = signatures_compatible(anchor, signature)
        if compatible is True:
            compatible_count += 1
            status = "COMPATIBLE"
        elif compatible is False:
            incompatible_count += 1
            status = "INCOMPATIBLE"
        else:
            unknown_count += 1
            status = "UNKNOWN"
        evidence.append(
            {
                "time_sec": round(time_sec, 6),
                "status": status,
                "similarity": round(signature_similarity(anchor, signature), 6),
                "signature": signature.to_payload(),
            }
        )

    judged = compatible_count + incompatible_count
    minimum_samples = _env_int("PLAYER_REID_TEAM_COLOR_MIN_SAMPLES", 2, 1, 12)
    maximum_incompatible_fraction = _env_float(
        "PLAYER_REID_TEAM_COLOR_MAX_INCOMPATIBLE_FRACTION", 0.20, 0.0, 1.0
    )
    incompatible_fraction = incompatible_count / float(judged) if judged else 1.0
    passed = (
        judged >= minimum_samples
        and compatible_count >= minimum_samples
        and incompatible_fraction <= maximum_incompatible_fraction
    )
    reason_codes: list[str] = []
    if judged < minimum_samples or compatible_count < minimum_samples:
        reason_codes.append("INSUFFICIENT_KIT_COLOR_EVIDENCE")
    if incompatible_count:
        reason_codes.append("KIT_COLOR_INCONSISTENT_WITH_ANCHOR")
    return {
        "version": GUARD_VERSION,
        "passed": passed,
        "compatible_samples": compatible_count,
        "incompatible_samples": incompatible_count,
        "unknown_samples": unknown_count,
        "incompatible_fraction": round(incompatible_fraction, 6),
        "reason_codes": reason_codes,
        "evidence": evidence,
    }


def _abstain_segment(segment: Mapping[str, Any], guard: Mapping[str, Any], reason: str) -> dict[str, Any]:
    updated = dict(segment)
    reid = dict(updated.get("reid") or {})
    original_candidate = reid.get("selected_candidate_id") or updated.get("selected_track_id")
    reasons = list(dict.fromkeys([*(reid.get("reason_codes") or []), reason]))
    reid.update(
        {
            "status": "ABSTAINED",
            "selected_candidate_id": None,
            "reason_codes": reasons,
            "kit_color_guard": dict(guard),
            "pre_guard_selected_candidate_id": original_candidate,
        }
    )
    updated.update(
        {
            "selected_track_id": None,
            "identity_id": None,
            "identity_status": "ABSTAINED",
            "coverage_pct": 0.0,
            "lost_segments": [],
            "bboxes": [],
            "reid": reid,
        }
    )
    return updated


def _coverage_pct(segments: Sequence[Mapping[str, Any]], duration: float, fps: float) -> float:
    if duration <= 0 or fps <= 0:
        return 0.0
    observed = {
        int(round(float(bbox.get("t")) * fps))
        for segment in segments
        for bbox in (segment.get("bboxes") or [])
        if isinstance(bbox, Mapping) and bbox.get("t") is not None
    }
    total = max(1, int(round(duration * fps)))
    return min(100.0, len(observed) / float(total) * 100.0)


def _largest_gap(segments: Sequence[Mapping[str, Any]], duration: float) -> float:
    times = sorted(
        {
            max(0.0, min(duration, float(bbox.get("t"))))
            for segment in segments
            for bbox in (segment.get("bboxes") or [])
            if isinstance(bbox, Mapping) and bbox.get("t") is not None
        }
    )
    if not times:
        return max(0.0, duration)
    gaps = [times[0], max(0.0, duration - times[-1])]
    gaps.extend(max(0.0, right - left) for left, right in zip(times, times[1:]))
    return max(gaps, default=0.0)


def apply_team_color_guard(
    output: Mapping[str, Any],
    *,
    input_video_path: str | Path,
    player_ref: Mapping[str, Any],
    frame_reader: Callable[[float], np.ndarray | None] | None = None,
) -> dict[str, Any]:
    guarded = dict(output)
    raw_segments = output.get("segments")
    if not isinstance(raw_segments, list):
        return guarded

    owned_reader: _VideoReader | None = None
    if frame_reader is None:
        owned_reader = _VideoReader(input_video_path)
        frame_reader = owned_reader.read
    try:
        acquisition = (
            output.get("anchor_acquisition")
            if isinstance(output.get("anchor_acquisition"), Mapping)
            else {}
        )
        seed_anchor = acquisition.get("seed_anchor")
        guard_anchor = (
            seed_anchor if isinstance(seed_anchor, Mapping) else player_ref
        )
        anchor = _anchor_signature(frame_reader, guard_anchor)
        segments = [dict(item) if isinstance(item, Mapping) else {} for item in raw_segments]
        accepted_indices: list[int] = []
        for index, segment in enumerate(segments):
            reid = segment.get("reid") if isinstance(segment.get("reid"), Mapping) else {}
            identity_status = str(reid.get("status") or segment.get("identity_status") or "").upper()
            if (
                bool(segment.get("bboxes"))
                and segment.get("selected_track_id") is not None
                and identity_status != "ABSTAINED"
            ):
                accepted_indices.append(index)

        try:
            anchor_time = float(
                guard_anchor.get("t", guard_anchor.get("best_time_sec"))
            )
        except (TypeError, ValueError):
            anchor_time = -1.0
        anchor_indices = [
            index
            for index in accepted_indices
            if str(segments[index].get("direction") or "").lower() == "anchor"
        ]
        if not anchor_indices and anchor_time >= 0:
            anchor_indices = [
                index
                for index in accepted_indices
                if float(segments[index].get("window_start") or 0.0)
                <= anchor_time
                <= float(segments[index].get("window_end") or 0.0)
            ][:1]

        decisions: list[dict[str, Any]] = []
        anchor_failed = False
        reason_codes = ["TEAM_COLOR_GUARD_EXPERIMENTAL"]
        anchor_geometry = _anchor_geometry_evidence(
            segments,
            anchor_indices,
            guard_anchor,
        )
        if not anchor_geometry["passed"]:
            anchor_failed = True
            reason_codes.extend(anchor_geometry["reason_codes"])

        if anchor is None:
            anchor_failed = True
            reason_codes.append("ANCHOR_KIT_COLOR_UNAVAILABLE")
        else:
            for index in accepted_indices:
                segment_guard = _segment_color_evidence(frame_reader, segments[index], anchor)
                decisions.append({"window_index": index, **segment_guard})
                if not segment_guard["passed"]:
                    segments[index] = _abstain_segment(
                        segments[index], segment_guard, "KIT_COLOR_GUARD_REJECTED"
                    )
                    if index in anchor_indices:
                        anchor_failed = True

        if anchor_failed:
            reason_codes.append("ANCHOR_TRACK_COLOR_UNVERIFIED")
            for index in accepted_indices:
                if segments[index].get("bboxes"):
                    guard = next(
                        (item for item in decisions if item.get("window_index") == index),
                        {
                            "version": GUARD_VERSION,
                            "passed": False,
                            "reason_codes": ["ANCHOR_TRACK_COLOR_UNVERIFIED"],
                            "evidence": [],
                        },
                    )
                    segments[index] = _abstain_segment(
                        segments[index], guard, "ANCHOR_TRACK_COLOR_UNVERIFIED"
                    )

        duration = max(
            [float(segment.get("window_end") or 0.0) for segment in segments] or [0.0]
        )
        fps = float(output.get("fps") or 1.0)
        segments_with_player = sum(1 for segment in segments if segment.get("bboxes"))
        coverage = _coverage_pct(segments, duration, fps)
        largest_gap = _largest_gap(segments, duration)

        summary = dict(output.get("reid_summary") or {})
        summary["team_color_guard"] = {
            "version": GUARD_VERSION,
            "status": "ANCHOR_REJECTED" if anchor_failed else "APPLIED",
            "validated": False,
            "anchor_signature": anchor.to_payload() if anchor is not None else None,
            "seed_anchor_id": acquisition.get("seed_anchor_id"),
            "anchor_geometry": anchor_geometry,
            "segments_checked": len(accepted_indices),
            "segments_rejected": max(0, len(accepted_indices) - segments_with_player),
            "post_guard_segments_with_player": segments_with_player,
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "decisions": decisions,
        }
        summary["status"] = "ANCHOR_REJECTED" if anchor_failed else "EXPERIMENTAL_GUARDED"
        summary["validated"] = False
        summary["reason_codes"] = list(
            dict.fromkeys([*(summary.get("reason_codes") or []), *reason_codes])
        )
        tracking_failed = bool(anchor_failed or segments_with_player == 0)

        guarded.update(
            {
                "identity_mode": "appearance_reid_v1+kit_color_guard_v1",
                "segments": segments,
                "segments_with_player": segments_with_player,
                "coverage_pct_total": round(coverage, 2),
                "coverage_pct": round(coverage, 2),
                "largest_gap_sec": (
                    None if tracking_failed else round(largest_gap, 2)
                ),
                "tracking_success": not tracking_failed,
                "tracking_status": (
                    "ANCHOR_REJECTED"
                    if anchor_failed
                    else (
                        "NO_PLAYER_TRACK"
                        if segments_with_player == 0
                        else "SUCCEEDED"
                    )
                ),
                "action_required": (
                    "RESELECT_PLAYER" if tracking_failed else None
                ),
                "reid_summary": summary,
            }
        )
        guarded["notes"] = (
            str(output.get("notes") or "").rstrip()
            + " Accepted identity links are additionally gated by manual-anchor kit colour."
        ).strip()
        return guarded
    finally:
        if owned_reader is not None:
            owned_reader.close()


def _repersist_guarded_output(output: dict[str, Any], job_id: str) -> dict[str, Any]:
    try:
        from app.reid import windowed_tracking
        from app.workers import tracking as legacy

        endpoint_url = legacy.S3_ENDPOINT_URL
        bucket = os.environ.get("S3_BUCKET", "").strip()
        expires_seconds = int(os.environ.get("SIGNED_URL_EXPIRES_SECONDS", "3600"))
        if not endpoint_url or not bucket:
            return output
        payload = {
            key: value
            for key, value in output.items()
            if key not in {"tracking_key", "tracking_url"}
        }
        return windowed_tracking._persist_tracking_output(
            job_id,
            payload,
            endpoint_url=endpoint_url,
            bucket=bucket,
            expires_seconds=expires_seconds,
        )
    except Exception:
        logger.exception("Unable to persist kit-colour-guarded tracking output job_id=%s", job_id)
        return output


def guard_windowed_reid(implementation: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(implementation, "__algonext_team_color_guard__", False):
        return implementation

    def guarded(*args: Any, **kwargs: Any) -> Any:
        output = implementation(*args, **kwargs)
        if not team_color_guard_enabled() or not isinstance(output, Mapping):
            return output
        if output.get("tracking_success") is False:
            return output
        if not isinstance(output.get("segments"), list):
            return output
        if len(args) < 3 or not isinstance(args[2], Mapping):
            logger.warning("Kit-colour guard skipped because the player reference is unavailable")
            return output
        try:
            corrected = apply_team_color_guard(
                output,
                input_video_path=args[1],
                player_ref=args[2],
            )
        except Exception:
            logger.exception("Kit-colour guard failed; clearing accepted identity links")
            corrected = dict(output)
            segments = []
            for raw in output.get("segments") or []:
                segment = dict(raw) if isinstance(raw, Mapping) else {}
                if segment.get("bboxes"):
                    segment = _abstain_segment(
                        segment,
                        {
                            "version": GUARD_VERSION,
                            "passed": False,
                            "reason_codes": ["TEAM_COLOR_GUARD_ERROR"],
                            "evidence": [],
                        },
                        "TEAM_COLOR_GUARD_ERROR",
                    )
                segments.append(segment)
            corrected["segments"] = segments
            corrected["segments_with_player"] = 0
            corrected["coverage_pct"] = 0.0
            corrected["coverage_pct_total"] = 0.0
            corrected["largest_gap_sec"] = None
            corrected["tracking_success"] = False
            corrected["tracking_status"] = "TEAM_COLOR_GUARD_ERROR"
            corrected["action_required"] = "RETRY_ANALYSIS"
            summary = dict(output.get("reid_summary") or {})
            summary.update(
                {
                    "status": "TEAM_COLOR_GUARD_ERROR",
                    "validated": False,
                    "team_color_guard": {
                        "version": GUARD_VERSION,
                        "status": "ERROR_FAIL_CLOSED",
                        "validated": False,
                        "reason_codes": ["TEAM_COLOR_GUARD_ERROR"],
                    },
                }
            )
            corrected["reid_summary"] = summary
        job_id = str(args[0]) if args else ""
        return _repersist_guarded_output(corrected, job_id) if job_id else corrected

    guarded.__name__ = getattr(implementation, "__name__", "guarded_windowed_reid")
    guarded.__doc__ = "Windowed Player ReID with a fail-closed manual-anchor kit-colour gate."
    setattr(guarded, "__algonext_team_color_guard__", True)
    setattr(guarded, "__algonext_original_reid__", implementation)
    return guarded
