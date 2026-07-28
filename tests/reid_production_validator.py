"""Pure production-regression validator used by GitHub Actions and unit tests."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_DURATION_SEC = 5931.775
EXPECTED_WINDOW_SEC = 60.0
EXPECTED_OVERLAP_SEC = 5.0
EXPECTED_WINDOWS = 108
EXPECTED_ANCHOR_MODEL = "yolo11s.pt"
EXPECTED_MAX_WORKER_AGE_SEC = 60.0
MIN_ANCHOR_FPS = 5
SAMPLE_TARGET = 60.0
MIN_RELEASE_COVERAGE_PCT = 5.0
MIN_RELEASE_RETAINED_SEGMENTS = 5
MIN_RELEASE_AUTONOMOUS_SEGMENTS = 3
MIN_RELEASE_AUTONOMOUS_BBOXES = 60
MIN_GUARD_COMPATIBLE_SAMPLES = 2
MIN_GUARD_SIMILARITY = 0.60
MIXED_GUARD_MIN_SIMILARITY = 0.90
MIXED_GUARD_MIN_ANCHOR_SHARE = 0.35
MIXED_GUARD_MAX_DOMINANT_GAP = 0.20
MAX_GUARD_INCOMPATIBLE_FRACTION = 0.20
EXPECTED_ANCHOR_MIN_IOU = 0.08
EXPECTED_ANCHOR_MAX_TIME_DELTA_SEC = 1.25
GUARD_VERSION = "kit-color-guard-v1"
COLOR_FAMILY_ORDER = (
    "RED_WARM",
    "YELLOW",
    "GREEN",
    "CYAN_BLUE",
    "PURPLE",
    "WHITE",
    "BLACK",
    "NEUTRAL",
)
COLOR_FAMILIES = set(COLOR_FAMILY_ORDER)

FORBIDDEN_SCORE_KEYS = {
    "match_rating_10",
    "matchRating10",
    "impact_100",
    "impact100",
    "impact_adj",
    "impact_components",
    "highlight_adj",
    "baseline_rating",
    "role_group",
    "overall_score",
    "overallScore",
    "role_score",
    "roleScore",
}
FORBIDDEN_LEGACY_METRIC_KEYS = {
    "trackingQualityIndex",
    "sampleFramesCount",
    "stabilityScore",
    "coveragePct",
    "candidate_metrics",
    "candidateMetrics",
    "preview_candidate_metrics",
    "previewCandidateMetrics",
}
ARTIFACT_SENSITIVE_KEY_FRAGMENTS = {
    "url",
    "uri",
    "token",
    "credential",
    "authorization",
    "signature",
    "secret",
    "password",
    "cookie",
}
ARTIFACT_SENSITIVE_COMPACT_KEYS = {
    "apikey",
    "awsaccesskeyid",
}
ARTIFACT_HTTP_RE = re.compile(r"https?://", re.IGNORECASE)
ARTIFACT_SECRET_STRING_RE = re.compile(
    r"x-amz-|"
    r"(?:awsaccesskeyid|access[_-]?token|id[_-]?token|api[_-]?key|"
    r"client[_-]?secret|password|credential|security[_-]?token|"
    r"signature|token|authorization)(?:\s*[:=]|%3d)|"
    r"bearer\s+\S+",
    re.IGNORECASE,
)


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is missing or malformed")
    return value


def _finite(value: Any, label: str) -> float:
    _require(
        value is not None and not isinstance(value, bool),
        f"{label} is missing or malformed: {value!r}",
    )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is malformed: {value!r}") from exc
    _require(math.isfinite(parsed), f"{label} is non-finite: {value!r}")
    return parsed


def _integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer: {value!r}",
    )
    return int(value)


def _close(actual: Any, expected: float, label: str, *, tolerance: float) -> None:
    parsed = _finite(actual, label)
    _require(
        math.isclose(parsed, expected, abs_tol=tolerance),
        f"{label} changed: expected={expected} actual={parsed}",
    )


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _envelope_data(envelope: Any, label: str) -> Mapping[str, Any]:
    source = _mapping(envelope, label)
    data = source.get("data")
    return _mapping(data, f"{label}.data") if data is not None else source


def validate_runtime_attestation(
    envelope: Mapping[str, Any],
    *,
    expected_revision: str,
) -> dict[str, Any]:
    source = _mapping(envelope, "runtime")
    _require(source.get("ok") is True, "Runtime envelope is not successful")
    payload = _mapping(source.get("data"), "runtime.data")
    revision = str(payload.get("revision") or "")
    worker = _mapping(payload.get("worker"), "runtime.worker")
    worker_revision = str(worker.get("revision") or "")
    dependencies = _mapping(
        payload.get("dependencies"),
        "runtime.dependencies",
    )
    _require(payload.get("ready") is True, "Runtime endpoint is not ready")
    _require(
        payload.get("required") is True,
        "Runtime worker readiness enforcement is disabled",
    )
    _require(
        dependencies.get("redis") == "ready" and dependencies.get("worker") == "ready",
        "Runtime dependencies are not ready",
    )
    _require(worker.get("state") == "ready", "Runtime worker state is not ready")
    _require(
        payload.get("worker_revision_matches_api") is True,
        "Runtime worker revision is not attested against the API",
    )
    worker_age = _finite(
        payload.get("worker_age_seconds"),
        "runtime.worker_age_seconds",
    )
    max_worker_age = _finite(
        payload.get("max_worker_age_seconds"),
        "runtime.max_worker_age_seconds",
    )
    _close(
        max_worker_age,
        EXPECTED_MAX_WORKER_AGE_SEC,
        "runtime.max_worker_age_seconds",
        tolerance=1e-6,
    )
    _require(
        0.0 <= worker_age <= EXPECTED_MAX_WORKER_AGE_SEC,
        "Runtime worker heartbeat is stale or malformed",
    )
    _require(
        revision == expected_revision,
        f"API revision mismatch: expected={expected_revision} actual={revision}",
    )
    _require(
        worker_revision == expected_revision,
        "Worker revision mismatch: "
        f"expected={expected_revision} actual={worker_revision}",
    )
    return {
        "ready": True,
        "api_revision": revision,
        "worker_revision": worker_revision,
        "expected_revision": expected_revision,
        "worker_age_seconds": worker_age,
        "max_worker_age_seconds": max_worker_age,
    }


def _expected_selections(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    selections = payload.get("selections")
    _require(
        isinstance(selections, list) and len(selections) == 2,
        "Expected exactly two canonical fixture selections",
    )
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(selections):
        source = _mapping(item, f"selection[{index}]")
        bbox = _mapping(source.get("bbox"), f"selection[{index}].bbox")
        normalized.append(
            {
                "frame_key": str(source.get("frameKey") or ""),
                "t": _finite(
                    source.get("frameTimeSec"),
                    f"selection[{index}].frameTimeSec",
                ),
                "bbox": {
                    key: _finite(bbox.get(key), f"selection[{index}].bbox.{key}")
                    for key in ("x", "y", "w", "h")
                },
            }
        )
    return sorted(normalized, key=lambda item: item["t"])


def _attempt_id(envelope: Mapping[str, Any], *, label: str) -> str:
    payload = _envelope_data(envelope, label)
    value = str(payload.get("analysis_attempt_id") or "").strip()
    _require(bool(value), f"{label} analysis_attempt_id is missing")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValidationError(
            f"{label} analysis_attempt_id is not a UUID: {value!r}"
        ) from exc
    _require(
        str(parsed) == value.lower(), f"{label} analysis_attempt_id is not canonical"
    )
    return value


def _job_id(envelope: Mapping[str, Any], *, label: str) -> str:
    payload = _envelope_data(envelope, label)
    value = str(payload.get("job_id") or payload.get("id") or "").strip()
    _require(bool(value), f"{label} job id is missing")
    return value


def _analysis_attempt_values(
    value: Any,
    *,
    path: str,
) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            nested_path = f"{path}.{key}"
            if key in {"analysis_attempt_id", "analysisAttemptId"}:
                found[nested_path] = nested
            found.update(_analysis_attempt_values(nested, path=nested_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.update(
                _analysis_attempt_values(
                    nested,
                    path=f"{path}[{index}]",
                )
            )
    return found


def _scan_forbidden_values(value: Any, path: str = "result") -> dict[str, Any]:
    leaks: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            nested_path = f"{path}.{key}"
            if (
                key in FORBIDDEN_SCORE_KEYS or key in FORBIDDEN_LEGACY_METRIC_KEYS
            ) and nested not in (None, {}, []):
                leaks[nested_path] = nested
            leaks.update(_scan_forbidden_values(nested, nested_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            leaks.update(_scan_forbidden_values(nested, f"{path}[{index}]"))
    return leaks


def _artifact_key_is_sensitive(key: Any, nested: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
    if normalized in {"signature", "anchorsignature"} and isinstance(nested, Mapping):
        return False
    return normalized in ARTIFACT_SENSITIVE_COMPACT_KEYS or any(
        fragment in normalized for fragment in ARTIFACT_SENSITIVE_KEY_FRAGMENTS
    )


def sanitize_artifact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: sanitize_artifact_value(nested)
            for key, nested in value.items()
            if not _artifact_key_is_sensitive(key, nested)
        }
    if isinstance(value, list):
        return [sanitize_artifact_value(item) for item in value]
    if isinstance(value, str):
        if ARTIFACT_HTTP_RE.search(value):
            return "[REDACTED_URL]"
        if ARTIFACT_SECRET_STRING_RE.search(value):
            return "[REDACTED_SENSITIVE_VALUE]"
    return value


def assert_sanitized_artifact(value: Any, *, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require(
                not _artifact_key_is_sensitive(key, nested),
                f"{path}.{key} retains a sensitive key",
            )
            assert_sanitized_artifact(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_sanitized_artifact(nested, path=f"{path}[{index}]")
    elif isinstance(value, str):
        _require(
            ARTIFACT_HTTP_RE.search(value) is None
            and ARTIFACT_SECRET_STRING_RE.search(value) is None,
            f"{path} retains a URL or credential marker",
        )


def _bbox_geometry(
    bbox: Mapping[str, Any],
    *,
    label: str,
    window_start: float,
    window_end: float,
    duration: float,
    tolerance: float,
) -> dict[str, float]:
    values = {
        key: _finite(bbox.get(key), f"{label}.{key}")
        for key in ("t", "x", "y", "w", "h")
    }
    _require(
        -tolerance <= values["t"] <= duration + tolerance,
        f"{label}.t is outside the fixture duration",
    )
    _require(
        window_start - tolerance <= values["t"] <= window_end + tolerance,
        f"{label}.t is outside its tracking window",
    )
    epsilon = 1e-6
    _require(
        values["x"] >= 0.0
        and values["y"] >= 0.0
        and values["w"] > 0.0
        and values["h"] > 0.0
        and values["x"] <= 1.0 + epsilon
        and values["y"] <= 1.0 + epsilon
        and values["x"] + values["w"] <= 1.0 + epsilon
        and values["y"] + values["h"] <= 1.0 + epsilon,
        f"{label} has invalid normalized geometry",
    )
    return values


def _image_motion(points: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    centers = sorted(
        (
            float(item["t"]),
            float(item["x"]) + float(item["w"]) / 2.0,
            float(item["y"]) + float(item["h"]) / 2.0,
        )
        for item in points
    )
    path_length = 0.0
    speeds: list[float] = []
    bursts = 0
    above_threshold = False
    for previous, current in zip(centers, centers[1:]):
        t0, x0, y0 = previous
        t1, x1, y1 = current
        delta = t1 - t0
        if delta <= 0:
            continue
        distance = math.hypot(x1 - x0, y1 - y0)
        path_length += distance
        speed = distance / delta
        speeds.append(speed)
        is_above = speed >= 0.02
        if is_above and not above_threshold:
            bursts += 1
        above_threshold = is_above
    tracked_span = centers[-1][0] - centers[0][0] if len(centers) >= 2 else 0.0
    average_speed = path_length / tracked_span if tracked_span > 0 else 0.0
    sorted_speeds = sorted(speeds)
    if sorted_speeds:
        p95_index = min(
            len(sorted_speeds) - 1,
            int(round(0.95 * (len(sorted_speeds) - 1))),
        )
        p95_speed = sorted_speeds[p95_index]
    else:
        p95_speed = 0.0
    return {
        "metric_space": "image_plane_normalized",
        "camera_motion_compensated": False,
        "pitch_calibrated": False,
        "observed_samples": len(centers),
        "tracked_span_sec": round(max(0.0, tracked_span), 3),
        "normalized_path_length": round(path_length, 6),
        "avg_center_speed_norm_per_sec": round(average_speed, 6),
        "p95_center_speed_norm_per_sec": round(p95_speed, 6),
        "motion_bursts_proxy": bursts,
    }


def _strict_window_index(
    value: Any,
    *,
    label: str,
    total: int,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    parsed = _integer(value, label)
    _require(0 <= parsed < total, f"{label} is out of range: {parsed}")
    return parsed


def _track_id_set(segment: Mapping[str, Any]) -> set[str]:
    values: list[Any] = []
    if segment.get("selected_track_id") is not None:
        values.append(segment.get("selected_track_id"))
    selected_track_ids = segment.get("selected_track_ids")
    if isinstance(selected_track_ids, list):
        values.extend(selected_track_ids)
    return {str(value) for value in values if value is not None}


def _color_signature(value: Any, *, label: str) -> Mapping[str, Any]:
    signature = _mapping(value, label)
    _require(
        signature.get("version") == GUARD_VERSION,
        f"{label}.version changed",
    )
    dominant_family = str(signature.get("dominant_family") or "")
    _require(
        dominant_family in COLOR_FAMILIES,
        f"{label}.dominant_family is invalid",
    )
    confidence = _finite(signature.get("confidence"), f"{label}.confidence")
    quality = _finite(signature.get("quality"), f"{label}.quality")
    _require(
        0.0 <= confidence <= 1.0 and 0.0 <= quality <= 1.0,
        f"{label} confidence/quality is outside [0, 1]",
    )
    distribution = _mapping(
        signature.get("distribution"),
        f"{label}.distribution",
    )
    _require(
        set(distribution) == COLOR_FAMILIES,
        f"{label}.distribution families changed",
    )
    weights = [
        _finite(distribution.get(family), f"{label}.distribution.{family}")
        for family in sorted(COLOR_FAMILIES)
    ]
    _require(
        all(0.0 <= weight <= 1.0 for weight in weights)
        and math.isclose(sum(weights), 1.0, abs_tol=1e-4),
        f"{label}.distribution is not normalized",
    )
    expected_dominant = max(
        COLOR_FAMILY_ORDER,
        key=lambda family: float(distribution[family]),
    )
    _require(
        dominant_family == expected_dominant,
        f"{label}.dominant_family disagrees with its distribution",
    )
    return signature


def _signature_similarity(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> float:
    first_distribution = _mapping(
        first.get("distribution"),
        "first signature distribution",
    )
    second_distribution = _mapping(
        second.get("distribution"),
        "second signature distribution",
    )
    return max(
        0.0,
        min(
            1.0,
            sum(
                math.sqrt(
                    max(0.0, float(first_distribution[family]))
                    * max(0.0, float(second_distribution[family]))
                )
                for family in COLOR_FAMILY_ORDER
            ),
        ),
    )


def _signatures_compatible(
    anchor: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    confidence_gate: float,
) -> bool | None:
    if (
        float(anchor["confidence"]) < confidence_gate
        or float(observed["confidence"]) < confidence_gate
    ):
        return None
    similarity = _signature_similarity(anchor, observed)
    anchor_family = str(anchor["dominant_family"])
    observed_family = str(observed["dominant_family"])
    if anchor_family != observed_family:
        observed_distribution = _mapping(
            observed.get("distribution"),
            "observed signature distribution",
        )
        anchor_family_share = float(observed_distribution[anchor_family])
        dominant_share = max(
            float(observed_distribution[family]) for family in COLOR_FAMILY_ORDER
        )
        if (
            similarity >= MIXED_GUARD_MIN_SIMILARITY
            and anchor_family_share >= MIXED_GUARD_MIN_ANCHOR_SHARE
            and dominant_share - anchor_family_share <= MIXED_GUARD_MAX_DOMINANT_GAP
        ):
            return None
        return False
    return similarity >= MIN_GUARD_SIMILARITY


def _guard_geometry(value: Any, *, label: str) -> Mapping[str, Any]:
    geometry = _mapping(value, label)
    _require(geometry.get("passed") is True, f"{label} did not pass")
    _require(
        geometry.get("reason_codes") == [],
        f"{label} contains rejection reasons",
    )
    nearest_time = _finite(
        geometry.get("nearest_time_sec"),
        f"{label}.nearest_time_sec",
    )
    time_delta = _finite(
        geometry.get("time_delta_sec"),
        f"{label}.time_delta_sec",
    )
    overlap = _finite(geometry.get("iou"), f"{label}.iou")
    minimum_iou = _finite(
        geometry.get("minimum_iou"),
        f"{label}.minimum_iou",
    )
    maximum_delta = _finite(
        geometry.get("maximum_time_delta_sec"),
        f"{label}.maximum_time_delta_sec",
    )
    _require(
        nearest_time >= 0.0
        and 0.0 <= time_delta <= maximum_delta
        and 0.0 <= minimum_iou <= overlap <= 1.0,
        f"{label} thresholds are inconsistent",
    )
    return geometry


def _bbox_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    x1 = max(float(first["x"]), float(second["x"]))
    y1 = max(float(first["y"]), float(second["y"]))
    x2 = min(
        float(first["x"]) + float(first["w"]),
        float(second["x"]) + float(second["w"]),
    )
    y2 = min(
        float(first["y"]) + float(first["h"]),
        float(second["y"]) + float(second["h"]),
    )
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        float(first["w"]) * float(first["h"])
        + float(second["w"]) * float(second["h"])
        - intersection
    )
    return intersection / union if union > 0.0 else 0.0


def _validate_anchor_candidate_geometry(
    value: Any,
    *,
    segment: Mapping[str, Any],
    expected_anchor: Mapping[str, Any],
    label: str,
) -> tuple[Mapping[str, Any], bool]:
    geometry = _mapping(value, label)
    bboxes = [
        bbox for bbox in (segment.get("bboxes") or []) if isinstance(bbox, Mapping)
    ]
    _require(bool(bboxes), f"{label} has no retained anchor bboxes")
    expected_time = float(expected_anchor["t"])
    nearest_bbox = min(
        bboxes,
        key=lambda bbox: abs(
            _finite(bbox.get("t"), f"{label}.retained_bbox.t") - expected_time
        ),
    )
    nearest_time = _finite(nearest_bbox.get("t"), f"{label}.retained_bbox.t")
    time_delta = abs(nearest_time - expected_time)
    overlap = _bbox_iou(expected_anchor["bbox"], nearest_bbox)
    reason_codes: list[str] = []
    if time_delta > EXPECTED_ANCHOR_MAX_TIME_DELTA_SEC:
        reason_codes.append("ANCHOR_SAMPLE_TOO_FAR")
    if overlap < EXPECTED_ANCHOR_MIN_IOU:
        reason_codes.append("ANCHOR_BBOX_MISMATCH")
    expected_passed = not reason_codes

    _require(
        geometry.get("passed") is expected_passed,
        f"{label}.passed disagrees with retained geometry",
    )
    _require(
        geometry.get("reason_codes") == reason_codes,
        f"{label}.reason_codes disagree with retained geometry",
    )
    _close(
        geometry.get("nearest_time_sec"),
        round(nearest_time, 6),
        f"{label}.nearest_time_sec",
        tolerance=1e-5,
    )
    _close(
        geometry.get("time_delta_sec"),
        round(time_delta, 6),
        f"{label}.time_delta_sec",
        tolerance=1e-5,
    )
    _close(
        geometry.get("iou"),
        round(overlap, 6),
        f"{label}.iou",
        tolerance=1e-5,
    )
    _close(
        geometry.get("minimum_iou"),
        EXPECTED_ANCHOR_MIN_IOU,
        f"{label}.minimum_iou",
        tolerance=1e-9,
    )
    _close(
        geometry.get("maximum_time_delta_sec"),
        EXPECTED_ANCHOR_MAX_TIME_DELTA_SEC,
        f"{label}.maximum_time_delta_sec",
        tolerance=1e-9,
    )
    return geometry, expected_passed


def _validate_segment_guard(
    value: Any,
    *,
    segment: Mapping[str, Any],
    anchor_signature: Mapping[str, Any],
    confidence_gate: float,
    label: str,
    require_bbox_membership: bool,
    expected_passed: bool,
) -> Mapping[str, Any]:
    guard = _mapping(value, label)
    _require(guard.get("version") == GUARD_VERSION, f"{label}.version changed")
    _require(
        isinstance(guard.get("passed"), bool)
        and guard.get("passed") is expected_passed,
        f"{label}.passed changed",
    )
    _require(
        guard.get("sampling_mode") in {"ANCHOR_NEIGHBORHOOD", "SEGMENT_EVEN"},
        f"{label}.sampling_mode changed",
    )
    compatible = _integer(
        guard.get("compatible_samples"),
        f"{label}.compatible_samples",
    )
    incompatible = _integer(
        guard.get("incompatible_samples"),
        f"{label}.incompatible_samples",
    )
    unknown = _integer(
        guard.get("unknown_samples"),
        f"{label}.unknown_samples",
    )
    _require(
        compatible >= 0 and incompatible >= 0 and unknown >= 0,
        f"{label} sample counters are malformed",
    )
    judged = compatible + incompatible
    expected_fraction = incompatible / float(judged) if judged else 1.0
    _close(
        guard.get("incompatible_fraction"),
        round(expected_fraction, 6),
        f"{label}.incompatible_fraction",
        tolerance=1e-6,
    )
    evidence = guard.get("evidence")
    _require(
        isinstance(evidence, list) and bool(evidence),
        f"{label}.evidence is empty",
    )
    statuses = {"COMPATIBLE": 0, "INCOMPATIBLE": 0, "UNKNOWN": 0}
    bbox_times = [
        _finite(bbox.get("t"), f"{label}.bbox.t")
        for bbox in (segment.get("bboxes") or [])
        if isinstance(bbox, Mapping)
    ]
    window_start = _finite(segment.get("window_start"), f"{label}.window_start")
    window_end = _finite(segment.get("window_end"), f"{label}.window_end")
    for index, raw_evidence in enumerate(evidence):
        item = _mapping(raw_evidence, f"{label}.evidence[{index}]")
        status = str(item.get("status") or "")
        _require(status in statuses, f"{label}.evidence[{index}].status changed")
        timestamp = _finite(
            item.get("time_sec"),
            f"{label}.evidence[{index}].time_sec",
        )
        _require(
            window_start <= timestamp <= window_end,
            f"{label}.evidence[{index}] is outside its window",
        )
        if require_bbox_membership:
            _require(
                any(
                    math.isclose(timestamp, bbox_time, abs_tol=1e-5)
                    for bbox_time in bbox_times
                ),
                f"{label}.evidence[{index}] is not tied to a retained bbox",
            )
        signature_value = item.get("signature")
        similarity_value = item.get("similarity")
        _require(
            (signature_value is None) == (similarity_value is None),
            f"{label}.evidence[{index}] signature/similarity pairing changed",
        )
        if signature_value is None:
            expected_status = "UNKNOWN"
        else:
            observed_signature = _color_signature(
                signature_value,
                label=f"{label}.evidence[{index}].signature",
            )
            expected_similarity = _signature_similarity(
                anchor_signature,
                observed_signature,
            )
            _close(
                similarity_value,
                round(expected_similarity, 6),
                f"{label}.evidence[{index}].similarity",
                tolerance=1e-6,
            )
            compatibility = _signatures_compatible(
                anchor_signature,
                observed_signature,
                confidence_gate=confidence_gate,
            )
            expected_status = (
                "COMPATIBLE"
                if compatibility is True
                else ("INCOMPATIBLE" if compatibility is False else "UNKNOWN")
            )
        _require(
            status == expected_status,
            f"{label}.evidence[{index}].status disagrees with its signature",
        )
        statuses[status] += 1
    _require(
        statuses["COMPATIBLE"] == compatible
        and statuses["INCOMPATIBLE"] == incompatible
        and statuses["UNKNOWN"] == unknown,
        f"{label} evidence counters disagree",
    )
    _require(
        len(evidence) == compatible + incompatible + unknown,
        f"{label} evidence length disagrees",
    )
    derived_passed = (
        judged >= MIN_GUARD_COMPATIBLE_SAMPLES
        and compatible >= MIN_GUARD_COMPATIBLE_SAMPLES
        and expected_fraction <= MAX_GUARD_INCOMPATIBLE_FRACTION
    )
    _require(
        expected_passed is derived_passed,
        f"{label}.passed disagrees with independently derived evidence",
    )
    expected_reason_codes: list[str] = []
    if (
        judged < MIN_GUARD_COMPATIBLE_SAMPLES
        or compatible < MIN_GUARD_COMPATIBLE_SAMPLES
    ):
        expected_reason_codes.append("INSUFFICIENT_KIT_COLOR_EVIDENCE")
    if incompatible:
        expected_reason_codes.append("KIT_COLOR_INCONSISTENT_WITH_ANCHOR")
    _require(
        guard.get("reason_codes") == expected_reason_codes,
        f"{label}.reason_codes disagree with independently derived evidence",
    )
    return guard


def _retained_identity(
    segment: Mapping[str, Any],
    *,
    expected_identity_id: str,
) -> bool:
    reid = segment.get("reid")
    reid = reid if isinstance(reid, Mapping) else {}
    guard = reid.get("kit_color_guard")
    return bool(
        segment.get("bboxes")
        and segment.get("selected_track_id") is not None
        and segment.get("identity_id") == expected_identity_id
        and segment.get("identity_status") == "ACCEPTED"
        and reid.get("status") == "ACCEPTED"
        and reid.get("identity_id") == expected_identity_id
        and reid.get("validated") is False
        and isinstance(guard, Mapping)
        and guard.get("passed") is True
    )


def _compare_anchor(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
    include_frame_key: bool = True,
) -> None:
    if include_frame_key:
        _require(
            actual.get("frame_key") == expected["frame_key"],
            f"{label}.frame_key changed",
        )
    _close(actual.get("t"), expected["t"], f"{label}.t", tolerance=0.01)
    for key, value in expected["bbox"].items():
        _close(actual.get(key), value, f"{label}.{key}", tolerance=1e-9)


def _validate_guard_attestation(
    *,
    guard: Mapping[str, Any],
    segments_by_window: Mapping[int, Mapping[str, Any]],
    accepted_segments: Sequence[Mapping[str, Any]],
    matches_by_anchor: Mapping[int, Mapping[str, Any]],
    expected_by_anchor: Mapping[int, Mapping[str, Any]],
    anchor_acquisition: Mapping[str, Any],
) -> None:
    _require(
        guard.get("version") == GUARD_VERSION
        and guard.get("validated") is False
        and guard.get("status") == "APPLIED",
        "Kit-colour guard status is invalid",
    )
    _require(
        guard.get("prototype_status") == "SELECTED",
        "Kit-colour guard prototype was not selected",
    )
    reason_codes = guard.get("reason_codes")
    _require(
        isinstance(reason_codes, list)
        and "TEAM_COLOR_GUARD_EXPERIMENTAL" in reason_codes,
        "Kit-colour guard reason codes are incomplete",
    )
    seed_anchor_id = _integer(
        anchor_acquisition.get("seed_anchor_id"),
        "anchor_acquisition.seed_anchor_id",
    )
    _require(
        guard.get("seed_anchor_id") == seed_anchor_id,
        "Kit-colour guard seed anchor differs from acquisition",
    )
    guard_anchor_id = _integer(
        guard.get("guard_anchor_id"),
        "team_color_guard.guard_anchor_id",
    )
    _require(
        guard_anchor_id in matches_by_anchor and guard_anchor_id in expected_by_anchor,
        "Kit-colour guard anchor is not canonical",
    )
    guard_match = matches_by_anchor[guard_anchor_id]
    _require(
        guard_match.get("status") == "MATCHED",
        "Kit-colour guard anchor was not matched",
    )
    guard_window = _strict_window_index(
        guard_match.get("window_index"),
        label="team_color_guard.guard_anchor_window",
        total=EXPECTED_WINDOWS,
    )
    guard_segment = segments_by_window[guard_window]
    _require(
        str(guard_match.get("local_track_id")) in _track_id_set(guard_segment),
        "Kit-colour guard anchor local track is not retained",
    )

    anchor_signature = _color_signature(
        guard.get("anchor_signature"),
        label="team_color_guard.anchor_signature",
    )
    confidence_gate = _finite(
        guard.get("prototype_confidence_gate"),
        "team_color_guard.prototype_confidence_gate",
    )
    _require(
        0.0 <= confidence_gate <= 1.0
        and _finite(
            anchor_signature.get("confidence"),
            "team_color_guard.anchor_signature.confidence",
        )
        >= confidence_gate,
        "Kit-colour guard prototype confidence is insufficient",
    )
    anchor_candidates = guard.get("anchor_candidates")
    _require(
        isinstance(anchor_candidates, list) and len(anchor_candidates) == 2,
        "Kit-colour guard anchor candidates are incomplete",
    )
    candidates_by_anchor: dict[int, Mapping[str, Any]] = {}
    selected_candidates: list[Mapping[str, Any]] = []
    usable_candidates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index, raw_candidate in enumerate(anchor_candidates):
        label = f"team_color_guard.anchor_candidates[{index}]"
        candidate = _mapping(
            raw_candidate,
            label,
        )
        anchor_id = _integer(
            candidate.get("anchor_id"),
            f"{label}.anchor_id",
        )
        _require(
            anchor_id in matches_by_anchor and anchor_id not in candidates_by_anchor,
            "Kit-colour guard anchor candidate IDs changed",
        )
        _require(
            candidate.get("match_status") == "MATCHED",
            f"Kit-colour guard anchor {anchor_id} was not matched",
        )
        match = matches_by_anchor[anchor_id]
        window_index = _strict_window_index(
            match.get("window_index"),
            label=f"{label}.matched_window_index",
            total=EXPECTED_WINDOWS,
        )
        window_indices = candidate.get("window_indices")
        _require(
            window_indices == [window_index],
            f"Kit-colour guard anchor {anchor_id} window binding changed",
        )
        segment = segments_by_window[window_index]
        _require(
            str(match.get("local_track_id")) in _track_id_set(segment),
            f"Kit-colour guard anchor {anchor_id} local track is not retained",
        )
        _require(
            candidate.get("is_seed") is (anchor_id == seed_anchor_id),
            f"Kit-colour guard anchor {anchor_id} seed binding changed",
        )
        geometry, geometry_passed = _validate_anchor_candidate_geometry(
            candidate.get("geometry"),
            segment=segment,
            expected_anchor=expected_by_anchor[anchor_id],
            label=f"{label}.geometry",
        )
        signature_value = candidate.get("signature")
        signature = (
            None
            if signature_value is None
            else _color_signature(signature_value, label=f"{label}.signature")
        )
        expected_reason_codes: list[str] = []
        if not geometry_passed:
            expected_reason_codes.extend(geometry["reason_codes"])
        elif signature is None:
            expected_reason_codes.append("ANCHOR_KIT_COLOR_UNAVAILABLE")
        elif float(signature["confidence"]) < confidence_gate:
            expected_reason_codes.append("ANCHOR_KIT_COLOR_LOW_CONFIDENCE")
        _require(
            candidate.get("reason_codes") == expected_reason_codes,
            f"Kit-colour guard anchor {anchor_id} reason codes disagree",
        )
        usable = bool(
            geometry_passed
            and signature is not None
            and float(signature["confidence"]) >= confidence_gate
        )
        expected_state = (
            "SELECTED"
            if usable and anchor_id == guard_anchor_id
            else ("USABLE" if usable else "REJECTED")
        )
        _require(
            candidate.get("state") == expected_state,
            f"Kit-colour guard anchor {anchor_id} state disagrees with its evidence",
        )
        if candidate.get("state") == "SELECTED":
            selected_candidates.append(candidate)
        if usable:
            usable_candidates.append((candidate, signature))
        candidates_by_anchor[anchor_id] = candidate
    _require(
        set(candidates_by_anchor) == set(matches_by_anchor),
        "Kit-colour guard anchor candidates differ from anchor matches",
    )
    _require(
        len(selected_candidates) == 1
        and selected_candidates[0].get("anchor_id") == guard_anchor_id,
        "Kit-colour guard selected anchor binding changed",
    )
    selected_candidate = selected_candidates[0]
    selected_by_rank, selected_signature = max(
        usable_candidates,
        key=lambda item: (
            float(item[1]["confidence"]),
            float(item[1]["quality"]),
            item[0].get("is_seed") is True,
        ),
    )
    _require(
        selected_by_rank.get("anchor_id") == guard_anchor_id,
        "Kit-colour guard selected candidate is not the backend-ranked prototype",
    )
    computed_conflicts: list[dict[str, Any]] = []
    for left_index, (left_candidate, left_signature) in enumerate(usable_candidates):
        for right_candidate, right_signature in usable_candidates[left_index + 1 :]:
            compatible = (
                False
                if left_signature["dominant_family"]
                != right_signature["dominant_family"]
                else _signatures_compatible(
                    left_signature,
                    right_signature,
                    confidence_gate=confidence_gate,
                )
            )
            if compatible is False:
                computed_conflicts.append(
                    {
                        "left_anchor_id": left_candidate["anchor_id"],
                        "right_anchor_id": right_candidate["anchor_id"],
                        "similarity": round(
                            _signature_similarity(
                                left_signature,
                                right_signature,
                            ),
                            6,
                        ),
                    }
                )
    _require(
        guard.get("anchor_conflicts") == computed_conflicts,
        "Kit-colour guard anchor conflicts disagree with candidate signatures",
    )
    _require(
        not computed_conflicts,
        "Kit-colour guard reported conflicting usable anchor signatures",
    )
    anchor_geometry = _guard_geometry(
        guard.get("anchor_geometry"),
        label="team_color_guard.anchor_geometry",
    )
    _require(
        selected_signature == anchor_signature
        and selected_candidate.get("signature") == anchor_signature
        and selected_candidate.get("geometry") == anchor_geometry,
        "Kit-colour guard selected prototype evidence differs",
    )

    decisions = guard.get("decisions")
    _require(
        isinstance(decisions, list) and bool(decisions),
        "Kit-colour guard decisions are missing",
    )
    segments_checked = _integer(
        guard.get("segments_checked"),
        "team_color_guard.segments_checked",
    )
    segments_rejected = _integer(
        guard.get("segments_rejected"),
        "team_color_guard.segments_rejected",
    )
    post_guard_segments = _integer(
        guard.get("post_guard_segments_with_player"),
        "team_color_guard.post_guard_segments_with_player",
    )
    _require(
        segments_checked == len(decisions)
        and post_guard_segments == len(accepted_segments)
        and segments_rejected == segments_checked - post_guard_segments
        and segments_checked >= post_guard_segments,
        "Kit-colour guard counters disagree",
    )
    decisions_by_window: dict[int, Mapping[str, Any]] = {}
    for index, raw_decision in enumerate(decisions):
        decision = _mapping(
            raw_decision,
            f"team_color_guard.decisions[{index}]",
        )
        window_index = _strict_window_index(
            decision.get("window_index"),
            label=f"team_color_guard.decisions[{index}].window_index",
            total=EXPECTED_WINDOWS,
        )
        _require(
            window_index not in decisions_by_window,
            "Duplicate kit-colour guard decision",
        )
        segment = segments_by_window[window_index]
        decision_passed = decision.get("passed") is True
        _require(
            isinstance(decision.get("passed"), bool),
            f"Kit-colour guard decision {window_index} lacks a boolean pass",
        )
        _validate_segment_guard(
            decision,
            segment=segment,
            anchor_signature=anchor_signature,
            confidence_gate=confidence_gate,
            label=f"team_color_guard.decisions[{window_index}]",
            require_bbox_membership=bool(segment.get("bboxes")),
            expected_passed=decision_passed,
        )
        decisions_by_window[window_index] = decision

    for segment in accepted_segments:
        window_index = _integer(
            segment.get("window_index"),
            "retained guard window_index",
        )
        decision = decisions_by_window.get(window_index)
        _require(
            decision is not None and decision.get("passed") is True,
            f"Retained window {window_index} lacks a passing guard decision",
        )
        reid = _mapping(segment.get("reid"), f"window {window_index}.reid")
        segment_guard = _validate_segment_guard(
            reid.get("kit_color_guard"),
            segment=segment,
            anchor_signature=anchor_signature,
            confidence_gate=confidence_gate,
            label=f"window {window_index}.kit_color_guard",
            require_bbox_membership=True,
            expected_passed=True,
        )
        decision_payload = {
            key: value for key, value in decision.items() if key != "window_index"
        }
        _require(
            dict(segment_guard) == decision_payload,
            f"Window {window_index} guard differs from its decision",
        )


def validate_regression_result(
    *,
    job_envelope: Mapping[str, Any],
    selection_payload: Mapping[str, Any],
    enqueue_envelope: Mapping[str, Any],
    fixture_before_envelope: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _expected_selections(selection_payload)
    enqueue_attempt = _attempt_id(enqueue_envelope, label="enqueue")
    before = _envelope_data(fixture_before_envelope, "fixture_before")
    before_job_id = _job_id(fixture_before_envelope, label="fixture_before")
    enqueue_job_id = _job_id(enqueue_envelope, label="enqueue")
    final_job_id = _job_id(job_envelope, label="job_final")
    _require(
        before_job_id == enqueue_job_id == final_job_id,
        "Regression fixture/enqueue/final job ids differ",
    )
    previous_attempt_values = _analysis_attempt_values(
        before,
        path="fixture_before",
    )
    previous_attempts = {
        str(value).strip()
        for value in previous_attempt_values.values()
        if value is not None and str(value).strip()
    }
    _require(
        enqueue_attempt not in previous_attempts,
        "Enqueue reused a previous nested analysis_attempt_id",
    )

    job = _envelope_data(job_envelope, "job_final")
    result = _mapping(job.get("result"), "job_final.result")
    tracking = _mapping(result.get("tracking"), "result.tracking")
    outcome = _mapping(result.get("analysis_outcome"), "result.analysis_outcome")
    target = _mapping(job.get("target"), "job_final.target")
    target_tracking = _mapping(
        target.get("tracking"),
        "job_final.target.tracking",
    )
    progress = _mapping(job.get("progress"), "job_final.progress")
    for label, value in (
        ("result", result.get("analysis_attempt_id")),
        ("tracking", tracking.get("analysis_attempt_id")),
        ("analysis_outcome", outcome.get("analysis_attempt_id")),
        ("target", target.get("analysis_attempt_id")),
        ("target.tracking", target_tracking.get("analysis_attempt_id")),
        ("progress", progress.get("analysis_attempt_id")),
    ):
        _require(
            value == enqueue_attempt,
            f"{label} analysis_attempt_id does not match enqueue",
        )
    final_attempt_values = _analysis_attempt_values(job, path="job_final")
    mismatched_attempts = {
        path: value
        for path, value in final_attempt_values.items()
        if value != enqueue_attempt
    }
    _require(
        not mismatched_attempts,
        f"Final result contains mismatched analysis attempts: {mismatched_attempts}",
    )
    _require(
        target.get("confirmed") is True and target.get("full_match_mode") is True,
        "Final target lost canonical full-match confirmation",
    )
    _require(
        progress.get("step") == "DONE"
        and _integer(progress.get("pct"), "job_final.progress.pct") == 100,
        "Final progress is not terminal",
    )
    _require(
        job.get("error") in (None, "") and job.get("failure_reason") in (None, ""),
        "Final job contains a failure/error",
    )

    summary = _mapping(tracking.get("reid_summary"), "tracking.reid_summary")
    guard = _mapping(summary.get("team_color_guard"), "team_color_guard")
    _require(job.get("status") == "PARTIAL", "Regression must finish PARTIAL")
    _require(tracking.get("mode") == "full_match_windowed", "Unexpected tracking mode")
    _require(tracking.get("tracking_success") is True, "Tracking did not succeed")
    _require(
        tracking.get("action_required") is None, "Successful tracking requires action"
    )
    _require(tracking.get("metrics_scope") == "selected_player", "Wrong metrics scope")
    _require(
        _integer(tracking.get("anchors_total"), "anchors_total") == 2
        and _integer(tracking.get("anchors_matched"), "anchors_matched") == 2,
        "Both fixture anchors must match",
    )
    _require(
        _integer(tracking.get("segments_total"), "segments_total") == EXPECTED_WINDOWS
        and _integer(tracking.get("windows_processed"), "windows_processed")
        == EXPECTED_WINDOWS,
        "All fixture windows must be processed",
    )
    _require(
        _integer(summary.get("processing_failures"), "processing_failures") == 0,
        "Window processing failures were reported",
    )

    runtime_profile = _mapping(
        tracking.get("runtime_profile"),
        "tracking.runtime_profile",
    )
    _close(
        runtime_profile.get("duration_sec"),
        EXPECTED_DURATION_SEC,
        "runtime_profile.duration_sec",
        tolerance=0.01,
    )
    _close(
        runtime_profile.get("window_sec"),
        EXPECTED_WINDOW_SEC,
        "runtime_profile.window_sec",
        tolerance=0.01,
    )
    _close(
        runtime_profile.get("overlap_sec"),
        EXPECTED_OVERLAP_SEC,
        "runtime_profile.overlap_sec",
        tolerance=0.01,
    )
    fps = _integer(runtime_profile.get("fps"), "runtime_profile.fps")
    _require(fps in {1, 2}, f"Unexpected runtime fps: {fps}")
    _require(
        _integer(tracking.get("fps"), "tracking.fps") == fps,
        "Tracking fps differs from runtime profile",
    )

    anchor_matches = tracking.get("anchor_matches")
    _require(
        isinstance(anchor_matches, list) and len(anchor_matches) == 2,
        "anchor_matches must contain exactly two entries",
    )
    sorted_matches = sorted(
        (_mapping(item, "anchor_match") for item in anchor_matches),
        key=lambda item: _integer(item.get("anchor_id"), "anchor_match.anchor_id"),
    )
    matches_by_anchor: dict[int, Mapping[str, Any]] = {}
    matched_anchor_indices: set[int] = set()
    for index, (match, selection) in enumerate(
        zip(sorted_matches, expected),
        start=1,
    ):
        _require(match.get("anchor_id") == index, "Anchor IDs changed")
        _require(match.get("status") == "MATCHED", f"Anchor {index} did not match")
        _require(
            match.get("frame_key") == selection["frame_key"],
            f"Anchor {index} frame_key changed",
        )
        _close(
            match.get("time_sec"),
            selection["t"],
            f"anchor_matches[{index}].time_sec",
            tolerance=0.01,
        )
        _require(
            match.get("source")
            == ("primary_player_ref" if index == 1 else "selection"),
            f"Anchor {index} source changed",
        )
        _require(
            match.get("local_track_id") is not None,
            f"Anchor {index} lacks a local track",
        )
        matches_by_anchor[index] = match
        matched_anchor_indices.add(
            _strict_window_index(
                match.get("window_index"),
                label=f"anchor_matches[{index}].window_index",
                total=EXPECTED_WINDOWS,
            )
        )
    _require(len(matched_anchor_indices) == 2, "Anchor windows are not distinct")
    summary_identity_id = str(summary.get("identity_id") or "").strip()
    _require(bool(summary_identity_id), "ReID summary identity_id is missing")
    summary_matches = summary.get("anchor_matches")
    _require(
        isinstance(summary_matches, list) and summary_matches == anchor_matches,
        "ReID summary anchor matches differ from tracking",
    )

    anchors_used = _mapping(tracking.get("anchors_used"), "tracking.anchors_used")
    used_selections = anchors_used.get("selections")
    _require(
        isinstance(used_selections, list) and len(used_selections) == 2,
        "anchors_used.selections changed",
    )
    for index, (used, selection) in enumerate(
        zip(
            sorted(
                (_mapping(item, "anchors_used.selection") for item in used_selections),
                key=lambda item: _finite(item.get("t"), "anchors_used.selection.t"),
            ),
            expected,
        )
    ):
        _compare_anchor(used, selection, label=f"anchors_used.selections[{index}]")
    player_ref = _mapping(anchors_used.get("player_ref"), "anchors_used.player_ref")
    _compare_anchor(
        player_ref,
        expected[0],
        label="anchors_used.player_ref",
        include_frame_key=False,
    )

    anchor_acquisition = _mapping(
        tracking.get("anchor_acquisition"),
        "tracking.anchor_acquisition",
    )
    anchor_fps = _integer(anchor_acquisition.get("fps"), "anchor_acquisition.fps")
    _require(anchor_fps >= MIN_ANCHOR_FPS, "Anchor acquisition fps is too low")
    _require(
        anchor_acquisition.get("detector_model") == EXPECTED_ANCHOR_MODEL,
        "Anchor acquisition detector is not yolo11s.pt",
    )
    _require(
        _integer(
            anchor_acquisition.get("windows_processed"),
            "anchor_acquisition.windows_processed",
        )
        == 2,
        "Anchor acquisition did not process both anchor windows",
    )
    _require(
        anchor_acquisition.get("seed_anchor_id") == 1,
        "Unexpected seed anchor id",
    )
    seed_window = _strict_window_index(
        anchor_acquisition.get("seed_window_index"),
        label="anchor_acquisition.seed_window_index",
        total=EXPECTED_WINDOWS,
    )
    _require(
        seed_window == sorted_matches[0].get("window_index"),
        "Seed window differs from the primary anchor window",
    )
    seed_anchor = _mapping(
        anchor_acquisition.get("seed_anchor"),
        "anchor_acquisition.seed_anchor",
    )
    _require(seed_anchor.get("anchor_id") == 1, "Seed anchor id changed")
    _require(seed_anchor.get("window_index") == seed_window, "Seed window changed")
    _compare_anchor(seed_anchor, expected[0], label="anchor_acquisition.seed_anchor")

    segments = tracking.get("segments")
    _require(
        isinstance(segments, list) and len(segments) == EXPECTED_WINDOWS,
        "Tracking segments payload is incomplete",
    )
    segments_by_window: dict[int, Mapping[str, Any]] = {}
    graph: dict[int, dict[str, Any]] = {}
    all_bboxes: list[dict[str, float]] = []
    for list_index, raw_segment in enumerate(segments):
        segment = _mapping(raw_segment, f"segments[{list_index}]")
        window_index = _strict_window_index(
            segment.get("window_index"),
            label=f"segments[{list_index}].window_index",
            total=EXPECTED_WINDOWS,
        )
        _require(window_index not in segments_by_window, "Duplicate window_index")
        window_start = _finite(
            segment.get("window_start"), f"window {window_index} start"
        )
        window_end = _finite(segment.get("window_end"), f"window {window_index} end")
        expected_start = window_index * (EXPECTED_WINDOW_SEC - EXPECTED_OVERLAP_SEC)
        expected_end = min(EXPECTED_DURATION_SEC, expected_start + EXPECTED_WINDOW_SEC)
        _close(
            window_start, expected_start, f"window {window_index} start", tolerance=0.01
        )
        _close(window_end, expected_end, f"window {window_index} end", tolerance=0.01)
        direction = str(segment.get("direction") or "").lower()
        processing_direction = str(segment.get("processing_direction") or "").lower()
        parent = _strict_window_index(
            segment.get("parent_window_index"),
            label=f"window {window_index} parent_window_index",
            total=EXPECTED_WINDOWS,
            allow_none=True,
        )
        if direction == "anchor":
            _require(
                processing_direction in {"anchor", "forward", "backward"},
                f"Anchor window {window_index} has malformed direction",
            )
            if processing_direction == "anchor":
                _require(parent is None, "Seed anchor must not have a parent")
            else:
                expected_parent = window_index + (
                    -1 if processing_direction == "forward" else 1
                )
                _require(parent == expected_parent, "Anchor parent is non-contiguous")
        else:
            _require(direction in {"forward", "backward"}, "Malformed window direction")
            _require(processing_direction == direction, "Processing direction changed")
            expected_parent = window_index + (
                -1 if processing_direction == "forward" else 1
            )
            _require(parent == expected_parent, "Window parent is non-contiguous")
        segment_bboxes = segment.get("bboxes") or []
        _require(isinstance(segment_bboxes, list), "Segment bboxes are malformed")
        for bbox_index, raw_bbox in enumerate(segment_bboxes):
            bbox = _mapping(raw_bbox, f"window {window_index} bbox {bbox_index}")
            all_bboxes.append(
                _bbox_geometry(
                    bbox,
                    label=f"window {window_index} bbox {bbox_index}",
                    window_start=window_start,
                    window_end=window_end,
                    duration=EXPECTED_DURATION_SEC,
                    tolerance=1.0 / fps,
                )
            )
        segments_by_window[window_index] = segment
        graph[window_index] = {
            "direction": direction,
            "processing_direction": processing_direction,
            "parent_window_index": parent,
        }
    _require(
        set(segments_by_window) == set(range(EXPECTED_WINDOWS)),
        "Window graph is incomplete",
    )
    anchor_window_indices = {
        index for index, node in graph.items() if node["direction"] == "anchor"
    }
    _require(
        anchor_window_indices == matched_anchor_indices,
        "Manual-anchor windows differ from anchor_matches",
    )
    for anchor_id, match in matches_by_anchor.items():
        window_index = _integer(
            match.get("window_index"),
            f"anchor {anchor_id} window_index",
        )
        _require(
            str(match.get("local_track_id"))
            in _track_id_set(segments_by_window[window_index]),
            f"Anchor {anchor_id} local track is not in its retained segment",
        )
    _require(
        summary.get("anchor_window_index") == seed_window
        and str(summary.get("anchor_local_track_id"))
        == str(matches_by_anchor[1].get("local_track_id")),
        "ReID summary seed track binding changed",
    )

    accepted = [segment for segment in segments if segment.get("bboxes")]
    _require(bool(accepted), "No selected-player segment was retained")
    for anchor_index in matched_anchor_indices:
        _require(
            _retained_identity(
                segments_by_window[anchor_index],
                expected_identity_id=summary_identity_id,
            ),
            f"Anchor window {anchor_index} lacks guarded identity proof",
        )
    connected_autonomous: set[int] = set()
    for raw_segment in accepted:
        segment = _mapping(raw_segment, "accepted segment")
        window_index = _integer(segment.get("window_index"), "accepted window_index")
        node = graph[window_index]
        if node["direction"] == "anchor":
            continue
        _require(
            _retained_identity(
                segment,
                expected_identity_id=summary_identity_id,
            ),
            f"Window {window_index} lacks guarded identity proof",
        )
        chain_direction = node["processing_direction"]
        current = window_index
        visited: set[int] = set()
        while True:
            _require(current not in visited, "Cycle in retained ReID chain")
            visited.add(current)
            current_segment = segments_by_window[current]
            current_node = graph[current]
            _require(
                _retained_identity(
                    current_segment,
                    expected_identity_id=summary_identity_id,
                ),
                f"Retained chain crosses rejected parent {current}",
            )
            if current_node["direction"] == "anchor":
                _require(
                    current in matched_anchor_indices, "Chain reaches unverified anchor"
                )
                _require(
                    current_node["processing_direction"] in {"anchor", chain_direction},
                    "Retained chain changes direction at anchor",
                )
                connected_autonomous.add(window_index)
                break
            _require(
                current_node["direction"] == chain_direction
                and current_node["processing_direction"] == chain_direction,
                "Retained chain changes direction",
            )
            parent = current_node["parent_window_index"]
            _require(parent is not None, "Retained chain ends without an anchor")
            current = parent

    _require(
        _integer(tracking.get("bboxes_count"), "bboxes_count") == len(all_bboxes),
        "Bounding-box counter mismatch",
    )
    _require(
        _integer(tracking.get("segments_with_player"), "segments_with_player")
        == len(accepted),
        "Retained-segment counter mismatch",
    )

    anchor_windows = [
        (
            _finite(segments_by_window[index].get("window_start"), "anchor start"),
            _finite(segments_by_window[index].get("window_end"), "anchor end"),
        )
        for index in matched_anchor_indices
    ]
    autonomous_times: set[float] = set()
    autonomous_segments: list[Mapping[str, Any]] = []
    for raw_segment in accepted:
        segment = _mapping(raw_segment, "accepted segment")
        window_index = _integer(segment.get("window_index"), "accepted window_index")
        if graph[window_index]["direction"] == "anchor":
            continue
        _require(
            window_index in connected_autonomous,
            f"Disconnected window {window_index} was retained",
        )
        reid = _mapping(segment.get("reid"), f"window {window_index}.reid")
        _require(
            reid.get("tracklet_scope") == "MOTION_CONTINUOUS_STRONG_OVERLAP",
            f"Window {window_index} lacks scoped overlap tracklet",
        )
        outside = {
            round(_finite(bbox.get("t"), "autonomous bbox timestamp"), 6)
            for bbox in segment.get("bboxes") or []
            if all(
                _finite(bbox.get("t"), "autonomous bbox timestamp") < start - 1.0 / fps
                or _finite(bbox.get("t"), "autonomous bbox timestamp") > end + 1.0 / fps
                for start, end in anchor_windows
            )
        }
        _require(
            _integer(
                reid.get("autonomous_bboxes_count"),
                f"window {window_index}.autonomous_bboxes_count",
            )
            == len(outside),
            "Segment autonomous counter mismatch",
        )
        if outside:
            autonomous_segments.append(segment)
            autonomous_times.update(outside)
    _require(
        len(accepted) >= MIN_RELEASE_RETAINED_SEGMENTS,
        "Release tracking retained too few windows",
    )
    _require(
        len(autonomous_segments) >= MIN_RELEASE_AUTONOMOUS_SEGMENTS,
        "Release tracking retained too few autonomous windows",
    )
    _require(
        len(autonomous_times) >= MIN_RELEASE_AUTONOMOUS_BBOXES,
        "Release tracking retained too few autonomous observations",
    )
    _require(
        _integer(
            tracking.get("autonomous_bboxes_count"),
            "tracking.autonomous_bboxes_count",
        )
        == len(autonomous_times),
        "Autonomous bbox counter mismatch",
    )
    _require(
        _integer(
            tracking.get("autonomous_segments_with_player"),
            "tracking.autonomous_segments_with_player",
        )
        == len(autonomous_segments),
        "Autonomous segment counter mismatch",
    )
    _require(
        tracking.get("tracking_scope_status") == "CROSS_WINDOW_EVIDENCE",
        "Tracking scope is not cross-window",
    )

    unique_frames = {int(round(float(item["t"]) * fps)) for item in all_bboxes}
    expected_frames = max(1, int(round(EXPECTED_DURATION_SEC * fps)))
    coverage = _clamp(len(unique_frames) / float(expected_frames) * 100.0)
    rounded_coverage = round(coverage, 2)
    for key in ("coverage_pct", "coverage_pct_total"):
        _close(tracking.get(key), rounded_coverage, f"tracking.{key}", tolerance=0.01)
    _require(
        coverage >= MIN_RELEASE_COVERAGE_PCT,
        "Release tracking coverage is sparse",
    )
    times = sorted({float(item["t"]) for item in all_bboxes})
    gaps = [max(0.0, times[0])]
    gaps.extend(
        max(0.0, current - previous) for previous, current in zip(times, times[1:])
    )
    gaps.append(max(0.0, EXPECTED_DURATION_SEC - times[-1]))
    largest_gap = round(max(gaps), 2)
    _close(
        tracking.get("largest_gap_sec"),
        largest_gap,
        "tracking.largest_gap_sec",
        tolerance=0.01,
    )
    _require(
        tracking.get("tracking_status") == "SUCCEEDED",
        "Release tracking status is not SUCCEEDED",
    )
    _require(
        tracking.get("partial") is False,
        "Release tracking is marked partial",
    )
    _require(
        tracking.get("partial_reason") is None,
        "Release tracking contains a partial reason",
    )

    leaks = _scan_forbidden_values(result)
    _require(not leaks, f"Legacy or unvalidated metrics leaked: {leaks}")
    _require(
        result.get("player_evaluation_available") is False,
        "Evaluation must be withheld",
    )
    _require(
        result.get("legacy_scores_suppressed") is True,
        "Legacy scores are not suppressed",
    )
    _require(
        result.get("report") is None and result.get("player_runs") is None,
        "Report/player_runs leaked",
    )
    for key in ("radar", "breakdown", "skills_computed"):
        _require(result.get(key) in ({}, None), f"Unvalidated {key} leaked")
    _require(result.get("skills_missing") in ([], None), "skills_missing leaked")

    score_provenance = _mapping(result.get("score_provenance"), "score_provenance")
    tracking_quality = _mapping(result.get("tracking_quality"), "tracking_quality")
    quality_provenance = _mapping(
        tracking_quality.get("provenance"),
        "tracking_quality.provenance",
    )
    _require(
        score_provenance.get("validated_player_score") is False
        and score_provenance.get("metrics_scope") == "selected_player",
        "Score provenance is invalid",
    )
    _require(quality_provenance == score_provenance, "Quality provenance differs")
    tracking_signals = _mapping(result.get("tracking_signals"), "tracking_signals")
    _require(
        tracking_quality.get("signals") == tracking_signals, "Tracking signals differ"
    )
    expected_motion = _image_motion(all_bboxes)
    _require(
        tracking_signals.get("image_motion") == expected_motion,
        "Image-motion metrics are not derived from selected-player bboxes",
    )
    evidence_metrics = _mapping(result.get("evidence_metrics"), "evidence_metrics")
    _require(
        evidence_metrics.get("image_motion") == expected_motion,
        "Evidence image-motion metrics differ",
    )
    _close(
        tracking_signals.get("coverage_pct"),
        rounded_coverage,
        "tracking_signals.coverage_pct",
        tolerance=0.01,
    )
    _close(
        tracking_signals.get("coverage_ratio"),
        round(coverage / 100.0, 6),
        "tracking_signals.coverage_ratio",
        tolerance=1e-6,
    )
    _require(
        _integer(tracking_signals.get("segments_total"), "signals.segments_total")
        == EXPECTED_WINDOWS
        and _integer(
            tracking_signals.get("segments_with_player"),
            "signals.segments_with_player",
        )
        == len(accepted),
        "Tracking signal segment counters differ",
    )
    lost_count = sum(
        len(segment.get("lost_segments") or [])
        for segment in segments
        if isinstance(segment, Mapping)
    )
    continuity = _clamp(100.0 - lost_count * 12.5)
    sample_sufficiency = _clamp(len(all_bboxes) / SAMPLE_TARGET * 100.0)
    expected_evaluation_status = "TRACKING_ONLY"
    expected_score_kind = "tracking_quality"
    _require(
        result.get("evaluation_status") == expected_evaluation_status,
        "Evaluation status changed",
    )
    _require(result.get("score_kind") == expected_score_kind, "Score kind changed")
    _require(
        score_provenance.get("kind") == expected_score_kind, "Provenance kind changed"
    )
    _require(
        tracking_quality.get("status") == expected_evaluation_status,
        "Quality status changed",
    )
    _require(
        tracking_quality.get("score_kind") == expected_score_kind,
        "Quality score kind changed",
    )
    _require(
        tracking_quality.get("player_evaluation_available") is False,
        "Quality claims evaluation",
    )
    result_summary = _mapping(result.get("summary"), "result.summary")
    _require(
        result_summary.get("evaluation_status") == expected_evaluation_status,
        "Summary status changed",
    )
    _require(
        result_summary.get("player_evaluation_available") is False,
        "Summary claims evaluation",
    )

    quality_index = result.get("tracking_quality_index")
    _require(
        tracking_quality.get("tracking_quality_index") == quality_index
        and result_summary.get("tracking_quality_index") == quality_index,
        "Tracking quality index copies disagree",
    )
    _require(
        _integer(tracking_signals.get("samples_used"), "signals.samples_used")
        == len(all_bboxes),
        "Evaluated samples differ from selected-player bboxes",
    )
    _close(
        tracking_signals.get("sample_sufficiency_pct"),
        round(sample_sufficiency, 2),
        "signals.sample_sufficiency_pct",
        tolerance=0.01,
    )
    _close(
        tracking_signals.get("tracklet_continuity_pct"),
        round(continuity, 2),
        "signals.tracklet_continuity_pct",
        tolerance=0.01,
    )
    _require(
        tracking_signals.get("tracklet_continuity_source") == "lost_segments_proxy",
        "Continuity provenance changed",
    )
    _close(
        tracking_signals.get("largest_gap_sec"),
        largest_gap,
        "signals.largest_gap_sec",
        tolerance=0.01,
    )
    expected_quality = round(
        _clamp(coverage * 0.50 + continuity * 0.30 + sample_sufficiency * 0.20),
        1,
    )
    _close(quality_index, expected_quality, "tracking_quality_index", tolerance=0.01)
    expected_confidence = (
        "medium"
        if coverage >= 50.0 and continuity >= 70.0 and len(all_bboxes) >= 60
        else "low"
    )
    _require(
        tracking_quality.get("tracking_confidence") == expected_confidence,
        "Tracking confidence changed",
    )

    expected_outcome = {
        "analysis_attempt_id": enqueue_attempt,
        "pipeline_state": "DONE",
        "tracking_state": "SUCCEEDED",
        "metrics_scope": "selected_player",
        "observed_samples": len(all_bboxes),
        "segments_with_player": len(accepted),
        "autonomous_segments_with_player": len(autonomous_segments),
        "autonomous_bboxes_count": len(autonomous_times),
        "tracking_scope_status": "CROSS_WINDOW_EVIDENCE",
        "windows_processed": EXPECTED_WINDOWS,
        "windows_total": EXPECTED_WINDOWS,
        "anchors_total": 2,
        "anchors_matched": 2,
        "action_required": None,
    }
    mismatches = {
        key: {"expected": value, "actual": outcome.get(key)}
        for key, value in expected_outcome.items()
        if outcome.get(key) != value
    }
    _require(not mismatches, f"analysis_outcome mismatch: {mismatches}")
    for source_name, source in (
        ("reid_summary", summary),
        ("analysis_outcome", outcome),
    ):
        _require(
            _integer(
                source.get("autonomous_bboxes_count"),
                f"{source_name}.autonomous_bboxes_count",
            )
            == len(autonomous_times),
            f"{source_name} autonomous bbox counter mismatch",
        )
        _require(
            _integer(
                source.get("autonomous_segments_with_player"),
                f"{source_name}.autonomous_segments_with_player",
            )
            == len(autonomous_segments),
            f"{source_name} autonomous segment counter mismatch",
        )
        _require(
            source.get("tracking_scope_status") == "CROSS_WINDOW_EVIDENCE",
            f"{source_name} tracking scope mismatch",
        )
    _require(summary.get("validated") is False, "ReID summary claims validation")
    _validate_guard_attestation(
        guard=guard,
        segments_by_window=segments_by_window,
        accepted_segments=accepted,
        matches_by_anchor=matches_by_anchor,
        expected_by_anchor={
            index: selection for index, selection in enumerate(expected, start=1)
        },
        anchor_acquisition=anchor_acquisition,
    )

    return {
        "job_id": job.get("job_id") or job.get("id"),
        "analysis_attempt_id": enqueue_attempt,
        "status": job.get("status"),
        "guard_status": guard.get("status"),
        "segments_total": EXPECTED_WINDOWS,
        "segments_with_player": len(accepted),
        "cross_window_segments": len(autonomous_segments),
        "autonomous_bboxes_count": len(autonomous_times),
        "segments_rejected": int(guard.get("segments_rejected") or 0),
        "anchors_matched": 2,
        "bboxes_count": len(all_bboxes),
        "coverage_pct": rounded_coverage,
        "largest_gap_sec": largest_gap,
        "anchor_geometry": guard.get("anchor_geometry"),
        "anchor_signature": guard.get("anchor_signature"),
        "warnings": job.get("warnings") or [],
        "scores_withheld": True,
    }


def _read_json(path: str) -> Mapping[str, Any]:
    return _mapping(json.loads(Path(path).read_text()), path)


def _write_outputs(report: Mapping[str, Any], *, summary_path: str) -> None:
    Path(summary_path).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a") as handle:
            handle.write(f"guard_status={report['guard_status']}\n")
            handle.write(f"segments_with_player={report['segments_with_player']}\n")
            handle.write(f"segments_rejected={report['segments_rejected']}\n")
            handle.write(f"coverage_pct={report['coverage_pct']}\n")
            handle.write(f"analysis_attempt_id={report['analysis_attempt_id']}\n")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a") as handle:
            handle.write("## ReID production regression\n\n```json\n")
            handle.write(json.dumps(report, indent=2, ensure_ascii=False))
            handle.write("\n```\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    attest = subparsers.add_parser("attest-runtime")
    attest.add_argument("--payload", required=True)
    attest.add_argument("--expected-revision", required=True)
    attest.add_argument("--output")
    validate = subparsers.add_parser("validate-result")
    validate.add_argument("--job-final", required=True)
    validate.add_argument("--selection", required=True)
    validate.add_argument("--enqueue", required=True)
    validate.add_argument("--fixture-before", required=True)
    validate.add_argument("--summary", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "attest-runtime":
            report = validate_runtime_attestation(
                _read_json(args.payload),
                expected_revision=args.expected_revision,
            )
            if args.output:
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            return 0
        report = validate_regression_result(
            job_envelope=_read_json(args.job_final),
            selection_payload=_read_json(args.selection),
            enqueue_envelope=_read_json(args.enqueue),
            fixture_before_envelope=_read_json(args.fixture_before),
        )
        _write_outputs(report, summary_path=args.summary)
        return 0
    except (ValidationError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
