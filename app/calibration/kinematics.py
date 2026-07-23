from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from app.calibration.homography import PitchCalibration


@dataclass(frozen=True)
class MotionThresholds:
    maximum_gap_sec: float = 1.0
    maximum_speed_mps: float = 12.5
    maximum_acceleration_mps2: float = 12.0
    sprint_threshold_mps: float = 7.0
    minimum_sprint_duration_sec: float = 1.0
    pitch_margin_m: float = 2.0
    smoothing_window: int = 3
    minimum_projected_points: int = 10

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_gap_sec",
            "maximum_speed_mps",
            "maximum_acceleration_mps2",
            "sprint_threshold_mps",
            "minimum_sprint_duration_sec",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        margin = float(self.pitch_margin_m)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("pitch_margin_m must be finite and >= 0")
        if self.sprint_threshold_mps > self.maximum_speed_mps:
            raise ValueError(
                "sprint_threshold_mps must not exceed maximum_speed_mps"
            )
        if self.smoothing_window < 1 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd integer")
        if self.minimum_projected_points < 2:
            raise ValueError("minimum_projected_points must be >= 2")


@dataclass(frozen=True)
class CalibratedTrackPoint:
    time_sec: float
    x_m: float
    y_m: float
    calibration_id: str
    confidence: float | None = None
    image_x: float | None = None
    image_y: float | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must not be empty")
        for field_name in ("time_sec", "x_m", "y_m"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.time_sec < 0:
            raise ValueError("time_sec must be >= 0")
        for field_name in ("confidence", "image_x", "image_y"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite when present")

    def to_payload(self) -> dict[str, Any]:
        return {
            "time_sec": round(self.time_sec, 6),
            "x_m": round(self.x_m, 6),
            "y_m": round(self.y_m, 6),
            "calibration_id": self.calibration_id,
            "confidence": self.confidence,
            "image_x": self.image_x,
            "image_y": self.image_y,
        }


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def collect_tracking_bboxes(
    tracking: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    if not isinstance(tracking, Mapping):
        return []
    segments = tracking.get("segments")
    if isinstance(segments, list):
        bboxes: list[Mapping[str, Any]] = []
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            if segment.get("identity_status") == "ABSTAINED":
                continue
            for bbox in segment.get("bboxes") or []:
                if isinstance(bbox, Mapping):
                    bboxes.append(bbox)
        return bboxes
    return [
        bbox
        for bbox in (tracking.get("bboxes") or [])
        if isinstance(bbox, Mapping)
    ]


def _calibration_span(calibration: PitchCalibration) -> float:
    if calibration.start_sec is None or calibration.end_sec is None:
        return float("inf")
    return max(0.0, calibration.end_sec - calibration.start_sec)


def _select_calibration(
    calibrations: Sequence[PitchCalibration],
    time_sec: float,
) -> PitchCalibration | None:
    candidates = [
        calibration
        for calibration in calibrations
        if calibration.validated and calibration.contains_time(time_sec)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda calibration: (
            _calibration_span(calibration),
            calibration.rmse_m,
            calibration.camera_segment_id,
        ),
    )


def project_tracking_footpoints(
    bboxes: Iterable[Mapping[str, Any]],
    calibrations: Sequence[PitchCalibration],
    *,
    thresholds: MotionThresholds | None = None,
) -> tuple[list[CalibratedTrackPoint], dict[str, int]]:
    thresholds = thresholds or MotionThresholds()
    validated_calibrations = [
        calibration for calibration in calibrations if calibration.validated
    ]
    counters = {
        "input_bboxes": 0,
        "invalid_bbox": 0,
        "missing_calibration": 0,
        "outside_pitch": 0,
        "projected_points": 0,
    }
    points: list[CalibratedTrackPoint] = []
    for bbox in bboxes:
        counters["input_bboxes"] += 1
        time_sec = _finite(bbox.get("t"))
        x = _finite(bbox.get("x"))
        y = _finite(bbox.get("y"))
        width = _finite(bbox.get("w"))
        height = _finite(bbox.get("h"))
        if (
            time_sec is None
            or x is None
            or y is None
            or width is None
            or height is None
            or time_sec < 0
            or width <= 0
            or height <= 0
        ):
            counters["invalid_bbox"] += 1
            continue
        calibration = _select_calibration(validated_calibrations, time_sec)
        if calibration is None:
            counters["missing_calibration"] += 1
            continue
        foot_x = x + width / 2.0
        foot_y = y + height
        try:
            x_m, y_m = calibration.project_image_point(foot_x, foot_y)
        except Exception:
            counters["invalid_bbox"] += 1
            continue
        margin = thresholds.pitch_margin_m
        if not (
            -margin <= x_m <= calibration.pitch.length_m + margin
            and -margin <= y_m <= calibration.pitch.width_m + margin
        ):
            counters["outside_pitch"] += 1
            continue
        confidence = _finite(bbox.get("conf"))
        points.append(
            CalibratedTrackPoint(
                time_sec=time_sec,
                x_m=max(0.0, min(calibration.pitch.length_m, x_m)),
                y_m=max(0.0, min(calibration.pitch.width_m, y_m)),
                calibration_id=calibration.camera_segment_id,
                confidence=confidence,
                image_x=foot_x,
                image_y=foot_y,
            )
        )

    best_by_time: dict[int, CalibratedTrackPoint] = {}
    for point in points:
        time_key = int(round(point.time_sec * 1000.0))
        existing = best_by_time.get(time_key)
        existing_confidence = existing.confidence if existing else None
        if existing is None or (point.confidence or 0.0) >= (
            existing_confidence or 0.0
        ):
            best_by_time[time_key] = point
    deduplicated = sorted(
        best_by_time.values(),
        key=lambda point: point.time_sec,
    )
    counters["projected_points"] = len(deduplicated)
    return deduplicated, counters


def _smooth_points(
    points: Sequence[CalibratedTrackPoint],
    window: int,
) -> list[CalibratedTrackPoint]:
    if window <= 1 or len(points) < 3:
        return list(points)
    radius = window // 2
    smoothed: list[CalibratedTrackPoint] = []
    for index, point in enumerate(points):
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        neighbours = [
            item
            for item in points[start:end]
            if item.calibration_id == point.calibration_id
            and abs(item.time_sec - point.time_sec) <= 1.0
        ]
        smoothed.append(
            CalibratedTrackPoint(
                time_sec=point.time_sec,
                x_m=float(median(item.x_m for item in neighbours)),
                y_m=float(median(item.y_m for item in neighbours)),
                calibration_id=point.calibration_id,
                confidence=point.confidence,
                image_x=point.image_x,
                image_y=point.image_y,
            )
        )
    return smoothed


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95.0))


def calculate_calibrated_motion(
    tracking: Mapping[str, Any] | None,
    calibrations: Sequence[PitchCalibration],
    *,
    thresholds: MotionThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or MotionThresholds()
    validated_calibrations = [
        calibration for calibration in calibrations if calibration.validated
    ]
    reason_codes: list[str] = []
    if not validated_calibrations:
        reason_codes.append("NO_VALIDATED_PITCH_CALIBRATION")
        return {
            "schema_version": "calibrated-motion-diagnostic-v1",
            "status": "UNAVAILABLE",
            "metric_space": "pitch_metres",
            "pitch_calibration_validated": False,
            "athletic_metric_validated": False,
            "reason_codes": reason_codes,
            "thresholds": asdict(thresholds),
        }

    bboxes = collect_tracking_bboxes(tracking)
    points, counters = project_tracking_footpoints(
        bboxes,
        validated_calibrations,
        thresholds=thresholds,
    )
    if len(points) < thresholds.minimum_projected_points:
        reason_codes.append("INSUFFICIENT_CALIBRATED_TRACK_POINTS")
    smoothed = _smooth_points(points, thresholds.smoothing_window)

    total_distance = 0.0
    observed_duration = 0.0
    accepted_transitions = 0
    rejected_camera_changes = 0
    rejected_gaps = 0
    rejected_speed = 0
    rejected_acceleration = 0
    speed_samples: list[float] = []
    accepted_steps: list[tuple[float, float, float, str]] = []
    previous_speed: float | None = None

    for previous, current in zip(smoothed, smoothed[1:]):
        if previous.calibration_id != current.calibration_id:
            rejected_camera_changes += 1
            previous_speed = None
            continue
        dt = current.time_sec - previous.time_sec
        if dt <= 0 or dt > thresholds.maximum_gap_sec:
            rejected_gaps += 1
            previous_speed = None
            continue
        distance = math.hypot(
            current.x_m - previous.x_m,
            current.y_m - previous.y_m,
        )
        speed = distance / dt
        if speed > thresholds.maximum_speed_mps:
            rejected_speed += 1
            previous_speed = None
            continue
        if previous_speed is not None:
            acceleration = abs(speed - previous_speed) / dt
            if acceleration > thresholds.maximum_acceleration_mps2:
                rejected_acceleration += 1
                previous_speed = None
                continue
        total_distance += distance
        observed_duration += dt
        accepted_transitions += 1
        speed_samples.append(speed)
        accepted_steps.append(
            (previous.time_sec, current.time_sec, speed, current.calibration_id)
        )
        previous_speed = speed

    minimum_transitions = max(1, thresholds.minimum_projected_points // 2)
    if accepted_transitions < minimum_transitions:
        reason_codes.append("INSUFFICIENT_ACCEPTED_MOTION_TRANSITIONS")

    sprint_count = 0
    sprint_duration = 0.0
    active_start: float | None = None
    active_end: float | None = None
    active_calibration_id: str | None = None

    def close_active_sprint() -> None:
        nonlocal sprint_count, sprint_duration, active_start, active_end
        nonlocal active_calibration_id
        if (
            active_start is not None
            and active_end is not None
            and active_end - active_start
            >= thresholds.minimum_sprint_duration_sec
        ):
            sprint_count += 1
            sprint_duration += active_end - active_start
        active_start = None
        active_end = None
        active_calibration_id = None

    for start, end, speed, calibration_id in accepted_steps:
        if (
            active_calibration_id is not None
            and calibration_id != active_calibration_id
        ):
            close_active_sprint()
        if active_end is not None and start - active_end > 1e-6:
            close_active_sprint()
        if speed >= thresholds.sprint_threshold_mps:
            if active_start is None:
                active_start = start
                active_calibration_id = calibration_id
            active_end = end
        else:
            close_active_sprint()
    close_active_sprint()

    average_speed_mps = (
        total_distance / observed_duration if observed_duration > 0 else 0.0
    )
    if accepted_transitions == 0:
        status = "UNAVAILABLE"
    else:
        status = "AVAILABLE" if not reason_codes else "PARTIAL"
    calibration_ids = sorted({point.calibration_id for point in points})
    coverage_ratio = (
        counters["projected_points"] / float(counters["input_bboxes"])
        if counters["input_bboxes"] > 0
        else 0.0
    )
    return {
        "schema_version": "calibrated-motion-diagnostic-v1",
        "status": status,
        "metric_space": "pitch_metres",
        "pitch_calibration_validated": True,
        "athletic_metric_validated": False,
        "identity_validation_required": True,
        "observed_path_length_m": round(total_distance, 3),
        "observed_duration_sec": round(observed_duration, 3),
        "average_observed_speed_kmh": round(average_speed_mps * 3.6, 3),
        "p95_observed_speed_kmh": round(_p95(speed_samples) * 3.6, 3),
        "maximum_accepted_speed_kmh": round(
            (max(speed_samples) if speed_samples else 0.0) * 3.6,
            3,
        ),
        "sprint_bouts_proxy": sprint_count,
        "sprint_duration_sec_proxy": round(sprint_duration, 3),
        "quality": {
            **counters,
            "calibration_coverage_ratio": round(coverage_ratio, 6),
            "accepted_transitions": accepted_transitions,
            "rejected_camera_changes": rejected_camera_changes,
            "rejected_gaps": rejected_gaps,
            "rejected_speed_outliers": rejected_speed,
            "rejected_acceleration_outliers": rejected_acceleration,
            "calibration_ids": calibration_ids,
        },
        "reason_codes": reason_codes,
        "thresholds": asdict(thresholds),
        "limitations": [
            "Distances cover only accepted, observed transitions and are not extrapolated to a full match.",
            "Sprint bouts remain a proxy until frame timing, player identity and calibration stability are validated on real matches.",
            "A validated homography is required for every camera segment used by the calculation.",
        ],
        "points": [point.to_payload() for point in smoothed],
    }
