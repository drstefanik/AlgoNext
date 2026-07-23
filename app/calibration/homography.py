from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from app.calibration.model import PitchDimensions
from app.calibration.schema import (
    CALIBRATION_RESULT_SCHEMA_VERSION,
    CalibrationRequest,
)


class CalibrationFitError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalibrationThresholds:
    minimum_correspondences: int = 6
    ransac_reprojection_threshold_m: float = 1.5
    minimum_inlier_ratio: float = 0.75
    maximum_rmse_m: float = 1.5
    maximum_p95_error_m: float = 3.0
    minimum_image_hull_area_ratio: float = 0.02
    minimum_field_hull_area_ratio: float = 0.08
    maximum_condition_number: float = 1_000_000.0
    minimum_projective_denominator: float = 0.005

    def __post_init__(self) -> None:
        if self.minimum_correspondences < 4:
            raise ValueError("minimum_correspondences must be >= 4")
        for field_name in (
            "ransac_reprojection_threshold_m",
            "maximum_rmse_m",
            "maximum_p95_error_m",
            "maximum_condition_number",
            "minimum_projective_denominator",
        ):
            if float(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "minimum_inlier_ratio",
            "minimum_image_hull_area_ratio",
            "minimum_field_hull_area_ratio",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")


def _matrix_to_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(matrix, dtype=np.float64)
    )


def _matrix_from_value(value: Any, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{field} must be a finite 3x3 matrix")
    return matrix


def _convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(abs(cv2.contourArea(hull)))


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, matrix)
    return projected.reshape(-1, 2)


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("inf")
    return float(np.percentile(values, percentile))


def _weighted_rmse(errors: np.ndarray, weights: np.ndarray) -> float:
    if errors.size == 0:
        return float("inf")
    safe_weights = np.maximum(0.01, np.asarray(weights, dtype=np.float64))
    return float(
        math.sqrt(
            float(np.sum(safe_weights * np.square(errors)))
            / float(np.sum(safe_weights))
        )
    )


def _condition_number(
    matrix: np.ndarray,
    pitch: PitchDimensions,
) -> float:
    field_normalizer = np.array(
        [
            [1.0 / pitch.length_m, 0.0, 0.0],
            [0.0, 1.0 / pitch.width_m, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = field_normalizer @ matrix
    scale = normalized[2, 2]
    if abs(scale) > 1e-12:
        normalized = normalized / scale
    return float(np.linalg.cond(normalized))


def _minimum_denominator(matrix: np.ndarray) -> float:
    samples = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    denominators = (
        matrix[2, 0] * samples[:, 0]
        + matrix[2, 1] * samples[:, 1]
        + matrix[2, 2]
    )
    return float(np.min(np.abs(denominators)))


@dataclass(frozen=True)
class PitchCalibration:
    camera_segment_id: str
    status: str
    matrix_image_to_field: tuple[tuple[float, ...], ...]
    matrix_field_to_image: tuple[tuple[float, ...], ...]
    pitch: PitchDimensions
    source: str
    start_sec: float | None
    end_sec: float | None
    total_correspondences: int
    inlier_count: int
    inlier_mask: tuple[bool, ...]
    inlier_ratio: float
    rmse_m: float
    median_error_m: float
    p95_error_m: float
    maximum_error_m: float
    image_hull_area_ratio: float
    field_hull_area_ratio: float
    condition_number: float
    minimum_projective_denominator: float
    reason_codes: tuple[str, ...]
    thresholds: CalibrationThresholds
    schema_version: str = CALIBRATION_RESULT_SCHEMA_VERSION
    method: str = "opencv-findHomography-ransac-v1"
    validated: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"VALIDATED", "REJECTED"}:
            raise ValueError("calibration status must be VALIDATED or REJECTED")
        if self.validated != (self.status == "VALIDATED"):
            raise ValueError("validated must match status")
        _matrix_from_value(
            self.matrix_image_to_field,
            "matrix_image_to_field",
        )
        _matrix_from_value(
            self.matrix_field_to_image,
            "matrix_field_to_image",
        )

    def contains_time(self, time_sec: float) -> bool:
        timestamp = float(time_sec)
        if self.start_sec is not None and timestamp < self.start_sec:
            return False
        if self.end_sec is not None and timestamp >= self.end_sec:
            return False
        return True

    def project_image_point(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        matrix = _matrix_from_value(
            self.matrix_image_to_field,
            "matrix_image_to_field",
        )
        projected = _project(
            matrix,
            np.array([[float(x), float(y)]], dtype=np.float64),
        )[0]
        return float(projected[0]), float(projected[1])

    def project_field_point(
        self,
        x_m: float,
        y_m: float,
    ) -> tuple[float, float]:
        matrix = _matrix_from_value(
            self.matrix_field_to_image,
            "matrix_field_to_image",
        )
        projected = _project(
            matrix,
            np.array([[float(x_m), float(y_m)]], dtype=np.float64),
        )[0]
        return float(projected[0]), float(projected[1])

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_segment_id": self.camera_segment_id,
            "status": self.status,
            "validated": self.validated,
            "method": self.method,
            "source": self.source,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "pitch": {
                "length_m": self.pitch.length_m,
                "width_m": self.pitch.width_m,
            },
            "matrix_image_to_field": [
                list(row) for row in self.matrix_image_to_field
            ],
            "matrix_field_to_image": [
                list(row) for row in self.matrix_field_to_image
            ],
            "quality": {
                "total_correspondences": self.total_correspondences,
                "inlier_count": self.inlier_count,
                "inlier_mask": list(self.inlier_mask),
                "inlier_ratio": round(self.inlier_ratio, 9),
                "rmse_m": round(self.rmse_m, 9),
                "median_error_m": round(self.median_error_m, 9),
                "p95_error_m": round(self.p95_error_m, 9),
                "maximum_error_m": round(self.maximum_error_m, 9),
                "image_hull_area_ratio": round(
                    self.image_hull_area_ratio, 9
                ),
                "field_hull_area_ratio": round(
                    self.field_hull_area_ratio, 9
                ),
                "condition_number": round(self.condition_number, 9),
                "minimum_projective_denominator": round(
                    self.minimum_projective_denominator, 12
                ),
            },
            "thresholds": asdict(self.thresholds),
            "reason_codes": list(self.reason_codes),
            "provenance": {
                "coordinate_input": "image_normalized_0_1",
                "coordinate_output": "pitch_metres",
                "opencv_version": cv2.__version__,
                "quality_gate": "pitch-calibration-gate-v1",
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PitchCalibration":
        pitch_payload = payload.get("pitch")
        if not isinstance(pitch_payload, Mapping):
            raise ValueError("pitch must be an object")
        quality = payload.get("quality")
        if not isinstance(quality, Mapping):
            raise ValueError("quality must be an object")
        thresholds_payload = payload.get("thresholds")
        if not isinstance(thresholds_payload, Mapping):
            raise ValueError("thresholds must be an object")
        thresholds = CalibrationThresholds(
            **{
                key: thresholds_payload[key]
                for key in asdict(CalibrationThresholds())
                if key in thresholds_payload
            }
        )
        status = str(payload.get("status") or "")
        inlier_mask_value = quality.get("inlier_mask") or []
        if not isinstance(inlier_mask_value, list):
            raise ValueError("quality.inlier_mask must be an array")
        return cls(
            camera_segment_id=str(payload.get("camera_segment_id") or ""),
            status=status,
            validated=bool(payload.get("validated")),
            method=str(payload.get("method") or "opencv-findHomography-ransac-v1"),
            matrix_image_to_field=_matrix_to_tuple(
                _matrix_from_value(
                    payload.get("matrix_image_to_field"),
                    "matrix_image_to_field",
                )
            ),
            matrix_field_to_image=_matrix_to_tuple(
                _matrix_from_value(
                    payload.get("matrix_field_to_image"),
                    "matrix_field_to_image",
                )
            ),
            pitch=PitchDimensions(
                length_m=float(pitch_payload.get("length_m")),
                width_m=float(pitch_payload.get("width_m")),
            ),
            source=str(payload.get("source") or "unknown"),
            start_sec=(
                float(payload["start_sec"])
                if payload.get("start_sec") is not None
                else None
            ),
            end_sec=(
                float(payload["end_sec"])
                if payload.get("end_sec") is not None
                else None
            ),
            total_correspondences=int(quality.get("total_correspondences") or 0),
            inlier_count=int(quality.get("inlier_count") or 0),
            inlier_mask=tuple(bool(value) for value in inlier_mask_value),
            inlier_ratio=float(quality.get("inlier_ratio") or 0.0),
            rmse_m=float(quality.get("rmse_m") or 0.0),
            median_error_m=float(quality.get("median_error_m") or 0.0),
            p95_error_m=float(quality.get("p95_error_m") or 0.0),
            maximum_error_m=float(quality.get("maximum_error_m") or 0.0),
            image_hull_area_ratio=float(
                quality.get("image_hull_area_ratio") or 0.0
            ),
            field_hull_area_ratio=float(
                quality.get("field_hull_area_ratio") or 0.0
            ),
            condition_number=float(quality.get("condition_number") or 0.0),
            minimum_projective_denominator=float(
                quality.get("minimum_projective_denominator") or 0.0
            ),
            reason_codes=tuple(
                str(value) for value in (payload.get("reason_codes") or [])
            ),
            thresholds=thresholds,
            schema_version=str(
                payload.get("schema_version")
                or CALIBRATION_RESULT_SCHEMA_VERSION
            ),
        )


def fit_pitch_calibration(
    request: CalibrationRequest,
    *,
    thresholds: CalibrationThresholds | None = None,
) -> PitchCalibration:
    thresholds = thresholds or CalibrationThresholds()
    source_points = np.array(
        [
            [correspondence.image.x, correspondence.image.y]
            for correspondence in request.correspondences
        ],
        dtype=np.float64,
    )
    field_points = np.array(
        [
            [correspondence.field.x_m, correspondence.field.y_m]
            for correspondence in request.correspondences
        ],
        dtype=np.float64,
    )
    weights = np.array(
        [correspondence.weight for correspondence in request.correspondences],
        dtype=np.float64,
    )

    image_hull_area = _convex_hull_area(source_points)
    field_hull_area = _convex_hull_area(field_points)
    if image_hull_area <= 1e-9:
        raise CalibrationFitError("image correspondences are collinear")
    if field_hull_area <= 1e-6:
        raise CalibrationFitError("field correspondences are collinear")

    matrix, mask = cv2.findHomography(
        source_points,
        field_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=thresholds.ransac_reprojection_threshold_m,
        maxIters=10_000,
        confidence=0.999,
    )
    if matrix is None or not np.isfinite(matrix).all():
        raise CalibrationFitError("OpenCV could not estimate a finite homography")
    if abs(float(matrix[2, 2])) > 1e-12:
        matrix = matrix / float(matrix[2, 2])
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise CalibrationFitError("estimated homography is singular") from exc
    if abs(float(inverse[2, 2])) > 1e-12:
        inverse = inverse / float(inverse[2, 2])

    projected = _project(matrix, source_points)
    errors = np.linalg.norm(projected - field_points, axis=1)
    if mask is None:
        inlier_mask = np.ones(len(source_points), dtype=bool)
    else:
        inlier_mask = np.asarray(mask, dtype=np.uint8).reshape(-1).astype(bool)
    if inlier_mask.size != len(source_points):
        raise CalibrationFitError("OpenCV returned an invalid inlier mask")
    inlier_errors = errors[inlier_mask]
    inlier_weights = weights[inlier_mask]
    inlier_count = int(np.count_nonzero(inlier_mask))
    total_count = len(source_points)
    inlier_ratio = inlier_count / float(total_count)
    rmse = _weighted_rmse(inlier_errors, inlier_weights)
    median_error = (
        float(np.median(inlier_errors)) if inlier_errors.size else float("inf")
    )
    p95_error = _percentile(inlier_errors, 95.0)
    maximum_error = (
        float(np.max(inlier_errors)) if inlier_errors.size else float("inf")
    )
    image_hull_ratio = max(0.0, min(1.0, image_hull_area))
    field_hull_ratio = max(
        0.0,
        min(1.0, field_hull_area / request.pitch.area_m2),
    )
    condition_number = _condition_number(matrix, request.pitch)
    minimum_denominator = _minimum_denominator(matrix)

    reason_codes: list[str] = []
    if total_count < thresholds.minimum_correspondences:
        reason_codes.append("INSUFFICIENT_CALIBRATION_POINTS")
    if inlier_ratio < thresholds.minimum_inlier_ratio:
        reason_codes.append("LOW_CALIBRATION_INLIER_RATIO")
    if rmse > thresholds.maximum_rmse_m:
        reason_codes.append("HIGH_CALIBRATION_RMSE")
    if p95_error > thresholds.maximum_p95_error_m:
        reason_codes.append("HIGH_CALIBRATION_P95_ERROR")
    if image_hull_ratio < thresholds.minimum_image_hull_area_ratio:
        reason_codes.append("LOW_IMAGE_POINT_COVERAGE")
    if field_hull_ratio < thresholds.minimum_field_hull_area_ratio:
        reason_codes.append("LOW_FIELD_POINT_COVERAGE")
    if (
        not math.isfinite(condition_number)
        or condition_number > thresholds.maximum_condition_number
    ):
        reason_codes.append("ILL_CONDITIONED_HOMOGRAPHY")
    if minimum_denominator < thresholds.minimum_projective_denominator:
        reason_codes.append("PROJECTIVE_HORIZON_INTERSECTS_FRAME")

    status = "VALIDATED" if not reason_codes else "REJECTED"
    return PitchCalibration(
        camera_segment_id=request.camera_segment_id,
        status=status,
        validated=status == "VALIDATED",
        matrix_image_to_field=_matrix_to_tuple(matrix),
        matrix_field_to_image=_matrix_to_tuple(inverse),
        pitch=request.pitch,
        source=request.source,
        start_sec=request.start_sec,
        end_sec=request.end_sec,
        total_correspondences=total_count,
        inlier_count=inlier_count,
        inlier_mask=tuple(bool(value) for value in inlier_mask),
        inlier_ratio=inlier_ratio,
        rmse_m=rmse,
        median_error_m=median_error,
        p95_error_m=p95_error,
        maximum_error_m=maximum_error,
        image_hull_area_ratio=image_hull_ratio,
        field_hull_area_ratio=field_hull_ratio,
        condition_number=condition_number,
        minimum_projective_denominator=minimum_denominator,
        reason_codes=tuple(reason_codes),
        thresholds=thresholds,
    )
