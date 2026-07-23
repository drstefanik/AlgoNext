from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from app.calibration.model import PitchDimensions
from app.calibration.schema import (
    CALIBRATION_RESULT_SCHEMA_VERSION,
    CalibrationRequest,
)

CALIBRATION_METHOD = "opencv-findHomography-ransac-v1"
CALIBRATION_QUALITY_GATE = "pitch-calibration-gate-v1"
RANSAC_SEED = 0
_RANSAC_LOCK = threading.Lock()


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
        if (
            isinstance(self.minimum_correspondences, bool)
            or not isinstance(self.minimum_correspondences, int)
            or self.minimum_correspondences < 4
        ):
            raise ValueError("minimum_correspondences must be an integer >= 4")
        for field_name in (
            "ransac_reprojection_threshold_m",
            "maximum_rmse_m",
            "maximum_p95_error_m",
            "maximum_condition_number",
            "minimum_projective_denominator",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        for field_name in (
            "minimum_inlier_ratio",
            "minimum_image_hull_area_ratio",
            "minimum_field_hull_area_ratio",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]")


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
    """Return the smallest absolute projective denominator on the unit frame.

    The denominator is affine in image x/y, so its extrema occur at the four
    corners. A sign change between corners proves that the projective horizon
    crosses the frame and therefore the true minimum is zero, even when none of
    a finite set of sample points lies exactly on that horizon.
    """

    corners = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    denominators = (
        matrix[2, 0] * corners[:, 0]
        + matrix[2, 1] * corners[:, 1]
        + matrix[2, 2]
    )
    if not np.isfinite(denominators).all():
        return float("nan")
    minimum = float(np.min(denominators))
    maximum = float(np.max(denominators))
    if minimum <= 0.0 <= maximum:
        return 0.0
    return float(np.min(np.abs(denominators)))


def _require_finite(value: float, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    field: str,
) -> None:
    actual = set(value.keys())
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}")


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
    method: str = CALIBRATION_METHOD
    validated: bool = False

    def __post_init__(self) -> None:
        _require_non_empty_string(self.camera_segment_id, "camera_segment_id")
        _require_non_empty_string(self.source, "source")
        if self.schema_version != CALIBRATION_RESULT_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal "
                f"{CALIBRATION_RESULT_SCHEMA_VERSION!r}"
            )
        if self.method != CALIBRATION_METHOD:
            raise ValueError(f"method must equal {CALIBRATION_METHOD!r}")
        if self.status not in {"VALIDATED", "REJECTED"}:
            raise ValueError("calibration status must be VALIDATED or REJECTED")
        if not isinstance(self.validated, bool):
            raise ValueError("validated must be a boolean")
        if self.validated != (self.status == "VALIDATED"):
            raise ValueError("validated must match status")
        if any(
            not isinstance(code, str) or not code.strip()
            for code in self.reason_codes
        ):
            raise ValueError("reason_codes must contain non-empty strings")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason_codes must not contain duplicates")
        if self.validated and self.reason_codes:
            raise ValueError("validated calibration cannot contain reason codes")
        if not self.validated and not self.reason_codes:
            raise ValueError("rejected calibration must contain reason codes")
        if self.total_correspondences < 4:
            raise ValueError("total_correspondences must be >= 4")
        if len(self.inlier_mask) != self.total_correspondences:
            raise ValueError("inlier_mask length must match total_correspondences")
        if any(not isinstance(value, bool) for value in self.inlier_mask):
            raise ValueError("inlier_mask must contain booleans")
        if self.inlier_count != sum(self.inlier_mask):
            raise ValueError("inlier_count must match inlier_mask")
        if self.inlier_count < 4:
            raise ValueError("inlier_count must be >= 4")
        if not 0.0 <= _require_finite(self.inlier_ratio, "inlier_ratio") <= 1.0:
            raise ValueError("inlier_ratio must be in [0, 1]")
        for field_name in (
            "rmse_m",
            "median_error_m",
            "p95_error_m",
            "maximum_error_m",
            "image_hull_area_ratio",
            "field_hull_area_ratio",
            "condition_number",
            "minimum_projective_denominator",
        ):
            value = _require_finite(float(getattr(self, field_name)), field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if not 0.0 <= self.image_hull_area_ratio <= 1.0:
            raise ValueError("image_hull_area_ratio must be in [0, 1]")
        if not 0.0 <= self.field_hull_area_ratio <= 1.0:
            raise ValueError("field_hull_area_ratio must be in [0, 1]")
        if self.start_sec is not None:
            _require_finite(self.start_sec, "start_sec")
            if self.start_sec < 0:
                raise ValueError("start_sec must be >= 0")
        if self.end_sec is not None:
            _require_finite(self.end_sec, "end_sec")
            if self.end_sec < 0:
                raise ValueError("end_sec must be >= 0")
        if (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec <= self.start_sec
        ):
            raise ValueError("end_sec must be greater than start_sec")
        _matrix_from_value(
            self.matrix_image_to_field,
            "matrix_image_to_field",
        )
        _matrix_from_value(
            self.matrix_field_to_image,
            "matrix_field_to_image",
        )

    def contains_time(self, time_sec: float) -> bool:
        timestamp = _require_finite(time_sec, "time_sec")
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
            np.array(
                [[_require_finite(x, "x"), _require_finite(y, "y")]],
                dtype=np.float64,
            ),
        )[0]
        if not np.isfinite(projected).all():
            raise ValueError("image point projects to a non-finite field point")
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
            np.array(
                [
                    [
                        _require_finite(x_m, "x_m"),
                        _require_finite(y_m, "y_m"),
                    ]
                ],
                dtype=np.float64,
            ),
        )[0]
        if not np.isfinite(projected).all():
            raise ValueError("field point projects to a non-finite image point")
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
                "quality_gate": CALIBRATION_QUALITY_GATE,
                "ransac_seed": RANSAC_SEED,
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PitchCalibration":
        if not isinstance(payload, Mapping):
            raise ValueError("calibration payload must be an object")
        schema_version = payload.get("schema_version")
        if schema_version != CALIBRATION_RESULT_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal "
                f"{CALIBRATION_RESULT_SCHEMA_VERSION!r}"
            )
        method = payload.get("method")
        if method != CALIBRATION_METHOD:
            raise ValueError(f"method must equal {CALIBRATION_METHOD!r}")
        validated_value = payload.get("validated")
        if not isinstance(validated_value, bool):
            raise ValueError("validated must be a boolean")

        pitch_payload = payload.get("pitch")
        if not isinstance(pitch_payload, Mapping):
            raise ValueError("pitch must be an object")
        _require_exact_keys(
            pitch_payload,
            {"length_m", "width_m"},
            "pitch",
        )

        quality = payload.get("quality")
        if not isinstance(quality, Mapping):
            raise ValueError("quality must be an object")
        quality_fields = {
            "total_correspondences",
            "inlier_count",
            "inlier_mask",
            "inlier_ratio",
            "rmse_m",
            "median_error_m",
            "p95_error_m",
            "maximum_error_m",
            "image_hull_area_ratio",
            "field_hull_area_ratio",
            "condition_number",
            "minimum_projective_denominator",
        }
        _require_exact_keys(quality, quality_fields, "quality")

        threshold_defaults = asdict(CalibrationThresholds())
        threshold_fields = set(threshold_defaults)
        thresholds_payload = payload.get("thresholds")
        if not isinstance(thresholds_payload, Mapping):
            raise ValueError("thresholds must be an object")
        _require_exact_keys(
            thresholds_payload,
            threshold_fields,
            "thresholds",
        )
        thresholds = CalibrationThresholds(
            **{
                key: thresholds_payload[key]
                for key in threshold_defaults
            }
        )

        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be an object")
        _require_exact_keys(
            provenance,
            {
                "coordinate_input",
                "coordinate_output",
                "opencv_version",
                "quality_gate",
                "ransac_seed",
            },
            "provenance",
        )
        if provenance.get("coordinate_input") != "image_normalized_0_1":
            raise ValueError("provenance.coordinate_input is invalid")
        if provenance.get("coordinate_output") != "pitch_metres":
            raise ValueError("provenance.coordinate_output is invalid")
        _require_non_empty_string(
            provenance.get("opencv_version"),
            "provenance.opencv_version",
        )
        if provenance.get("quality_gate") != CALIBRATION_QUALITY_GATE:
            raise ValueError("provenance.quality_gate is invalid")
        if provenance.get("ransac_seed") != RANSAC_SEED:
            raise ValueError("provenance.ransac_seed is invalid")

        status = str(payload.get("status") or "")
        inlier_mask_value = quality.get("inlier_mask")
        if not isinstance(inlier_mask_value, list) or any(
            not isinstance(value, bool) for value in inlier_mask_value
        ):
            raise ValueError("quality.inlier_mask must be an array of booleans")
        reason_codes_value = payload.get("reason_codes")
        if not isinstance(reason_codes_value, list) or any(
            not isinstance(value, str) for value in reason_codes_value
        ):
            raise ValueError("reason_codes must be an array of strings")
        return cls(
            camera_segment_id=_require_non_empty_string(
                payload.get("camera_segment_id"),
                "camera_segment_id",
            ),
            status=status,
            validated=validated_value,
            method=method,
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
                length_m=float(pitch_payload["length_m"]),
                width_m=float(pitch_payload["width_m"]),
            ),
            source=_require_non_empty_string(payload.get("source"), "source"),
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
            total_correspondences=int(quality["total_correspondences"]),
            inlier_count=int(quality["inlier_count"]),
            inlier_mask=tuple(inlier_mask_value),
            inlier_ratio=float(quality["inlier_ratio"]),
            rmse_m=float(quality["rmse_m"]),
            median_error_m=float(quality["median_error_m"]),
            p95_error_m=float(quality["p95_error_m"]),
            maximum_error_m=float(quality["maximum_error_m"]),
            image_hull_area_ratio=float(quality["image_hull_area_ratio"]),
            field_hull_area_ratio=float(quality["field_hull_area_ratio"]),
            condition_number=float(quality["condition_number"]),
            minimum_projective_denominator=float(
                quality["minimum_projective_denominator"]
            ),
            reason_codes=tuple(reason_codes_value),
            thresholds=thresholds,
            schema_version=schema_version,
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

    with _RANSAC_LOCK:
        cv2.setRNGSeed(RANSAC_SEED)
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
    if abs(float(matrix[2, 2])) <= 1e-12:
        raise CalibrationFitError("estimated homography has an invalid scale")
    matrix = matrix / float(matrix[2, 2])
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise CalibrationFitError("estimated homography is singular") from exc
    if not np.isfinite(inverse).all() or abs(float(inverse[2, 2])) <= 1e-12:
        raise CalibrationFitError("estimated inverse homography is invalid")
    inverse = inverse / float(inverse[2, 2])

    projected = _project(matrix, source_points)
    if not np.isfinite(projected).all():
        raise CalibrationFitError("homography produced non-finite projected points")
    errors = np.linalg.norm(projected - field_points, axis=1)
    if not np.isfinite(errors).all():
        raise CalibrationFitError("homography produced non-finite reprojection errors")
    if mask is None:
        inlier_mask = np.ones(len(source_points), dtype=bool)
    else:
        inlier_mask = np.asarray(mask, dtype=np.uint8).reshape(-1).astype(bool)
    if inlier_mask.size != len(source_points):
        raise CalibrationFitError("OpenCV returned an invalid inlier mask")
    inlier_count = int(np.count_nonzero(inlier_mask))
    if inlier_count < 4:
        raise CalibrationFitError(
            "estimated homography has fewer than four RANSAC inliers"
        )
    inlier_errors = errors[inlier_mask]
    inlier_weights = weights[inlier_mask]
    total_count = len(source_points)
    inlier_ratio = inlier_count / float(total_count)
    rmse = _weighted_rmse(inlier_errors, inlier_weights)
    median_error = float(np.median(inlier_errors))
    p95_error = _percentile(inlier_errors, 95.0)
    maximum_error = float(np.max(inlier_errors))
    image_hull_ratio = max(0.0, min(1.0, image_hull_area))
    field_hull_ratio = max(
        0.0,
        min(1.0, field_hull_area / request.pitch.area_m2),
    )
    condition_number = _condition_number(matrix, request.pitch)
    minimum_denominator = _minimum_denominator(matrix)
    for field_name, value in (
        ("rmse", rmse),
        ("median_error", median_error),
        ("p95_error", p95_error),
        ("maximum_error", maximum_error),
        ("condition_number", condition_number),
        ("minimum_projective_denominator", minimum_denominator),
    ):
        if not math.isfinite(value):
            raise CalibrationFitError(
                f"estimated homography produced non-finite {field_name}"
            )

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
    if condition_number > thresholds.maximum_condition_number:
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
