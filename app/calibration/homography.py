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


def _payload_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return parsed


def _payload_optional_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _payload_number(value, field, minimum=minimum)


def _payload_integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _matrix_from_payload(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must be a 3x3 numeric array")
    rows: list[list[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{field}[{row_index}] must contain three numbers")
        rows.append(
            [
                _payload_number(item, f"{field}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            ]
        )
    return np.asarray(rows, dtype=np.float64)


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


def _rmse(errors: np.ndarray) -> float:
    if errors.size == 0:
        return float("inf")
    return float(math.sqrt(float(np.mean(np.square(errors)))))


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
    """Return the smallest absolute projective denominator on the unit frame."""

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


def _require_allowed_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    field: str,
) -> None:
    unknown = sorted(set(value.keys()) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    field: str,
) -> None:
    _require_allowed_keys(value, expected, field)
    missing = sorted(expected - set(value.keys()))
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(missing)}")


def _normalized_identity_product(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    product = first @ second
    scale = float(product[2, 2])
    if not np.isfinite(product).all() or abs(scale) <= 1e-12:
        raise ValueError("homography matrices do not form a finite inverse pair")
    return product / scale


def _gate_reason_codes(
    *,
    total_correspondences: int,
    inlier_ratio: float,
    rmse_m: float,
    p95_error_m: float,
    image_hull_area_ratio: float,
    field_hull_area_ratio: float,
    condition_number: float,
    minimum_projective_denominator: float,
    thresholds: CalibrationThresholds,
) -> tuple[str, ...]:
    reason_codes: list[str] = []
    if total_correspondences < thresholds.minimum_correspondences:
        reason_codes.append("INSUFFICIENT_CALIBRATION_POINTS")
    if inlier_ratio < thresholds.minimum_inlier_ratio:
        reason_codes.append("LOW_CALIBRATION_INLIER_RATIO")
    if rmse_m > thresholds.maximum_rmse_m:
        reason_codes.append("HIGH_CALIBRATION_RMSE")
    if p95_error_m > thresholds.maximum_p95_error_m:
        reason_codes.append("HIGH_CALIBRATION_P95_ERROR")
    if image_hull_area_ratio < thresholds.minimum_image_hull_area_ratio:
        reason_codes.append("LOW_IMAGE_POINT_COVERAGE")
    if field_hull_area_ratio < thresholds.minimum_field_hull_area_ratio:
        reason_codes.append("LOW_FIELD_POINT_COVERAGE")
    if condition_number > thresholds.maximum_condition_number:
        reason_codes.append("ILL_CONDITIONED_HOMOGRAPHY")
    if minimum_projective_denominator < thresholds.minimum_projective_denominator:
        reason_codes.append("PROJECTIVE_HORIZON_INTERSECTS_FRAME")
    return tuple(reason_codes)


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
        if not isinstance(self.pitch, PitchDimensions):
            raise ValueError("pitch must be PitchDimensions")
        if not isinstance(self.thresholds, CalibrationThresholds):
            raise ValueError("thresholds must be CalibrationThresholds")
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
        if (
            isinstance(self.total_correspondences, bool)
            or not isinstance(self.total_correspondences, int)
            or self.total_correspondences < 4
        ):
            raise ValueError("total_correspondences must be an integer >= 4")
        if (
            isinstance(self.inlier_count, bool)
            or not isinstance(self.inlier_count, int)
            or self.inlier_count < 4
        ):
            raise ValueError("inlier_count must be an integer >= 4")
        if self.inlier_count > self.total_correspondences:
            raise ValueError("inlier_count cannot exceed total_correspondences")
        if len(self.inlier_mask) != self.total_correspondences:
            raise ValueError("inlier_mask length must match total_correspondences")
        if any(not isinstance(value, bool) for value in self.inlier_mask):
            raise ValueError("inlier_mask must contain booleans")
        if self.inlier_count != sum(self.inlier_mask):
            raise ValueError("inlier_count must match inlier_mask")

        inlier_ratio = _require_finite(self.inlier_ratio, "inlier_ratio")
        if not 0.0 <= inlier_ratio <= 1.0:
            raise ValueError("inlier_ratio must be in [0, 1]")
        expected_ratio = self.inlier_count / float(self.total_correspondences)
        if not math.isclose(inlier_ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("inlier_ratio must match inlier counts")

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
        if self.median_error_m > self.maximum_error_m:
            raise ValueError("median_error_m cannot exceed maximum_error_m")
        if self.p95_error_m > self.maximum_error_m:
            raise ValueError("p95_error_m cannot exceed maximum_error_m")

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

        image_to_field = _matrix_from_value(
            self.matrix_image_to_field,
            "matrix_image_to_field",
        )
        field_to_image = _matrix_from_value(
            self.matrix_field_to_image,
            "matrix_field_to_image",
        )
        identity = np.eye(3, dtype=np.float64)
        if not np.allclose(
            _normalized_identity_product(image_to_field, field_to_image),
            identity,
            rtol=1e-6,
            atol=1e-6,
        ) or not np.allclose(
            _normalized_identity_product(field_to_image, image_to_field),
            identity,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("homography matrices are not inverse pairs")

        computed_condition = _condition_number(image_to_field, self.pitch)
        if not math.isclose(
            computed_condition,
            self.condition_number,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("condition_number does not match homography")
        computed_denominator = _minimum_denominator(image_to_field)
        if not math.isclose(
            computed_denominator,
            self.minimum_projective_denominator,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "minimum_projective_denominator does not match homography"
            )

        expected_reasons = _gate_reason_codes(
            total_correspondences=self.total_correspondences,
            inlier_ratio=self.inlier_ratio,
            rmse_m=self.rmse_m,
            p95_error_m=self.p95_error_m,
            image_hull_area_ratio=self.image_hull_area_ratio,
            field_hull_area_ratio=self.field_hull_area_ratio,
            condition_number=self.condition_number,
            minimum_projective_denominator=self.minimum_projective_denominator,
            thresholds=self.thresholds,
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("reason_codes do not match the calibration gate")
        expected_status = "VALIDATED" if not expected_reasons else "REJECTED"
        if self.status != expected_status:
            raise ValueError("status does not match the calibration gate")

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
        _require_allowed_keys(
            payload,
            {
                "schema_version",
                "camera_segment_id",
                "status",
                "validated",
                "method",
                "source",
                "start_sec",
                "end_sec",
                "pitch",
                "matrix_image_to_field",
                "matrix_field_to_image",
                "quality",
                "thresholds",
                "reason_codes",
                "provenance",
            },
            "calibration",
        )
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
            minimum_correspondences=_payload_integer(
                thresholds_payload["minimum_correspondences"],
                "thresholds.minimum_correspondences",
                minimum=4,
            ),
            ransac_reprojection_threshold_m=_payload_number(
                thresholds_payload["ransac_reprojection_threshold_m"],
                "thresholds.ransac_reprojection_threshold_m",
            ),
            minimum_inlier_ratio=_payload_number(
                thresholds_payload["minimum_inlier_ratio"],
                "thresholds.minimum_inlier_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            maximum_rmse_m=_payload_number(
                thresholds_payload["maximum_rmse_m"],
                "thresholds.maximum_rmse_m",
            ),
            maximum_p95_error_m=_payload_number(
                thresholds_payload["maximum_p95_error_m"],
                "thresholds.maximum_p95_error_m",
            ),
            minimum_image_hull_area_ratio=_payload_number(
                thresholds_payload["minimum_image_hull_area_ratio"],
                "thresholds.minimum_image_hull_area_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            minimum_field_hull_area_ratio=_payload_number(
                thresholds_payload["minimum_field_hull_area_ratio"],
                "thresholds.minimum_field_hull_area_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            maximum_condition_number=_payload_number(
                thresholds_payload["maximum_condition_number"],
                "thresholds.maximum_condition_number",
            ),
            minimum_projective_denominator=_payload_number(
                thresholds_payload["minimum_projective_denominator"],
                "thresholds.minimum_projective_denominator",
            ),
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
        if (
            _payload_integer(
                provenance.get("ransac_seed"),
                "provenance.ransac_seed",
            )
            != RANSAC_SEED
        ):
            raise ValueError("provenance.ransac_seed is invalid")

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
            status=_require_non_empty_string(payload.get("status"), "status"),
            validated=validated_value,
            method=method,
            matrix_image_to_field=_matrix_to_tuple(
                _matrix_from_payload(
                    payload.get("matrix_image_to_field"),
                    "matrix_image_to_field",
                )
            ),
            matrix_field_to_image=_matrix_to_tuple(
                _matrix_from_payload(
                    payload.get("matrix_field_to_image"),
                    "matrix_field_to_image",
                )
            ),
            pitch=PitchDimensions(
                length_m=_payload_number(
                    pitch_payload["length_m"],
                    "pitch.length_m",
                    minimum=90.0,
                    maximum=120.0,
                ),
                width_m=_payload_number(
                    pitch_payload["width_m"],
                    "pitch.width_m",
                    minimum=45.0,
                    maximum=90.0,
                ),
            ),
            source=_require_non_empty_string(payload.get("source"), "source"),
            start_sec=_payload_optional_number(
                payload.get("start_sec"),
                "start_sec",
                minimum=0.0,
            ),
            end_sec=_payload_optional_number(
                payload.get("end_sec"),
                "end_sec",
                minimum=0.0,
            ),
            total_correspondences=_payload_integer(
                quality["total_correspondences"],
                "quality.total_correspondences",
                minimum=4,
            ),
            inlier_count=_payload_integer(
                quality["inlier_count"],
                "quality.inlier_count",
                minimum=4,
            ),
            inlier_mask=tuple(inlier_mask_value),
            inlier_ratio=_payload_number(
                quality["inlier_ratio"],
                "quality.inlier_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            rmse_m=_payload_number(
                quality["rmse_m"],
                "quality.rmse_m",
                minimum=0.0,
            ),
            median_error_m=_payload_number(
                quality["median_error_m"],
                "quality.median_error_m",
                minimum=0.0,
            ),
            p95_error_m=_payload_number(
                quality["p95_error_m"],
                "quality.p95_error_m",
                minimum=0.0,
            ),
            maximum_error_m=_payload_number(
                quality["maximum_error_m"],
                "quality.maximum_error_m",
                minimum=0.0,
            ),
            image_hull_area_ratio=_payload_number(
                quality["image_hull_area_ratio"],
                "quality.image_hull_area_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            field_hull_area_ratio=_payload_number(
                quality["field_hull_area_ratio"],
                "quality.field_hull_area_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            condition_number=_payload_number(
                quality["condition_number"],
                "quality.condition_number",
                minimum=0.0,
            ),
            minimum_projective_denominator=_payload_number(
                quality["minimum_projective_denominator"],
                "quality.minimum_projective_denominator",
                minimum=0.0,
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
    total_count = len(source_points)
    inlier_ratio = inlier_count / float(total_count)
    rmse = _rmse(inlier_errors)
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

    reason_codes = _gate_reason_codes(
        total_correspondences=total_count,
        inlier_ratio=inlier_ratio,
        rmse_m=rmse,
        p95_error_m=p95_error,
        image_hull_area_ratio=image_hull_ratio,
        field_hull_area_ratio=field_hull_ratio,
        condition_number=condition_number,
        minimum_projective_denominator=minimum_denominator,
        thresholds=thresholds,
    )
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
        reason_codes=reason_codes,
        thresholds=thresholds,
    )
