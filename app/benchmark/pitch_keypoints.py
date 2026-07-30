from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from app.calibration.model import PitchDimensions, landmark_coordinates, standard_landmarks

ANNOTATION_SCHEMA_VERSION = "pitch-keypoint-annotation-v1"
PREDICTION_SCHEMA_VERSION = "pitch-keypoint-prediction-v1"
REPORT_SCHEMA_VERSION = "pitch-keypoint-benchmark-report-v1"
LANDMARK_VOCABULARY = tuple(sorted(standard_landmarks().keys()))


class KeypointSchemaError(ValueError):
    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KeypointSchemaError(path, "expected an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise KeypointSchemaError(path, "expected an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KeypointSchemaError(path, "expected a non-empty string")
    return value.strip()


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise KeypointSchemaError(path, "expected a boolean")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise KeypointSchemaError(path, "expected an integer")
    if value < minimum:
        raise KeypointSchemaError(path, f"must be >= {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KeypointSchemaError(path, "expected a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise KeypointSchemaError(path, "expected a finite number")
    if minimum is not None and parsed < minimum:
        raise KeypointSchemaError(path, f"must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise KeypointSchemaError(path, f"must be <= {maximum}")
    return parsed


def _optional_number(value: Any, path: str, minimum: float = 0.0) -> float | None:
    if value is None:
        return None
    return _number(value, path, minimum=minimum)


def _require_allowed_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise KeypointSchemaError(path, f"unknown fields: {unknown}")


def _require_schema(value: Mapping[str, Any], expected: str, path: str) -> None:
    actual = _string(value.get("schema_version"), f"{path}.schema_version")
    if actual != expected:
        raise KeypointSchemaError(
            f"{path}.schema_version",
            f"expected {expected!r}, got {actual!r}",
        )


def _validate_landmark(name: str, path: str) -> str:
    if name not in LANDMARK_VOCABULARY:
        raise KeypointSchemaError(path, f"unknown pitch landmark {name!r}")
    return name


@dataclass(frozen=True)
class AnnotatedKeypoint:
    landmark: str
    x: float
    y: float
    visibility: str = "visible"

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "AnnotatedKeypoint":
        value = _mapping(payload, path)
        _require_allowed_keys(value, {"landmark", "x", "y", "visibility"}, path)
        landmark = _validate_landmark(
            _string(value.get("landmark"), f"{path}.landmark"),
            f"{path}.landmark",
        )
        visibility = _string(value.get("visibility", "visible"), f"{path}.visibility")
        if visibility not in {"visible", "occluded"}:
            raise KeypointSchemaError(
                f"{path}.visibility",
                "expected 'visible' or 'occluded'",
            )
        return cls(
            landmark=landmark,
            x=_number(value.get("x"), f"{path}.x", minimum=0.0, maximum=1.0),
            y=_number(value.get("y"), f"{path}.y", minimum=0.0, maximum=1.0),
            visibility=visibility,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "landmark": self.landmark,
            "x": round(self.x, 8),
            "y": round(self.y, 8),
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class PredictedKeypoint:
    landmark: str
    x: float
    y: float
    confidence: float

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "PredictedKeypoint":
        value = _mapping(payload, path)
        _require_allowed_keys(value, {"landmark", "x", "y", "confidence"}, path)
        return cls(
            landmark=_validate_landmark(
                _string(value.get("landmark"), f"{path}.landmark"),
                f"{path}.landmark",
            ),
            x=_number(value.get("x"), f"{path}.x", minimum=0.0, maximum=1.0),
            y=_number(value.get("y"), f"{path}.y", minimum=0.0, maximum=1.0),
            confidence=_number(
                value.get("confidence"),
                f"{path}.confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "landmark": self.landmark,
            "x": round(self.x, 8),
            "y": round(self.y, 8),
            "confidence": round(self.confidence, 8),
        }


def _validate_unique_landmarks(
    keypoints: Sequence[AnnotatedKeypoint] | Sequence[PredictedKeypoint],
    path: str,
) -> None:
    seen: set[str] = set()
    for keypoint in keypoints:
        if keypoint.landmark in seen:
            raise KeypointSchemaError(
                path,
                f"landmark {keypoint.landmark!r} appears more than once",
            )
        seen.add(keypoint.landmark)


@dataclass(frozen=True)
class AnnotationFrame:
    frame_id: str
    video_id: str
    shot_id: str
    time_sec: float
    width: int
    height: int
    is_pitch_view: bool
    keypoints: tuple[AnnotatedKeypoint, ...]
    split: str = "development"

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "AnnotationFrame":
        value = _mapping(payload, path)
        _require_allowed_keys(
            value,
            {
                "frame_id", "video_id", "shot_id", "time_sec", "width",
                "height", "is_pitch_view", "keypoints", "split",
            },
            path,
        )
        is_pitch_view = _boolean(value.get("is_pitch_view"), f"{path}.is_pitch_view")
        keypoints = tuple(
            AnnotatedKeypoint.from_payload(item, f"{path}.keypoints[{index}]")
            for index, item in enumerate(_array(value.get("keypoints"), f"{path}.keypoints"))
        )
        _validate_unique_landmarks(keypoints, f"{path}.keypoints")
        if not is_pitch_view and keypoints:
            raise KeypointSchemaError(
                f"{path}.keypoints",
                "non-pitch frames must not contain pitch keypoints",
            )
        split = _string(value.get("split", "development"), f"{path}.split")
        if split not in {"development", "validation", "test"}:
            raise KeypointSchemaError(
                f"{path}.split",
                "expected development, validation or test",
            )
        return cls(
            frame_id=_string(value.get("frame_id"), f"{path}.frame_id"),
            video_id=_string(value.get("video_id"), f"{path}.video_id"),
            shot_id=_string(value.get("shot_id"), f"{path}.shot_id"),
            time_sec=_number(value.get("time_sec"), f"{path}.time_sec", minimum=0.0),
            width=_integer(value.get("width"), f"{path}.width", minimum=1),
            height=_integer(value.get("height"), f"{path}.height", minimum=1),
            is_pitch_view=is_pitch_view,
            keypoints=keypoints,
            split=split,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "time_sec": round(self.time_sec, 6),
            "width": self.width,
            "height": self.height,
            "is_pitch_view": self.is_pitch_view,
            "split": self.split,
            "keypoints": [keypoint.to_payload() for keypoint in self.keypoints],
        }


@dataclass(frozen=True)
class PredictionFrame:
    frame_id: str
    video_id: str
    shot_id: str
    time_sec: float
    abstained: bool
    keypoints: tuple[PredictedKeypoint, ...]
    model_version: str
    configuration_hash: str
    reason_codes: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "PredictionFrame":
        value = _mapping(payload, path)
        _require_allowed_keys(
            value,
            {
                "frame_id", "video_id", "shot_id", "time_sec", "abstained",
                "keypoints", "model_version", "configuration_hash", "reason_codes",
            },
            path,
        )
        keypoints = tuple(
            PredictedKeypoint.from_payload(item, f"{path}.keypoints[{index}]")
            for index, item in enumerate(_array(value.get("keypoints"), f"{path}.keypoints"))
        )
        _validate_unique_landmarks(keypoints, f"{path}.keypoints")
        abstained = _boolean(value.get("abstained"), f"{path}.abstained")
        if abstained and keypoints:
            raise KeypointSchemaError(
                f"{path}.keypoints",
                "abstained predictions must not contain keypoints",
            )
        reason_values = _array(value.get("reason_codes", []), f"{path}.reason_codes")
        reason_codes = tuple(
            _string(item, f"{path}.reason_codes[{index}]")
            for index, item in enumerate(reason_values)
        )
        if len(set(reason_codes)) != len(reason_codes):
            raise KeypointSchemaError(f"{path}.reason_codes", "reason codes must be unique")
        if abstained and not reason_codes:
            raise KeypointSchemaError(
                f"{path}.reason_codes",
                "abstained predictions require at least one reason code",
            )
        return cls(
            frame_id=_string(value.get("frame_id"), f"{path}.frame_id"),
            video_id=_string(value.get("video_id"), f"{path}.video_id"),
            shot_id=_string(value.get("shot_id"), f"{path}.shot_id"),
            time_sec=_number(value.get("time_sec"), f"{path}.time_sec", minimum=0.0),
            abstained=abstained,
            keypoints=keypoints,
            model_version=_string(value.get("model_version"), f"{path}.model_version"),
            configuration_hash=_string(
                value.get("configuration_hash"),
                f"{path}.configuration_hash",
            ),
            reason_codes=reason_codes,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "shot_id": self.shot_id,
            "time_sec": round(self.time_sec, 6),
            "abstained": self.abstained,
            "model_version": self.model_version,
            "configuration_hash": self.configuration_hash,
            "reason_codes": list(self.reason_codes),
            "keypoints": [keypoint.to_payload() for keypoint in self.keypoints],
        }


@dataclass(frozen=True)
class AnnotationDataset:
    frames: tuple[AnnotationFrame, ...]
    schema_version: str = ANNOTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {ANNOTATION_SCHEMA_VERSION!r}")
        _validate_unique_frame_ids(self.frames, "frames")
        if not self.frames:
            raise ValueError("annotation dataset must contain at least one frame")

    @classmethod
    def from_payload(cls, payload: Any, path: str = "$") -> "AnnotationDataset":
        value = _mapping(payload, path)
        _require_allowed_keys(value, {"schema_version", "frames"}, path)
        _require_schema(value, ANNOTATION_SCHEMA_VERSION, path)
        frames = tuple(
            AnnotationFrame.from_payload(item, f"{path}.frames[{index}]")
            for index, item in enumerate(_array(value.get("frames"), f"{path}.frames"))
        )
        _validate_unique_frame_ids(frames, f"{path}.frames")
        if not frames:
            raise KeypointSchemaError(f"{path}.frames", "must contain at least one frame")
        return cls(frames=frames)


@dataclass(frozen=True)
class PredictionDataset:
    frames: tuple[PredictionFrame, ...]
    schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {PREDICTION_SCHEMA_VERSION!r}")
        _validate_unique_frame_ids(self.frames, "frames")

    @classmethod
    def from_payload(cls, payload: Any, path: str = "$") -> "PredictionDataset":
        value = _mapping(payload, path)
        _require_allowed_keys(value, {"schema_version", "frames"}, path)
        _require_schema(value, PREDICTION_SCHEMA_VERSION, path)
        frames = tuple(
            PredictionFrame.from_payload(item, f"{path}.frames[{index}]")
            for index, item in enumerate(_array(value.get("frames"), f"{path}.frames"))
        )
        _validate_unique_frame_ids(frames, f"{path}.frames")
        return cls(frames=frames)


def _validate_unique_frame_ids(frames: Sequence[Any], path: str) -> None:
    seen: set[str] = set()
    for frame in frames:
        if frame.frame_id in seen:
            raise KeypointSchemaError(path, f"frame_id {frame.frame_id!r} appears more than once")
        seen.add(frame.frame_id)


def load_json(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KeypointSchemaError(str(file_path), "file not found") from exc
    except json.JSONDecodeError as exc:
        raise KeypointSchemaError(
            str(file_path),
            f"invalid JSON at line {exc.lineno}, column {exc.colno}",
        ) from exc


def load_annotations(path: str | Path) -> AnnotationDataset:
    return AnnotationDataset.from_payload(load_json(path), path=str(path))


def load_predictions(path: str | Path) -> PredictionDataset:
    return PredictionDataset.from_payload(load_json(path), path=str(path))


@dataclass(frozen=True)
class EvaluationThresholds:
    confidence_threshold: float = 0.50
    match_radius_normalized: float = 0.03
    calibration_minimum_points: int = 6
    calibration_minimum_image_hull_ratio: float = 0.02
    calibration_minimum_field_hull_ratio: float = 0.08
    calibration_minimum_inlier_ratio: float = 0.75
    calibration_maximum_rmse_m: float = 1.5
    calibration_maximum_p95_m: float = 3.0

    def __post_init__(self) -> None:
        for field_name in (
            "confidence_threshold",
            "match_radius_normalized",
            "calibration_minimum_image_hull_ratio",
            "calibration_minimum_field_hull_ratio",
            "calibration_minimum_inlier_ratio",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]")
        for field_name in (
            "calibration_maximum_rmse_m",
            "calibration_maximum_p95_m",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.calibration_minimum_points < 4:
            raise ValueError("calibration_minimum_points must be >= 4")


@dataclass(frozen=True)
class QualityGateThresholds:
    semantic_f1_min: float = 0.75
    pck_02_min: float = 0.65
    p95_error_max: float = 0.035
    non_pitch_false_positive_rate_max: float = 0.05
    calibration_validated_rate_min: float = 0.70
    pitch_frame_prediction_coverage_min: float = 0.60

    def __post_init__(self) -> None:
        for field_name in (
            "semantic_f1_min",
            "pck_02_min",
            "p95_error_max",
            "non_pitch_false_positive_rate_max",
            "calibration_validated_rate_min",
            "pitch_frame_prediction_coverage_min",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]")


def normalized_keypoint_error(
    annotation: AnnotatedKeypoint,
    prediction: PredictedKeypoint,
) -> float:
    return math.hypot(annotation.x - prediction.x, annotation.y - prediction.y) / math.sqrt(2.0)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _convex_hull_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float32)
    hull = cv2.convexHull(array)
    return float(abs(cv2.contourArea(hull)))


def _calibration_diagnostic(
    annotation: AnnotationFrame,
    prediction: PredictionFrame | None,
    accepted_predictions: Mapping[str, PredictedKeypoint],
    thresholds: EvaluationThresholds,
    pitch: PitchDimensions,
) -> dict[str, Any]:
    visible_annotation = {
        item.landmark: item
        for item in annotation.keypoints
        if item.visibility == "visible"
    }
    usable_labels = sorted(set(visible_annotation).intersection(accepted_predictions))
    diagnostic: dict[str, Any] = {
        "attempted": False,
        "solved": False,
        "validated": False,
        "point_count": len(usable_labels),
        "inlier_ratio": 0.0,
        "rmse_m": None,
        "p95_m": None,
        "image_hull_ratio": 0.0,
        "field_hull_ratio": 0.0,
        "reason_codes": [],
    }
    if prediction is None or prediction.abstained:
        diagnostic["reason_codes"].append("PREDICTION_ABSTAINED")
        return diagnostic
    if len(usable_labels) < thresholds.calibration_minimum_points:
        diagnostic["reason_codes"].append("INSUFFICIENT_SEMANTIC_KEYPOINTS")
        return diagnostic
    image_points = np.asarray(
        [[accepted_predictions[label].x, accepted_predictions[label].y] for label in usable_labels],
        dtype=np.float64,
    )
    field_points = np.asarray(
        [landmark_coordinates(label, pitch) for label in usable_labels],
        dtype=np.float64,
    )
    diagnostic["attempted"] = True
    diagnostic["image_hull_ratio"] = _convex_hull_area(
        [tuple(point) for point in image_points]
    )
    diagnostic["field_hull_ratio"] = _safe_divide(
        _convex_hull_area([tuple(point) for point in field_points]),
        pitch.area_m2,
    )
    matrix, mask = cv2.findHomography(
        image_points,
        field_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=thresholds.calibration_maximum_rmse_m,
        maxIters=10_000,
        confidence=0.999,
    )
    if matrix is None or not np.isfinite(matrix).all():
        diagnostic["reason_codes"].append("HOMOGRAPHY_NOT_SOLVED")
        return diagnostic
    diagnostic["solved"] = True
    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - field_points, axis=1)
    if mask is None:
        inliers = np.ones(len(errors), dtype=bool)
    else:
        inliers = np.asarray(mask, dtype=np.uint8).reshape(-1).astype(bool)
    inlier_errors = errors[inliers]
    diagnostic["inlier_ratio"] = _safe_divide(float(np.count_nonzero(inliers)), len(errors))
    diagnostic["rmse_m"] = (
        float(math.sqrt(float(np.mean(np.square(inlier_errors)))))
        if inlier_errors.size
        else None
    )
    diagnostic["p95_m"] = (
        float(np.percentile(inlier_errors, 95.0)) if inlier_errors.size else None
    )
    reasons: list[str] = []
    if diagnostic["inlier_ratio"] < thresholds.calibration_minimum_inlier_ratio:
        reasons.append("LOW_CALIBRATION_INLIER_RATIO")
    if (
        diagnostic["rmse_m"] is None
        or diagnostic["rmse_m"] > thresholds.calibration_maximum_rmse_m
    ):
        reasons.append("HIGH_CALIBRATION_RMSE")
    if (
        diagnostic["p95_m"] is None
        or diagnostic["p95_m"] > thresholds.calibration_maximum_p95_m
    ):
        reasons.append("HIGH_CALIBRATION_P95_ERROR")
    if diagnostic["image_hull_ratio"] < thresholds.calibration_minimum_image_hull_ratio:
        reasons.append("LOW_IMAGE_POINT_COVERAGE")
    if diagnostic["field_hull_ratio"] < thresholds.calibration_minimum_field_hull_ratio:
        reasons.append("LOW_FIELD_POINT_COVERAGE")
    diagnostic["reason_codes"] = reasons
    diagnostic["validated"] = not reasons
    return diagnostic


def build_calibration_request(
    prediction: PredictionFrame,
    *,
    confidence_threshold: float = 0.50,
    minimum_points: int = 6,
    minimum_image_hull_ratio: float = 0.02,
    pitch: PitchDimensions | None = None,
) -> dict[str, Any] | None:
    pitch = pitch or PitchDimensions()
    if prediction.abstained:
        return None
    accepted = [
        point for point in prediction.keypoints if point.confidence >= confidence_threshold
    ]
    if len(accepted) < minimum_points:
        return None
    if _convex_hull_area([(point.x, point.y) for point in accepted]) < minimum_image_hull_ratio:
        return None
    return {
        "schema_version": "pitch-calibration-request-v1",
        "camera_segment_id": prediction.shot_id,
        "source": "semantic_keypoint_model",
        "start_sec": None,
        "end_sec": None,
        "pitch": {"length_m": pitch.length_m, "width_m": pitch.width_m},
        "correspondences": [
            {
                "image": {"x": point.x, "y": point.y},
                "landmark": point.landmark,
                "label": point.landmark,
            }
            for point in accepted
        ],
    }


def evaluate_keypoint_dataset(
    annotations: AnnotationDataset,
    predictions: PredictionDataset,
    *,
    thresholds: EvaluationThresholds | None = None,
    pitch: PitchDimensions | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or EvaluationThresholds()
    pitch = pitch or PitchDimensions()
    annotation_by_id = {frame.frame_id: frame for frame in annotations.frames}
    prediction_by_id = {frame.frame_id: frame for frame in predictions.frames}
    extra_prediction_ids = sorted(set(prediction_by_id) - set(annotation_by_id))
    if extra_prediction_ids:
        raise ValueError(
            "prediction dataset contains unknown frame_ids: "
            + ", ".join(extra_prediction_ids[:5])
        )

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    label_matches = 0
    localization_errors: list[float] = []
    visible_ground_truth = 0
    predicted_keypoints = 0
    pitch_frames = 0
    pitch_frames_with_prediction = 0
    non_pitch_frames = 0
    non_pitch_frames_with_prediction = 0
    abstained_frames = 0
    calibration_attempted = 0
    calibration_solved = 0
    calibration_validated = 0
    per_frame: list[dict[str, Any]] = []

    for annotation in annotations.frames:
        prediction = prediction_by_id.get(annotation.frame_id)
        if annotation.is_pitch_view:
            pitch_frames += 1
        else:
            non_pitch_frames += 1
        if prediction is None:
            prediction = PredictionFrame(
                frame_id=annotation.frame_id,
                video_id=annotation.video_id,
                shot_id=annotation.shot_id,
                time_sec=annotation.time_sec,
                abstained=True,
                keypoints=(),
                model_version="missing",
                configuration_hash="missing",
                reason_codes=("MISSING_PREDICTION_FRAME",),
            )
        if prediction.video_id != annotation.video_id or prediction.shot_id != annotation.shot_id:
            raise ValueError(f"metadata mismatch for frame {annotation.frame_id!r}")
        if abs(prediction.time_sec - annotation.time_sec) > 0.050:
            raise ValueError(f"time_sec mismatch for frame {annotation.frame_id!r}")
        if prediction.abstained:
            abstained_frames += 1

        accepted_predictions = {
            item.landmark: item
            for item in prediction.keypoints
            if item.confidence >= thresholds.confidence_threshold
        }
        if annotation.is_pitch_view and accepted_predictions:
            pitch_frames_with_prediction += 1
        if not annotation.is_pitch_view and accepted_predictions:
            non_pitch_frames_with_prediction += 1
        predicted_keypoints += len(accepted_predictions)
        annotation_points = {
            item.landmark: item
            for item in annotation.keypoints
            if item.visibility == "visible"
        }
        visible_ground_truth += len(annotation_points)

        frame_tp = 0
        frame_fp = 0
        frame_fn = 0
        frame_errors: list[float] = []
        for landmark, predicted in accepted_predictions.items():
            annotated = annotation_points.get(landmark)
            if annotated is None:
                false_positives += 1
                frame_fp += 1
                continue
            error = normalized_keypoint_error(annotated, predicted)
            localization_errors.append(error)
            frame_errors.append(error)
            label_matches += 1
            if error <= thresholds.match_radius_normalized:
                true_positives += 1
                frame_tp += 1
            else:
                false_positives += 1
                false_negatives += 1
                frame_fp += 1
                frame_fn += 1
        for landmark in annotation_points:
            if landmark not in accepted_predictions:
                false_negatives += 1
                frame_fn += 1

        calibration = _calibration_diagnostic(
            annotation,
            prediction,
            accepted_predictions,
            thresholds,
            pitch,
        )
        calibration_attempted += int(calibration["attempted"])
        calibration_solved += int(calibration["solved"])
        calibration_validated += int(calibration["validated"])
        per_frame.append(
            {
                "frame_id": annotation.frame_id,
                "split": annotation.split,
                "is_pitch_view": annotation.is_pitch_view,
                "prediction_abstained": prediction.abstained,
                "accepted_prediction_count": len(accepted_predictions),
                "ground_truth_count": len(annotation_points),
                "true_positives": frame_tp,
                "false_positives": frame_fp,
                "false_negatives": frame_fn,
                "mean_localization_error": (
                    round(float(np.mean(frame_errors)), 8) if frame_errors else None
                ),
                "calibration": calibration,
            }
        )

    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    metrics = {
        "semantic_precision": round(precision, 8),
        "semantic_recall": round(recall, 8),
        "semantic_f1": round(_f1(precision, recall), 8),
        "label_match_count": label_matches,
        "mean_normalized_error": (
            round(float(np.mean(localization_errors)), 8)
            if localization_errors
            else None
        ),
        "median_normalized_error": (
            round(float(median(localization_errors)), 8)
            if localization_errors
            else None
        ),
        "p95_normalized_error": (
            round(_percentile(localization_errors, 95.0) or 0.0, 8)
            if localization_errors
            else None
        ),
        "pck_01": round(
            _safe_divide(
                sum(error <= 0.01 for error in localization_errors),
                len(localization_errors),
            ),
            8,
        ),
        "pck_02": round(
            _safe_divide(
                sum(error <= 0.02 for error in localization_errors),
                len(localization_errors),
            ),
            8,
        ),
        "pck_05": round(
            _safe_divide(
                sum(error <= 0.05 for error in localization_errors),
                len(localization_errors),
            ),
            8,
        ),
        "pitch_frame_prediction_coverage": round(
            _safe_divide(pitch_frames_with_prediction, pitch_frames),
            8,
        ),
        "non_pitch_false_positive_rate": round(
            _safe_divide(non_pitch_frames_with_prediction, non_pitch_frames),
            8,
        ),
        "abstention_rate": round(
            _safe_divide(abstained_frames, len(annotations.frames)),
            8,
        ),
        "calibration_attempt_rate": round(
            _safe_divide(calibration_attempted, pitch_frames),
            8,
        ),
        "calibration_solved_rate": round(
            _safe_divide(calibration_solved, calibration_attempted),
            8,
        ),
        "calibration_validated_rate": round(
            _safe_divide(calibration_validated, calibration_attempted),
            8,
        ),
    }
    counts = {
        "frames": len(annotations.frames),
        "pitch_frames": pitch_frames,
        "non_pitch_frames": non_pitch_frames,
        "visible_ground_truth_keypoints": visible_ground_truth,
        "accepted_predicted_keypoints": predicted_keypoints,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "abstained_frames": abstained_frames,
        "calibration_attempted_frames": calibration_attempted,
        "calibration_solved_frames": calibration_solved,
        "calibration_validated_frames": calibration_validated,
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "thresholds": asdict(thresholds),
        "pitch": {"length_m": pitch.length_m, "width_m": pitch.width_m},
        "counts": counts,
        "metrics": metrics,
        "per_frame": per_frame,
        "note": (
            "This benchmark validates semantic pitch-keypoint predictions and "
            "calibration readiness. It does not validate athletic or player scoring."
        ),
    }


def evaluate_quality_gate(
    report: Mapping[str, Any],
    thresholds: QualityGateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or QualityGateThresholds()
    metrics = _mapping(report.get("metrics"), "$.metrics")
    definitions = [
        ("semantic_f1", ">=", thresholds.semantic_f1_min),
        ("pck_02", ">=", thresholds.pck_02_min),
        ("p95_normalized_error", "<=", thresholds.p95_error_max),
        (
            "non_pitch_false_positive_rate",
            "<=",
            thresholds.non_pitch_false_positive_rate_max,
        ),
        (
            "calibration_validated_rate",
            ">=",
            thresholds.calibration_validated_rate_min,
        ),
        (
            "pitch_frame_prediction_coverage",
            ">=",
            thresholds.pitch_frame_prediction_coverage_min,
        ),
    ]
    checks: list[dict[str, Any]] = []
    for name, comparator, threshold in definitions:
        raw = metrics.get(name)
        actual = float(raw) if isinstance(raw, (int, float)) else float("inf")
        if comparator == ">=":
            passed = math.isfinite(actual) and actual >= threshold
        else:
            passed = math.isfinite(actual) and actual <= threshold
        checks.append(
            {
                "metric": name,
                "actual": None if not math.isfinite(actual) else round(actual, 8),
                "comparator": comparator,
                "threshold": threshold,
                "passed": passed,
            }
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "thresholds": asdict(thresholds),
        "checks": checks,
        "note": (
            "These are initial engineering gates for semantic pitch keypoints. "
            "They are not evidence of validated player evaluation."
        ),
    }
