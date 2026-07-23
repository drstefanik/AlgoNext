from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.calibration.model import PitchDimensions, landmark_coordinates

CALIBRATION_REQUEST_SCHEMA_VERSION = "pitch-calibration-request-v1"
CALIBRATION_RESULT_SCHEMA_VERSION = "pitch-calibration-result-v1"


class CalibrationValidationError(ValueError):
    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationValidationError(path, "expected an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise CalibrationValidationError(path, "expected an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationValidationError(path, "expected a non-empty string")
    return value.strip()


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationValidationError(path, "expected a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CalibrationValidationError(path, "expected a finite number")
    if minimum is not None and parsed < minimum:
        raise CalibrationValidationError(path, f"must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise CalibrationValidationError(path, f"must be <= {maximum}")
    return parsed


def _optional_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, path, minimum=minimum, maximum=maximum)


@dataclass(frozen=True)
class ImagePoint:
    x: float
    y: float

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "ImagePoint":
        value = _mapping(payload, path)
        return cls(
            x=_number(value.get("x"), f"{path}.x", minimum=0.0, maximum=1.0),
            y=_number(value.get("y"), f"{path}.y", minimum=0.0, maximum=1.0),
        )

    def to_payload(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class FieldPoint:
    x_m: float
    y_m: float

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        path: str,
        dimensions: PitchDimensions,
    ) -> "FieldPoint":
        value = _mapping(payload, path)
        return cls(
            x_m=_number(
                value.get("x_m"),
                f"{path}.x_m",
                minimum=0.0,
                maximum=dimensions.length_m,
            ),
            y_m=_number(
                value.get("y_m"),
                f"{path}.y_m",
                minimum=0.0,
                maximum=dimensions.width_m,
            ),
        )

    def to_payload(self) -> dict[str, float]:
        return {"x_m": self.x_m, "y_m": self.y_m}


@dataclass(frozen=True)
class CalibrationCorrespondence:
    image: ImagePoint
    field: FieldPoint
    label: str | None = None
    weight: float = 1.0

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        path: str,
        dimensions: PitchDimensions,
    ) -> "CalibrationCorrespondence":
        value = _mapping(payload, path)
        label_value = value.get("label")
        if label_value is not None and (
            not isinstance(label_value, str) or not label_value.strip()
        ):
            raise CalibrationValidationError(
                f"{path}.label", "expected a non-empty string or null"
            )
        image = ImagePoint.from_payload(value.get("image"), f"{path}.image")
        field_payload = value.get("field")
        landmark = value.get("landmark")
        if field_payload is None and landmark is not None:
            landmark_name = _string(landmark, f"{path}.landmark")
            try:
                x_m, y_m = landmark_coordinates(landmark_name, dimensions)
            except KeyError as exc:
                raise CalibrationValidationError(
                    f"{path}.landmark", str(exc)
                ) from exc
            field = FieldPoint(x_m=x_m, y_m=y_m)
            label_value = label_value or landmark_name
        else:
            field = FieldPoint.from_payload(
                field_payload,
                f"{path}.field",
                dimensions,
            )
        return cls(
            image=image,
            field=field,
            label=label_value.strip() if isinstance(label_value, str) else None,
            weight=_number(
                value.get("weight", 1.0),
                f"{path}.weight",
                minimum=0.01,
                maximum=100.0,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "image": self.image.to_payload(),
            "field": self.field.to_payload(),
            "label": self.label,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class CalibrationRequest:
    camera_segment_id: str
    correspondences: tuple[CalibrationCorrespondence, ...]
    pitch: PitchDimensions = PitchDimensions()
    source: str = "manual"
    start_sec: float | None = None
    end_sec: float | None = None
    schema_version: str = CALIBRATION_REQUEST_SCHEMA_VERSION

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        path: str = "$",
    ) -> "CalibrationRequest":
        value = _mapping(payload, path)
        pitch_payload = value.get("pitch") or {}
        pitch_value = _mapping(pitch_payload, f"{path}.pitch")
        pitch = PitchDimensions(
            length_m=_number(
                pitch_value.get("length_m", 105.0),
                f"{path}.pitch.length_m",
                minimum=90.0,
                maximum=120.0,
            ),
            width_m=_number(
                pitch_value.get("width_m", 68.0),
                f"{path}.pitch.width_m",
                minimum=45.0,
                maximum=90.0,
            ),
        )
        correspondences = tuple(
            CalibrationCorrespondence.from_payload(
                item,
                f"{path}.correspondences[{index}]",
                pitch,
            )
            for index, item in enumerate(
                _list(value.get("correspondences"), f"{path}.correspondences")
            )
        )
        if len(correspondences) < 4:
            raise CalibrationValidationError(
                f"{path}.correspondences",
                "at least four correspondences are required",
            )
        seen_image: set[tuple[float, float]] = set()
        seen_field: set[tuple[float, float]] = set()
        for correspondence in correspondences:
            image_key = (
                round(correspondence.image.x, 8),
                round(correspondence.image.y, 8),
            )
            field_key = (
                round(correspondence.field.x_m, 6),
                round(correspondence.field.y_m, 6),
            )
            if image_key in seen_image:
                raise CalibrationValidationError(
                    f"{path}.correspondences",
                    "duplicate image point",
                )
            if field_key in seen_field:
                raise CalibrationValidationError(
                    f"{path}.correspondences",
                    "duplicate field point",
                )
            seen_image.add(image_key)
            seen_field.add(field_key)

        start_sec = _optional_number(
            value.get("start_sec"),
            f"{path}.start_sec",
            minimum=0.0,
        )
        end_sec = _optional_number(
            value.get("end_sec"),
            f"{path}.end_sec",
            minimum=0.0,
        )
        if start_sec is not None and end_sec is not None and end_sec <= start_sec:
            raise CalibrationValidationError(
                f"{path}.end_sec", "must be greater than start_sec"
            )

        schema_version = _string(
            value.get("schema_version", CALIBRATION_REQUEST_SCHEMA_VERSION),
            f"{path}.schema_version",
        )
        if schema_version != CALIBRATION_REQUEST_SCHEMA_VERSION:
            raise CalibrationValidationError(
                f"{path}.schema_version",
                f"expected {CALIBRATION_REQUEST_SCHEMA_VERSION!r}",
            )
        return cls(
            camera_segment_id=_string(
                value.get("camera_segment_id"),
                f"{path}.camera_segment_id",
            ),
            correspondences=correspondences,
            pitch=pitch,
            source=_string(value.get("source", "manual"), f"{path}.source"),
            start_sec=start_sec,
            end_sec=end_sec,
            schema_version=schema_version,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_segment_id": self.camera_segment_id,
            "source": self.source,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "pitch": {
                "length_m": self.pitch.length_m,
                "width_m": self.pitch.width_m,
            },
            "correspondences": [
                correspondence.to_payload()
                for correspondence in self.correspondences
            ],
        }


def load_calibration_request(path: str | Path) -> CalibrationRequest:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CalibrationValidationError(str(file_path), "file not found") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationValidationError(
            str(file_path),
            f"invalid JSON at line {exc.lineno}, column {exc.colno}",
        ) from exc
    return CalibrationRequest.from_payload(payload, path=str(file_path))
