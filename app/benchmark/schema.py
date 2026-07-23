from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class SchemaValidationError(ValueError):
    """Raised when a benchmark annotation or prediction violates the contract."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(path, "expected an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaValidationError(path, "expected an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(path, "expected a non-empty string")
    return value.strip()


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(path, "expected an integer")
    if value < minimum:
        raise SchemaValidationError(path, f"must be >= {minimum}")
    return value


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(path, "expected a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SchemaValidationError(path, "expected a finite number")
    if minimum is not None and parsed < minimum:
        raise SchemaValidationError(path, f"must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise SchemaValidationError(path, f"must be <= {maximum}")
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
class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "BoundingBox":
        value = _mapping(payload, path)
        x = _number(value.get("x"), f"{path}.x", minimum=0.0, maximum=1.0)
        y = _number(value.get("y"), f"{path}.y", minimum=0.0, maximum=1.0)
        w = _number(value.get("w"), f"{path}.w", minimum=0.0, maximum=1.0)
        h = _number(value.get("h"), f"{path}.h", minimum=0.0, maximum=1.0)
        if w <= 0.0:
            raise SchemaValidationError(f"{path}.w", "must be > 0")
        if h <= 0.0:
            raise SchemaValidationError(f"{path}.h", "must be > 0")
        epsilon = 1e-6
        if x + w > 1.0 + epsilon:
            raise SchemaValidationError(path, "x + w must be <= 1")
        if y + h > 1.0 + epsilon:
            raise SchemaValidationError(path, "y + h must be <= 1")
        return cls(x=x, y=y, w=w, h=h)


@dataclass(frozen=True)
class GroundTruthObject:
    identity: str
    bbox: BoundingBox
    ignore: bool = False

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "GroundTruthObject":
        value = _mapping(payload, path)
        ignore = value.get("ignore", False)
        if not isinstance(ignore, bool):
            raise SchemaValidationError(f"{path}.ignore", "expected a boolean")
        return cls(
            identity=_string(value.get("identity"), f"{path}.identity"),
            bbox=BoundingBox.from_payload(value.get("bbox"), f"{path}.bbox"),
            ignore=ignore,
        )


@dataclass(frozen=True)
class PredictedTrack:
    track_id: str
    bbox: BoundingBox
    confidence: float | None = None

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "PredictedTrack":
        value = _mapping(payload, path)
        track_id = value.get("track_id")
        if isinstance(track_id, (int, float)) and not isinstance(track_id, bool):
            track_id = str(track_id)
        return cls(
            track_id=_string(track_id, f"{path}.track_id"),
            bbox=BoundingBox.from_payload(value.get("bbox"), f"{path}.bbox"),
            confidence=_optional_number(
                value.get("confidence"),
                f"{path}.confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True)
class AnnotationFrame:
    frame_index: int
    time_sec: float | None
    objects: tuple[GroundTruthObject, ...]

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "AnnotationFrame":
        value = _mapping(payload, path)
        objects = tuple(
            GroundTruthObject.from_payload(item, f"{path}.objects[{index}]")
            for index, item in enumerate(_list(value.get("objects"), f"{path}.objects"))
        )
        identities: set[str] = set()
        for obj in objects:
            if obj.ignore:
                continue
            if obj.identity in identities:
                raise SchemaValidationError(
                    f"{path}.objects",
                    f"identity {obj.identity!r} appears more than once in the frame",
                )
            identities.add(obj.identity)
        return cls(
            frame_index=_integer(value.get("frame_index"), f"{path}.frame_index"),
            time_sec=_optional_number(
                value.get("time_sec"), f"{path}.time_sec", minimum=0.0
            ),
            objects=objects,
        )


@dataclass(frozen=True)
class PredictionFrame:
    frame_index: int
    time_sec: float | None
    tracks: tuple[PredictedTrack, ...]

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "PredictionFrame":
        value = _mapping(payload, path)
        tracks = tuple(
            PredictedTrack.from_payload(item, f"{path}.tracks[{index}]")
            for index, item in enumerate(_list(value.get("tracks"), f"{path}.tracks"))
        )
        track_ids: set[str] = set()
        for track in tracks:
            if track.track_id in track_ids:
                raise SchemaValidationError(
                    f"{path}.tracks",
                    f"track_id {track.track_id!r} appears more than once in the frame",
                )
            track_ids.add(track.track_id)
        return cls(
            frame_index=_integer(value.get("frame_index"), f"{path}.frame_index"),
            time_sec=_optional_number(
                value.get("time_sec"), f"{path}.time_sec", minimum=0.0
            ),
            tracks=tracks,
        )


def _validate_unique_frames(
    frames: Sequence[AnnotationFrame] | Sequence[PredictionFrame], path: str
) -> None:
    seen: set[int] = set()
    for frame in frames:
        if frame.frame_index in seen:
            raise SchemaValidationError(
                path, f"frame_index {frame.frame_index} appears more than once"
            )
        seen.add(frame.frame_index)


@dataclass(frozen=True)
class SequenceAnnotation:
    video_id: str
    fps: float | None
    frames: tuple[AnnotationFrame, ...]
    schema_version: str = "tracking-annotation-v1"

    @classmethod
    def from_payload(cls, payload: Any, path: str = "$") -> "SequenceAnnotation":
        value = _mapping(payload, path)
        frames = tuple(
            AnnotationFrame.from_payload(item, f"{path}.frames[{index}]")
            for index, item in enumerate(_list(value.get("frames"), f"{path}.frames"))
        )
        _validate_unique_frames(frames, f"{path}.frames")
        if not frames:
            raise SchemaValidationError(f"{path}.frames", "must contain at least one frame")
        return cls(
            video_id=_string(value.get("video_id"), f"{path}.video_id"),
            fps=_optional_number(value.get("fps"), f"{path}.fps", minimum=1e-9),
            frames=tuple(sorted(frames, key=lambda frame: frame.frame_index)),
            schema_version=_string(
                value.get("schema_version", "tracking-annotation-v1"),
                f"{path}.schema_version",
            ),
        )


@dataclass(frozen=True)
class SequencePrediction:
    video_id: str
    frames: tuple[PredictionFrame, ...]
    schema_version: str = "tracking-prediction-v1"

    @classmethod
    def from_payload(cls, payload: Any, path: str = "$") -> "SequencePrediction":
        value = _mapping(payload, path)
        frames = tuple(
            PredictionFrame.from_payload(item, f"{path}.frames[{index}]")
            for index, item in enumerate(_list(value.get("frames"), f"{path}.frames"))
        )
        _validate_unique_frames(frames, f"{path}.frames")
        return cls(
            video_id=_string(value.get("video_id"), f"{path}.video_id"),
            frames=tuple(sorted(frames, key=lambda frame: frame.frame_index)),
            schema_version=_string(
                value.get("schema_version", "tracking-prediction-v1"),
                f"{path}.schema_version",
            ),
        )


def load_json(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaValidationError(str(file_path), "file not found") from exc
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            str(file_path), f"invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def load_annotation(path: str | Path) -> SequenceAnnotation:
    return SequenceAnnotation.from_payload(load_json(path), path=str(path))


def load_prediction(path: str | Path) -> SequencePrediction:
    return SequencePrediction.from_payload(load_json(path), path=str(path))
