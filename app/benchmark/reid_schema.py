from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ReIDSchemaValidationError(ValueError):
    """Raised when a ReID benchmark payload violates its contract."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReIDSchemaValidationError(path, "expected an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReIDSchemaValidationError(path, "expected an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReIDSchemaValidationError(path, "expected a non-empty string")
    return value.strip()


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _optional_boolean(value: Any, path: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ReIDSchemaValidationError(path, "expected a boolean or null")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReIDSchemaValidationError(path, "expected an integer")
    if value < minimum:
        raise ReIDSchemaValidationError(path, f"must be >= {minimum}")
    return value


def _optional_integer(value: Any, path: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, path, minimum=minimum)


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReIDSchemaValidationError(path, "expected a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ReIDSchemaValidationError(path, "expected a finite number")
    if minimum is not None and parsed < minimum:
        raise ReIDSchemaValidationError(path, f"must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ReIDSchemaValidationError(path, f"must be <= {maximum}")
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


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    items = _list(value, path)
    normalized = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(items)
    )
    if len(set(normalized)) != len(normalized):
        raise ReIDSchemaValidationError(path, "must not contain duplicates")
    return normalized


VISIBILITY_VISIBLE = "VISIBLE"
VISIBILITY_NOT_VISIBLE = "NOT_VISIBLE"
VISIBILITY_UNCERTAIN = "UNCERTAIN"
VISIBILITIES = {
    VISIBILITY_VISIBLE,
    VISIBILITY_NOT_VISIBLE,
    VISIBILITY_UNCERTAIN,
}

CANDIDATE_PRESENT = "PRESENT"
CANDIDATE_ABSENT = "ABSENT"
CANDIDATE_UNVERIFIABLE = "UNVERIFIABLE"
CANDIDATE_STATES = {
    CANDIDATE_PRESENT,
    CANDIDATE_ABSENT,
    CANDIDATE_UNVERIFIABLE,
}

DECISION_ACCEPTED = "ACCEPTED"
DECISION_ABSTAINED = "ABSTAINED"
DECISION_FAILED = "FAILED"
DECISIONS = {DECISION_ACCEPTED, DECISION_ABSTAINED, DECISION_FAILED}


@dataclass(frozen=True)
class EvidenceBox:
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "EvidenceBox":
        value = _mapping(payload, path)
        x = _number(value.get("x"), f"{path}.x", minimum=0.0, maximum=1.0)
        y = _number(value.get("y"), f"{path}.y", minimum=0.0, maximum=1.0)
        w = _number(value.get("w"), f"{path}.w", minimum=0.0, maximum=1.0)
        h = _number(value.get("h"), f"{path}.h", minimum=0.0, maximum=1.0)
        if w <= 0.0 or h <= 0.0:
            raise ReIDSchemaValidationError(path, "w and h must be > 0")
        epsilon = 1e-6
        if x + w > 1.0 + epsilon or y + h > 1.0 + epsilon:
            raise ReIDSchemaValidationError(
                path, "box must remain inside the normalized frame"
            )
        return cls(x=x, y=y, w=w, h=h)


@dataclass(frozen=True)
class ReIDEvidenceFrame:
    time_sec: float
    frame_index: int | None = None
    bbox: EvidenceBox | None = None
    image_path: str | None = None
    note: str | None = None

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "ReIDEvidenceFrame":
        value = _mapping(payload, path)
        bbox_payload = value.get("bbox")
        return cls(
            time_sec=_number(value.get("time_sec"), f"{path}.time_sec", minimum=0.0),
            frame_index=_optional_integer(
                value.get("frame_index"), f"{path}.frame_index", minimum=0
            ),
            bbox=(
                EvidenceBox.from_payload(bbox_payload, f"{path}.bbox")
                if bbox_payload is not None
                else None
            ),
            image_path=_optional_string(value.get("image_path"), f"{path}.image_path"),
            note=_optional_string(value.get("note"), f"{path}.note"),
        )


@dataclass(frozen=True)
class ReIDWindowAnnotation:
    window_index: int
    window_start: float
    window_end: float
    target_visibility: str
    candidate_state: str | None = None
    target_candidate_id: str | None = None
    selected_track_is_target: bool | None = None
    evidence_frames: tuple[ReIDEvidenceFrame, ...] = ()
    notes: str | None = None

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "ReIDWindowAnnotation":
        value = _mapping(payload, path)
        visibility = _string(
            value.get("target_visibility"), f"{path}.target_visibility"
        ).upper()
        if visibility not in VISIBILITIES:
            raise ReIDSchemaValidationError(
                f"{path}.target_visibility",
                "expected one of " + ", ".join(sorted(VISIBILITIES)),
            )

        raw_candidate_state = value.get("candidate_state")
        candidate_state = (
            _string(raw_candidate_state, f"{path}.candidate_state").upper()
            if raw_candidate_state is not None
            else None
        )
        if candidate_state is not None and candidate_state not in CANDIDATE_STATES:
            raise ReIDSchemaValidationError(
                f"{path}.candidate_state",
                "expected one of " + ", ".join(sorted(CANDIDATE_STATES)),
            )
        if visibility == VISIBILITY_VISIBLE and candidate_state is None:
            raise ReIDSchemaValidationError(
                f"{path}.candidate_state",
                "is required when target_visibility is VISIBLE",
            )
        if visibility != VISIBILITY_VISIBLE and candidate_state is not None:
            raise ReIDSchemaValidationError(
                f"{path}.candidate_state",
                "is only allowed when target_visibility is VISIBLE",
            )

        target_candidate_id = _optional_string(
            value.get("target_candidate_id"), f"{path}.target_candidate_id"
        )
        if candidate_state == CANDIDATE_PRESENT and target_candidate_id is None:
            raise ReIDSchemaValidationError(
                f"{path}.target_candidate_id",
                "is required when candidate_state is PRESENT",
            )
        if candidate_state != CANDIDATE_PRESENT and target_candidate_id is not None:
            raise ReIDSchemaValidationError(
                f"{path}.target_candidate_id",
                "is only allowed when candidate_state is PRESENT",
            )

        selected_track_is_target = _optional_boolean(
            value.get("selected_track_is_target"),
            f"{path}.selected_track_is_target",
        )
        if selected_track_is_target is True and visibility != VISIBILITY_VISIBLE:
            raise ReIDSchemaValidationError(
                f"{path}.selected_track_is_target",
                "cannot be true unless target_visibility is VISIBLE",
            )
        if (
            selected_track_is_target is True
            and candidate_state == CANDIDATE_ABSENT
        ):
            raise ReIDSchemaValidationError(
                f"{path}.selected_track_is_target",
                "cannot be true when candidate_state is ABSENT",
            )

        window_start = _number(
            value.get("window_start"), f"{path}.window_start", minimum=0.0
        )
        window_end = _number(
            value.get("window_end"), f"{path}.window_end", minimum=0.0
        )
        if window_end <= window_start:
            raise ReIDSchemaValidationError(
                f"{path}.window_end", "must be greater than window_start"
            )

        evidence_payload = value.get("evidence_frames", [])
        evidence_frames = tuple(
            ReIDEvidenceFrame.from_payload(
                item, f"{path}.evidence_frames[{index}]"
            )
            for index, item in enumerate(
                _list(evidence_payload, f"{path}.evidence_frames")
            )
        )
        return cls(
            window_index=_integer(value.get("window_index"), f"{path}.window_index"),
            window_start=window_start,
            window_end=window_end,
            target_visibility=visibility,
            candidate_state=candidate_state,
            target_candidate_id=target_candidate_id,
            selected_track_is_target=selected_track_is_target,
            evidence_frames=evidence_frames,
            notes=_optional_string(value.get("notes"), f"{path}.notes"),
        )


@dataclass(frozen=True)
class ReIDSequenceAnnotation:
    video_id: str
    identity: str
    windows: tuple[ReIDWindowAnnotation, ...]
    fps: float | None = None
    schema_version: str = "reid-window-annotation-v1"

    @classmethod
    def from_payload(cls, payload: Any, path: str = "$") -> "ReIDSequenceAnnotation":
        value = _mapping(payload, path)
        windows = tuple(
            ReIDWindowAnnotation.from_payload(item, f"{path}.windows[{index}]")
            for index, item in enumerate(
                _list(value.get("windows"), f"{path}.windows")
            )
        )
        if not windows:
            raise ReIDSchemaValidationError(f"{path}.windows", "must not be empty")
        indices = [window.window_index for window in windows]
        if len(set(indices)) != len(indices):
            raise ReIDSchemaValidationError(
                f"{path}.windows", "window_index must be unique"
            )
        return cls(
            video_id=_string(value.get("video_id"), f"{path}.video_id"),
            identity=_string(value.get("identity"), f"{path}.identity"),
            windows=tuple(sorted(windows, key=lambda window: window.window_index)),
            fps=_optional_number(value.get("fps"), f"{path}.fps", minimum=1e-9),
            schema_version=_string(
                value.get("schema_version", "reid-window-annotation-v1"),
                f"{path}.schema_version",
            ),
        )


@dataclass(frozen=True)
class ReIDWindowPrediction:
    window_index: int
    window_start: float
    window_end: float
    decision: str
    selected_candidate_id: str | None
    best_candidate_id: str | None
    best_score: float
    margin: float
    candidate_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Any, path: str) -> "ReIDWindowPrediction":
        value = _mapping(payload, path)
        decision = _string(value.get("decision"), f"{path}.decision").upper()
        if decision not in DECISIONS:
            raise ReIDSchemaValidationError(
                f"{path}.decision",
                "expected one of " + ", ".join(sorted(DECISIONS)),
            )
        selected_candidate_id = _optional_string(
            value.get("selected_candidate_id"), f"{path}.selected_candidate_id"
        )
        if decision == DECISION_ACCEPTED and selected_candidate_id is None:
            raise ReIDSchemaValidationError(
                f"{path}.selected_candidate_id", "is required for ACCEPTED decisions"
            )
        if decision != DECISION_ACCEPTED and selected_candidate_id is not None:
            raise ReIDSchemaValidationError(
                f"{path}.selected_candidate_id",
                "is only allowed for ACCEPTED decisions",
            )

        window_start = _number(
            value.get("window_start"), f"{path}.window_start", minimum=0.0
        )
        window_end = _number(
            value.get("window_end"), f"{path}.window_end", minimum=0.0
        )
        if window_end <= window_start:
            raise ReIDSchemaValidationError(
                f"{path}.window_end", "must be greater than window_start"
            )

        candidate_ids = _string_tuple(
            value.get("candidate_ids", []), f"{path}.candidate_ids"
        )
        best_candidate_id = _optional_string(
            value.get("best_candidate_id"), f"{path}.best_candidate_id"
        )
        if best_candidate_id is not None and best_candidate_id not in candidate_ids:
            raise ReIDSchemaValidationError(
                f"{path}.best_candidate_id", "must appear in candidate_ids"
            )
        if (
            selected_candidate_id is not None
            and candidate_ids
            and selected_candidate_id not in candidate_ids
        ):
            raise ReIDSchemaValidationError(
                f"{path}.selected_candidate_id", "must appear in candidate_ids"
            )

        return cls(
            window_index=_integer(value.get("window_index"), f"{path}.window_index"),
            window_start=window_start,
            window_end=window_end,
            decision=decision,
            selected_candidate_id=selected_candidate_id,
            best_candidate_id=best_candidate_id,
            best_score=_number(
                value.get("best_score", 0.0),
                f"{path}.best_score",
                minimum=0.0,
                maximum=1.0,
            ),
            margin=_number(
                value.get("margin", 0.0),
                f"{path}.margin",
                minimum=0.0,
                maximum=1.0,
            ),
            candidate_ids=candidate_ids,
            reason_codes=_string_tuple(
                value.get("reason_codes", []), f"{path}.reason_codes"
            ),
        )


@dataclass(frozen=True)
class ReIDSequencePrediction:
    video_id: str
    windows: tuple[ReIDWindowPrediction, ...]
    schema_version: str = "reid-window-prediction-v1"

    @classmethod
    def from_payload(cls, payload: Any, path: str = "$") -> "ReIDSequencePrediction":
        value = _mapping(payload, path)
        windows = tuple(
            ReIDWindowPrediction.from_payload(item, f"{path}.windows[{index}]")
            for index, item in enumerate(
                _list(value.get("windows"), f"{path}.windows")
            )
        )
        indices = [window.window_index for window in windows]
        if len(set(indices)) != len(indices):
            raise ReIDSchemaValidationError(
                f"{path}.windows", "window_index must be unique"
            )
        return cls(
            video_id=_string(value.get("video_id"), f"{path}.video_id"),
            windows=tuple(sorted(windows, key=lambda window: window.window_index)),
            schema_version=_string(
                value.get("schema_version", "reid-window-prediction-v1"),
                f"{path}.schema_version",
            ),
        )


def _load_json(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReIDSchemaValidationError(str(file_path), "file not found") from exc
    except json.JSONDecodeError as exc:
        raise ReIDSchemaValidationError(
            str(file_path), f"invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def load_reid_annotation(path: str | Path) -> ReIDSequenceAnnotation:
    return ReIDSequenceAnnotation.from_payload(_load_json(path), path=str(path))


def load_reid_prediction(path: str | Path) -> ReIDSequencePrediction:
    return ReIDSequencePrediction.from_payload(_load_json(path), path=str(path))
