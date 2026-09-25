import math
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, conlist, model_validator, ConfigDict


def _finite_number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected a finite number") from exc
    if isinstance(value, bool) or not math.isfinite(parsed):
        raise ValueError("Expected a finite number")
    return parsed


class JobCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    video_url: Optional[str] = None
    video_bucket: Optional[str] = None
    video_key: Optional[str] = None
    lgi_match_id: Optional[str] = None
    role: str = Field(min_length=1)
    category: str = Field(min_length=1)
    team_name: Optional[str] = None
    player_name: Optional[str] = None
    shirt_number: Optional[int] = Field(default=None, ge=0, le=99)
    full_match_mode: Optional[bool] = None

    @model_validator(mode="after")
    def require_video_source(self) -> "JobCreate":
        sources = [
            bool(self.video_url),
            bool(self.video_key),
            bool(self.lgi_match_id),
        ]
        if sum(sources) != 1:
            raise ValueError(
                "Provide exactly one of video_url, video_key, or lgi_match_id"
            )
        return self


class JobOut(BaseModel):
    job_id: str
    status: str


class JobStatusOut(BaseModel):
    job_id: str
    id: Optional[str] = None
    status: str
    progress: Dict
    error: Optional[str]
    failure_reason: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class SelectionBox(BaseModel):
    model_config = ConfigDict(populate_by_name=True, allow_inf_nan=False)
    frame_time_sec: float = Field(ge=0)
    frame_key: Optional[str] = Field(default=None, alias="frameKey")
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Missing selection payload")
        frame_time_sec = data.get("frame_time_sec")
        if frame_time_sec is None:
            frame_time_sec = data.get("frameTimeSec")
        if frame_time_sec is None:
            raise ValueError("Missing selection frame_time_sec")

        bbox = data.get("bbox_xywh")
        if not isinstance(bbox, dict):
            bbox = data.get("bbox")
        if not isinstance(bbox, dict):
            bbox = {
                "x": data.get("x"),
                "y": data.get("y"),
                "w": data.get("w"),
                "h": data.get("h"),
            }
        if not isinstance(bbox, dict) or not {"x", "y", "w", "h"}.issubset(bbox):
            raise ValueError("Missing selection bbox")

        bbox = PlayerRefPayload._validate_bbox_xywh(
            {
                "x": _finite_number(bbox["x"]),
                "y": _finite_number(bbox["y"]),
                "w": _finite_number(bbox["w"]),
                "h": _finite_number(bbox["h"]),
            }
        )

        frame_key = data.get("frame_key") or data.get("frameKey")

        return {
            "frame_time_sec": _finite_number(frame_time_sec),
            "frame_key": frame_key,
            "x": bbox["x"],
            "y": bbox["y"],
            "w": bbox["w"],
            "h": bbox["h"],
        }


class SelectionPayload(BaseModel):
    selections: conlist(SelectionBox, min_length=1, max_length=5)


class PlayerRefPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, allow_inf_nan=False
    )
    frame_time_sec: float = Field(ge=0, alias="frameTimeSec")
    bbox_xywh: Dict[str, float]
    bbox_xyxy: Dict[str, float]

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Missing frame_time_sec/frameTimeSec")
        frame_time_sec = data.get("frame_time_sec", data.get("frameTimeSec"))
        if frame_time_sec is None:
            raise ValueError("Missing frame_time_sec/frameTimeSec")

        bbox_xywh = cls._extract_bbox_xywh(data)
        if bbox_xywh is None:
            bbox_xywh = cls._extract_bbox_xywh_from_xyxy(
                data.get("bbox_xyxy", data.get("bboxXYXY"))
            )
        if bbox_xywh is None:
            raise ValueError("Missing bbox fields")

        bbox_xywh = cls._validate_bbox_xywh(bbox_xywh)
        bbox_xyxy = cls._bbox_xywh_to_xyxy(bbox_xywh)
        return {
            "frame_time_sec": _finite_number(frame_time_sec),
            "bbox_xywh": bbox_xywh,
            "bbox_xyxy": bbox_xyxy,
        }

    @staticmethod
    def _extract_bbox_xywh(data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        bbox_xywh = data.get("bbox_xywh", data.get("bboxXYWH"))
        if isinstance(bbox_xywh, dict) and {"x", "y", "w", "h"}.issubset(bbox_xywh):
            return {
                "x": _finite_number(bbox_xywh["x"]),
                "y": _finite_number(bbox_xywh["y"]),
                "w": _finite_number(bbox_xywh["w"]),
                "h": _finite_number(bbox_xywh["h"]),
            }
        if {"x", "y", "w", "h"}.issubset(data.keys()):
            return {
                "x": _finite_number(data["x"]),
                "y": _finite_number(data["y"]),
                "w": _finite_number(data["w"]),
                "h": _finite_number(data["h"]),
            }
        bbox = data.get("bbox")
        if isinstance(bbox, dict) and {"x", "y", "w", "h"}.issubset(bbox.keys()):
            return {
                "x": _finite_number(bbox["x"]),
                "y": _finite_number(bbox["y"]),
                "w": _finite_number(bbox["w"]),
                "h": _finite_number(bbox["h"]),
            }
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = bbox
            return {
                "x": _finite_number(x),
                "y": _finite_number(y),
                "w": _finite_number(w),
                "h": _finite_number(h),
            }
        return None

    @staticmethod
    def _extract_bbox_xywh_from_xyxy(bbox_xyxy: Any) -> Optional[Dict[str, float]]:
        if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) == 4:
            x1, y1, x2, y2 = bbox_xyxy
            x1_f, y1_f, x2_f, y2_f = map(_finite_number, (x1, y1, x2, y2))
            return {
                "x": x1_f,
                "y": y1_f,
                "w": x2_f - x1_f,
                "h": y2_f - y1_f,
            }
        return None

    @staticmethod
    def _bbox_xywh_to_xyxy(bbox_xywh: Dict[str, float]) -> Dict[str, float]:
        return {
            "x1": bbox_xywh["x"],
            "y1": bbox_xywh["y"],
            "x2": bbox_xywh["x"] + bbox_xywh["w"],
            "y2": bbox_xywh["y"] + bbox_xywh["h"],
        }

    @staticmethod
    def _validate_bbox_xywh(bbox_xywh: Dict[str, float]) -> Dict[str, float]:
        x, y, w, h = (
            _finite_number(bbox_xywh["x"]),
            _finite_number(bbox_xywh["y"]),
            _finite_number(bbox_xywh["w"]),
            _finite_number(bbox_xywh["h"]),
        )
        if not all(math.isfinite(value) for value in (x, y, w, h)):
            raise ValueError("BBox coordinates must be finite")
        if w <= 0 or h <= 0:
            raise ValueError("Invalid bbox dimensions")
        if x < 0 or y < 0 or x > 1 or y > 1:
            raise ValueError("Invalid bbox dimensions")
        if x + w > 1 or y + h > 1:
            raise ValueError("Invalid bbox dimensions")
        return {"x": x, "y": y, "w": w, "h": h}


class TrackSelectionBox(BaseModel):
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, allow_inf_nan=False
    )
    frame_time_sec: float = Field(ge=0, alias="time_sec")
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Missing selection payload")
        frame_time_sec = next(
            (
                data[key]
                for key in ("frame_time_sec", "time_sec", "frameTimeSec")
                if data.get(key) is not None
            ),
            None,
        )
        if frame_time_sec is None:
            raise ValueError("Missing selection time_sec")

        bbox = data.get("bbox")
        if not isinstance(bbox, dict):
            bbox = {
                "x": data.get("x"),
                "y": data.get("y"),
                "w": data.get("w"),
                "h": data.get("h"),
            }
        if not isinstance(bbox, dict) or not {"x", "y", "w", "h"}.issubset(bbox):
            raise ValueError("Missing selection bbox")

        bbox = PlayerRefPayload._validate_bbox_xywh(bbox)
        return {
            "frame_time_sec": _finite_number(frame_time_sec),
            "x": _finite_number(bbox["x"]),
            "y": _finite_number(bbox["y"]),
            "w": _finite_number(bbox["w"]),
            "h": _finite_number(bbox["h"]),
        }


class TrackSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    track_id: int | str = Field(alias="trackId")
    selection: TrackSelectionBox | None = None


class PickPlayerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    frame_key: str = Field(alias="frameKey")
    track_id: int | str = Field(alias="trackId")

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Missing pick-player payload")
        frame_key = data.get("frame_key") or data.get("frameKey")
        track_id = (
            data.get("track_id")
            if data.get("track_id") is not None
            else data.get("trackId")
        )
        if not frame_key:
            raise ValueError("Missing frame_key")
        if track_id is None:
            raise ValueError("Missing track_id")
        return {"frame_key": frame_key, "track_id": track_id}


class TargetSelectionPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, allow_inf_nan=False
    )
    frame_key: Optional[str] = Field(default=None, alias="frameKey")
    time_sec: Optional[float] = Field(default=None, alias="timeSec", ge=0)
    bbox: Dict[str, float]
    track_id: Optional[int | str] = Field(default=0, alias="trackId")
    force: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Missing target selection payload")
        frame_key = data.get("frame_key") or data.get("frameKey") or data.get("key")
        time_sec = next(
            (
                data[key]
                for key in ("time_sec", "timeSec", "frame_time_sec", "frameTimeSec")
                if data.get(key) is not None
            ),
            None,
        )
        force = bool(data.get("force")) if "force" in data else False
        if "track_id" in data:
            track_id = data.get("track_id")
        else:
            track_id = data.get("trackId")
        if track_id is None:
            track_id = 0
        bbox = data.get("bbox")
        if not isinstance(bbox, dict):
            bbox = {
                "x": data.get("x"),
                "y": data.get("y"),
                "w": data.get("w"),
                "h": data.get("h"),
            }
        if not isinstance(bbox, dict) or not {"x", "y", "w", "h"}.issubset(bbox):
            raise ValueError("Missing target bbox")
        bbox = PlayerRefPayload._validate_bbox_xywh(
            {
                "x": _finite_number(bbox["x"]),
                "y": _finite_number(bbox["y"]),
                "w": _finite_number(bbox["w"]),
                "h": _finite_number(bbox["h"]),
            }
        )
        if frame_key is None and time_sec is None:
            raise ValueError("Missing frame_key or time_sec")
        return {
            "frame_key": frame_key,
            "time_sec": _finite_number(time_sec) if time_sec is not None else None,
            "bbox": bbox,
            "track_id": track_id,
            "force": force,
        }
