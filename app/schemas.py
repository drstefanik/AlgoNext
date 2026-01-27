from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, conlist, model_validator, ConfigDict


class JobCreate(BaseModel):
    video_url: Optional[str] = None
    video_bucket: Optional[str] = None
    video_key: Optional[str] = None
    role: str
    category: str
    team_name: str = Field(min_length=1)
    player_name: Optional[str] = None
    shirt_number: Optional[int] = Field(default=None, ge=0, le=99)

    @model_validator(mode="after")
    def require_video_source(self) -> "JobCreate":
        if not self.video_url and not self.video_key:
            raise ValueError("video_url or video_key is required")
        if self.video_url and self.video_key:
            raise ValueError("Provide either video_url or video_key, not both")
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
    model_config = ConfigDict(populate_by_name=True)
    frame_time_sec: float = Field(ge=0)
    frame_key: Optional[str] = Field(default=None, alias="frameKey")
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class SelectionPayload(BaseModel):
    selections: conlist(SelectionBox, min_length=1, max_length=5)


class PlayerRefPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
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
            "frame_time_sec": float(frame_time_sec),
            "bbox_xywh": bbox_xywh,
            "bbox_xyxy": bbox_xyxy,
        }

    @staticmethod
    def _extract_bbox_xywh(data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        bbox_xywh = data.get("bbox_xywh", data.get("bboxXYWH"))
        if isinstance(bbox_xywh, dict) and {"x", "y", "w", "h"}.issubset(bbox_xywh):
            return {
                "x": float(bbox_xywh["x"]),
                "y": float(bbox_xywh["y"]),
                "w": float(bbox_xywh["w"]),
                "h": float(bbox_xywh["h"]),
            }
        if {"x", "y", "w", "h"}.issubset(data.keys()):
            return {
                "x": float(data["x"]),
                "y": float(data["y"]),
                "w": float(data["w"]),
                "h": float(data["h"]),
            }
        bbox = data.get("bbox")
        if isinstance(bbox, dict) and {"x", "y", "w", "h"}.issubset(bbox.keys()):
            return {
                "x": float(bbox["x"]),
                "y": float(bbox["y"]),
                "w": float(bbox["w"]),
                "h": float(bbox["h"]),
            }
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = bbox
            return {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}
        return None

    @staticmethod
    def _extract_bbox_xywh_from_xyxy(bbox_xyxy: Any) -> Optional[Dict[str, float]]:
        if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) == 4:
            x1, y1, x2, y2 = bbox_xyxy
            x1_f, y1_f, x2_f, y2_f = map(float, (x1, y1, x2, y2))
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
            float(bbox_xywh["x"]),
            float(bbox_xywh["y"]),
            float(bbox_xywh["w"]),
            float(bbox_xywh["h"]),
        )
        if w <= 0 or h <= 0:
            raise ValueError("Invalid bbox dimensions")
        if x < 0 or y < 0 or x > 1 or y > 1:
            raise ValueError("Invalid bbox dimensions")
        if x + w > 1 or y + h > 1:
            raise ValueError("Invalid bbox dimensions")
        return {"x": x, "y": y, "w": w, "h": h}


class TrackSelectionBox(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
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
        frame_time_sec = (
            data.get("frame_time_sec")
            or data.get("time_sec")
            or data.get("frameTimeSec")
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

        return {
            "frame_time_sec": float(frame_time_sec),
            "x": float(bbox["x"]),
            "y": float(bbox["y"]),
            "w": float(bbox["w"]),
            "h": float(bbox["h"]),
        }


class TrackSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    track_id: int | str = Field(alias="trackId")
    selection: TrackSelectionBox


class TargetSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    frame_key: Optional[str] = Field(default=None, alias="frameKey")
    time_sec: Optional[float] = Field(default=None, alias="timeSec")
    bbox: Dict[str, float]

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Missing target selection payload")
        frame_key = data.get("frame_key") or data.get("frameKey")
        time_sec = data.get("time_sec") or data.get("timeSec")
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
                "x": float(bbox["x"]),
                "y": float(bbox["y"]),
                "w": float(bbox["w"]),
                "h": float(bbox["h"]),
            }
        )
        if frame_key is None and time_sec is None:
            raise ValueError("Missing frame_key or time_sec")
        return {
            "frame_key": frame_key,
            "time_sec": float(time_sec) if time_sec is not None else None,
            "bbox": bbox,
        }
