from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class PitchDimensions:
    length_m: float = 105.0
    width_m: float = 68.0

    def __post_init__(self) -> None:
        if self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("pitch dimensions must be positive")
        if not 90.0 <= float(self.length_m) <= 120.0:
            raise ValueError("pitch length must be between 90 and 120 metres")
        if not 45.0 <= float(self.width_m) <= 90.0:
            raise ValueError("pitch width must be between 45 and 90 metres")

    @property
    def area_m2(self) -> float:
        return float(self.length_m * self.width_m)

    def normalize(self, x_m: float, y_m: float) -> Tuple[float, float]:
        return float(x_m) / self.length_m, float(y_m) / self.width_m

    def denormalize(self, x: float, y: float) -> Tuple[float, float]:
        return float(x) * self.length_m, float(y) * self.width_m


@dataclass(frozen=True)
class PitchLandmark:
    name: str
    x_m: float
    y_m: float
    kind: str = "line_intersection"


def standard_landmarks(
    dimensions: PitchDimensions | None = None,
) -> Dict[str, PitchLandmark]:
    pitch = dimensions or PitchDimensions()
    length = pitch.length_m
    width = pitch.width_m
    halfway = length / 2.0
    centre_y = width / 2.0

    penalty_area_depth = 16.5
    penalty_area_width = min(40.32, width)
    penalty_y_top = (width - penalty_area_width) / 2.0
    penalty_y_bottom = width - penalty_y_top

    goal_area_depth = 5.5
    goal_area_width = min(18.32, width)
    goal_y_top = (width - goal_area_width) / 2.0
    goal_y_bottom = width - goal_y_top

    penalty_spot_distance = 11.0
    centre_circle_radius = 9.15

    entries = [
        PitchLandmark("corner_left_top", 0.0, 0.0),
        PitchLandmark("corner_left_bottom", 0.0, width),
        PitchLandmark("corner_right_top", length, 0.0),
        PitchLandmark("corner_right_bottom", length, width),
        PitchLandmark("halfway_top", halfway, 0.0),
        PitchLandmark("halfway_bottom", halfway, width),
        PitchLandmark("centre_spot", halfway, centre_y, "spot"),
        PitchLandmark(
            "centre_circle_left", halfway - centre_circle_radius, centre_y, "arc_tangent"
        ),
        PitchLandmark(
            "centre_circle_right", halfway + centre_circle_radius, centre_y, "arc_tangent"
        ),
        PitchLandmark(
            "left_penalty_area_top", penalty_area_depth, penalty_y_top
        ),
        PitchLandmark(
            "left_penalty_area_bottom", penalty_area_depth, penalty_y_bottom
        ),
        PitchLandmark(
            "right_penalty_area_top", length - penalty_area_depth, penalty_y_top
        ),
        PitchLandmark(
            "right_penalty_area_bottom", length - penalty_area_depth, penalty_y_bottom
        ),
        PitchLandmark("left_goal_area_top", goal_area_depth, goal_y_top),
        PitchLandmark("left_goal_area_bottom", goal_area_depth, goal_y_bottom),
        PitchLandmark("right_goal_area_top", length - goal_area_depth, goal_y_top),
        PitchLandmark(
            "right_goal_area_bottom", length - goal_area_depth, goal_y_bottom
        ),
        PitchLandmark("left_penalty_spot", penalty_spot_distance, centre_y, "spot"),
        PitchLandmark(
            "right_penalty_spot", length - penalty_spot_distance, centre_y, "spot"
        ),
    ]
    return {entry.name: entry for entry in entries}


def landmark_coordinates(
    name: str,
    dimensions: PitchDimensions | None = None,
) -> Tuple[float, float]:
    landmarks = standard_landmarks(dimensions)
    try:
        landmark = landmarks[name]
    except KeyError as exc:
        raise KeyError(f"unknown pitch landmark: {name}") from exc
    return landmark.x_m, landmark.y_m
