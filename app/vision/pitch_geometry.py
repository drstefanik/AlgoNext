from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import cv2
import numpy as np

PITCH_GEOMETRY_SCHEMA_VERSION = "pitch-geometry-proposal-v1"


@dataclass(frozen=True)
class PitchGeometryThresholds:
    max_processing_width: int = 960
    green_hue_min: int = 25
    green_hue_max: int = 95
    green_saturation_min: int = 35
    green_value_min: int = 30
    white_saturation_max: int = 85
    white_value_min: int = 150
    minimum_pitch_coverage: float = 0.22
    non_pitch_green_ceiling: float = 0.08
    minimum_line_length_fraction: float = 0.08
    minimum_line_support: float = 0.32
    minimum_line_count: int = 3
    minimum_orientation_families: int = 2
    minimum_intersections: int = 2
    maximum_lines: int = 24
    duplicate_angle_deg: float = 5.0
    duplicate_distance_fraction: float = 0.025
    intersection_merge_fraction: float = 0.025

    def __post_init__(self) -> None:
        if self.max_processing_width < 160:
            raise ValueError("max_processing_width must be >= 160")
        for field_name in (
            "minimum_pitch_coverage",
            "non_pitch_green_ceiling",
            "minimum_line_length_fraction",
            "minimum_line_support",
            "duplicate_distance_fraction",
            "intersection_merge_fraction",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]")
        for field_name in (
            "minimum_line_count",
            "minimum_orientation_families",
            "minimum_intersections",
            "maximum_lines",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.maximum_lines < self.minimum_line_count:
            raise ValueError("maximum_lines must be >= minimum_line_count")


@dataclass(frozen=True)
class DetectedLine:
    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float
    length_fraction: float
    mask_support: float
    confidence: float

    def to_payload(self) -> dict[str, float]:
        return {
            "x1": round(self.x1, 6),
            "y1": round(self.y1, 6),
            "x2": round(self.x2, 6),
            "y2": round(self.y2, 6),
            "angle_deg": round(self.angle_deg, 4),
            "length_fraction": round(self.length_fraction, 6),
            "mask_support": round(self.mask_support, 6),
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class KeypointProposal:
    x: float
    y: float
    kind: str
    confidence: float
    source_line_indices: tuple[int, int]
    semantic_landmark: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "kind": self.kind,
            "confidence": round(self.confidence, 6),
            "source_line_indices": list(self.source_line_indices),
            "semantic_landmark": self.semantic_landmark,
        }


@dataclass(frozen=True)
class PitchEvidence:
    green_ratio: float
    white_line_ratio: float
    edge_density: float
    line_count: int
    orientation_family_count: int
    intersection_count: int
    pitch_probability: float
    classification: str

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "green_ratio",
            "white_line_ratio",
            "edge_density",
            "pitch_probability",
        ):
            payload[key] = round(float(payload[key]), 6)
        return payload


@dataclass(frozen=True)
class PitchGeometryProposal:
    status: str
    evidence: PitchEvidence
    lines: tuple[DetectedLine, ...]
    keypoints: tuple[KeypointProposal, ...]
    reason_codes: tuple[str, ...]
    semantic_landmarks_available: bool = False
    calibration_ready: bool = False
    schema_version: str = PITCH_GEOMETRY_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "evidence": self.evidence.to_payload(),
            "lines": [line.to_payload() for line in self.lines],
            "keypoints": [point.to_payload() for point in self.keypoints],
            "semantic_landmarks_available": self.semantic_landmarks_available,
            "calibration_ready": self.calibration_ready,
            "reason_codes": list(self.reason_codes),
        }


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    if frame.shape[0] < 32 or frame.shape[1] < 32:
        raise ValueError("frame is too small for pitch geometry analysis")
    if frame.dtype != np.uint8:
        raise ValueError("frame must use uint8 BGR pixels")


def _resize_for_processing(
    frame: np.ndarray,
    thresholds: PitchGeometryThresholds,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= thresholds.max_processing_width:
        return frame.copy()
    scale = thresholds.max_processing_width / float(width)
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(
        frame,
        (thresholds.max_processing_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def build_pitch_masks(
    frame: np.ndarray,
    thresholds: PitchGeometryThresholds | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = thresholds or PitchGeometryThresholds()
    _validate_frame(frame)
    processed = _resize_for_processing(frame, thresholds)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array(
            [
                thresholds.green_hue_min,
                thresholds.green_saturation_min,
                thresholds.green_value_min,
            ],
            dtype=np.uint8,
        ),
        np.array(
            [thresholds.green_hue_max, 255, 255],
            dtype=np.uint8,
        ),
    )
    kernel_size = max(3, int(round(min(processed.shape[:2]) * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel, iterations=1)
    green = _largest_component(green)

    pitch_support = cv2.dilate(green, kernel, iterations=1)
    white = cv2.inRange(
        hsv,
        np.array([0, 0, thresholds.white_value_min], dtype=np.uint8),
        np.array([179, thresholds.white_saturation_max, 255], dtype=np.uint8),
    )
    white = cv2.bitwise_and(white, pitch_support)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, line_kernel, iterations=1)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, line_kernel, iterations=1)
    return processed, green, white


def _line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _angle_distance(first: float, second: float) -> float:
    delta = abs(first - second) % 180.0
    return min(delta, 180.0 - delta)


def _sample_line_support(
    mask: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    length = max(2, int(round(math.hypot(x2 - x1, y2 - y1))))
    xs = np.linspace(x1, x2, length)
    ys = np.linspace(y1, y2, length)
    xs = np.clip(np.rint(xs).astype(np.int32), 0, mask.shape[1] - 1)
    ys = np.clip(np.rint(ys).astype(np.int32), 0, mask.shape[0] - 1)
    return float(np.count_nonzero(mask[ys, xs])) / float(length)


def _line_equation(line: DetectedLine, width: int, height: int) -> tuple[float, float, float]:
    x1 = line.x1 * width
    y1 = line.y1 * height
    x2 = line.x2 * width
    y2 = line.y2 * height
    a = y1 - y2
    b = x2 - x1
    norm = math.hypot(a, b)
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0
    return a / norm, b / norm, (x1 * y2 - x2 * y1) / norm


def _line_distance_pixels(
    candidate: DetectedLine,
    reference: DetectedLine,
    width: int,
    height: int,
) -> float:
    a, b, c = _line_equation(reference, width, height)
    if abs(a) + abs(b) <= 1e-12:
        return float("inf")
    points = (
        (candidate.x1 * width, candidate.y1 * height),
        (candidate.x2 * width, candidate.y2 * height),
    )
    return sum(abs(a * x + b * y + c) for x, y in points) / 2.0


def detect_pitch_lines(
    frame: np.ndarray,
    thresholds: PitchGeometryThresholds | None = None,
) -> tuple[tuple[DetectedLine, ...], PitchEvidence, np.ndarray, np.ndarray]:
    thresholds = thresholds or PitchGeometryThresholds()
    processed, green_mask, white_mask = build_pitch_masks(frame, thresholds)
    height, width = processed.shape[:2]
    frame_area = float(height * width)
    diagonal = math.hypot(width, height)
    green_ratio = float(np.count_nonzero(green_mask)) / frame_area
    white_line_ratio = float(np.count_nonzero(white_mask)) / frame_area

    edges = cv2.Canny(white_mask, 40, 120, apertureSize=3, L2gradient=True)
    edge_density = float(np.count_nonzero(edges)) / frame_area
    minimum_length = max(
        20,
        int(round(min(height, width) * thresholds.minimum_line_length_fraction)),
    )
    hough_threshold = max(15, int(round(min(height, width) * 0.025)))
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=hough_threshold,
        minLineLength=minimum_length,
        maxLineGap=max(8, int(round(min(height, width) * 0.025))),
    )

    candidates: list[DetectedLine] = []
    if raw is not None:
        for values in raw.reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in values)
            length = math.hypot(x2 - x1, y2 - y1)
            length_fraction = length / max(1.0, diagonal)
            support = _sample_line_support(white_mask, x1, y1, x2, y2)
            if support < thresholds.minimum_line_support:
                continue
            confidence = _clamp(
                0.55 * min(1.0, length_fraction / 0.35) + 0.45 * support
            )
            candidates.append(
                DetectedLine(
                    x1=_clamp(x1 / float(width)),
                    y1=_clamp(y1 / float(height)),
                    x2=_clamp(x2 / float(width)),
                    y2=_clamp(y2 / float(height)),
                    angle_deg=_line_angle_deg(x1, y1, x2, y2),
                    length_fraction=length_fraction,
                    mask_support=support,
                    confidence=confidence,
                )
            )

    selected: list[DetectedLine] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.confidence, item.length_fraction),
        reverse=True,
    ):
        duplicate = False
        for existing in selected:
            if (
                _angle_distance(candidate.angle_deg, existing.angle_deg)
                > thresholds.duplicate_angle_deg
            ):
                continue
            distance = _line_distance_pixels(candidate, existing, width, height)
            if distance <= thresholds.duplicate_distance_fraction * diagonal:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= thresholds.maximum_lines:
            break

    families = cluster_line_orientations(selected)
    intersections = intersect_pitch_lines(
        selected,
        merge_fraction=thresholds.intersection_merge_fraction,
    )

    green_score = _clamp((green_ratio - 0.10) / 0.45)
    white_score = _clamp(white_line_ratio / 0.025)
    line_score = _clamp(len(selected) / 8.0)
    family_score = _clamp((len(families) - 1) / 2.0)
    intersection_score = _clamp(len(intersections) / 6.0)
    probability = _clamp(
        0.45 * green_score
        + 0.15 * white_score
        + 0.20 * line_score
        + 0.10 * family_score
        + 0.10 * intersection_score
    )
    if (
        green_ratio >= thresholds.minimum_pitch_coverage
        and len(selected) >= thresholds.minimum_line_count
        and len(families) >= thresholds.minimum_orientation_families
        and probability >= 0.50
    ):
        classification = "PITCH_CANDIDATE"
    elif green_ratio <= thresholds.non_pitch_green_ceiling or probability < 0.18:
        classification = "NON_PITCH"
    else:
        classification = "UNKNOWN"

    evidence = PitchEvidence(
        green_ratio=green_ratio,
        white_line_ratio=white_line_ratio,
        edge_density=edge_density,
        line_count=len(selected),
        orientation_family_count=len(families),
        intersection_count=len(intersections),
        pitch_probability=probability,
        classification=classification,
    )
    return tuple(selected), evidence, green_mask, white_mask


def cluster_line_orientations(
    lines: Sequence[DetectedLine],
    *,
    tolerance_deg: float = 12.0,
) -> tuple[tuple[float, ...], ...]:
    clusters: list[list[float]] = []
    for line in sorted(lines, key=lambda item: item.angle_deg):
        placed = False
        for cluster in clusters:
            mean = _orientation_mean(cluster)
            if _angle_distance(line.angle_deg, mean) <= tolerance_deg:
                cluster.append(line.angle_deg)
                placed = True
                break
        if not placed:
            clusters.append([line.angle_deg])
    if len(clusters) > 1:
        first_mean = _orientation_mean(clusters[0])
        last_mean = _orientation_mean(clusters[-1])
        if _angle_distance(first_mean, last_mean) <= tolerance_deg:
            clusters[0].extend(clusters.pop())
    return tuple(tuple(cluster) for cluster in clusters)


def _orientation_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    radians = np.deg2rad(np.asarray(values, dtype=np.float64) * 2.0)
    angle = math.degrees(
        math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean()))
    )
    return (angle / 2.0) % 180.0


def _intersection(first: DetectedLine, second: DetectedLine) -> tuple[float, float] | None:
    x1, y1, x2, y2 = first.x1, first.y1, first.x2, first.y2
    x3, y3, x4, y4 = second.x1, second.y1, second.x2, second.y2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) <= 1e-9:
        return None
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    x = (
        determinant_first * (x3 - x4)
        - (x1 - x2) * determinant_second
    ) / denominator
    y = (
        determinant_first * (y3 - y4)
        - (y1 - y2) * determinant_second
    ) / denominator
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    if not (-0.03 <= x <= 1.03 and -0.03 <= y <= 1.03):
        return None
    return _clamp(x), _clamp(y)


def intersect_pitch_lines(
    lines: Sequence[DetectedLine],
    *,
    merge_fraction: float = 0.025,
) -> tuple[KeypointProposal, ...]:
    proposals: list[KeypointProposal] = []
    for first_index, first in enumerate(lines):
        for second_index in range(first_index + 1, len(lines)):
            second = lines[second_index]
            separation = _angle_distance(first.angle_deg, second.angle_deg)
            if separation < 20.0 or separation > 160.0:
                continue
            point = _intersection(first, second)
            if point is None:
                continue
            x, y = point
            angle_quality = math.sin(math.radians(separation))
            confidence = _clamp(
                math.sqrt(first.confidence * second.confidence) * angle_quality
            )
            candidate = KeypointProposal(
                x=x,
                y=y,
                kind="line_intersection",
                confidence=confidence,
                source_line_indices=(first_index, second_index),
            )
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(proposals)
                    if math.hypot(existing.x - x, existing.y - y) <= merge_fraction
                ),
                None,
            )
            if duplicate_index is None:
                proposals.append(candidate)
            elif candidate.confidence > proposals[duplicate_index].confidence:
                proposals[duplicate_index] = candidate
    proposals.sort(key=lambda item: item.confidence, reverse=True)
    return tuple(proposals[:32])


def estimate_pitch_evidence(
    frame: np.ndarray,
    thresholds: PitchGeometryThresholds | None = None,
) -> PitchEvidence:
    _, evidence, _, _ = detect_pitch_lines(frame, thresholds)
    return evidence


def detect_pitch_geometry(
    frame: np.ndarray,
    thresholds: PitchGeometryThresholds | None = None,
) -> PitchGeometryProposal:
    thresholds = thresholds or PitchGeometryThresholds()
    lines, evidence, _, _ = detect_pitch_lines(frame, thresholds)
    keypoints = intersect_pitch_lines(
        lines,
        merge_fraction=thresholds.intersection_merge_fraction,
    )
    reasons: list[str] = []
    if evidence.classification != "PITCH_CANDIDATE":
        reasons.append("FRAME_NOT_CONFIDENT_PITCH_VIEW")
    if evidence.line_count < thresholds.minimum_line_count:
        reasons.append("INSUFFICIENT_PITCH_LINES")
    if evidence.orientation_family_count < thresholds.minimum_orientation_families:
        reasons.append("INSUFFICIENT_LINE_ORIENTATION_FAMILIES")
    if evidence.intersection_count < thresholds.minimum_intersections:
        reasons.append("INSUFFICIENT_LINE_INTERSECTIONS")
    reasons.append("SEMANTIC_LANDMARKS_NOT_ASSIGNED")
    status = "CANDIDATE" if len(reasons) == 1 else "INSUFFICIENT"
    return PitchGeometryProposal(
        status=status,
        evidence=evidence,
        lines=lines,
        keypoints=keypoints,
        reason_codes=tuple(dict.fromkeys(reasons)),
        semantic_landmarks_available=False,
        calibration_ready=False,
    )
