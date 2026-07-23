from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import cv2
import numpy as np

from app.reid.association import AppearanceDescriptor, DESCRIPTOR_VERSION


@dataclass(frozen=True)
class CropQuality:
    score: float
    width: int
    height: int
    sharpness: float
    saturation: float


def _normalized_histogram(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
) -> np.ndarray:
    histogram = cv2.calcHist(
        [values], [0], None, [bins], list(value_range)
    ).reshape(-1)
    total = float(histogram.sum())
    if total > 0:
        histogram = histogram / total
    return histogram.astype(np.float32)


def crop_from_normalized_bbox(
    frame: np.ndarray,
    bbox: Mapping[str, float],
    *,
    padding: float = 0.04,
) -> np.ndarray | None:
    if (
        frame is None
        or frame.ndim != 3
        or frame.shape[0] < 2
        or frame.shape[1] < 2
    ):
        return None
    height, width = frame.shape[:2]
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        box_width = float(bbox["w"])
        box_height = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if box_width <= 0 or box_height <= 0:
        return None
    x1 = max(0, int(round((x - padding * box_width) * width)))
    y1 = max(0, int(round((y - padding * box_height) * height)))
    x2 = min(
        width,
        int(round((x + box_width * (1 + padding)) * width)),
    )
    y2 = min(
        height,
        int(round((y + box_height * (1 + padding)) * height)),
    )
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2].copy()


def evaluate_crop_quality(crop: np.ndarray) -> CropQuality:
    if crop is None or crop.ndim != 3:
        return CropQuality(0.0, 0, 0, 0.0, 0.0)
    height, width = crop.shape[:2]
    if width < 8 or height < 16:
        return CropQuality(0.0, width, height, 0.0, 0.0)
    resized = cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    saturation = float(hsv[:, :, 1].mean()) / 255.0
    size_score = min(1.0, (width * height) / float(32 * 64))
    sharpness_score = min(1.0, sharpness / 180.0)
    saturation_score = (
        min(1.0, saturation / 0.35) if saturation > 0 else 0.0
    )
    aspect_ratio = width / float(max(1, height))
    aspect_score = max(
        0.0,
        1.0 - abs(aspect_ratio - 0.42) / 0.42,
    )
    score = (
        size_score * 0.35
        + sharpness_score * 0.25
        + saturation_score * 0.20
        + aspect_score * 0.20
    )
    return CropQuality(
        score=float(max(0.0, min(1.0, score))),
        width=width,
        height=height,
        sharpness=sharpness,
        saturation=saturation,
    )


def extract_appearance_descriptor(
    crop: np.ndarray,
) -> AppearanceDescriptor | None:
    quality = evaluate_crop_quality(crop)
    if quality.score <= 0.0:
        return None
    normalized = cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    regions = [
        hsv[8:34, :, :],
        hsv[30:58, :, :],
    ]
    features: list[np.ndarray] = []
    for region in regions:
        features.extend(
            [
                _normalized_histogram(region[:, :, 0], 12, (0, 180)),
                _normalized_histogram(region[:, :, 1], 6, (0, 256)),
                _normalized_histogram(region[:, :, 2], 6, (0, 256)),
            ]
        )
        pixels = region.reshape(-1, 3)
        means = pixels.mean(axis=0) / np.array([180.0, 255.0, 255.0])
        stds = pixels.std(axis=0) / np.array([180.0, 255.0, 255.0])
        features.append(means.astype(np.float32))
        features.append(stds.astype(np.float32))
    vector = np.concatenate(features).astype(np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    vector = vector / norm
    return AppearanceDescriptor(
        vector=tuple(float(value) for value in vector),
        sample_count=1,
        quality=quality.score,
        version=DESCRIPTOR_VERSION,
    )


def aggregate_appearance_descriptors(
    descriptors: Iterable[AppearanceDescriptor],
) -> AppearanceDescriptor | None:
    values = list(descriptors)
    if not values:
        return None
    version = values[0].version
    dimension = len(values[0].vector)
    for descriptor in values:
        if descriptor.version != version or len(descriptor.vector) != dimension:
            raise ValueError(
                "all appearance descriptors must use the same version and dimension"
            )
    weights = np.array(
        [
            max(0.05, descriptor.quality) * descriptor.sample_count
            for descriptor in values
        ],
        dtype=np.float64,
    )
    matrix = np.array(
        [descriptor.vector for descriptor in values],
        dtype=np.float64,
    )
    vector = np.average(matrix, axis=0, weights=weights)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return None
    vector = vector / vector_norm
    total_samples = sum(descriptor.sample_count for descriptor in values)
    quality = sum(
        descriptor.quality * descriptor.sample_count
        for descriptor in values
    ) / total_samples
    return AppearanceDescriptor(
        vector=tuple(float(value) for value in vector),
        sample_count=total_samples,
        quality=float(max(0.0, min(1.0, quality))),
        version=version,
    )
