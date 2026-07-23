from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from app.vision.pitch_geometry import (
    PitchEvidence,
    PitchGeometryThresholds,
    estimate_pitch_evidence,
)

SHOT_ANALYSIS_SCHEMA_VERSION = "camera-shot-analysis-v1"


@dataclass(frozen=True)
class ShotSegmentationThresholds:
    sample_fps: float = 2.0
    hard_cut_floor: float = 0.34
    adaptive_mad_multiplier: float = 6.0
    minimum_boundary_separation_sec: float = 0.60
    minimum_shot_duration_sec: float = 1.0
    calibration_minimum_duration_sec: float = 2.0
    short_insert_maximum_duration_sec: float = 2.5
    minimum_pitch_frame_ratio: float = 0.50
    maximum_samples: int = 30_000

    def __post_init__(self) -> None:
        for field_name in (
            "sample_fps",
            "hard_cut_floor",
            "adaptive_mad_multiplier",
            "minimum_boundary_separation_sec",
            "minimum_shot_duration_sec",
            "calibration_minimum_duration_sec",
            "short_insert_maximum_duration_sec",
            "minimum_pitch_frame_ratio",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be > 0")
        if not 0.0 <= self.hard_cut_floor <= 1.0:
            raise ValueError("hard_cut_floor must be in [0, 1]")
        if not 0.0 <= self.minimum_pitch_frame_ratio <= 1.0:
            raise ValueError("minimum_pitch_frame_ratio must be in [0, 1]")
        if self.calibration_minimum_duration_sec < self.minimum_shot_duration_sec:
            raise ValueError(
                "calibration_minimum_duration_sec must be >= minimum_shot_duration_sec"
            )
        if isinstance(self.maximum_samples, bool) or self.maximum_samples < 2:
            raise ValueError("maximum_samples must be an integer >= 2")


@dataclass(frozen=True)
class FrameSample:
    time_sec: float
    frame: np.ndarray

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_sec)) or self.time_sec < 0:
            raise ValueError("time_sec must be finite and >= 0")
        if not isinstance(self.frame, np.ndarray):
            raise TypeError("frame must be a numpy array")


@dataclass(frozen=True)
class FrameObservation:
    time_sec: float
    distance_from_previous: float
    brightness: float
    contrast: float
    pitch: PitchEvidence

    def to_payload(self) -> dict[str, Any]:
        return {
            "time_sec": round(self.time_sec, 6),
            "distance_from_previous": round(self.distance_from_previous, 6),
            "brightness": round(self.brightness, 6),
            "contrast": round(self.contrast, 6),
            "pitch": self.pitch.to_payload(),
        }


@dataclass(frozen=True)
class ShotBoundary:
    time_sec: float
    before_sample_index: int
    after_sample_index: int
    score: float
    threshold: float
    kind: str = "HARD_CUT"

    def to_payload(self) -> dict[str, Any]:
        return {
            "time_sec": round(self.time_sec, 6),
            "before_sample_index": self.before_sample_index,
            "after_sample_index": self.after_sample_index,
            "score": round(self.score, 6),
            "threshold": round(self.threshold, 6),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class CameraShot:
    shot_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    sample_start_index: int
    sample_end_index: int
    representative_sample_index: int
    representative_time_sec: float
    classification: str
    pitch_probability: float
    pitch_frame_ratio: float
    short_insert: bool
    calibration_candidate: bool
    exclude_from_calibration: bool
    reason_codes: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        for key in (
            "start_sec",
            "end_sec",
            "duration_sec",
            "representative_time_sec",
            "pitch_probability",
            "pitch_frame_ratio",
        ):
            payload[key] = round(float(payload[key]), 6)
        return payload


@dataclass(frozen=True)
class ShotAnalysis:
    samples: tuple[FrameObservation, ...]
    boundaries: tuple[ShotBoundary, ...]
    shots: tuple[CameraShot, ...]
    adaptive_threshold: float
    sampled_fps: float
    source_duration_sec: float | None = None
    schema_version: str = SHOT_ANALYSIS_SCHEMA_VERSION

    def to_payload(self, *, include_samples: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "sampled_fps": round(self.sampled_fps, 6),
            "source_duration_sec": (
                round(self.source_duration_sec, 6)
                if self.source_duration_sec is not None
                else None
            ),
            "adaptive_threshold": round(self.adaptive_threshold, 6),
            "sample_count": len(self.samples),
            "boundaries": [boundary.to_payload() for boundary in self.boundaries],
            "shots": [shot.to_payload() for shot in self.shots],
        }
        if include_samples:
            payload["samples"] = [sample.to_payload() for sample in self.samples]
        return payload


@dataclass(frozen=True)
class _FrameSignature:
    histogram: np.ndarray
    grayscale: np.ndarray
    edges: np.ndarray
    mean_bgr: np.ndarray


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _validate_bgr_frame(frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError("frame must use uint8 BGR pixels")
    if frame.shape[0] < 16 or frame.shape[1] < 16:
        raise ValueError("frame is too small")


def _signature(frame: np.ndarray) -> _FrameSignature:
    _validate_bgr_frame(frame)
    thumbnail = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    grayscale = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 140, apertureSize=3, L2gradient=True)
    mean_bgr = thumbnail.reshape(-1, 3).mean(axis=0) / 255.0
    return _FrameSignature(
        histogram=histogram.astype(np.float32),
        grayscale=grayscale,
        edges=edges,
        mean_bgr=mean_bgr.astype(np.float32),
    )


def frame_distance(first: np.ndarray, second: np.ndarray) -> float:
    return signature_distance(_signature(first), _signature(second))


def signature_distance(first: _FrameSignature, second: _FrameSignature) -> float:
    histogram_distance = float(
        cv2.compareHist(first.histogram, second.histogram, cv2.HISTCMP_BHATTACHARYYA)
    )
    grayscale_distance = float(
        np.mean(
            np.abs(
                first.grayscale.astype(np.float32)
                - second.grayscale.astype(np.float32)
            )
        )
        / 255.0
    )
    first_edges = first.edges > 0
    second_edges = second.edges > 0
    union = int(np.count_nonzero(first_edges | second_edges))
    edge_distance = (
        float(np.count_nonzero(first_edges ^ second_edges)) / float(union)
        if union > 0
        else 0.0
    )
    mean_colour_distance = float(
        np.linalg.norm(first.mean_bgr - second.mean_bgr) / math.sqrt(3.0)
    )
    return _clamp(
        0.45 * histogram_distance
        + 0.30 * grayscale_distance
        + 0.15 * edge_distance
        + 0.10 * mean_colour_distance
    )


def _adaptive_threshold(
    distances: Sequence[float],
    thresholds: ShotSegmentationThresholds,
) -> float:
    values = np.asarray(
        [value for value in distances[1:] if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if values.size == 0:
        return thresholds.hard_cut_floor
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    adaptive = median + thresholds.adaptive_mad_multiplier * robust_sigma
    return _clamp(max(thresholds.hard_cut_floor, adaptive), 0.0, 0.95)


def _local_maximum(distances: Sequence[float], index: int) -> bool:
    value = distances[index]
    previous = distances[index - 1] if index > 1 else -1.0
    following = distances[index + 1] if index + 1 < len(distances) else -1.0
    return value >= previous and value >= following


def _detect_boundaries(
    samples: Sequence[FrameSample],
    distances: Sequence[float],
    threshold: float,
    thresholds: ShotSegmentationThresholds,
) -> tuple[ShotBoundary, ...]:
    boundaries: list[ShotBoundary] = []
    last_boundary_time = -float("inf")
    for index in range(1, len(samples)):
        score = distances[index]
        if score < threshold or not _local_maximum(distances, index):
            continue
        boundary_time = (samples[index - 1].time_sec + samples[index].time_sec) / 2.0
        if boundary_time - last_boundary_time < thresholds.minimum_boundary_separation_sec:
            if boundaries and score > boundaries[-1].score:
                boundaries[-1] = ShotBoundary(
                    time_sec=boundary_time,
                    before_sample_index=index - 1,
                    after_sample_index=index,
                    score=score,
                    threshold=threshold,
                )
                last_boundary_time = boundary_time
            continue
        boundaries.append(
            ShotBoundary(
                time_sec=boundary_time,
                before_sample_index=index - 1,
                after_sample_index=index,
                score=score,
                threshold=threshold,
            )
        )
        last_boundary_time = boundary_time
    return tuple(boundaries)


def _sample_interval(samples: Sequence[FrameSample], sampled_fps: float) -> float:
    if len(samples) >= 2:
        deltas = [
            current.time_sec - previous.time_sec
            for previous, current in zip(samples, samples[1:])
            if current.time_sec > previous.time_sec
        ]
        if deltas:
            return float(np.median(np.asarray(deltas, dtype=np.float64)))
    return 1.0 / max(sampled_fps, 1e-6)


def _build_shots(
    frame_samples: Sequence[FrameSample],
    observations: Sequence[FrameObservation],
    boundaries: Sequence[ShotBoundary],
    thresholds: ShotSegmentationThresholds,
    source_duration_sec: float | None,
) -> tuple[CameraShot, ...]:
    start_indices = [0, *[boundary.after_sample_index for boundary in boundaries]]
    end_indices = [
        *[boundary.before_sample_index for boundary in boundaries],
        len(frame_samples) - 1,
    ]
    interval = _sample_interval(frame_samples, thresholds.sample_fps)
    shots: list[CameraShot] = []
    for shot_number, (start_index, end_index) in enumerate(
        zip(start_indices, end_indices),
        start=1,
    ):
        if end_index < start_index:
            continue
        shot_observations = observations[start_index : end_index + 1]
        start_sec = (
            boundaries[shot_number - 2].time_sec
            if shot_number > 1
            else frame_samples[0].time_sec
        )
        if shot_number <= len(boundaries):
            end_sec = boundaries[shot_number - 1].time_sec
        else:
            inferred_end = frame_samples[end_index].time_sec + interval
            end_sec = (
                min(inferred_end, source_duration_sec)
                if source_duration_sec is not None
                else inferred_end
            )
        end_sec = max(start_sec, end_sec)
        duration = end_sec - start_sec
        pitch_probabilities = [
            item.pitch.pitch_probability for item in shot_observations
        ]
        mean_pitch_probability = (
            float(np.mean(pitch_probabilities)) if pitch_probabilities else 0.0
        )
        pitch_frames = sum(
            item.pitch.classification == "PITCH_CANDIDATE"
            for item in shot_observations
        )
        non_pitch_frames = sum(
            item.pitch.classification == "NON_PITCH" for item in shot_observations
        )
        pitch_frame_ratio = pitch_frames / float(max(1, len(shot_observations)))
        if (
            pitch_frame_ratio >= thresholds.minimum_pitch_frame_ratio
            and mean_pitch_probability >= 0.50
        ):
            classification = "PITCH_CANDIDATE"
        elif non_pitch_frames > len(shot_observations) / 2.0:
            classification = "NON_PITCH"
        else:
            classification = "UNKNOWN"
        representative_relative = max(
            range(len(shot_observations)),
            key=lambda index: (
                shot_observations[index].pitch.pitch_probability,
                shot_observations[index].pitch.line_count,
                -abs(index - (len(shot_observations) - 1) / 2.0),
            ),
        )
        representative_index = start_index + representative_relative
        short_insert = duration < thresholds.short_insert_maximum_duration_sec
        reasons: list[str] = []
        if classification != "PITCH_CANDIDATE":
            reasons.append("SHOT_NOT_CONFIDENT_PITCH_VIEW")
        if duration < thresholds.calibration_minimum_duration_sec:
            reasons.append("SHOT_TOO_SHORT_FOR_CALIBRATION")
        if len(shot_observations) < 2:
            reasons.append("INSUFFICIENT_SHOT_SAMPLES")
        calibration_candidate = not reasons
        shots.append(
            CameraShot(
                shot_id=f"shot-{shot_number:04d}",
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=duration,
                sample_start_index=start_index,
                sample_end_index=end_index,
                representative_sample_index=representative_index,
                representative_time_sec=frame_samples[representative_index].time_sec,
                classification=classification,
                pitch_probability=mean_pitch_probability,
                pitch_frame_ratio=pitch_frame_ratio,
                short_insert=short_insert,
                calibration_candidate=calibration_candidate,
                exclude_from_calibration=not calibration_candidate,
                reason_codes=tuple(reasons),
            )
        )
    return tuple(shots)


def analyze_frame_sequence(
    samples: Iterable[FrameSample | tuple[float, np.ndarray]],
    *,
    thresholds: ShotSegmentationThresholds | None = None,
    pitch_thresholds: PitchGeometryThresholds | None = None,
    source_duration_sec: float | None = None,
) -> ShotAnalysis:
    thresholds = thresholds or ShotSegmentationThresholds()
    pitch_thresholds = pitch_thresholds or PitchGeometryThresholds()
    parsed: list[FrameSample] = []
    for item in samples:
        sample = (
            item
            if isinstance(item, FrameSample)
            else FrameSample(float(item[0]), item[1])
        )
        _validate_bgr_frame(sample.frame)
        parsed.append(sample)
    if not parsed:
        raise ValueError("at least one frame sample is required")
    parsed.sort(key=lambda item: item.time_sec)
    if any(
        current.time_sec <= previous.time_sec
        for previous, current in zip(parsed, parsed[1:])
    ):
        raise ValueError("frame sample timestamps must be strictly increasing")
    if source_duration_sec is not None:
        if not math.isfinite(float(source_duration_sec)) or source_duration_sec <= 0:
            raise ValueError("source_duration_sec must be finite and > 0")
        if source_duration_sec < parsed[-1].time_sec:
            raise ValueError("source_duration_sec cannot precede the final sample")

    signatures = [_signature(sample.frame) for sample in parsed]
    distances = [0.0]
    for previous, current in zip(signatures, signatures[1:]):
        distances.append(signature_distance(previous, current))
    adaptive_threshold = _adaptive_threshold(distances, thresholds)

    observations: list[FrameObservation] = []
    for sample, distance in zip(parsed, distances):
        gray = cv2.cvtColor(sample.frame, cv2.COLOR_BGR2GRAY)
        evidence = estimate_pitch_evidence(sample.frame, pitch_thresholds)
        observations.append(
            FrameObservation(
                time_sec=sample.time_sec,
                distance_from_previous=distance,
                brightness=float(gray.mean()) / 255.0,
                contrast=float(gray.std()) / 255.0,
                pitch=evidence,
            )
        )
    boundaries = _detect_boundaries(
        parsed,
        distances,
        adaptive_threshold,
        thresholds,
    )
    shots = _build_shots(
        parsed,
        observations,
        boundaries,
        thresholds,
        source_duration_sec,
    )
    return ShotAnalysis(
        samples=tuple(observations),
        boundaries=boundaries,
        shots=shots,
        adaptive_threshold=adaptive_threshold,
        sampled_fps=thresholds.sample_fps,
        source_duration_sec=source_duration_sec,
    )


def sample_video_frames(
    video_path: str | Path,
    *,
    thresholds: ShotSegmentationThresholds | None = None,
) -> tuple[list[FrameSample], float]:
    thresholds = thresholds or ShotSegmentationThresholds()
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / source_fps if frame_count > 0 else 0.0
        interval = max(1, int(round(source_fps / thresholds.sample_fps)))
        samples: list[FrameSample] = []
        frame_index = 0
        while len(samples) < thresholds.maximum_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                samples.append(
                    FrameSample(frame_index / source_fps, frame.copy())
                )
            frame_index += 1
        if not samples:
            raise RuntimeError("video contains no readable frames")
        if duration <= 0:
            duration = samples[-1].time_sec + 1.0 / thresholds.sample_fps
        return samples, duration
    finally:
        capture.release()


def analyze_video_shots(
    video_path: str | Path,
    *,
    thresholds: ShotSegmentationThresholds | None = None,
    pitch_thresholds: PitchGeometryThresholds | None = None,
) -> ShotAnalysis:
    thresholds = thresholds or ShotSegmentationThresholds()
    samples, duration = sample_video_frames(video_path, thresholds=thresholds)
    return analyze_frame_sequence(
        samples,
        thresholds=thresholds,
        pitch_thresholds=pitch_thresholds,
        source_duration_sec=duration,
    )
