from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.vision.pitch_geometry import (
    PitchGeometryProposal,
    PitchGeometryThresholds,
    detect_pitch_geometry,
)
from app.vision.shot_segmentation import (
    CameraShot,
    FrameSample,
    ShotAnalysis,
    ShotSegmentationThresholds,
    analyze_frame_sequence,
    sample_video_frames,
)

CAMERA_ANALYSIS_SCHEMA_VERSION = "camera-analysis-v1"


@dataclass(frozen=True)
class CameraAnalysisThresholds:
    geometry_frames_per_shot: int = 3
    minimum_candidate_geometry_frames: int = 1
    minimum_mean_pitch_probability: float = 0.55

    def __post_init__(self) -> None:
        for field_name in (
            "geometry_frames_per_shot",
            "minimum_candidate_geometry_frames",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.minimum_candidate_geometry_frames > self.geometry_frames_per_shot:
            raise ValueError(
                "minimum_candidate_geometry_frames cannot exceed geometry_frames_per_shot"
            )
        if (
            not math.isfinite(float(self.minimum_mean_pitch_probability))
            or not 0.0 <= self.minimum_mean_pitch_probability <= 1.0
        ):
            raise ValueError("minimum_mean_pitch_probability must be in [0, 1]")


@dataclass(frozen=True)
class GeometryFrameAnalysis:
    sample_index: int
    time_sec: float
    proposal: PitchGeometryProposal

    def to_payload(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "time_sec": round(self.time_sec, 6),
            "proposal": self.proposal.to_payload(),
        }


@dataclass(frozen=True)
class CameraSegmentAnalysis:
    shot: CameraShot
    status: str
    geometry_frames: tuple[GeometryFrameAnalysis, ...]
    geometry_candidate_count: int
    automatic_calibration_available: bool
    reason_codes: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "shot": self.shot.to_payload(),
            "status": self.status,
            "geometry_candidate_count": self.geometry_candidate_count,
            "geometry_frames": [frame.to_payload() for frame in self.geometry_frames],
            "automatic_calibration_available": self.automatic_calibration_available,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class CameraAnalysisResult:
    shot_analysis: ShotAnalysis
    segments: tuple[CameraSegmentAnalysis, ...]
    automatic_calibration_available: bool
    reason_codes: tuple[str, ...]
    thresholds: CameraAnalysisThresholds
    schema_version: str = CAMERA_ANALYSIS_SCHEMA_VERSION

    def to_payload(self, *, include_samples: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shot_analysis": self.shot_analysis.to_payload(
                include_samples=include_samples
            ),
            "segments": [segment.to_payload() for segment in self.segments],
            "automatic_calibration_available": self.automatic_calibration_available,
            "reason_codes": list(self.reason_codes),
            "thresholds": asdict(self.thresholds),
            "summary": {
                "shot_count": len(self.segments),
                "pitch_candidate_shots": sum(
                    segment.shot.classification == "PITCH_CANDIDATE"
                    for segment in self.segments
                ),
                "geometry_candidate_shots": sum(
                    segment.status == "GEOMETRY_CANDIDATE"
                    for segment in self.segments
                ),
                "excluded_shots": sum(
                    segment.status == "EXCLUDED" for segment in self.segments
                ),
            },
        }


def _parse_samples(
    samples: Iterable[FrameSample | tuple[float, np.ndarray]],
) -> list[FrameSample]:
    parsed = [
        item if isinstance(item, FrameSample) else FrameSample(float(item[0]), item[1])
        for item in samples
    ]
    if not parsed:
        raise ValueError("at least one frame sample is required")
    parsed.sort(key=lambda item: item.time_sec)
    return parsed


def _representative_indices(
    shot: CameraShot,
    count: int,
) -> list[int]:
    start = shot.sample_start_index
    end = shot.sample_end_index
    available = end - start + 1
    if available <= count:
        return list(range(start, end + 1))
    positions = np.linspace(start, end, count)
    indices = [int(round(value)) for value in positions]
    indices.append(shot.representative_sample_index)
    unique: list[int] = []
    for index in indices:
        index = max(start, min(end, index))
        if index not in unique:
            unique.append(index)
    unique.sort(
        key=lambda index: (
            index != shot.representative_sample_index,
            abs(index - shot.representative_sample_index),
        )
    )
    return unique[:count]


def analyze_camera_sequence(
    samples: Iterable[FrameSample | tuple[float, np.ndarray]],
    *,
    shot_thresholds: ShotSegmentationThresholds | None = None,
    geometry_thresholds: PitchGeometryThresholds | None = None,
    analysis_thresholds: CameraAnalysisThresholds | None = None,
    source_duration_sec: float | None = None,
) -> CameraAnalysisResult:
    shot_thresholds = shot_thresholds or ShotSegmentationThresholds()
    geometry_thresholds = geometry_thresholds or PitchGeometryThresholds()
    analysis_thresholds = analysis_thresholds or CameraAnalysisThresholds()
    parsed = _parse_samples(samples)
    shot_analysis = analyze_frame_sequence(
        parsed,
        thresholds=shot_thresholds,
        pitch_thresholds=geometry_thresholds,
        source_duration_sec=source_duration_sec,
    )

    segments: list[CameraSegmentAnalysis] = []
    for shot in shot_analysis.shots:
        if shot.exclude_from_calibration:
            segments.append(
                CameraSegmentAnalysis(
                    shot=shot,
                    status="EXCLUDED",
                    geometry_frames=(),
                    geometry_candidate_count=0,
                    automatic_calibration_available=False,
                    reason_codes=shot.reason_codes,
                )
            )
            continue
        frame_analyses: list[GeometryFrameAnalysis] = []
        for sample_index in _representative_indices(
            shot,
            analysis_thresholds.geometry_frames_per_shot,
        ):
            sample = parsed[sample_index]
            proposal = detect_pitch_geometry(sample.frame, geometry_thresholds)
            frame_analyses.append(
                GeometryFrameAnalysis(
                    sample_index=sample_index,
                    time_sec=sample.time_sec,
                    proposal=proposal,
                )
            )
        candidate_count = sum(
            item.proposal.status == "CANDIDATE" for item in frame_analyses
        )
        reasons: list[str] = []
        if candidate_count < analysis_thresholds.minimum_candidate_geometry_frames:
            reasons.append("INSUFFICIENT_GEOMETRY_CANDIDATE_FRAMES")
        if shot.pitch_probability < analysis_thresholds.minimum_mean_pitch_probability:
            reasons.append("LOW_MEAN_PITCH_PROBABILITY")
        # The current baseline proposes unlabeled line intersections only.
        reasons.append("SEMANTIC_PITCH_KEYPOINT_MODEL_REQUIRED")
        status = (
            "GEOMETRY_CANDIDATE"
            if candidate_count >= analysis_thresholds.minimum_candidate_geometry_frames
            and shot.pitch_probability
            >= analysis_thresholds.minimum_mean_pitch_probability
            else "INSUFFICIENT"
        )
        segments.append(
            CameraSegmentAnalysis(
                shot=shot,
                status=status,
                geometry_frames=tuple(frame_analyses),
                geometry_candidate_count=candidate_count,
                automatic_calibration_available=False,
                reason_codes=tuple(dict.fromkeys(reasons)),
            )
        )

    reasons = ["SEMANTIC_PITCH_KEYPOINT_MODEL_REQUIRED"]
    if not any(segment.status == "GEOMETRY_CANDIDATE" for segment in segments):
        reasons.append("NO_GEOMETRY_CANDIDATE_SHOTS")
    return CameraAnalysisResult(
        shot_analysis=shot_analysis,
        segments=tuple(segments),
        automatic_calibration_available=False,
        reason_codes=tuple(reasons),
        thresholds=analysis_thresholds,
    )


def analyze_camera_video(
    video_path: str | Path,
    *,
    shot_thresholds: ShotSegmentationThresholds | None = None,
    geometry_thresholds: PitchGeometryThresholds | None = None,
    analysis_thresholds: CameraAnalysisThresholds | None = None,
) -> CameraAnalysisResult:
    shot_thresholds = shot_thresholds or ShotSegmentationThresholds()
    samples, duration = sample_video_frames(video_path, thresholds=shot_thresholds)
    return analyze_camera_sequence(
        samples,
        shot_thresholds=shot_thresholds,
        geometry_thresholds=geometry_thresholds,
        analysis_thresholds=analysis_thresholds,
        source_duration_sec=duration,
    )
