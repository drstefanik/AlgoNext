from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from app.benchmark.adapters import prediction_from_algonext_tracking
from app.benchmark.reid_adapters import prediction_from_algonext_reid
from app.benchmark.reid_metrics import (
    ReIDGateThresholds,
    evaluate_reid_quality_gate,
    evaluate_reid_sequence,
)
from app.benchmark.reid_schema import ReIDSequenceAnnotation
from app.benchmark.schema import SequenceAnnotation
from app.benchmark.tracking_metrics import (
    GateThresholds,
    evaluate_dataset,
    evaluate_quality_gate,
)


def evaluate_reid_benchmark_suite(
    tracking: Mapping[str, Any],
    *,
    frame_annotation: SequenceAnnotation,
    window_annotation: ReIDSequenceAnnotation,
    iou_threshold: float = 0.50,
    tracking_thresholds: GateThresholds | None = None,
    reid_thresholds: ReIDGateThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate frame-level ID metrics and window-level ReID decisions together."""

    if frame_annotation.video_id != window_annotation.video_id:
        raise ValueError(
            "annotation video_id mismatch: "
            f"{frame_annotation.video_id!r} != {window_annotation.video_id!r}"
        )
    video_id = frame_annotation.video_id
    frame_prediction = prediction_from_algonext_tracking(
        tracking,
        video_id=video_id,
        evaluation_fps=frame_annotation.fps,
    )
    frame_report = evaluate_dataset(
        [(frame_annotation, frame_prediction)],
        iou_threshold=iou_threshold,
    )
    tracking_thresholds = tracking_thresholds or GateThresholds()
    frame_gate = evaluate_quality_gate(frame_report, tracking_thresholds)
    frame_report["quality_gate"] = frame_gate

    window_prediction = prediction_from_algonext_reid(tracking, video_id=video_id)
    window_report = evaluate_reid_sequence(window_annotation, window_prediction)
    reid_thresholds = reid_thresholds or ReIDGateThresholds()
    window_gate = evaluate_reid_quality_gate(window_report, reid_thresholds)
    window_report["quality_gate"] = window_gate

    return {
        "schema_version": "reid-benchmark-suite-v1",
        "video_id": video_id,
        "identity": window_annotation.identity,
        "parameters": {
            "iou_threshold": iou_threshold,
            "evaluation_fps": frame_annotation.fps,
            "tracking_gate_thresholds": asdict(tracking_thresholds),
            "reid_gate_thresholds": asdict(reid_thresholds),
        },
        "frame_tracking": frame_report,
        "window_reid": window_report,
        "quality_gate": {
            "passed": bool(frame_gate["passed"] and window_gate["passed"]),
            "components": {
                "frame_tracking": frame_gate,
                "window_reid": window_gate,
            },
            "note": (
                "Both frame-level identity tracking and window-level ReID gates "
                "must pass. Passing does not validate football or physical scoring."
            ),
        },
    }
