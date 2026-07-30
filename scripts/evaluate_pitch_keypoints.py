#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.benchmark.pitch_keypoints import (  # noqa: E402
    AnnotationDataset,
    EvaluationThresholds,
    PredictionDataset,
    QualityGateThresholds,
    evaluate_keypoint_dataset,
    evaluate_quality_gate,
    load_annotations,
    load_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate semantic football-pitch keypoints, localization, negative "
            "frames and calibration readiness."
        )
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.50)
    parser.add_argument("--match-radius", type=float, default=0.03)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--semantic-f1-min", type=float, default=0.75)
    parser.add_argument("--pck-02-min", type=float, default=0.65)
    parser.add_argument("--p95-error-max", type=float, default=0.035)
    parser.add_argument("--non-pitch-fpr-max", type=float, default=0.05)
    parser.add_argument("--calibration-validated-rate-min", type=float, default=0.70)
    parser.add_argument("--pitch-coverage-min", type=float, default=0.60)
    return parser.parse_args()


def _json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"path does not exist: {path}")
    files = sorted(item for item in path.glob("*.json") if item.is_file())
    if not files:
        raise ValueError(f"directory contains no JSON files: {path}")
    return files


def _load_dataset_pair(
    annotations_path: Path,
    predictions_path: Path,
) -> tuple[AnnotationDataset, PredictionDataset]:
    annotation_files = _json_files(annotations_path)
    prediction_files = _json_files(predictions_path)
    if len(annotation_files) == 1 and len(prediction_files) == 1:
        return load_annotations(annotation_files[0]), load_predictions(prediction_files[0])
    annotation_by_name = {path.name: path for path in annotation_files}
    prediction_by_name = {path.name: path for path in prediction_files}
    if set(annotation_by_name) != set(prediction_by_name):
        missing_predictions = sorted(set(annotation_by_name) - set(prediction_by_name))
        extra_predictions = sorted(set(prediction_by_name) - set(annotation_by_name))
        raise ValueError(
            "annotation/prediction filenames differ; "
            f"missing_predictions={missing_predictions}, extra_predictions={extra_predictions}"
        )
    annotation_frames = []
    prediction_frames = []
    for name in sorted(annotation_by_name):
        annotation_frames.extend(load_annotations(annotation_by_name[name]).frames)
        prediction_frames.extend(load_predictions(prediction_by_name[name]).frames)
    return (
        AnnotationDataset(frames=tuple(annotation_frames)),
        PredictionDataset(frames=tuple(prediction_frames)),
    )


def _assert_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_finite_json(payload)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    try:
        annotations, predictions = _load_dataset_pair(
            Path(args.annotations),
            Path(args.predictions),
        )
        report = evaluate_keypoint_dataset(
            annotations,
            predictions,
            thresholds=EvaluationThresholds(
                confidence_threshold=args.confidence_threshold,
                match_radius_normalized=args.match_radius,
            ),
        )
        gate = evaluate_quality_gate(
            report,
            QualityGateThresholds(
                semantic_f1_min=args.semantic_f1_min,
                pck_02_min=args.pck_02_min,
                p95_error_max=args.p95_error_max,
                non_pitch_false_positive_rate_max=args.non_pitch_fpr_max,
                calibration_validated_rate_min=args.calibration_validated_rate_min,
                pitch_frame_prediction_coverage_min=args.pitch_coverage_min,
            ),
        )
        payload = {**report, "quality_gate": gate}
        _write_json_atomic(Path(args.json_out), payload)
    except (OSError, TypeError, ValueError) as exc:
        print(f"pitch-keypoint evaluation failed: {exc}", file=sys.stderr)
        return 2

    metrics = report["metrics"]
    print(
        "pitch-keypoint benchmark "
        f"frames={report['counts']['frames']} "
        f"f1={metrics['semantic_f1']:.3f} "
        f"pck02={metrics['pck_02']:.3f} "
        f"calibration_validated={metrics['calibration_validated_rate']:.3f} "
        f"gate={'PASS' if gate['passed'] else 'FAIL'}"
    )
    if args.fail_on_gate and not gate["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
