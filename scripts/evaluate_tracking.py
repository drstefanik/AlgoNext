#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, TypeVar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.schema import (
    SchemaValidationError,
    SequenceAnnotation,
    SequencePrediction,
    load_annotation,
    load_prediction,
)
from app.benchmark.tracking_metrics import (
    GateThresholds,
    evaluate_dataset,
    evaluate_quality_gate,
)


SequenceType = TypeVar("SequenceType", SequenceAnnotation, SequencePrediction)


def _load_sequences(
    path: Path,
    loader: Callable[[Path], SequenceType],
) -> dict[str, SequenceType]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise SchemaValidationError(str(path), "no JSON files found")

    sequences: dict[str, SequenceType] = {}
    for file_path in files:
        sequence = loader(file_path)
        if sequence.video_id in sequences:
            raise SchemaValidationError(
                str(path),
                f"video_id {sequence.video_id!r} appears in more than one file",
            )
        sequences[sequence.video_id] = sequence
    return sequences


def _format_percent(value: float) -> str:
    return f"{value * 100.0:6.2f}%"


def _human_report(report: dict, gate: dict) -> str:
    metrics = report["aggregate"]["metrics"]
    counts = report["aggregate"]["counts"]
    rows = [
        ("Sequences", str(report["sequence_count"])),
        ("Annotated frames", str(counts["annotation_frames"])),
        ("GT detections", str(counts["ground_truth_detections"])),
        ("Predicted detections", str(counts["prediction_detections"])),
        ("Detection precision", _format_percent(metrics["detection_precision"])),
        ("Detection recall", _format_percent(metrics["detection_recall"])),
        ("Detection F1", _format_percent(metrics["detection_f1"])),
        ("Mean matched IoU", _format_percent(metrics["mean_matched_iou"])),
        ("Detection accuracy", _format_percent(metrics["detection_accuracy"])),
        ("Association accuracy", _format_percent(metrics["association_accuracy"])),
        ("HOTA-style @ IoU", _format_percent(metrics["hota_style_at_threshold"])),
        ("Track coverage", _format_percent(metrics["track_coverage"])),
        ("ID precision", _format_percent(metrics["identity_precision"])),
        ("ID recall", _format_percent(metrics["identity_recall"])),
        ("IDF1", _format_percent(metrics["idf1"])),
        (
            "ID switches / 100 matches",
            f"{metrics['id_switches_per_100_matches']:.2f}",
        ),
        ("Fragmentations", str(counts["fragmentations"])),
        ("Mostly tracked identities", _format_percent(metrics["mostly_tracked_ratio"])),
        ("Mostly lost identities", _format_percent(metrics["mostly_lost_ratio"])),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [
        "AlgoNext tracking benchmark",
        "=" * 31,
        *[f"{label:<{width}}  {value}" for label, value in rows],
        "",
        f"Quality gate: {'PASS' if gate['passed'] else 'FAIL'}",
    ]
    for check in gate["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"  [{marker}] {check['metric']} "
            f"{check['comparator']} {check['threshold']} "
            f"(actual {check['actual']})"
        )
    lines.extend(["", gate["note"]])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate player tracking predictions against frame-level identity "
            "annotations. The output measures tracking quality only."
        )
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Annotation JSON file or directory of JSON files.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Prediction JSON file or directory of JSON files.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="Minimum normalized bounding-box IoU for a frame match (default: 0.50).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable JSON report.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write the complete JSON report to this path.",
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit with status 1 when any engineering quality gate fails.",
    )
    parser.add_argument("--detection-f1-min", type=float, default=0.75)
    parser.add_argument("--idf1-min", type=float, default=0.65)
    parser.add_argument("--track-coverage-min", type=float, default=0.60)
    parser.add_argument(
        "--id-switches-per-100-max",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--hota-style-min",
        type=float,
        default=0.55,
        help=(
            "Minimum transparent HOTA-style score at the configured IoU. "
            "This is not an official TrackEval HOTA result."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        annotations = _load_sequences(args.annotations, load_annotation)
        predictions = _load_sequences(args.predictions, load_prediction)
        missing_predictions = sorted(set(annotations) - set(predictions))
        extra_predictions = sorted(set(predictions) - set(annotations))
        if missing_predictions:
            raise SchemaValidationError(
                str(args.predictions),
                "missing predictions for video_id: " + ", ".join(missing_predictions),
            )
        if extra_predictions:
            raise SchemaValidationError(
                str(args.predictions),
                "predictions have no matching annotation for video_id: "
                + ", ".join(extra_predictions),
            )

        report = evaluate_dataset(
            [
                (annotations[video_id], predictions[video_id])
                for video_id in sorted(annotations)
            ],
            iou_threshold=args.iou_threshold,
        )
        gate = evaluate_quality_gate(
            report,
            GateThresholds(
                detection_f1_min=args.detection_f1_min,
                idf1_min=args.idf1_min,
                track_coverage_min=args.track_coverage_min,
                id_switches_per_100_max=args.id_switches_per_100_max,
                hota_style_min=args.hota_style_min,
            ),
        )
        report["quality_gate"] = gate
    except (SchemaValidationError, ValueError) as exc:
        print(f"Benchmark input error: {exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        print(_human_report(report, gate))

    return 1 if args.fail_on_gate and not gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
