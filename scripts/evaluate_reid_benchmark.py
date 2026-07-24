#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.reid_metrics import ReIDGateThresholds
from app.benchmark.reid_schema import ReIDSchemaValidationError, load_reid_annotation
from app.benchmark.reid_suite import evaluate_reid_benchmark_suite
from app.benchmark.schema import SchemaValidationError, load_annotation
from app.benchmark.tracking_metrics import GateThresholds


def _pct(value: float) -> str:
    return f"{value * 100.0:6.2f}%"


def _human_report(report: dict) -> str:
    frame = report["frame_tracking"]["aggregate"]
    window = report["window_reid"]
    frame_metrics = frame["metrics"]
    window_metrics = window["metrics"]
    window_counts = window["counts"]
    rows = [
        ("Frame annotation count", str(frame["counts"]["annotation_frames"])),
        ("Detection F1", _pct(frame_metrics["detection_f1"])),
        ("IDF1", _pct(frame_metrics["idf1"])),
        ("Track coverage", _pct(frame_metrics["track_coverage"])),
        ("ID switches / 100", f"{frame_metrics['id_switches_per_100_matches']:.2f}"),
        ("HOTA-style", _pct(frame_metrics["hota_style_at_threshold"])),
        ("Scorable windows", str(window_counts["scorable_windows"])),
        ("Accepted precision", _pct(window_metrics["accepted_precision"])),
        ("False-link rate", _pct(window_metrics["false_link_rate"])),
        (
            "Association recall | candidate",
            _pct(window_metrics["association_recall_given_candidate"]),
        ),
        ("Visible-window recall", _pct(window_metrics["visible_window_recall"])),
        ("Candidate recall | visible", _pct(window_metrics["candidate_recall_visible"])),
        ("Processing failure rate", _pct(window_metrics["processing_failure_rate"])),
    ]
    width = max(len(label) for label, _ in rows)
    gate = report["quality_gate"]
    lines = [
        "AlgoNext ReID benchmark suite",
        "=" * 32,
        *[f"{label:<{width}}  {value}" for label, value in rows],
        "",
        f"Combined quality gate: {'PASS' if gate['passed'] else 'FAIL'}",
        f"  Frame tracking: {'PASS' if gate['components']['frame_tracking']['passed'] else 'FAIL'}",
        f"  Window ReID:    {'PASS' if gate['components']['window_reid']['passed'] else 'FAIL'}",
        "",
        gate["note"],
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete AlgoNext ReID benchmark: frame-level IDF1 plus "
            "human-reviewed window association decisions."
        )
    )
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument("--frame-annotations", required=True, type=Path)
    parser.add_argument("--window-annotations", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")

    parser.add_argument("--detection-f1-min", type=float, default=0.75)
    parser.add_argument("--idf1-min", type=float, default=0.65)
    parser.add_argument("--track-coverage-min", type=float, default=0.60)
    parser.add_argument("--id-switches-per-100-max", type=float, default=5.0)
    parser.add_argument("--hota-style-min", type=float, default=0.55)

    parser.add_argument("--minimum-scorable-windows", type=int, default=30)
    parser.add_argument("--accepted-judgement-coverage-min", type=float, default=0.90)
    parser.add_argument("--accepted-precision-min", type=float, default=0.95)
    parser.add_argument("--false-link-rate-max", type=float, default=0.05)
    parser.add_argument(
        "--association-recall-given-candidate-min", type=float, default=0.60
    )
    parser.add_argument("--visible-window-recall-min", type=float, default=0.45)
    parser.add_argument("--candidate-annotation-coverage-min", type=float, default=0.70)
    parser.add_argument("--candidate-recall-visible-min", type=float, default=0.70)
    parser.add_argument("--processing-failure-rate-max", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tracking = json.loads(args.tracking.read_text(encoding="utf-8"))
        if not isinstance(tracking, dict):
            raise ValueError("tracking input must be a JSON object")
        frame_annotation = load_annotation(args.frame_annotations)
        window_annotation = load_reid_annotation(args.window_annotations)
        report = evaluate_reid_benchmark_suite(
            tracking,
            frame_annotation=frame_annotation,
            window_annotation=window_annotation,
            iou_threshold=args.iou_threshold,
            tracking_thresholds=GateThresholds(
                detection_f1_min=args.detection_f1_min,
                idf1_min=args.idf1_min,
                track_coverage_min=args.track_coverage_min,
                id_switches_per_100_max=args.id_switches_per_100_max,
                hota_style_min=args.hota_style_min,
            ),
            reid_thresholds=ReIDGateThresholds(
                minimum_scorable_windows=args.minimum_scorable_windows,
                accepted_judgement_coverage_min=(
                    args.accepted_judgement_coverage_min
                ),
                accepted_precision_min=args.accepted_precision_min,
                false_link_rate_max=args.false_link_rate_max,
                association_recall_given_candidate_min=(
                    args.association_recall_given_candidate_min
                ),
                visible_window_recall_min=args.visible_window_recall_min,
                candidate_annotation_coverage_min=(
                    args.candidate_annotation_coverage_min
                ),
                candidate_recall_visible_min=args.candidate_recall_visible_min,
                processing_failure_rate_max=args.processing_failure_rate_max,
            ),
        )
    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        SchemaValidationError,
        ReIDSchemaValidationError,
    ) as exc:
        print(f"ReID suite input error: {exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded if args.json else _human_report(report))
    return 1 if args.fail_on_gate and not report["quality_gate"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
