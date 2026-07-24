#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.reid_adapters import prediction_from_algonext_reid
from app.benchmark.reid_metrics import (
    ReIDGateThresholds,
    evaluate_reid_quality_gate,
    evaluate_reid_sequence,
)
from app.benchmark.reid_schema import (
    ReIDSchemaValidationError,
    load_reid_annotation,
    load_reid_prediction,
)


def _percent(value: float) -> str:
    return f"{value * 100.0:6.2f}%"


def _human_report(report: dict, gate: dict) -> str:
    counts = report["counts"]
    metrics = report["metrics"]
    rows = [
        ("Annotation windows", str(counts.get("annotation_windows", 0))),
        ("Scorable windows", str(counts.get("scorable_windows", 0))),
        ("Visible windows", str(counts.get("visible_windows", 0))),
        ("Candidate-present windows", str(counts.get("candidate_present_windows", 0))),
        ("Candidate-absent windows", str(counts.get("candidate_absent_windows", 0))),
        ("Candidate-unverifiable windows", str(counts.get("candidate_unverifiable_windows", 0))),
        ("True accepts", str(counts.get("true_accepts", 0))),
        ("False accepts / links", str(counts.get("false_accepts", 0))),
        ("Unjudged accepts", str(counts.get("accepted_unjudged_windows", 0))),
        ("Missed associations", str(counts.get("missed_associations", 0))),
        ("Accepted judgement coverage", _percent(metrics["accepted_judgement_coverage"])),
        ("Accepted precision", _percent(metrics["accepted_precision"])),
        ("False-link rate", _percent(metrics["false_link_rate"])),
        (
            "Association recall | candidate",
            _percent(metrics["association_recall_given_candidate"]),
        ),
        ("Visible-window recall", _percent(metrics["visible_window_recall"])),
        ("Candidate annotation coverage", _percent(metrics["candidate_annotation_coverage"])),
        ("Candidate recall | visible", _percent(metrics["candidate_recall_visible"])),
        ("Non-visible abstention", _percent(metrics["nonvisible_abstention_rate"])),
        ("Abstention rate", _percent(metrics["abstention_rate"])),
        ("Processing failure rate", _percent(metrics["processing_failure_rate"])),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [
        "AlgoNext ReID window benchmark",
        "=" * 34,
        *[f"{label:<{width}}  {value}" for label, value in rows],
        "",
        f"Quality gate: {'PASS' if gate['passed'] else 'FAIL'}",
    ]
    for check in gate["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"  [{marker}] {check['metric']} {check['comparator']} "
            f"{check['threshold']} (actual {check['actual']})"
        )
    lines.extend(["", gate["note"]])
    return "\n".join(lines)


def add_gate_arguments(parser: argparse.ArgumentParser) -> None:
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


def thresholds_from_args(args: argparse.Namespace) -> ReIDGateThresholds:
    return ReIDGateThresholds(
        minimum_scorable_windows=args.minimum_scorable_windows,
        accepted_judgement_coverage_min=args.accepted_judgement_coverage_min,
        accepted_precision_min=args.accepted_precision_min,
        false_link_rate_max=args.false_link_rate_max,
        association_recall_given_candidate_min=(
            args.association_recall_given_candidate_min
        ),
        visible_window_recall_min=args.visible_window_recall_min,
        candidate_annotation_coverage_min=args.candidate_annotation_coverage_min,
        candidate_recall_visible_min=args.candidate_recall_visible_min,
        processing_failure_rate_max=args.processing_failure_rate_max,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AlgoNext cross-window ReID decisions against human-reviewed "
            "window annotations."
        )
    )
    parser.add_argument("--annotations", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tracking", type=Path, help="Raw AlgoNext tracking.json")
    source.add_argument(
        "--predictions", type=Path, help="reid-window-prediction-v1 JSON"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    add_gate_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        annotation = load_reid_annotation(args.annotations)
        if args.predictions:
            prediction = load_reid_prediction(args.predictions)
        else:
            tracking = json.loads(args.tracking.read_text(encoding="utf-8"))
            if not isinstance(tracking, dict):
                raise ValueError("tracking input must be a JSON object")
            prediction = prediction_from_algonext_reid(
                tracking, video_id=annotation.video_id
            )
        report = evaluate_reid_sequence(annotation, prediction)
        gate = evaluate_reid_quality_gate(report, thresholds_from_args(args))
        report["quality_gate"] = gate
    except (OSError, json.JSONDecodeError, ValueError, ReIDSchemaValidationError) as exc:
        print(f"ReID benchmark input error: {exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded if args.json else _human_report(report, gate))
    return 1 if args.fail_on_gate and not gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
