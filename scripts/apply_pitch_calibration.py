#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.calibration.homography import PitchCalibration
from app.calibration.kinematics import MotionThresholds, calculate_calibrated_motion


def load_json(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {file_path} at line {exc.lineno}, column {exc.colno}"
        ) from exc


def load_calibrations(path: str | Path) -> list[PitchCalibration]:
    value = load_json(path)
    if isinstance(value, dict) and isinstance(value.get("calibrations"), list):
        items = value["calibrations"]
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    calibrations = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"calibration[{index}] must be an object")
        calibrations.append(PitchCalibration.from_payload(item))
    return calibrations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project tracked player footpoints through validated pitch "
            "homographies and compute observed motion diagnostics."
        )
    )
    parser.add_argument("--tracking", required=True, help="AlgoNext tracking JSON")
    parser.add_argument(
        "--calibration",
        required=True,
        help="One calibration result, an array, or {calibrations:[...]}",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-gap-sec", type=float, default=1.0)
    parser.add_argument("--maximum-speed-mps", type=float, default=12.5)
    parser.add_argument("--maximum-acceleration-mps2", type=float, default=12.0)
    parser.add_argument("--sprint-threshold-mps", type=float, default=7.0)
    parser.add_argument("--minimum-sprint-duration-sec", type=float, default=1.0)
    parser.add_argument("--minimum-points", type=int, default=10)
    parser.add_argument("--fail-if-unavailable", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        tracking = load_json(args.tracking)
        if not isinstance(tracking, dict):
            raise ValueError("tracking JSON must be an object")
        calibrations = load_calibrations(args.calibration)
        thresholds = MotionThresholds(
            maximum_gap_sec=args.maximum_gap_sec,
            maximum_speed_mps=args.maximum_speed_mps,
            maximum_acceleration_mps2=args.maximum_acceleration_mps2,
            sprint_threshold_mps=args.sprint_threshold_mps,
            minimum_sprint_duration_sec=args.minimum_sprint_duration_sec,
            minimum_projected_points=args.minimum_points,
        )
        result = calculate_calibrated_motion(
            tracking,
            calibrations,
            thresholds=thresholds,
        )
    except (ValueError, TypeError) as exc:
        print(f"calibrated motion failed: {exc}", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "calibrated motion "
        f"status={result.get('status')} "
        f"points={(result.get('quality') or {}).get('projected_points', 0)} "
        f"path_m={result.get('observed_path_length_m', 0)} "
        f"avg_kmh={result.get('average_observed_speed_kmh', 0)}"
    )
    if result.get("reason_codes"):
        print("reason_codes=" + ",".join(result["reason_codes"]))
    if args.fail_if_unavailable and result.get("status") == "UNAVAILABLE":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
