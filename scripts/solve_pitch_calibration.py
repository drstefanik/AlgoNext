#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.calibration.homography import (
    CalibrationFitError,
    CalibrationThresholds,
    fit_pitch_calibration,
)
from app.calibration.schema import (
    CalibrationValidationError,
    load_calibration_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an image-normalized to pitch-metre homography and evaluate "
            "the pitch-calibration quality gate."
        )
    )
    parser.add_argument("--input", required=True, help="Calibration request JSON")
    parser.add_argument("--output", required=True, help="Result JSON path")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--minimum-correspondences", type=int, default=6)
    parser.add_argument("--ransac-threshold-m", type=float, default=1.5)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.75)
    parser.add_argument("--maximum-rmse-m", type=float, default=1.5)
    parser.add_argument("--maximum-p95-error-m", type=float, default=3.0)
    parser.add_argument("--minimum-image-coverage", type=float, default=0.02)
    parser.add_argument("--minimum-field-coverage", type=float, default=0.08)
    parser.add_argument("--maximum-condition-number", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = CalibrationThresholds(
        minimum_correspondences=args.minimum_correspondences,
        ransac_reprojection_threshold_m=args.ransac_threshold_m,
        minimum_inlier_ratio=args.minimum_inlier_ratio,
        maximum_rmse_m=args.maximum_rmse_m,
        maximum_p95_error_m=args.maximum_p95_error_m,
        minimum_image_hull_area_ratio=args.minimum_image_coverage,
        minimum_field_hull_area_ratio=args.minimum_field_coverage,
        maximum_condition_number=args.maximum_condition_number,
    )
    try:
        request = load_calibration_request(args.input)
        calibration = fit_pitch_calibration(request, thresholds=thresholds)
    except (CalibrationValidationError, CalibrationFitError, ValueError) as exc:
        print(f"pitch calibration failed: {exc}", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(calibration.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "pitch calibration "
        f"status={calibration.status} "
        f"points={calibration.inlier_count}/{calibration.total_correspondences} "
        f"rmse_m={calibration.rmse_m:.3f} "
        f"p95_m={calibration.p95_error_m:.3f} "
        f"image_coverage={calibration.image_hull_area_ratio:.3f} "
        f"field_coverage={calibration.field_hull_area_ratio:.3f}"
    )
    if calibration.reason_codes:
        print("reason_codes=" + ",".join(calibration.reason_codes))
    if args.fail_on_gate and not calibration.validated:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
