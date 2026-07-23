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

from app.vision.camera_analysis import (  # noqa: E402
    CameraAnalysisThresholds,
    analyze_camera_video,
)
from app.vision.pitch_geometry import PitchGeometryThresholds  # noqa: E402
from app.vision.shot_segmentation import ShotSegmentationThresholds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment broadcast video into camera shots, exclude non-pitch views, "
            "and generate conservative pitch-line/keypoint proposals."
        )
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--hard-cut-floor", type=float, default=0.34)
    parser.add_argument("--geometry-frames-per-shot", type=int, default=3)
    parser.add_argument("--minimum-pitch-coverage", type=float, default=0.22)
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument(
        "--fail-if-no-geometry-candidate",
        action="store_true",
        help="Exit 1 when no shot contains sufficient unlabeled pitch geometry.",
    )
    return parser.parse_args()


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
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
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
        result = analyze_camera_video(
            args.video,
            shot_thresholds=ShotSegmentationThresholds(
                sample_fps=args.sample_fps,
                hard_cut_floor=args.hard_cut_floor,
            ),
            geometry_thresholds=PitchGeometryThresholds(
                minimum_pitch_coverage=args.minimum_pitch_coverage,
            ),
            analysis_thresholds=CameraAnalysisThresholds(
                geometry_frames_per_shot=args.geometry_frames_per_shot,
            ),
        )
        payload = result.to_payload(include_samples=args.include_samples)
        _write_json_atomic(Path(args.output), payload)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"camera analysis failed: {exc}", file=sys.stderr)
        return 2

    summary = payload["summary"]
    print(
        "camera analysis "
        f"shots={summary['shot_count']} "
        f"pitch_shots={summary['pitch_candidate_shots']} "
        f"geometry_shots={summary['geometry_candidate_shots']} "
        f"auto_calibration={payload['automatic_calibration_available']}"
    )
    if args.fail_if_no_geometry_candidate and summary["geometry_candidate_shots"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
