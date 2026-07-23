#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.adapters import prediction_from_algonext_tracking
from app.benchmark.schema import SchemaValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an AlgoNext tracking.json artifact into the strict "
            "tracking-prediction-v1 benchmark format."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--evaluation-fps",
        type=float,
        help="Override the FPS used to derive benchmark frame indices.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tracking = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(tracking, dict):
            raise ValueError("tracking input must be a JSON object")
        prediction = prediction_from_algonext_tracking(
            tracking,
            video_id=args.video_id,
            evaluation_fps=args.evaluation_fps,
        )
    except (OSError, json.JSONDecodeError, ValueError, SchemaValidationError) as exc:
        print(f"Conversion error: {exc}", file=sys.stderr)
        return 2

    payload = {
        "schema_version": prediction.schema_version,
        "video_id": prediction.video_id,
        "frames": [
            {
                "frame_index": frame.frame_index,
                **({"time_sec": frame.time_sec} if frame.time_sec is not None else {}),
                "tracks": [
                    {
                        "track_id": track.track_id,
                        **(
                            {"confidence": track.confidence}
                            if track.confidence is not None
                            else {}
                        ),
                        "bbox": {
                            "x": track.bbox.x,
                            "y": track.bbox.y,
                            "w": track.bbox.w,
                            "h": track.bbox.h,
                        },
                    }
                    for track in frame.tracks
                ],
            }
            for frame in prediction.frames
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(prediction.frames)} scored frames to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
