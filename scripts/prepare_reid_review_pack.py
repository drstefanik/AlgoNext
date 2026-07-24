#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.reid_review import materialize_reid_review_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract review frames and build a self-contained browser annotation "
            "pack for an AlgoNext ReID tracking artifact."
        )
    )
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-window", type=int, default=3)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tracking = json.loads(args.tracking.read_text(encoding="utf-8"))
        if not isinstance(tracking, dict):
            raise ValueError("tracking input must be a JSON object")
        result = materialize_reid_review_pack(
            tracking,
            video_path=args.video,
            output_dir=args.output_dir,
            video_id=args.video_id,
            identity=args.identity,
            samples_per_window=args.samples_per_window,
            ffmpeg_binary=args.ffmpeg_binary,
        )
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(f"Review-pack preparation error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote {result['frames_extracted']} review frames and {result['html_path']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
