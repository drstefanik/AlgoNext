#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.reid_review import build_reid_annotation_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a human-review template for every window in an AlgoNext "
            "tracking.json artifact."
        )
    )
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-per-window", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tracking = json.loads(args.tracking.read_text(encoding="utf-8"))
        if not isinstance(tracking, dict):
            raise ValueError("tracking input must be a JSON object")
        template = build_reid_annotation_template(
            tracking,
            video_id=args.video_id,
            identity=args.identity,
            samples_per_window=args.samples_per_window,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Annotation preparation error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(template['windows'])} review windows to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
