#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.benchmark.reid_job import (
    discover_job_artifacts,
    download_file,
    fetch_json,
    unwrap_job_payload,
)
from app.benchmark.reid_review import materialize_reid_review_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a completed AlgoNext job's tracking/video artifacts and build "
            "the local ReID annotation pack."
        )
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--video-path",
        type=Path,
        help="Reuse a local copy of the input video instead of downloading it.",
    )
    parser.add_argument("--samples-per-window", type=int, default=3)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    endpoint = (
        args.api_base.rstrip("/")
        + "/jobs/"
        + args.job_id.strip()
    )
    try:
        envelope = fetch_json(endpoint, timeout_seconds=args.request_timeout)
        job = unwrap_job_payload(envelope)
        artifacts = discover_job_artifacts(job)
        (destination / "job-manifest.json").write_text(
            json.dumps(
                {
                    "job_id": artifacts["job_id"],
                    "status": job.get("status"),
                    "identity": artifacts["identity"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        tracking_path = download_file(
            artifacts["tracking_url"],
            destination / "tracking.json",
            timeout_seconds=args.request_timeout,
        )
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        if not isinstance(tracking, dict):
            raise ValueError("downloaded tracking artifact is not a JSON object")

        if args.video_path:
            video_path = args.video_path
            if not video_path.is_file():
                raise ValueError(f"local video file not found: {video_path}")
        else:
            video_path = download_file(
                artifacts["video_url"],
                destination / "input.mp4",
                timeout_seconds=args.request_timeout,
            )

        pack = materialize_reid_review_pack(
            tracking,
            video_path=video_path,
            output_dir=destination / "review-pack",
            video_id=artifacts["job_id"],
            identity=artifacts["identity"],
            samples_per_window=args.samples_per_window,
            ffmpeg_binary=args.ffmpeg_binary,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Job benchmark preparation error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Review pack ready: {pack['html_path']} ({pack['frames_extracted']} frames)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
