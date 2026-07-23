import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

from app.vision.camera_analysis import analyze_camera_video
from tests.synthetic_pitch import make_non_pitch_frame, make_pitch_frame

ROOT = Path(__file__).resolve().parents[1]


def write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        4.0,
        (640, 360),
    )
    if not writer.isOpened():
        raise RuntimeError("MJPG video writer is unavailable")
    try:
        for _ in range(12):
            writer.write(make_pitch_frame())
        for _ in range(8):
            writer.write(make_non_pitch_frame())
        for _ in range(12):
            writer.write(make_pitch_frame(brightness=0.92))
    finally:
        writer.release()


class CameraAnalysisVideoTests(unittest.TestCase):
    def test_video_reader_detects_multiple_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "camera.avi"
            write_test_video(video_path)
            result = analyze_camera_video(video_path)

            self.assertGreaterEqual(len(result.segments), 3)
            self.assertGreaterEqual(
                sum(segment.status == "GEOMETRY_CANDIDATE" for segment in result.segments),
                1,
            )
            self.assertGreaterEqual(
                sum(segment.status == "EXCLUDED" for segment in result.segments),
                1,
            )

    def test_cli_writes_finite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "camera.avi"
            output_path = Path(directory) / "analysis.json"
            write_test_video(video_path)
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_camera_segments.py"),
                    "--video",
                    str(video_path),
                    "--output",
                    str(output_path),
                    "--fail-if-no-geometry-candidate",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "camera-analysis-v1")
            self.assertFalse(payload["automatic_calibration_available"])


if __name__ == "__main__":
    unittest.main()
