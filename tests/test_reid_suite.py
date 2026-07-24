import json
import unittest
from pathlib import Path

from app.benchmark.reid_metrics import ReIDGateThresholds
from app.benchmark.reid_schema import load_reid_annotation
from app.benchmark.reid_suite import evaluate_reid_benchmark_suite
from app.benchmark.schema import load_annotation


FIXTURE = Path(__file__).parent / "fixtures" / "reid_benchmark"


class ReIDBenchmarkSuiteTests(unittest.TestCase):
    def test_fixture_passes_frame_idf1_and_window_reid_gates(self):
        tracking = json.loads((FIXTURE / "tracking.json").read_text(encoding="utf-8"))
        report = evaluate_reid_benchmark_suite(
            tracking,
            frame_annotation=load_annotation(FIXTURE / "frame-annotations.json"),
            window_annotation=load_reid_annotation(
                FIXTURE / "window-annotations.json"
            ),
            reid_thresholds=ReIDGateThresholds(minimum_scorable_windows=3),
        )

        self.assertEqual(report["schema_version"], "reid-benchmark-suite-v1")
        self.assertEqual(report["frame_tracking"]["aggregate"]["metrics"]["idf1"], 1.0)
        self.assertEqual(report["window_reid"]["metrics"]["accepted_precision"], 1.0)
        self.assertTrue(report["quality_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
