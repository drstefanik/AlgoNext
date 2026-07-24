import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.reid import progress_reporting


class ProgressReportingTests(unittest.TestCase):
    def setUp(self):
        progress_reporting._profiles.clear()

    def profile(self):
        return SimpleNamespace(
            duration_sec=6000.0,
            fps=1,
            window_sec=60.0,
            overlap_sec=5.0,
            detector_model="yolo11n.pt",
            target_samples=6000,
            estimated_samples=6546,
            to_payload=lambda: {
                "duration_sec": 6000.0,
                "fps": 1,
                "window_sec": 60.0,
                "overlap_sec": 5.0,
                "detector_model": "yolo11n.pt",
                "target_samples": 6000,
                "estimated_samples": 6546,
            },
        )

    def test_adapter_reports_window_count(self):
        calls = []
        persisted = []

        def current(job_id, pct, message):
            calls.append((job_id, pct, message))

        tracking = SimpleNamespace(_update_tracking_progress=current)
        profile = self.profile()
        progress_reporting._profiles["job-1"] = profile
        progress_reporting.install_progress_stats_adapter(tracking)

        with patch.object(
            progress_reporting,
            "_persist_progress",
            side_effect=lambda job_id, **kwargs: persisted.append((job_id, kwargs)),
        ):
            tracking._update_tracking_progress(
                "job-1", 25, "Tracking player with experimental ReID"
            )

        self.assertEqual(calls[0][1], 25)
        self.assertEqual(persisted[0][1]["windows_total"], 109)
        self.assertEqual(persisted[0][1]["windows_completed"], 54)
        self.assertEqual(persisted[0][1]["window_progress_pct"], 50.0)
        self.assertIn("54/109", persisted[0][1]["message"])

    def test_begin_and_end_manage_profile_context(self):
        profile = self.profile()
        with patch.object(progress_reporting, "_persist_progress") as persist:
            progress_reporting.begin_full_match_progress("job-1", profile)
        self.assertIs(progress_reporting._profiles["job-1"], profile)
        self.assertEqual(persist.call_args.kwargs["windows_total"], 109)
        progress_reporting.end_full_match_progress("job-1")
        self.assertNotIn("job-1", progress_reporting._profiles)


if __name__ == "__main__":
    unittest.main()
