import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.reid.full_match_runtime import select_full_match_profile
from app.reid.runtime import install_windowed_reid


class ReIDRuntimeTests(unittest.TestCase):
    def test_disabled_flag_still_installs_cpu_budget_wrapper(self):
        captured = {}

        def original(*args, **kwargs):
            captured.update(kwargs)
            return {"mode": "legacy"}

        module = SimpleNamespace(track_player_windowed=original)
        environment = {
            "PLAYER_REID_ENABLED": "0",
            "FULL_MATCH_TARGET_SAMPLES": "6000",
            "FULL_MATCH_MIN_FPS": "1",
            "FULL_MATCH_MAX_FPS": "2",
            "FULL_MATCH_WINDOW_SEC": "60",
            "FULL_MATCH_OVERLAP_SEC": "5",
            "FULL_MATCH_DETECTOR_MODEL": "yolo11n.pt",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertFalse(install_windowed_reid(module, lambda: "reid"))
            self.assertIsNot(module.track_player_windowed, original)
            output = module.track_player_windowed(
                "job-disabled",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                video_duration_sec=6000.0,
                fps=5,
                window_sec=45.0,
                overlap_sec=10.0,
            )

        self.assertEqual(captured["fps"], 1)
        self.assertEqual(captured["window_sec"], 60.0)
        self.assertEqual(captured["overlap_sec"], 5.0)
        self.assertEqual(captured["detector_model"], "yolo11n.pt")
        self.assertEqual(output["reid_summary"]["status"], "DISABLED")
        self.assertEqual(output["runtime_profile"]["fps"], 1)

    def test_enabled_flag_installs_wrapper_and_passes_fallback(self):
        calls = []

        def original(value):
            calls.append("legacy")
            return {"mode": "legacy", "value": value}

        def implementation(value, *, fallback):
            self.assertIs(fallback, original)
            calls.append("reid")
            return {"mode": "reid", "value": value}

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "1"}, clear=False):
            self.assertTrue(install_windowed_reid(module, implementation))
            self.assertEqual(module.track_player_windowed(7)["mode"], "reid")
            self.assertTrue(install_windowed_reid(module, implementation))
        self.assertEqual(calls, ["reid"])

    def test_enabled_wrapper_forwards_every_selection_unchanged(self):
        captured = {}
        selections = [
            {
                "frame_key": "frame-0.jpg",
                "frame_time_sec": 0.0,
                "x": 0.10,
                "y": 0.20,
                "w": 0.10,
                "h": 0.25,
            },
            {
                "frame_key": "frame-90.jpg",
                "frame_time_sec": 90.0,
                "x": 0.40,
                "y": 0.20,
                "w": 0.10,
                "h": 0.25,
            },
            {
                "frame_key": "frame-180.jpg",
                "frame_time_sec": 180.0,
                "x": 0.70,
                "y": 0.20,
                "w": 0.10,
                "h": 0.25,
            },
        ]

        def original(*args, **kwargs):
            return {"mode": "legacy"}

        def implementation(*args, fallback, **kwargs):
            self.assertIs(fallback, original)
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {
                "mode": "reid",
                "anchors_used": {"selections": args[3]},
            }

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "1"}, clear=False):
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(
                "job-multi",
                "/tmp/input.mp4",
                {"t": 0.0},
                selections,
                video_duration_sec=240.0,
                fps=5,
            )

        self.assertIs(captured["args"][3], selections)
        self.assertEqual(captured["args"][3], selections)
        self.assertEqual(
            output["anchors_used"]["selections"],
            selections,
        )

    def test_runtime_failure_falls_back_when_enabled(self):
        def original(value):
            return {"mode": "legacy", "value": value}

        def implementation(value, *, fallback):
            raise RuntimeError("boom")

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_ENABLED": "1",
                "PLAYER_REID_FAIL_OPEN": "1",
            },
            clear=False,
        ):
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(9)
        self.assertEqual(output["mode"], "legacy")
        self.assertEqual(output["reid_summary"]["status"], "FALLBACK_LEGACY")

    def test_runtime_failure_can_fail_closed(self):
        def original():
            return {"mode": "legacy"}

        def implementation(*, fallback):
            raise RuntimeError("boom")

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_ENABLED": "1",
                "PLAYER_REID_FAIL_OPEN": "0",
            },
            clear=False,
        ):
            install_windowed_reid(module, implementation)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                module.track_player_windowed()

    def test_tracking_timeout_returns_partial_without_restarting_legacy(self):
        class TrackingTimeoutError(RuntimeError):
            pass

        calls = []

        def original(*args, **kwargs):
            calls.append("legacy")
            return {"mode": "legacy"}

        def implementation(*args, fallback, **kwargs):
            raise TrackingTimeoutError("timeout")

        module = SimpleNamespace(
            track_player_windowed=original,
            TrackingTimeoutError=TrackingTimeoutError,
        )
        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_ENABLED": "1",
                "PLAYER_REID_FAIL_OPEN": "1",
            },
            clear=False,
        ), patch("app.reid.runtime.mark_partial_timeout") as mark_partial:
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(
                "job-1",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                video_duration_sec=6000.0,
                fps=5,
            )

        self.assertEqual(calls, [])
        self.assertTrue(output["partial"])
        self.assertEqual(output["partial_reason"], "TRACKING_TIMEOUT")
        self.assertEqual(output["reid_summary"]["status"], "PARTIAL_TIMEOUT")
        self.assertEqual(output["runtime_profile"]["fps"], 1)
        mark_partial.assert_called_once()

    def test_disabled_legacy_timeout_is_also_partial(self):
        class TrackingTimeoutError(RuntimeError):
            pass

        def original(*args, **kwargs):
            raise TrackingTimeoutError("timeout")

        module = SimpleNamespace(
            track_player_windowed=original,
            TrackingTimeoutError=TrackingTimeoutError,
        )
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "0"}, clear=False), patch(
            "app.reid.runtime.mark_partial_timeout"
        ) as mark_partial:
            install_windowed_reid(module)
            output = module.track_player_windowed(
                "job-disabled-timeout",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                video_duration_sec=6000.0,
                fps=5,
            )
        self.assertTrue(output["partial"])
        self.assertEqual(output["identity_mode"], "disabled")
        self.assertEqual(output["reid_summary"]["status"], "DISABLED")
        self.assertIn(
            "TRACKING_BUDGET_EXHAUSTED", output["reid_summary"]["reason_codes"]
        )
        mark_partial.assert_called_once()

    def test_long_full_match_uses_cpu_budget_profile(self):
        captured = {}

        def original(*args, **kwargs):
            return {"mode": "legacy"}

        def implementation(*args, fallback, **kwargs):
            captured.update(kwargs)
            return {"mode": "reid"}

        module = SimpleNamespace(track_player_windowed=original)
        environment = {
            "PLAYER_REID_ENABLED": "1",
            "FULL_MATCH_TARGET_SAMPLES": "6000",
            "FULL_MATCH_MIN_FPS": "1",
            "FULL_MATCH_MAX_FPS": "2",
            "FULL_MATCH_WINDOW_SEC": "60",
            "FULL_MATCH_OVERLAP_SEC": "5",
            "FULL_MATCH_DETECTOR_MODEL": "yolo11n.pt",
        }
        with patch.dict(os.environ, environment, clear=True):
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(
                "job-2",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                video_duration_sec=6000.0,
                fps=5,
                window_sec=45.0,
                overlap_sec=10.0,
            )

        self.assertEqual(captured["fps"], 1)
        self.assertEqual(captured["window_sec"], 60.0)
        self.assertEqual(captured["overlap_sec"], 5.0)
        self.assertEqual(captured["detector_model"], "yolo11n.pt")
        self.assertEqual(output["runtime_profile"]["fps"], 1)
        self.assertLess(output["runtime_profile"]["estimated_samples"], 7000)

    def test_short_video_preserves_requested_quality_profile(self):
        with patch.dict(os.environ, {}, clear=True):
            profile = select_full_match_profile(
                video_duration_sec=600.0,
                requested_fps=5,
                requested_window_sec=45.0,
                requested_overlap_sec=10.0,
                requested_detector_model="yolo11s.pt",
            )

        self.assertEqual(profile.fps, 5)
        self.assertEqual(profile.window_sec, 45.0)
        self.assertEqual(profile.overlap_sec, 10.0)
        self.assertEqual(profile.detector_model, "yolo11s.pt")

    def test_progress_adapter_maps_window_stage_to_visible_range(self):
        progress_calls = []

        def original_tracker():
            return {"mode": "legacy"}

        def implementation(*, fallback):
            return {"mode": "reid"}

        def update_progress(job_id, pct, message):
            progress_calls.append((job_id, pct, message))

        module = SimpleNamespace(
            track_player_windowed=original_tracker,
            _update_tracking_progress=update_progress,
        )
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "1"}, clear=False):
            install_windowed_reid(module, implementation)
            module._update_tracking_progress(
                "job-3",
                25,
                "Tracking player with experimental ReID",
            )

        self.assertEqual(progress_calls[0][0], "job-3")
        self.assertEqual(progress_calls[0][1], 53)
        self.assertIn("50% finestre", progress_calls[0][2])


if __name__ == "__main__":
    unittest.main()
