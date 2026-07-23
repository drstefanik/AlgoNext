import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.reid.runtime import install_windowed_reid


class ReIDRuntimeTests(unittest.TestCase):
    def test_disabled_flag_does_not_patch_tracker(self):
        def original():
            return "legacy"

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "0"}, clear=False):
            self.assertFalse(install_windowed_reid(module, lambda: "reid"))
        self.assertIs(module.track_player_windowed, original)

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
        self.assertEqual(
            output["reid_summary"]["status"],
            "FALLBACK_LEGACY",
        )

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

    def test_tracking_timeout_never_restarts_legacy_tracker(self):
        class TrackingTimeoutError(RuntimeError):
            pass

        calls = []

        def original():
            calls.append("legacy")
            return {"mode": "legacy"}

        def implementation(*, fallback):
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
        ):
            install_windowed_reid(module, implementation)
            with self.assertRaises(TrackingTimeoutError):
                module.track_player_windowed()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
