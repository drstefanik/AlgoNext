import unittest
from types import SimpleNamespace

from app.core.tracking_outcome import apply_tracking_outcome


def set_progress(job, step, pct, message):
    job.progress = {
        "step": step,
        "pct": pct,
        "message": message,
    }


class AnchorReselectionPipelineTests(unittest.TestCase):
    def test_failed_anchors_stop_before_full_analysis_and_preserve_preview(self):
        job = SimpleNamespace(
            status="RUNNING",
            target={
                "confirmed": True,
                "full_match_mode": True,
                "selection": {"frame_key": "frame.jpg"},
                "selections": [{"frame_key": "frame.jpg"}],
            },
            player_ref={"track_id": 10},
            anchor={"t": 179.751},
            progress={"step": "TRACKING", "pct": 35},
            warnings=[],
            error=None,
            failure_reason=None,
            result={
                "candidates": {"candidates": [{"track_id": 10}]},
                "framesProcessed": 32,
                "overall_score": 99,
                "report": {"stale": True},
            },
        )
        tracking = {
            "tracking_success": False,
            "tracking_status": "ANCHOR_NOT_FOUND",
            "action_required": "RESELECT_PLAYER",
            "bboxes_count": 0,
            "segments_total": 108,
            "windows_processed": 1,
            "segments_with_player": 0,
            "anchors_total": 1,
            "anchors_matched": 0,
            "reid_summary": {
                "reason_codes": ["REID_ANCHORS_NOT_FOUND"],
            },
        }

        stop = apply_tracking_outcome(
            job,
            tracking,
            set_progress=set_progress,
        )

        self.assertTrue(stop)
        self.assertEqual(job.status, "WAITING_FOR_PLAYER")
        self.assertIsNone(job.player_ref)
        self.assertEqual(job.anchor, {})
        self.assertFalse(job.target["confirmed"])
        self.assertTrue(job.target["full_match_mode"])
        self.assertNotIn("selection", job.target)
        self.assertNotIn("selections", job.target)
        self.assertEqual(job.progress["step"], "WAITING_FOR_PLAYER")
        self.assertEqual(job.progress["pct"], 35)
        self.assertEqual(
            job.result["candidates"]["candidates"][0]["track_id"],
            10,
        )
        self.assertNotIn("overall_score", job.result)
        self.assertNotIn("report", job.result)
        self.assertEqual(
            job.result["analysis_outcome"]["windows_processed"],
            1,
        )
        self.assertIn("PLAYER_RESELECTION_REQUIRED", job.warnings)

    def test_partial_timeout_is_not_converted_to_anchor_reselection(self):
        job = SimpleNamespace(
            status="RUNNING",
            target={"confirmed": True, "full_match_mode": True},
            player_ref={"track_id": 10},
            anchor={"t": 179.751},
            progress={"step": "TRACKING", "pct": 35},
            warnings=["TRACKING_PARTIAL_TIMEOUT"],
            error=None,
            failure_reason=None,
            result={},
        )

        stop = apply_tracking_outcome(
            job,
            {
                "partial": True,
                "partial_reason": "TRACKING_TIMEOUT",
                "bboxes_count": 0,
                "segments_total": 108,
                "segments_with_player": 0,
                "reid_summary": {
                    "status": "PARTIAL_TIMEOUT",
                    "reason_codes": ["TRACKING_BUDGET_EXHAUSTED"],
                },
            },
            set_progress=set_progress,
        )

        self.assertFalse(stop)
        self.assertEqual(job.status, "RUNNING")
        self.assertEqual(
            job.result["analysis_outcome"]["tracking_state"],
            "INCOMPLETE",
        )
        self.assertNotIn("PLAYER_RESELECTION_REQUIRED", job.warnings)

    def test_post_anchor_technical_failure_preserves_selection_for_retry(self):
        target = {
            "confirmed": True,
            "full_match_mode": True,
            "selection": {"frame_key": "frame.jpg"},
            "selections": [{"frame_key": "frame.jpg"}],
        }
        player_ref = {"track_id": 10, "t": 179.751}
        job = SimpleNamespace(
            status="RUNNING",
            target=target,
            player_ref=player_ref,
            anchor={"t": 179.751},
            progress={"step": "TRACKING", "pct": 35},
            warnings=[],
            error=None,
            failure_reason=None,
            result={},
        )

        stop = apply_tracking_outcome(
            job,
            {
                "tracking_success": False,
                "tracking_status": "TEAM_COLOR_GUARD_ERROR",
                "action_required": "RETRY_ANALYSIS",
                "bboxes_count": 0,
                "segments_total": 108,
                "segments_with_player": 0,
                "anchors_total": 1,
                "anchors_matched": 1,
                "reid_summary": {
                    "reason_codes": ["TEAM_COLOR_GUARD_ERROR"],
                },
            },
            set_progress=set_progress,
        )

        self.assertTrue(stop)
        self.assertEqual(job.status, "FAILED")
        self.assertIs(job.target, target)
        self.assertIs(job.player_ref, player_ref)
        self.assertIn("PLAYER_TRACKING_RETRY_REQUIRED", job.warnings)
        self.assertNotIn("PLAYER_RESELECTION_REQUIRED", job.warnings)


if __name__ == "__main__":
    unittest.main()
