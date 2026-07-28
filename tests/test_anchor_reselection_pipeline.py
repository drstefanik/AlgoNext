import unittest
from types import SimpleNamespace

from app.core.tracking_outcome import (
    StaleAnalysisAttemptError,
    apply_tracking_outcome,
)


def set_progress(job, step, pct, message):
    job.progress = {
        "step": step,
        "pct": pct,
        "message": message,
    }


class AnchorReselectionPipelineTests(unittest.TestCase):
    def test_failed_anchors_stop_before_full_analysis_and_preserve_preview(self):
        attempt_id = "63d748f7-66a4-485d-adca-d3c7a6067cb0"
        job = SimpleNamespace(
            status="RUNNING",
            target={
                "confirmed": True,
                "analysis_attempt_id": attempt_id,
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
            "analysis_attempt_id": attempt_id,
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
        self.assertEqual(
            job.target["tracking"]["analysis_attempt_id"],
            attempt_id,
        )
        self.assertEqual(job.progress["step"], "WAITING_FOR_PLAYER")
        self.assertEqual(job.progress["pct"], 35)
        self.assertEqual(
            job.progress["message"],
            "Player reference not found. Select a clearer frame.",
        )
        self.assertEqual(job.failure_reason, "PLAYER_RESELECTION_REQUIRED")
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
        self.assertEqual(
            job.result["analysis_outcome"]["pipeline_state"],
            "STOPPED",
        )
        self.assertEqual(
            job.result["analysis_outcome"]["tracking_state"],
            "FAILED",
        )
        self.assertIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
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

    def test_successful_tracking_does_not_declare_pipeline_done_before_finalize(self):
        attempt_id = "c712b2e7-30cf-4d43-9c3a-6c7017b48468"
        job = SimpleNamespace(
            status="RUNNING",
            target={
                "confirmed": True,
                "full_match_mode": True,
                "analysis_attempt_id": attempt_id,
            },
            player_ref={"track_id": 10},
            anchor={"t": 719.003},
            progress={"step": "TRACKING", "pct": 70},
            warnings=[],
            error=None,
            failure_reason=None,
            result={},
        )

        stop = apply_tracking_outcome(
            job,
            {
                "analysis_attempt_id": attempt_id,
                "tracking_success": True,
                "tracking_status": "SUCCEEDED",
                "action_required": None,
                "bboxes_count": 240,
                "segments_total": 108,
                "windows_processed": 108,
                "segments_with_player": 12,
                "anchors_total": 2,
                "anchors_matched": 2,
            },
            set_progress=set_progress,
        )

        self.assertFalse(stop)
        self.assertEqual(
            job.result["analysis_outcome"]["pipeline_state"],
            "RUNNING",
        )
        self.assertEqual(
            job.result["analysis_outcome"]["tracking_state"],
            "SUCCEEDED",
        )
        self.assertEqual(job.result["analysis_attempt_id"], attempt_id)
        self.assertEqual(
            job.result["tracking"]["analysis_attempt_id"],
            attempt_id,
        )
        self.assertEqual(
            job.result["analysis_outcome"]["analysis_attempt_id"],
            attempt_id,
        )

    def test_stale_attempt_cannot_mutate_newer_job(self):
        current_attempt = "63d748f7-66a4-485d-adca-d3c7a6067cb0"
        stale_attempt = "1fdaf4b6-3c5c-4923-b80d-c542df602e96"
        target = {
            "confirmed": True,
            "analysis_attempt_id": current_attempt,
        }
        result = {
            "analysis_attempt_id": current_attempt,
            "assets": {"input_video": {"key": "jobs/job/input.mp4"}},
        }
        job = SimpleNamespace(
            status="RUNNING",
            target=target,
            player_ref={"track_id": 10},
            anchor={"t": 719.003},
            progress={"step": "TRACKING", "pct": 70},
            warnings=[],
            error=None,
            failure_reason=None,
            result=result,
        )

        with self.assertRaises(StaleAnalysisAttemptError):
            apply_tracking_outcome(
                job,
                {
                    "analysis_attempt_id": stale_attempt,
                    "tracking_success": True,
                    "tracking_status": "SUCCEEDED",
                    "bboxes_count": 100,
                },
                set_progress=set_progress,
            )

        self.assertEqual(job.status, "RUNNING")
        self.assertIs(job.target, target)
        self.assertIs(job.result, result)
        self.assertEqual(job.progress, {"step": "TRACKING", "pct": 70})

    def test_guard_rejection_after_matched_anchors_reports_real_outcome(self):
        job = SimpleNamespace(
            status="RUNNING",
            target={
                "confirmed": True,
                "full_match_mode": True,
                "selection": {"frame_key": "frame.jpg"},
                "selections": [{"frame_key": "frame.jpg"}],
            },
            player_ref={"track_id": 10},
            anchor={"t": 719.003},
            progress={"step": "TRACKING", "pct": 99},
            warnings=[],
            error=None,
            failure_reason=None,
            result={},
        )
        tracking = {
            "tracking_success": False,
            "tracking_status": "ANCHOR_REJECTED",
            "action_required": "RESELECT_PLAYER",
            "bboxes_count": 0,
            "segments_total": 108,
            "windows_processed": 108,
            "segments_with_player": 0,
            "autonomous_segments_with_player": 0,
            "autonomous_bboxes_count": 0,
            "tracking_scope_status": "ANCHOR_ONLY",
            "anchors_total": 2,
            "anchors_matched": 2,
            "reid_summary": {
                "status": "ANCHOR_REJECTED",
                "reason_codes": ["ANCHOR_TRACK_COLOR_UNVERIFIED"],
            },
        }

        stop = apply_tracking_outcome(
            job,
            tracking,
            set_progress=set_progress,
        )

        self.assertTrue(stop)
        self.assertEqual(job.status, "WAITING_FOR_PLAYER")
        self.assertEqual(job.failure_reason, "ANCHOR_REJECTED")
        self.assertNotIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
        self.assertIn("ANCHOR_TRACK_COLOR_UNVERIFIED", job.warnings)
        self.assertIn("ANCHOR_REJECTED", job.warnings)
        self.assertIn("PLAYER_RESELECTION_REQUIRED", job.warnings)
        self.assertEqual(job.progress["step"], "WAITING_FOR_PLAYER")
        self.assertEqual(job.progress["pct"], 100)
        self.assertNotIn("reference not found", job.progress["message"].lower())
        self.assertEqual(
            job.result["analysis_outcome"]["pipeline_state"],
            "DONE",
        )
        self.assertEqual(
            job.result["analysis_outcome"]["tracking_state"],
            "FAILED",
        )
        self.assertEqual(
            job.result["analysis_outcome"]["autonomous_segments_with_player"],
            0,
        )
        self.assertEqual(
            job.result["analysis_outcome"]["autonomous_bboxes_count"],
            0,
        )
        self.assertEqual(
            job.result["analysis_outcome"]["tracking_scope_status"],
            "ANCHOR_ONLY",
        )
        self.assertEqual(
            job.result["analysis_outcome"]["reason_codes"],
            ["ANCHOR_TRACK_COLOR_UNVERIFIED", "ANCHOR_REJECTED"],
        )

    def test_sanitized_anchor_only_diagnostics_do_not_become_anchor_not_found(self):
        job = SimpleNamespace(
            status="RUNNING",
            target={
                "confirmed": True,
                "full_match_mode": True,
                "selection": {"frame_key": "frame_0004.jpg"},
                "selections": [{"frame_key": "frame_0004.jpg"}],
            },
            player_ref={"t": 719.003},
            anchor={"t": 719.003},
            progress={"step": "TRACKING", "pct": 99},
            warnings=[],
            error=None,
            failure_reason=None,
            result={},
        )
        tracking = {
            "tracking_success": False,
            "tracking_status": "ANCHOR_ONLY",
            "action_required": "RESELECT_PLAYER",
            "bboxes_count": 0,
            "segments_total": 108,
            "windows_processed": 108,
            "segments_with_player": 0,
            "anchors_total": 0,
            "anchors_matched": 0,
            "pre_guard_anchor_diagnostics": {
                "diagnostic_only": True,
                "validated": False,
                "anchors_total": 2,
                "anchors_matched_before_guard": 2,
            },
            "reid_summary": {
                "status": "ANCHOR_ONLY",
                "reason_codes": [
                    "AUTONOMOUS_REID_NOT_PROVEN",
                    "TEAM_COLOR_GUARD_UNVERIFIED_FAILURE_OUTPUT",
                ],
            },
        }

        stop = apply_tracking_outcome(
            job,
            tracking,
            set_progress=set_progress,
        )

        self.assertTrue(stop)
        self.assertEqual(job.status, "WAITING_FOR_PLAYER")
        self.assertEqual(job.failure_reason, "ANCHOR_ONLY")
        self.assertNotIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
        self.assertIn("AUTONOMOUS_REID_NOT_PROVEN", job.warnings)
        self.assertEqual(
            job.result["analysis_outcome"]["anchors_matched"],
            0,
        )
        self.assertEqual(
            job.result["analysis_outcome"]["anchors_matched_before_guard"],
            2,
        )
        self.assertIn(
            "tracking was rejected",
            job.progress["message"].lower(),
        )

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
