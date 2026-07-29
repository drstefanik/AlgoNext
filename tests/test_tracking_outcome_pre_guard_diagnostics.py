import unittest
from types import SimpleNamespace

from app.core.tracking_outcome import apply_tracking_outcome


def set_progress(job, step, pct, message):
    job.progress = {
        "step": step,
        "pct": pct,
        "message": message,
    }


def failed_job():
    attempt_id = "f0243750-3488-49a4-ada3-579859961671"
    return SimpleNamespace(
        status="RUNNING",
        target={
            "confirmed": True,
            "full_match_mode": True,
            "analysis_attempt_id": attempt_id,
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


def tracking_payload(*, matches):
    attempt_id = "f0243750-3488-49a4-ada3-579859961671"
    return {
        "analysis_attempt_id": attempt_id,
        "tracking_success": False,
        "tracking_status": "ANCHOR_ONLY",
        "action_required": "RESELECT_PLAYER",
        "bboxes_count": 0,
        "segments_total": 108,
        "windows_processed": 108,
        "segments_with_player": 0,
        "anchors_total": 0,
        "anchors_matched": 0,
        "reid_summary": {
            "status": "ANCHOR_ONLY",
            "reason_codes": [
                "AUTONOMOUS_REID_NOT_PROVEN",
                "TEAM_COLOR_GUARD_UNVERIFIED_FAILURE_OUTPUT",
            ],
            "pre_guard_anchor_diagnostics": {
                "diagnostic_only": True,
                "validated": False,
                "anchors_total": 2,
                "anchors_matched_before_guard": 999,
                "anchor_matches": matches,
            },
        },
    }


class TrackingOutcomePreGuardDiagnosticsTests(unittest.TestCase):
    def test_summary_fallback_is_propagated_and_record_count_is_used(self):
        job = failed_job()
        tracking = tracking_payload(
            matches=[
                {"anchor_id": 1, "matched_before_guard": True},
                {"anchor_id": 2, "matched_before_guard": True},
            ]
        )

        stop = apply_tracking_outcome(
            job,
            tracking,
            set_progress=set_progress,
        )

        self.assertTrue(stop)
        self.assertEqual(job.status, "WAITING_FOR_PLAYER")
        self.assertEqual(job.failure_reason, "ANCHOR_ONLY")
        self.assertNotIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
        self.assertEqual(
            job.result["analysis_outcome"]["anchors_matched_before_guard"],
            2,
        )
        self.assertEqual(
            job.result["tracking"]["pre_guard_anchor_diagnostics"][
                "anchor_matches"
            ],
            tracking["reid_summary"]["pre_guard_anchor_diagnostics"][
                "anchor_matches"
            ],
        )

    def test_unproved_declared_count_does_not_claim_an_anchor_match(self):
        job = failed_job()
        tracking = tracking_payload(
            matches=[
                {"anchor_id": 1, "matched_before_guard": False},
                {"anchor_id": 2},
            ]
        )

        stop = apply_tracking_outcome(
            job,
            tracking,
            set_progress=set_progress,
        )

        self.assertTrue(stop)
        self.assertEqual(
            job.result["analysis_outcome"]["anchors_matched_before_guard"],
            0,
        )
        self.assertEqual(job.failure_reason, "PLAYER_RESELECTION_REQUIRED")
        self.assertIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)


if __name__ == "__main__":
    unittest.main()
