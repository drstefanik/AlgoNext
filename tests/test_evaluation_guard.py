import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.core.evaluation_guard import sanitize_analysis_job
from app.core.models import AnalysisJob


class EvaluationGuardTests(unittest.TestCase):
    def test_legacy_scores_and_scoring_warnings_are_sanitized(self):
        job = AnalysisJob(
            id="job-legacy",
            status="COMPLETED",
            category="U15",
            role="Midfielder",
            warnings=[
                "INCOMPLETE_RADAR",
                "MISSING_OVERALL_SCORE",
                "MISSING_CLIPS",
            ],
            result={
                "overall_score": 88.0,
                "role_score": 88.0,
                "match_rating_10": 7.8,
                "radar": {"Finishing": 90.0},
                "evidence_metrics": {
                    "candidate_metrics": {
                        "coveragePct": 0.125,
                        "stabilityScore": 0.333,
                        "sampleFramesCount": 4,
                    },
                    "distance_covered_m": 1000.0,
                },
            },
        )

        sanitize_analysis_job(job)

        self.assertIsNone(job.result["overall_score"])
        self.assertIsNone(job.result["role_score"])
        self.assertIsNone(job.result["match_rating_10"])
        self.assertEqual(job.result["evaluation_status"], "TRACKING_ONLY")
        self.assertAlmostEqual(job.result["tracking_quality_index"], 17.6, places=1)
        self.assertNotIn("distance_covered_m", job.result["evidence_metrics"])
        self.assertEqual(job.status, "PARTIAL")
        self.assertIn("MISSING_CLIPS", job.warnings)
        self.assertNotIn("INCOMPLETE_RADAR", job.warnings)
        self.assertNotIn("MISSING_OVERALL_SCORE", job.warnings)
        self.assertIn("TRACKING_EVIDENCE_INSUFFICIENT", job.warnings)
        self.assertIn("CROSS_SHOT_IDENTITY_UNVALIDATED", job.warnings)
        self.assertIn("PLAYER_EVALUATION_WITHHELD", job.warnings)

    def test_real_full_match_payload_is_corrected_on_read(self):
        job = AnalysisJob(
            id="job-full-match",
            status="PARTIAL",
            category="U17",
            role="Midfielder",
            warnings=["INCOMPLETE_RADAR"],
            result={
                "tracking": {
                    "coverage_pct_total": 0.49,
                    "coverage_pct": 0.49,
                    "bboxes_count": 29,
                    "segments_total": 108,
                    "segments_with_player": 7,
                    "largest_gap_sec": 2031.94,
                },
                "evidence_metrics": {
                    "candidate_metrics": {"stabilityScore": 0.0},
                },
            },
        )

        sanitize_analysis_job(job)

        signals = job.result["tracking_signals"]
        self.assertAlmostEqual(signals["coverage_pct"], 0.49)
        self.assertAlmostEqual(signals["coverage_ratio"], 0.0049)
        self.assertAlmostEqual(job.result["tracking"]["coverage_ratio"], 0.0049)
        self.assertAlmostEqual(job.result["tracking"]["coverage_ratio_total"], 0.0049)
        self.assertAlmostEqual(job.result["tracking_quality_index"], 9.9, places=1)
        self.assertEqual(
            job.warnings,
            [
                "TRACKING_EVIDENCE_INSUFFICIENT",
                "CROSS_SHOT_IDENTITY_UNVALIDATED",
                "LONG_TRACKING_GAPS",
                "PLAYER_EVALUATION_WITHHELD",
            ],
        )

    def test_future_validated_player_score_and_warnings_are_preserved(self):
        validated = {
            "player_evaluation_available": True,
            "overall_score": 81.0,
            "score_provenance": {
                "kind": "player_evaluation",
                "validated_player_score": True,
                "version": "future-model-v1",
            },
        }
        job = AnalysisJob(
            id="job-validated",
            status="COMPLETED",
            category="Senior",
            role="Midfielder",
            warnings=["MODEL_VALIDATED"],
            result=dict(validated),
        )

        sanitize_analysis_job(job)

        self.assertEqual(job.result, validated)
        self.assertEqual(job.warnings, ["MODEL_VALIDATED"])
        self.assertEqual(job.status, "COMPLETED")

    def test_failed_anchor_is_truthful_and_recoverable(self):
        job = AnalysisJob(
            id="job-anchor-failed",
            status="WAITING_FOR_PLAYER",
            category="U17",
            role="Midfielder",
            warnings=["PLAYER_RESELECTION_REQUIRED"],
            result={
                "tracking": {
                    "tracking_success": False,
                    "tracking_status": "ANCHOR_NOT_FOUND",
                    "action_required": "RESELECT_PLAYER",
                    "coverage_pct_total": 0.0,
                    "bboxes_count": 0,
                    "segments_total": 108,
                    "segments_with_player": 0,
                    "largest_gap_sec": 5931.775,
                    "anchors_total": 1,
                    "anchors_matched": 0,
                    "reid_summary": {
                        "status": "ANCHOR_NOT_FOUND",
                        "reason_codes": ["REID_ANCHORS_NOT_FOUND"],
                    },
                },
                "evidence_metrics": {
                    "candidate_metrics": {
                        "coveragePct": 0.125,
                        "stabilityScore": 0.333,
                        "sampleFramesCount": 4,
                    }
                },
            },
        )

        sanitize_analysis_job(job)

        self.assertEqual(job.status, "WAITING_FOR_PLAYER")
        self.assertEqual(job.result["evaluation_status"], "TRACKING_FAILED")
        self.assertIsNone(job.result["tracking_quality_index"])
        self.assertEqual(job.result["tracking_signals"]["samples_used"], 0)
        self.assertEqual(
            job.result["tracking_signals"]["tracklet_continuity_pct"],
            0.0,
        )
        self.assertNotIn(
            "candidate_metrics",
            job.result["evidence_metrics"],
        )
        self.assertIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
        self.assertIn("PLAYER_RESELECTION_REQUIRED", job.warnings)

    def test_matched_anchor_rejection_preserves_real_reason(self):
        cases = (
            ("ANCHOR_REJECTED", "ANCHOR_TRACK_COLOR_UNVERIFIED"),
            ("ANCHOR_ONLY", "AUTONOMOUS_REID_NOT_PROVEN"),
        )
        for tracking_status, reason_code in cases:
            with self.subTest(tracking_status=tracking_status):
                job = AnalysisJob(
                    id=f"job-{tracking_status.lower()}",
                    status="WAITING_FOR_PLAYER",
                    category="U17",
                    role="Midfielder",
                    warnings=["PLAYER_ANCHOR_NOT_FOUND"],
                    result={
                        "tracking": {
                            "tracking_success": False,
                            "tracking_status": tracking_status,
                            "action_required": "RESELECT_PLAYER",
                            "bboxes_count": 0,
                            "segments_total": 108,
                            "segments_with_player": 0,
                            "anchors_total": 2,
                            "anchors_matched": 2,
                            "reid_summary": {
                                "status": tracking_status,
                                "reason_codes": [reason_code],
                            },
                        },
                    },
                )

                sanitize_analysis_job(job)

                self.assertEqual(
                    job.result["evaluation_status"],
                    "TRACKING_FAILED",
                )
                self.assertNotIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
                self.assertIn(reason_code, job.warnings)
                self.assertIn(tracking_status, job.warnings)
                self.assertIn("PLAYER_RESELECTION_REQUIRED", job.warnings)

    def test_sanitized_matched_anchor_rejection_uses_diagnostics_only_for_warning(self):
        job = AnalysisJob(
            id="job-sanitized-anchor-only",
            status="WAITING_FOR_PLAYER",
            category="U17",
            role="Midfielder",
            warnings=["PLAYER_ANCHOR_NOT_FOUND"],
            result={
                "tracking": {
                    "tracking_success": False,
                    "tracking_status": "ANCHOR_ONLY",
                    "action_required": "RESELECT_PLAYER",
                    "bboxes_count": 0,
                    "segments_total": 108,
                    "segments_with_player": 0,
                    "anchors_total": 0,
                    "anchors_matched": 0,
                    "pre_guard_anchor_diagnostics": {
                        "diagnostic_only": True,
                        "validated": False,
                        "anchors_total": 2,
                        "anchors_matched_before_guard": 999,
                        "anchor_matches": [
                            {
                                "anchor_id": 1,
                                "matched_before_guard": True,
                            },
                            {
                                "anchor_id": 2,
                                "matched_before_guard": True,
                            },
                        ],
                    },
                    "reid_summary": {
                        "status": "ANCHOR_ONLY",
                        "reason_codes": ["AUTONOMOUS_REID_NOT_PROVEN"],
                    },
                },
            },
        )

        sanitize_analysis_job(job)

        self.assertFalse(job.result["tracking"]["tracking_success"])
        self.assertEqual(job.result["tracking"]["anchors_matched"], 0)
        self.assertNotIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
        self.assertIn("AUTONOMOUS_REID_NOT_PROVEN", job.warnings)
        self.assertIn("ANCHOR_ONLY", job.warnings)
        self.assertIn("PLAYER_RESELECTION_REQUIRED", job.warnings)

    def test_unmarked_pre_guard_diagnostics_cannot_change_warning_classification(self):
        for mutation in (
            {"diagnostic_only": False, "validated": False},
            {"diagnostic_only": True, "validated": True},
        ):
            with self.subTest(mutation=mutation):
                diagnostics = {
                    **mutation,
                    "anchors_total": 1,
                    "anchors_matched_before_guard": 1,
                    "anchor_matches": [{"matched_before_guard": True}],
                }
                job = AnalysisJob(
                    id="job-untrusted-anchor-diagnostics",
                    status="WAITING_FOR_PLAYER",
                    category="U17",
                    role="Midfielder",
                    result={
                        "tracking": {
                            "tracking_success": False,
                            "tracking_status": "ANCHOR_ONLY",
                            "action_required": "RESELECT_PLAYER",
                            "anchors_total": 0,
                            "anchors_matched": 0,
                            "pre_guard_anchor_diagnostics": diagnostics,
                            "reid_summary": {
                                "reason_codes": ["AUTONOMOUS_REID_NOT_PROVEN"],
                            },
                        }
                    },
                )

                sanitize_analysis_job(job)

                self.assertIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
                self.assertIn("PLAYER_RESELECTION_REQUIRED", job.warnings)

    def test_acquisition_error_keeps_retry_semantics(self):
        job = AnalysisJob(
            id="job-anchor-infra-error",
            status="FAILED",
            category="U17",
            role="Midfielder",
            warnings=["PLAYER_ANCHOR_ACQUISITION_FAILED"],
            result={
                "tracking": {
                    "tracking_success": False,
                    "tracking_status": "ANCHOR_ACQUISITION_ERROR",
                    "action_required": "RETRY_ANALYSIS",
                    "bboxes_count": 0,
                    "segments_total": 108,
                    "segments_with_player": 0,
                    "reid_summary": {
                        "status": "ANCHOR_ACQUISITION_ERROR",
                        "reason_codes": ["REID_ANCHOR_ACQUISITION_ERROR"],
                    },
                },
            },
        )

        sanitize_analysis_job(job)

        self.assertEqual(job.result["evaluation_status"], "TRACKING_FAILED")
        self.assertIn("PLAYER_ANCHOR_ACQUISITION_FAILED", job.warnings)
        self.assertNotIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
        self.assertNotIn("PLAYER_RESELECTION_REQUIRED", job.warnings)

    def test_partial_timeout_does_not_blame_player_reference(self):
        job = AnalysisJob(
            id="job-tracking-timeout",
            status="PARTIAL",
            category="U17",
            role="Midfielder",
            warnings=["TRACKING_PARTIAL_TIMEOUT"],
            result={
                "tracking": {
                    "partial": True,
                    "partial_reason": "TRACKING_TIMEOUT",
                    "segments_total": 108,
                    "segments_with_player": 0,
                    "bboxes_count": 0,
                    "reid_summary": {
                        "status": "PARTIAL_TIMEOUT",
                        "reason_codes": ["TRACKING_BUDGET_EXHAUSTED"],
                    },
                },
            },
        )

        sanitize_analysis_job(job)

        self.assertEqual(
            job.result["evaluation_status"],
            "TRACKING_INCOMPLETE",
        )
        self.assertIn("TRACKING_PARTIAL_TIMEOUT", job.warnings)
        self.assertNotIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
        self.assertNotIn("PLAYER_RESELECTION_REQUIRED", job.warnings)

    def test_post_anchor_technical_error_requests_retry_without_reselection(self):
        job = AnalysisJob(
            id="job-team-guard-error",
            status="FAILED",
            category="U17",
            role="Midfielder",
            warnings=["PLAYER_TRACKING_RETRY_REQUIRED"],
            result={
                "tracking": {
                    "tracking_success": False,
                    "tracking_status": "TEAM_COLOR_GUARD_ERROR",
                    "action_required": "RETRY_ANALYSIS",
                    "bboxes_count": 0,
                    "segments_total": 108,
                    "segments_with_player": 0,
                    "anchors_total": 1,
                    "anchors_matched": 1,
                    "reid_summary": {
                        "status": "TEAM_COLOR_GUARD_ERROR",
                        "reason_codes": ["TEAM_COLOR_GUARD_ERROR"],
                    },
                },
            },
        )

        sanitize_analysis_job(job)

        self.assertIn("PLAYER_TRACKING_RETRY_REQUIRED", job.warnings)
        self.assertNotIn("PLAYER_ANCHOR_NOT_FOUND", job.warnings)
        self.assertNotIn("PLAYER_RESELECTION_REQUIRED", job.warnings)


if __name__ == "__main__":
    unittest.main()
