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
        self.assertAlmostEqual(
            job.result["tracking"]["coverage_ratio_total"], 0.0049
        )
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


if __name__ == "__main__":
    unittest.main()
