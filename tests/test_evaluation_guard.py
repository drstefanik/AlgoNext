import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.core.evaluation_guard import sanitize_analysis_job
from app.core.models import AnalysisJob


class EvaluationGuardTests(unittest.TestCase):
    def test_legacy_scores_are_sanitized_before_exposure_or_persistence(self):
        job = AnalysisJob(
            id="job-legacy",
            status="COMPLETED",
            category="U15",
            role="Midfielder",
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

    def test_future_validated_player_score_is_preserved(self):
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
            result=dict(validated),
        )

        sanitize_analysis_job(job)

        self.assertEqual(job.result, validated)


if __name__ == "__main__":
    unittest.main()
