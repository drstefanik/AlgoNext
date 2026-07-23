import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.core.evaluation_http_guard import (
    REPORT_UNAVAILABLE_MESSAGE,
    build_unavailable_report,
    job_ready_for_report,
    validated_player_evaluation_available,
)


class EvaluationReportGuardTests(unittest.TestCase):
    def test_processing_job_remains_pending_instead_of_unavailable(self):
        job = SimpleNamespace(status="RUNNING", progress={"step": "TRACKING"})

        self.assertFalse(job_ready_for_report(job))

    def test_completed_or_done_job_is_eligible_for_abstention_guard(self):
        completed = SimpleNamespace(status="COMPLETED", progress={})
        done_step = SimpleNamespace(status="RUNNING", progress={"step": "DONE"})
        failed = SimpleNamespace(status="FAILED", progress={"step": "DONE"})

        self.assertTrue(job_ready_for_report(completed))
        self.assertTrue(job_ready_for_report(done_step))
        self.assertFalse(job_ready_for_report(failed))

    def test_tracking_only_result_cannot_generate_player_report(self):
        result = {
            "player_evaluation_available": False,
            "score_provenance": {
                "kind": "tracking_quality",
                "validated_player_score": False,
            },
        }

        self.assertFalse(validated_player_evaluation_available(result))

    def test_validated_provenance_is_required_as_a_complete_triplet(self):
        incomplete = {
            "player_evaluation_available": True,
            "score_provenance": {
                "kind": "player_evaluation",
                "validated_player_score": False,
            },
        }
        validated = {
            "player_evaluation_available": True,
            "score_provenance": {
                "kind": "player_evaluation",
                "validated_player_score": True,
            },
        }

        self.assertFalse(validated_player_evaluation_available(incomplete))
        self.assertTrue(validated_player_evaluation_available(validated))

    def test_unavailable_report_has_zero_confidence_and_declared_limits(self):
        report = build_unavailable_report(
            {"limitations": ["Identity is not verified across camera cuts."]}
        )

        self.assertEqual(report["confidence"], 0.0)
        self.assertEqual(report["strengths"], [])
        self.assertEqual(report["risks"], [])
        self.assertEqual(
            report["limitations"],
            ["Identity is not verified across camera cuts."],
        )

    def test_unavailable_report_has_safe_default_limit(self):
        report = build_unavailable_report({})

        self.assertEqual(report["limitations"], [REPORT_UNAVAILABLE_MESSAGE])


if __name__ == "__main__":
    unittest.main()
