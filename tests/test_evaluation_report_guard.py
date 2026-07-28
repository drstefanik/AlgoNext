import asyncio
import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from starlette.requests import Request
from starlette.responses import Response

from app.core.evaluation_http_guard import (
    EvaluationReportGuardMiddleware,
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

    def test_get_sanitizes_only_a_detached_read_snapshot(self):
        job = SimpleNamespace(
            id="job-report-1",
            status="COMPLETED",
            progress={"step": "DONE", "pct": 100},
            result={
                "analysis_attempt_id": "attempt-a",
                "overall_score": 95,
                "player_evaluation_available": False,
                "score_provenance": {
                    "kind": "tracking_quality",
                    "validated_player_score": False,
                },
            },
            warnings=[],
            report_status="PENDING",
            report=None,
            ai_report=None,
        )

        class Session:
            def __init__(self):
                self.commits = 0
                self.closed = False

            def get(self, _model, job_id, **_kwargs):
                return job if job_id == job.id else None

            def commit(self):
                self.commits += 1

            def close(self):
                self.closed = True

        session = Session()
        before = copy.deepcopy(job.__dict__)
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": f"/jobs/{job.id}/report",
                "headers": [],
                "query_string": b"",
            }
        )
        middleware = EvaluationReportGuardMiddleware(lambda _scope: None)

        async def call_next(_request):
            raise AssertionError("GET guard should answer directly")

        with patch(
            "app.core.evaluation_http_guard.SessionLocal",
            return_value=session,
        ):
            response = asyncio.run(middleware.dispatch(request, call_next))

        payload = json.loads(response.body)
        self.assertEqual(payload["data"]["status"], "UNAVAILABLE")
        self.assertEqual(job.__dict__, before)
        self.assertEqual(session.commits, 0)
        self.assertTrue(session.closed)

    def test_post_refreshes_retry_b_before_sanitizing_attempt_a(self):
        cached_attempt_a = SimpleNamespace(
            id="job-report-race",
            status="COMPLETED",
            progress={"step": "DONE", "pct": 100},
            target={"analysis_attempt_id": "attempt-a"},
            result={
                "analysis_attempt_id": "attempt-a",
                "overall_score": 95,
                "player_evaluation_available": False,
                "score_provenance": {
                    "kind": "tracking_quality",
                    "validated_player_score": False,
                },
            },
            warnings=[],
            report_status="PENDING",
            report=None,
            ai_report=None,
        )
        authoritative_attempt_b = SimpleNamespace(
            id=cached_attempt_a.id,
            status="QUEUED",
            progress={
                "step": "QUEUED",
                "pct": 20,
                "analysis_attempt_id": "attempt-b",
            },
            target={"analysis_attempt_id": "attempt-b"},
            result={"analysis_attempt_id": "attempt-b"},
            warnings=[],
            report_status="PENDING",
            report=None,
            ai_report=None,
        )

        class ScalarResult:
            def scalar_one_or_none(self):
                return cached_attempt_a

        class StaleSession:
            def __init__(self, retry_session):
                self.retry_session = retry_session
                self.commits = 0
                self.adds = 0
                self.closed = False
                self.lock_options = []
                self.locked = []

            def execute(self, statement):
                options = dict(statement.get_execution_options())
                self.lock_options.append(options)
                self.locked.append(
                    getattr(statement, "_for_update_arg", None) is not None
                )
                if options.get("populate_existing"):
                    authoritative_job = self.retry_session.job
                    for field in (
                        "status",
                        "progress",
                        "target",
                        "result",
                        "warnings",
                        "report_status",
                        "report",
                        "ai_report",
                    ):
                        setattr(
                            cached_attempt_a,
                            field,
                            copy.deepcopy(getattr(authoritative_job, field)),
                        )
                return ScalarResult()

            def add(self, _job):
                self.adds += 1

            def commit(self):
                self.commits += 1

            def close(self):
                self.closed = True

        retry_session = SimpleNamespace(job=authoritative_attempt_b)
        session = StaleSession(retry_session)
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": f"/jobs/{cached_attempt_a.id}/report",
                "headers": [],
                "query_string": b"",
            }
        )
        middleware = EvaluationReportGuardMiddleware(lambda _scope: None)
        next_calls = 0

        async def call_next(_request):
            nonlocal next_calls
            next_calls += 1
            return Response("retry-b")

        with patch(
            "app.core.evaluation_http_guard.SessionLocal",
            return_value=session,
        ):
            response = asyncio.run(middleware.dispatch(request, call_next))

        self.assertEqual(response.body, b"retry-b")
        self.assertEqual(next_calls, 1)
        self.assertTrue(session.lock_options)
        self.assertTrue(
            all(options.get("populate_existing") for options in session.lock_options)
        )
        self.assertTrue(all(session.locked))
        self.assertEqual(
            cached_attempt_a.target["analysis_attempt_id"],
            "attempt-b",
        )
        self.assertEqual(
            cached_attempt_a.result,
            {"analysis_attempt_id": "attempt-b"},
        )
        self.assertEqual(session.adds, 0)
        self.assertEqual(session.commits, 0)
        self.assertTrue(session.closed)

    def test_post_commits_unavailable_report_while_holding_current_row(self):
        job = SimpleNamespace(
            id="job-report-current",
            status="COMPLETED",
            progress={"step": "DONE", "pct": 100},
            target={"analysis_attempt_id": "attempt-a"},
            result={
                "analysis_attempt_id": "attempt-a",
                "player_evaluation_available": False,
                "score_provenance": {
                    "kind": "tracking_quality",
                    "validated_player_score": False,
                },
            },
            warnings=[],
            report_status="PENDING",
            report_error=None,
            report=None,
            ai_report=None,
        )

        class ScalarResult:
            def scalar_one_or_none(self):
                return job

        class Session:
            def __init__(self):
                self.commits = 0
                self.adds = 0
                self.closed = False
                self.locked = False
                self.populate_existing = False

            def execute(self, statement):
                self.locked = getattr(statement, "_for_update_arg", None) is not None
                self.populate_existing = bool(
                    statement.get_execution_options().get("populate_existing")
                )
                return ScalarResult()

            def add(self, _job):
                self.adds += 1

            def commit(self):
                self.commits += 1

            def close(self):
                self.closed = True

        session = Session()
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": f"/jobs/{job.id}/report",
                "headers": [],
                "query_string": b"",
            }
        )
        middleware = EvaluationReportGuardMiddleware(lambda _scope: None)

        async def call_next(_request):
            raise AssertionError("POST guard should answer directly")

        with patch(
            "app.core.evaluation_http_guard.SessionLocal",
            return_value=session,
        ):
            response = asyncio.run(middleware.dispatch(request, call_next))

        payload = json.loads(response.body)
        self.assertEqual(payload["data"]["status"], "UNAVAILABLE")
        self.assertTrue(session.locked)
        self.assertTrue(session.populate_existing)
        self.assertEqual(session.adds, 1)
        self.assertEqual(session.commits, 1)
        self.assertEqual(job.report_status, "UNAVAILABLE")
        self.assertEqual(job.report["confidence"], 0.0)
        self.assertEqual(job.ai_report, job.report)
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
