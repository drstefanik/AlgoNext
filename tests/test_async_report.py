import copy
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import patch

from starlette.requests import Request

from app import api
from app.workers import ai_report as ai_report_worker


@dataclass
class DummyJob:
    id: str
    status: str
    category: str
    role: str
    target: dict = field(default_factory=dict)
    anchor: dict = field(default_factory=dict)
    player_ref: dict | None = field(default_factory=dict)
    progress: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    preview_frames: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    ai_report: dict | None = None
    report: dict | None = None
    report_status: str = "PENDING"
    report_error: str | None = None
    error: str | None = None
    failure_reason: str | None = None
    video_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DummySession:
    def __init__(self, job: DummyJob):
        self.job = job
        self.committed = False
        self.rollbacks = 0

    def get(self, model, job_id: str):
        if job_id == self.job.id:
            return self.job
        return None

    def add(self, _):
        return None

    def commit(self):
        self.committed = True

    def refresh(self, _):
        return None

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        return None


class AsyncReportTests(unittest.TestCase):
    def setUp(self):
        self.job = DummyJob(
            id="job-rpt-1",
            status="COMPLETED",
            category="soccer",
            role="player",
            target={"analysis_attempt_id": "attempt-a"},
            progress={"step": "DONE", "pct": 100},
            result={
                "analysis_attempt_id": "attempt-a",
                "player_evaluation_available": True,
                "score_provenance": {
                    "kind": "player_evaluation",
                    "validated_player_score": True,
                },
                "assets": {
                    "clips": [
                        {
                            "url": "https://example.com/clip1.mp4",
                            "start_sec": 1.0,
                            "end_sec": 4.0,
                        }
                    ]
                },
            },
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        self.session = DummySession(self.job)
        self.request = Request(
            {
                "type": "http",
                "headers": [
                    (b"x-analysis-attempt-id", b"attempt-a"),
                ],
            }
        )
        self.request.state.request_id = "req-rpt"

    def test_post_report_enqueues_task_and_sets_pending(self):
        with patch(
            "app.api._build_ai_report_payload",
            return_value={"clips": [{"url": "https://example.com/c.mp4"}]},
        ), patch("app.workers.ai_report.generate_report") as mock_task:
            mock_task.delay.return_value = None
            payload = api.enqueue_job_report(self.job.id, self.request, 1, self.session)

        self.assertTrue(self.session.committed)
        self.assertEqual(self.job.report_status, "PENDING")
        self.assertEqual(payload["data"]["status"], "PENDING")
        self.assertEqual(payload["data"]["analysis_attempt_id"], "attempt-a")
        mock_task.delay.assert_called_once_with(
            self.job.id,
            "attempt-a",
            force=True,
        )

    def test_get_report_done_includes_payload(self):
        self.job.report_status = "DONE"
        self.job.report = {"summary": "Good impact"}

        payload = api.get_job_report(self.job.id, self.request, self.session)

        self.assertEqual(payload["data"]["status"], "DONE")
        self.assertEqual(payload["data"]["report"]["summary"], "Good impact")

    def test_worker_marks_done_and_persists_report(self):
        expected = {
            "summary": "Report",
            "strengths": ["x"],
            "risks": ["y"],
            "key_moments": [],
            "training_plan_14_days": ["d1"],
            "limitations": ["z"],
            "confidence": 0.6,
        }

        with patch(
            "app.workers.ai_report.SessionLocal", return_value=self.session
        ), patch(
            "app.workers.ai_report._build_ai_report_payload",
            return_value={"clips": [{"url": "https://example.com/c.mp4"}]},
        ), patch(
            "app.workers.ai_report.generate_ai_report", return_value=(expected, None)
        ):
            ai_report_worker._generate_report_impl(
                self.job.id,
                expected_analysis_attempt_id="attempt-a",
                force=True,
            )

        self.assertEqual(self.job.report_status, "DONE")
        self.assertEqual(self.job.report, expected)

    def test_worker_abstains_when_player_evaluation_is_not_validated(self):
        self.job.result = {
            "player_evaluation_available": False,
            "score_provenance": {
                "kind": "tracking_quality",
                "validated_player_score": False,
            },
            "limitations": ["Identity not verified across shots."],
            "assets": {
                "clips": [
                    {
                        "url": "https://example.com/clip1.mp4",
                        "start_sec": 1.0,
                        "end_sec": 4.0,
                    }
                ]
            },
        }

        with patch(
            "app.workers.ai_report.SessionLocal", return_value=self.session
        ), patch("app.workers.ai_report.generate_ai_report") as generate_mock:
            ai_report_worker._generate_report_impl(
                self.job.id,
                expected_analysis_attempt_id="attempt-a",
                force=True,
            )

        generate_mock.assert_not_called()
        self.assertEqual(self.job.report_status, "UNAVAILABLE")
        self.assertEqual(self.job.report["confidence"], 0.0)
        self.assertIn("Identity not verified", self.job.report["limitations"][0])

    def test_report_from_attempt_a_cannot_overwrite_attempt_b_after_provider(self):
        attempt_b_result = {
            "analysis_attempt_id": "attempt-b",
            "assets": {"input_video": {"key": "jobs/job-rpt-1/input.mp4"}},
        }

        def rotate_attempt(_payload):
            self.job.target = {"analysis_attempt_id": "attempt-b"}
            self.job.result = attempt_b_result
            self.job.report_status = "PENDING"
            self.job.report = None
            self.job.ai_report = None
            return ({"summary": "stale attempt A"}, None)

        with patch(
            "app.workers.ai_report.SessionLocal", return_value=self.session
        ), patch(
            "app.workers.ai_report._build_ai_report_payload",
            return_value={"clips": [{"url": "https://example.com/c.mp4"}]},
        ), patch(
            "app.workers.ai_report.generate_ai_report",
            side_effect=rotate_attempt,
        ):
            ai_report_worker._generate_report_impl(
                self.job.id,
                expected_analysis_attempt_id="attempt-a",
                force=True,
            )

        self.assertEqual(self.job.target["analysis_attempt_id"], "attempt-b")
        self.assertEqual(self.job.result, attempt_b_result)
        self.assertEqual(self.job.report_status, "PENDING")
        self.assertIsNone(self.job.report)
        self.assertIsNone(self.job.ai_report)
        self.assertEqual(self.session.rollbacks, 1)

    def test_attempt_change_before_first_report_write_is_refreshed_under_lock(self):
        class Result:
            def __init__(self, job):
                self.job = job

            def scalar_one_or_none(self):
                return self.job

        class RefreshingSession(DummySession):
            def __init__(self, job):
                super().__init__(job)
                self.authoritative_target = copy.deepcopy(job.target)
                self.authoritative_result = copy.deepcopy(job.result)
                self.authoritative_report_status = job.report_status
                self.lock_options = []

            def get(self, _model, job_id, **_kwargs):
                if job_id == self.job.id:
                    return self.job
                return None

            def execute(self, statement):
                options = dict(statement.get_execution_options())
                self.lock_options.append(options)
                if options.get("populate_existing"):
                    self.job.target = copy.deepcopy(self.authoritative_target)
                    self.job.result = copy.deepcopy(self.authoritative_result)
                    self.job.report_status = self.authoritative_report_status
                    self.job.report = None
                    self.job.ai_report = None
                return Result(self.job)

        session = RefreshingSession(self.job)

        def rotate_before_running(_job):
            session.authoritative_target = {"analysis_attempt_id": "attempt-b"}
            session.authoritative_result = {
                "analysis_attempt_id": "attempt-b",
                "assets": {"input_video": {"key": "jobs/job-rpt-1/input.mp4"}},
            }
            session.authoritative_report_status = "PENDING"
            return {"clips": [{"url": "https://example.com/c.mp4"}]}

        with patch("app.workers.ai_report.SessionLocal", return_value=session), patch(
            "app.workers.ai_report._build_ai_report_payload",
            side_effect=rotate_before_running,
        ), patch("app.workers.ai_report.generate_ai_report") as generate_mock:
            ai_report_worker._generate_report_impl(
                self.job.id,
                expected_analysis_attempt_id="attempt-a",
                force=True,
            )

        generate_mock.assert_not_called()
        self.assertTrue(session.lock_options)
        self.assertTrue(
            all(options.get("populate_existing") for options in session.lock_options)
        )
        self.assertEqual(self.job.target["analysis_attempt_id"], "attempt-b")
        self.assertEqual(self.job.result["analysis_attempt_id"], "attempt-b")
        self.assertEqual(self.job.report_status, "PENDING")
        self.assertIsNone(self.job.report)
        self.assertIsNone(self.job.ai_report)
        self.assertEqual(session.rollbacks, 1)

    def test_legacy_report_task_cannot_adopt_nonce_bound_job(self):
        with patch(
            "app.workers.ai_report.SessionLocal", return_value=self.session
        ), patch("app.workers.ai_report.generate_ai_report") as generate_mock:
            ai_report_worker._generate_report_impl(
                self.job.id,
                expected_analysis_attempt_id=None,
                force=True,
            )

        generate_mock.assert_not_called()
        self.assertEqual(self.job.report_status, "PENDING")
        self.assertEqual(self.session.rollbacks, 1)


if __name__ == "__main__":
    unittest.main()
