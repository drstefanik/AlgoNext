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

    def close(self):
        return None


class AsyncReportTests(unittest.TestCase):
    def setUp(self):
        self.job = DummyJob(
            id="job-rpt-1",
            status="COMPLETED",
            category="soccer",
            role="player",
            progress={"step": "DONE", "pct": 100},
            result={
                "assets": {
                    "clips": [
                        {
                            "url": "https://example.com/clip1.mp4",
                            "start_sec": 1.0,
                            "end_sec": 4.0,
                        }
                    ]
                }
            },
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        self.session = DummySession(self.job)
        self.request = Request({"type": "http", "headers": []})
        self.request.state.request_id = "req-rpt"

    def test_post_report_enqueues_task_and_sets_pending(self):
        with patch("app.api._build_ai_report_payload", return_value={"clips": [{"url": "https://example.com/c.mp4"}]}), patch(
            "app.workers.ai_report.generate_report"
        ) as mock_task:
            mock_task.delay.return_value = None
            payload = api.enqueue_job_report(self.job.id, self.request, 1, self.session)

        self.assertTrue(self.session.committed)
        self.assertEqual(self.job.report_status, "PENDING")
        self.assertEqual(payload["data"]["status"], "PENDING")
        mock_task.delay.assert_called_once_with(self.job.id, force=True)

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

        with patch("app.workers.ai_report.SessionLocal", return_value=self.session), patch(
            "app.workers.ai_report._build_ai_report_payload",
            return_value={"clips": [{"url": "https://example.com/c.mp4"}]},
        ), patch("app.workers.ai_report.generate_ai_report", return_value=(expected, None)):
            ai_report_worker._generate_report_impl(self.job.id, force=True)

        self.assertEqual(self.job.report_status, "DONE")
        self.assertEqual(self.job.report, expected)


if __name__ == "__main__":
    unittest.main()
