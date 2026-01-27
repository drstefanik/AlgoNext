import json
import types
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import patch

from starlette.requests import Request

from app import api
from app.schemas import TargetSelectionPayload
from app.core import scoring


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
    error: str | None = None
    failure_reason: str | None = None
    video_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DummySession:
    def __init__(self, job: DummyJob):
        self.job = job
        self.committed = False
        self.refreshed = False

    def get(self, model, job_id: str):
        if job_id == self.job.id:
            return self.job
        return None

    def commit(self):
        self.committed = True

    def refresh(self, job):
        self.refreshed = True


class EnqueueAndResultFlowTests(unittest.TestCase):
    def setUp(self):
        self.job = DummyJob(
            id="job-456",
            status="WAITING_FOR_TARGET",
            category="soccer",
            role="player",
            progress={"step": "WAITING_FOR_TARGET", "pct": 14},
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        self.session = DummySession(self.job)
        self.request = Request({"type": "http", "headers": []})
        self.request.state.request_id = "req-123"

    def test_select_target_persists_confirmed_target_with_frame_key(self):
        self.job.preview_frames = [
            {
                "time_sec": 10.0,
                "key": "jobs/job-456/frames/frame_0001.jpg",
            }
        ]
        payload = TargetSelectionPayload.model_validate(
            {
                "time_sec": 10.05,
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            }
        )

        response = api.select_target(self.job.id, self.request, payload, self.session)

        self.assertTrue(self.session.committed)
        self.assertTrue(self.session.refreshed)
        self.assertEqual(self.job.status, "READY_TO_ENQUEUE")
        self.assertTrue(self.job.target.get("confirmed"))
        selection = self.job.target["selection"]
        self.assertEqual(
            selection["frame_key"], "jobs/job-456/frames/frame_0001.jpg"
        )
        self.assertTrue(response["data"]["target"]["confirmed"])

    def test_enqueue_returns_not_ready_for_missing_fields(self):
        response = api.enqueue_job(self.job.id, self.request, None, self.session)

        self.assertEqual(response.status_code, 400)
        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(payload["error"]["code"], "NOT_READY")
        self.assertEqual(payload["error"]["missing"], ["player_ref", "target"])
        self.assertEqual(payload["request_id"], "req-123")

    def test_enqueue_queues_when_ready(self):
        self.job.player_ref = {"track_id": 8}
        self.job.target = {"confirmed": True}

        dummy_module = types.SimpleNamespace(
            run_analysis=types.SimpleNamespace(delay=lambda *args, **kwargs: None)
        )
        with patch.dict("sys.modules", {"app.workers.pipeline": dummy_module}):
            response = api.enqueue_job(self.job.id, self.request, None, self.session)

        self.assertTrue(self.session.committed)
        self.assertEqual(self.job.status, "QUEUED")
        self.assertTrue(response["ok"])

    def test_compute_evaluation_returns_scores(self):
        tracking = {"coverage_pct": 72.5, "lost_segments": [{"start": 1, "end": 2}]}

        evaluation = scoring.compute_evaluation("player", {}, tracking)

        self.assertIsNotNone(evaluation["overall_score"])
        self.assertIsNotNone(evaluation["role_score"])
        self.assertTrue(evaluation["radar"])


if __name__ == "__main__":
    unittest.main()
