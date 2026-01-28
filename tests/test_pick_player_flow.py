import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import HTTPException
from starlette.requests import Request

from app import api
from app.schemas import TargetSelectionPayload


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


class PickPlayerFlowTests(unittest.TestCase):
    def setUp(self):
        self.job = DummyJob(
            id="job-789",
            status="WAITING_FOR_SELECTION",
            category="soccer",
            role="player",
            progress={"step": "WAITING_FOR_SELECTION", "pct": 10},
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        self.session = DummySession(self.job)
        self.request = Request({"type": "http", "headers": []})

        self.job.preview_frames = [
            {
                "time_sec": 15.0,
                "key": "jobs/job-789/frames/frame_0001.jpg",
                "tracks": [
                    {
                        "track_id": 9,
                        "bbox": {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.3},
                        "tier": "PRIMARY",
                    }
                ],
            }
        ]

    def test_pick_player_creates_player_ref_and_target_draft(self):
        payload = {"frame_key": "jobs/job-789/frames/frame_0001.jpg", "track_id": 9}

        response = api.pick_player(self.job.id, self.request, payload, self.session)

        self.assertTrue(self.session.committed)
        self.assertTrue(self.session.refreshed)
        self.assertEqual(self.job.status, "WAITING_FOR_TARGET")
        self.assertEqual(self.job.progress["step"], "WAITING_FOR_TARGET_CONFIRMATION")
        self.assertEqual(self.job.player_ref["track_id"], 9)
        self.assertEqual(self.job.player_ref["selection_source"], "preview_frame_pick")
        self.assertFalse(self.job.target.get("confirmed"))
        self.assertEqual(
            self.job.target["selection"]["frame_key"],
            "jobs/job-789/frames/frame_0001.jpg",
        )
        self.assertTrue(response["ok"])

    def test_confirm_target_valid(self):
        self.job.player_ref = {"track_id": 9}
        payload = TargetSelectionPayload.model_validate(
            {
                "frame_key": "jobs/job-789/frames/frame_0001.jpg",
                "time_sec": 15.0,
                "bbox": {"x": 0.12, "y": 0.22, "w": 0.2, "h": 0.3},
            }
        )

        response = api.save_target(self.job.id, payload, self.request, self.session)

        self.assertTrue(self.job.target.get("confirmed"))
        self.assertEqual(self.job.status, "READY_TO_ENQUEUE")
        self.assertTrue(response["ok"])

    def test_confirm_target_mismatch_raises(self):
        self.job.player_ref = {"track_id": 9}
        payload = TargetSelectionPayload.model_validate(
            {
                "frame_key": "jobs/job-789/frames/frame_0001.jpg",
                "time_sec": 15.0,
                "bbox": {"x": 0.7, "y": 0.7, "w": 0.2, "h": 0.2},
            }
        )

        with self.assertRaises(HTTPException) as ctx:
            api.save_target(self.job.id, payload, self.request, self.session)

        self.assertEqual(ctx.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
