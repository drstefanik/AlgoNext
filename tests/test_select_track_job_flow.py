import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import patch

from starlette.requests import Request

from app import api


class DummyS3Public:
    def generate_presigned_url(self, *args, **kwargs) -> str:
        return "https://example.com/signed.jpg"


@dataclass
class DummyJob:
    id: str
    status: str
    category: str
    role: str
    video_url: str | None = None
    video_bucket: str | None = None
    video_key: str | None = None
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

    def close(self):
        return None


class SelectTrackJobFlowTests(unittest.TestCase):
    def setUp(self):
        self.job = DummyJob(
            id="job-123",
            status="WAITING_FOR_SELECTION",
            category="soccer",
            role="player",
            video_url="https://example.com/video.mp4",
            progress={"step": "WAITING_FOR_SELECTION", "pct": 10},
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        self.session = DummySession(self.job)
        self.request = Request({"type": "http", "headers": []})

    def test_select_track_persists_player_ref_and_status(self):
        candidate = {
            "track_id": 8,
            "tier": "PRIMARY",
            "coverage_pct": 0.75,
            "sample_frames": [
                {
                    "key": "jobs/abc/frame.jpg",
                    "bucket": "bucket",
                    "time_sec": 1.5,
                    "bbox": {"x": 0.2, "y": 0.3, "w": 0.1, "h": 0.2},
                }
            ],
        }
        self.job.result = {"candidates": {"candidates": [candidate]}}

        payload = {
            "trackId": 8,
            "selection": {
                "frame_time_sec": 3954.517,
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            },
        }

        response = api.select_track(self.job.id, self.request, payload, self.session)

        self.assertTrue(self.session.committed)
        self.assertTrue(self.session.refreshed)
        self.assertEqual(response["data"]["status"], "WAITING_FOR_TARGET")
        self.assertEqual(self.job.status, "WAITING_FOR_TARGET")
        self.assertEqual(self.job.progress["step"], "WAITING_FOR_TARGET")
        self.assertIsNotNone(self.job.player_ref)
        self.assertEqual(self.job.player_ref["track_id"], 8)
        self.assertEqual(self.job.player_ref["tier"], "PRIMARY")
        self.assertEqual(self.job.target, {})
        self.assertNotEqual(
            self.job.updated_at, datetime(2024, 1, 1, tzinfo=timezone.utc)
        )

    def test_get_job_populates_player_ref_and_input_video_url(self):
        self.job.player_ref = {
            "track_id": 8,
            "tier": "PRIMARY",
            "sample_frames": [
                {
                    "key": "jobs/abc/frame.jpg",
                    "bucket": "bucket",
                    "time_sec": 1.5,
                }
            ],
        }
        self.job.status = "WAITING_FOR_TARGET"
        self.job.progress = {"step": "WAITING_FOR_TARGET", "pct": 14}

        context = {
            "s3_public": DummyS3Public(),
            "bucket": "bucket",
            "expires_seconds": 3600,
        }

        with patch.object(api, "SessionLocal", return_value=self.session), patch.object(
            api, "load_s3_context", return_value=context
        ):
            response = api.get_job(self.job.id, self.request)

        payload = response["data"]
        self.assertEqual(payload["status"], "WAITING_FOR_TARGET")
        self.assertIsNotNone(payload["player_ref"])
        self.assertEqual(payload["player_ref"]["track_id"], 8)
        self.assertTrue(payload["playerSaved"])
        self.assertFalse(payload["targetSaved"])
        sample_frame = payload["player_ref"]["sample_frames"][0]
        self.assertIn("signed_url", sample_frame)
        self.assertEqual(payload["assets"]["inputVideoUrl"], self.job.video_url)


if __name__ == "__main__":
    unittest.main()
