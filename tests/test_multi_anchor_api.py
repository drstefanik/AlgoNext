import types
import unittest
from dataclasses import dataclass, field
from unittest.mock import Mock, patch

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from app import api
from app.schemas import SelectionPayload


def _selection(index: int) -> dict:
    return {
        "frame_key": f"jobs/job-multi/frames/frame_{index + 1:04d}.jpg",
        "frame_time_sec": float(index * 30),
        "x": 0.10 + index * 0.01,
        "y": 0.20,
        "w": 0.10,
        "h": 0.25,
    }


@dataclass
class DummyJob:
    id: str = "job-multi"
    status: str = "WAITING_FOR_TARGET"
    category: str = "soccer"
    role: str = "player"
    target: dict = field(
        default_factory=lambda: {
            "full_match_mode": True,
            "player": {
                "player_name": "Mario Rossi",
                "team_name": "AS Roma",
                "shirt_number": 10,
            },
            "metadata": {"source": "full-match-create"},
            "confirmed": False,
            "selection": {"frame_key": "stale.jpg"},
            "selections": [],
            "tracking": {"status": "RUNNING", "stale": True},
        }
    )
    anchor: dict = field(default_factory=dict)
    player_ref: dict = field(
        default_factory=lambda: {
            "track_id": 9,
            "t": 0.0,
            "x": 0.10,
            "y": 0.20,
            "w": 0.10,
            "h": 0.25,
        }
    )
    progress: dict = field(
        default_factory=lambda: {"step": "WAITING_FOR_TARGET", "pct": 14}
    )
    result: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    error: str | None = None
    failure_reason: str | None = None
    preview_frames: list = field(
        default_factory=lambda: [
            {
                "key": f"jobs/job-multi/frames/frame_{index + 1:04d}.jpg",
                "time_sec": float(index * 30),
            }
            for index in range(5)
        ]
    )


class DummySession:
    def __init__(self, job: DummyJob):
        self.job = job
        self.commits = 0
        self.refreshes = 0

    def get(self, _model, job_id: str):
        return self.job if job_id == self.job.id else None

    def commit(self):
        self.commits += 1

    def refresh(self, _job):
        self.refreshes += 1


class MultiAnchorApiTests(unittest.TestCase):
    def setUp(self):
        self.request = Request({"type": "http", "headers": []})
        self.request.state.request_id = "req-multi"

    def test_selection_accepts_one_through_five_and_preserves_job_metadata(self):
        for count in range(1, 6):
            with self.subTest(count=count):
                job = DummyJob()
                session = DummySession(job)
                payload = SelectionPayload.model_validate(
                    {"selections": [_selection(index) for index in range(count)]}
                )
                enqueue = Mock()
                pipeline_module = types.SimpleNamespace(
                    run_analysis=types.SimpleNamespace(delay=enqueue)
                )

                with patch.dict(
                    "sys.modules",
                    {"app.workers.pipeline": pipeline_module},
                ):
                    response = api.save_selection(
                        job.id,
                        payload,
                        self.request,
                        session,
                    )

                self.assertTrue(response["ok"])
                self.assertEqual(job.status, "READY_TO_ENQUEUE")
                self.assertEqual(session.commits, 1)
                self.assertEqual(session.refreshes, 1)
                enqueue.assert_not_called()

                self.assertTrue(job.target["confirmed"])
                self.assertTrue(job.target["full_match_mode"])
                self.assertEqual(
                    job.target["player"]["player_name"],
                    "Mario Rossi",
                )
                self.assertEqual(
                    job.target["metadata"],
                    {"source": "full-match-create"},
                )
                self.assertEqual(
                    job.target["tracking"],
                    {"status": "PENDING"},
                )
                self.assertEqual(len(job.target["selections"]), count)
                self.assertEqual(
                    job.target["selections"][0]["frame_time_sec"],
                    0.0,
                )
                self.assertEqual(job.target["selection"]["time_sec"], 0.0)
                self.assertEqual(
                    job.target["selection"]["frame_key"],
                    "jobs/job-multi/frames/frame_0001.jpg",
                )

    def test_selection_rejects_unknown_preview_frame(self):
        job = DummyJob()
        session = DummySession(job)
        selection = _selection(0)
        selection["frame_key"] = "jobs/job-multi/frames/not-present.jpg"
        payload = SelectionPayload.model_validate({"selections": [selection]})

        with self.assertRaises(HTTPException) as raised:
            api.save_selection(job.id, payload, self.request, session)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(
            raised.exception.detail["error"]["code"],
            "INVALID_SELECTION",
        )
        self.assertEqual(session.commits, 0)

    def test_selection_rejects_duplicate_preview_frame(self):
        job = DummyJob()
        session = DummySession(job)
        duplicate = _selection(0)
        duplicate["frame_time_sec"] = 30.0
        payload = SelectionPayload.model_validate(
            {"selections": [_selection(0), duplicate]}
        )

        with self.assertRaises(HTTPException) as raised:
            api.save_selection(job.id, payload, self.request, session)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(
            raised.exception.detail["error"]["code"],
            "DUPLICATE_SELECTION_FRAME",
        )
        self.assertEqual(session.commits, 0)

    def test_selection_uses_canonical_preview_timestamp_for_frame_key(self):
        job = DummyJob()
        session = DummySession(job)
        selection = _selection(0)
        selection["frame_time_sec"] = 30.0
        payload = SelectionPayload.model_validate({"selections": [selection]})

        api.save_selection(job.id, payload, self.request, session)

        self.assertEqual(
            job.target["selections"][0]["frame_key"],
            "jobs/job-multi/frames/frame_0001.jpg",
        )
        self.assertEqual(
            job.target["selections"][0]["frame_time_sec"],
            0.0,
        )
        self.assertEqual(job.target["selection"]["time_sec"], 0.0)

    def test_selection_schema_rejects_zero_or_more_than_five_anchors(self):
        with self.assertRaises(ValidationError):
            SelectionPayload.model_validate({"selections": []})
        with self.assertRaises(ValidationError):
            SelectionPayload.model_validate(
                {"selections": [_selection(index) for index in range(6)]}
            )

    def test_new_player_pick_clears_failed_tracking_attempt(self):
        job = DummyJob(
            status="WAITING_FOR_PLAYER",
            result={
                "candidates": {"candidates": [{"track_id": 9}]},
                "tracking": {
                    "tracking_success": False,
                    "tracking_status": "ANCHOR_NOT_FOUND",
                    "action_required": "RESELECT_PLAYER",
                },
                "tracking_quality_index": 11.3,
                "report": {"stale": True},
            },
            warnings=[
                "PLAYER_ANCHOR_NOT_FOUND",
                "PLAYER_RESELECTION_REQUIRED",
            ],
            failure_reason="PLAYER_RESELECTION_REQUIRED",
        )

        reset = api._reset_failed_tracking_attempt(job)

        self.assertTrue(reset)
        self.assertEqual(
            job.result,
            {"candidates": {"candidates": [{"track_id": 9}]}},
        )
        self.assertEqual(job.warnings, [])
        self.assertIsNone(job.error)
        self.assertIsNone(job.failure_reason)
        self.assertNotIn("tracking", job.target)

    def test_acquisition_error_cannot_be_converted_to_reselection(self):
        job = DummyJob(
            status="FAILED",
            result={
                "tracking": {
                    "tracking_success": False,
                    "tracking_status": "ANCHOR_ACQUISITION_ERROR",
                    "action_required": "RETRY_ANALYSIS",
                    "reid_summary": {
                        "status": "ANCHOR_ACQUISITION_ERROR",
                        "reason_codes": ["REID_ANCHOR_ACQUISITION_ERROR"],
                    },
                },
            },
            warnings=["PLAYER_ANCHOR_ACQUISITION_FAILED"],
            failure_reason="PLAYER_ANCHOR_ACQUISITION_FAILED",
        )
        original_result = job.result

        reset = api._reset_failed_tracking_attempt(job)

        self.assertFalse(reset)
        self.assertIs(job.result, original_result)
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(
            job.failure_reason,
            "PLAYER_ANCHOR_ACQUISITION_FAILED",
        )


if __name__ == "__main__":
    unittest.main()
