import asyncio
import copy
import json
import types
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from app import api
from app.schemas import SelectionPayload, TargetSelectionPayload
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
    ai_report: dict | None = None
    report: dict | None = None
    report_status: str = "PENDING"
    report_error: str | None = None
    video_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DummySession:
    def __init__(self, job: DummyJob):
        self.job = job
        self.committed = False
        self.refreshed = False
        self.rollbacks = 0
        self.expirations = 0

    def get(self, model, job_id: str):
        if job_id == self.job.id:
            return self.job
        return None

    def commit(self):
        self.committed = True

    def refresh(self, job):
        self.refreshed = True

    def rollback(self):
        self.rollbacks += 1

    def expire_all(self):
        self.expirations += 1


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
        self.request = self.request_for_attempt()

    def request_for_attempt(self, analysis_attempt_id: str | None = None):
        headers = (
            [(b"x-analysis-attempt-id", analysis_attempt_id.encode())]
            if analysis_attempt_id
            else []
        )
        request = Request({"type": "http", "headers": headers})
        request.state.request_id = "req-123"
        return request

    def test_select_target_persists_confirmed_target_with_frame_key(self):
        self.job.preview_frames = [
            {
                "time_sec": 10.0,
                "key": "jobs/job-456/frames/frame_0001.jpg",
                "tracks": [
                    {
                        "track_id": 0,
                        "bbox": {
                            "x": 0.1,
                            "y": 0.1,
                            "w": 0.2,
                            "h": 0.2,
                        },
                    }
                ],
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
        self.assertEqual(selection["frame_key"], "jobs/job-456/frames/frame_0001.jpg")
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
        self.job.warnings = ["STALE_WARNING"]
        self.job.failure_reason = "STALE_FAILURE"
        self.job.result = {
            "tracking": {"tracking_success": True},
            "overall_score": 99,
            "report": {"stale": True},
            "assets": {
                "input_video": {"bucket": "fnh", "key": "jobs/job-456/input.mp4"},
                "clips": [{"key": "stale.mp4"}],
            },
        }
        self.job.ai_report = {"stale": True}
        self.job.report = {"stale": True}
        self.job.report_status = "DONE"
        self.job.report_error = "stale"

        delay_calls = []
        dummy_module = types.SimpleNamespace(
            run_analysis=types.SimpleNamespace(
                delay=lambda *args, **kwargs: delay_calls.append((args, kwargs))
            )
        )
        with patch.dict("sys.modules", {"app.workers.pipeline": dummy_module}):
            response = api.enqueue_job(self.job.id, self.request, None, self.session)

        self.assertTrue(self.session.committed)
        self.assertEqual(self.job.status, "QUEUED")
        self.assertTrue(response["ok"])
        attempt_id = response["data"]["analysis_attempt_id"]
        self.assertIsInstance(attempt_id, str)
        self.assertEqual(self.job.target["analysis_attempt_id"], attempt_id)
        self.assertEqual(
            self.job.target["tracking"],
            {"status": "PENDING", "analysis_attempt_id": attempt_id},
        )
        self.assertEqual(
            self.job.result,
            {
                "analysis_attempt_id": attempt_id,
                "assets": {
                    "input_video": {
                        "bucket": "fnh",
                        "key": "jobs/job-456/input.mp4",
                    }
                },
            },
        )
        self.assertEqual(self.job.warnings, [])
        self.assertIsNone(self.job.failure_reason)
        self.assertIsNone(self.job.ai_report)
        self.assertIsNone(self.job.report)
        self.assertEqual(self.job.report_status, "PENDING")
        self.assertIsNone(self.job.report_error)
        self.assertEqual(
            delay_calls,
            [((self.job.id, attempt_id), {})],
        )

    def test_enqueue_reconciles_pre_delivery_and_claimed_dispatch_errors(self):
        for claimed in (False, True):
            with self.subTest(claimed=claimed):
                self.job.player_ref = {"track_id": 8}
                self.job.target = {"confirmed": True}
                self.job.status = "READY_TO_ENQUEUE"
                self.job.progress = {"step": "READY_TO_ENQUEUE", "pct": 16}
                self.session = DummySession(self.job)

                def dispatch_then_raise(job_id, analysis_attempt_id):
                    if claimed:
                        self.job.status = "RUNNING"
                        self.job.progress = {
                            **self.job.progress,
                            "analysis_attempt_id": analysis_attempt_id,
                            "analysis_task_id": "celery-task-b",
                        }
                    raise RuntimeError("publish confirmation lost")

                dummy_module = types.SimpleNamespace(
                    run_analysis=types.SimpleNamespace(delay=dispatch_then_raise)
                )
                with patch.dict(
                    "sys.modules",
                    {"app.workers.pipeline": dummy_module},
                ):
                    if claimed:
                        response = api.enqueue_job(
                            self.job.id,
                            self.request,
                            None,
                            self.session,
                        )
                    else:
                        with self.assertRaises(HTTPException) as raised:
                            api.enqueue_job(
                                self.job.id,
                                self.request,
                                None,
                                self.session,
                            )

                self.assertEqual(self.session.rollbacks, 1)
                self.assertEqual(self.session.expirations, 1)
                if claimed:
                    self.assertTrue(response["data"]["dispatch_ambiguous"])
                    self.assertEqual(self.job.status, "RUNNING")
                    self.assertEqual(
                        self.job.progress["analysis_task_id"],
                        "celery-task-b",
                    )
                else:
                    self.assertEqual(raised.exception.status_code, 503)
                    self.assertEqual(
                        raised.exception.detail["error"]["code"],
                        "ANALYSIS_ENQUEUE_FAILED",
                    )
                    self.assertEqual(self.job.status, "FAILED")
                    self.assertEqual(
                        self.job.failure_reason,
                        "ANALYSIS_ENQUEUE_FAILED",
                    )

    def test_enqueue_rotates_attempt_only_for_a_new_allowed_attempt(self):
        self.job.player_ref = {"track_id": 8}
        self.job.target = {"confirmed": True}
        delay_calls = []
        dummy_module = types.SimpleNamespace(
            run_analysis=types.SimpleNamespace(
                delay=lambda *args, **kwargs: delay_calls.append((args, kwargs))
            )
        )
        with patch.dict("sys.modules", {"app.workers.pipeline": dummy_module}):
            first = api.enqueue_job(self.job.id, self.request, None, self.session)
            first_attempt = first["data"]["analysis_attempt_id"]

            self.request = self.request_for_attempt(first_attempt)
            active = api.enqueue_job(self.job.id, self.request, None, self.session)
            self.assertEqual(
                active["data"]["analysis_attempt_id"],
                first_attempt,
            )

            self.job.status = "PARTIAL"
            self.job.result["tracking"] = {"tracking_success": True}
            second = api.enqueue_job(self.job.id, self.request, None, self.session)

        second_attempt = second["data"]["analysis_attempt_id"]
        self.assertNotEqual(second_attempt, first_attempt)
        self.assertEqual(self.job.target["analysis_attempt_id"], second_attempt)
        self.assertNotIn("tracking", self.job.result)
        self.assertEqual(
            self.job.result["analysis_attempt_id"],
            second_attempt,
        )
        self.assertEqual(
            delay_calls,
            [
                ((self.job.id, first_attempt), {}),
                ((self.job.id, second_attempt), {}),
            ],
        )

    def test_active_analysis_rejects_every_selection_mutator_without_changes(self):
        selection = SelectionPayload.model_validate(
            {
                "selections": [
                    {
                        "frame_key": "jobs/job-456/frames/frame_0001.jpg",
                        "frame_time_sec": 10.0,
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.2,
                        "h": 0.2,
                    }
                ]
            }
        )
        target = {
            "frame_key": "jobs/job-456/frames/frame_0001.jpg",
            "time_sec": 10.0,
            "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            "track_id": 9,
        }
        self.job.preview_frames = [
            {
                "time_sec": 10.0,
                "key": "jobs/job-456/frames/frame_0001.jpg",
                "tracks": [
                    {
                        "track_id": 9,
                        "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                    }
                ],
            }
        ]
        self.job.player_ref = {"track_id": 8}
        self.job.target = {
            "confirmed": True,
            "analysis_attempt_id": "attempt-a",
            "tracking": {
                "status": "RUNNING",
                "analysis_attempt_id": "attempt-a",
            },
        }
        self.request = self.request_for_attempt("attempt-a")

        for status in ("QUEUED", "RUNNING", "PROCESSING"):
            for name, invoke in (
                (
                    "select-track",
                    lambda: api.select_track(
                        self.job.id,
                        self.request,
                        {"track_id": 9},
                        self.session,
                    ),
                ),
                (
                    "pick-player",
                    lambda: api.pick_player(
                        self.job.id,
                        self.request,
                        {
                            "frame_key": "jobs/job-456/frames/frame_0001.jpg",
                            "track_id": 9,
                        },
                        self.session,
                    ),
                ),
                (
                    "analyze-player-different-track",
                    lambda: api.analyze_player(
                        self.job.id,
                        self.request,
                        {
                            "frame_key": "jobs/job-456/frames/frame_0001.jpg",
                            "track_id": 9,
                        },
                        self.session,
                    ),
                ),
                (
                    "selection",
                    lambda: api.save_selection(
                        self.job.id,
                        selection,
                        self.request,
                        self.session,
                    ),
                ),
                (
                    "select-target",
                    lambda: api.select_target(
                        self.job.id,
                        self.request,
                        target,
                        self.session,
                    ),
                ),
                (
                    "target",
                    lambda: api.save_target(
                        self.job.id,
                        self.request,
                        target,
                        self.session,
                    ),
                ),
                (
                    "player-ref",
                    lambda: asyncio.run(
                        api.save_player_ref(
                            self.job.id,
                            self.request,
                            {
                                "frame_time_sec": 10.0,
                                "bbox_xywh": {
                                    "x": 0.1,
                                    "y": 0.1,
                                    "w": 0.2,
                                    "h": 0.2,
                                },
                            },
                            self.session,
                        )
                    ),
                ),
            ):
                with self.subTest(status=status, endpoint=name):
                    self.job.status = status
                    self.session.committed = False
                    self.session.refreshed = False
                    self.request._body = b""
                    before = copy.deepcopy(self.job.__dict__)

                    with self.assertRaises(HTTPException) as raised:
                        invoke()

                    self.assertEqual(raised.exception.status_code, 409)
                    self.assertEqual(
                        raised.exception.detail["error"]["code"],
                        "ANALYSIS_IN_PROGRESS",
                    )
                    self.assertEqual(
                        raised.exception.detail["error"]["details"],
                        {
                            "status": status,
                            "analysis_attempt_id": "attempt-a",
                        },
                    )
                    self.assertEqual(self.job.__dict__, before)
                    self.assertFalse(self.session.committed)
                    self.assertFalse(self.session.refreshed)

    def test_processing_enqueue_is_idempotent_and_does_not_rotate_attempt(self):
        self.job.status = "PROCESSING"
        self.job.player_ref = {"track_id": 8}
        self.job.target = {
            "confirmed": True,
            "analysis_attempt_id": "attempt-a",
            "full_match_mode": True,
            "player": {"player_name": "Mario Rossi", "team": "Blue"},
            "metadata": {"source": "full-match-create"},
        }
        self.request = self.request_for_attempt("attempt-a")
        before = copy.deepcopy(self.job.__dict__)

        response = api.enqueue_job(
            self.job.id,
            self.request,
            None,
            self.session,
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["data"]["analysis_attempt_id"], "attempt-a")
        self.assertEqual(self.job.__dict__, before)
        self.assertFalse(self.session.committed)

    def test_stale_attempt_cannot_mutate_newer_terminal_job(self):
        selection = SelectionPayload.model_validate(
            {
                "selections": [
                    {
                        "frame_key": "jobs/job-456/frames/frame_0001.jpg",
                        "frame_time_sec": 10.0,
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.2,
                        "h": 0.2,
                    }
                ]
            }
        )
        target = {
            "frame_key": "jobs/job-456/frames/frame_0001.jpg",
            "time_sec": 10.0,
            "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            "track_id": 9,
        }
        self.job.status = "PARTIAL"
        self.job.player_ref = {"track_id": 8}
        self.job.target = {
            "confirmed": True,
            "analysis_attempt_id": "attempt-b",
            "tracking": {
                "status": "PARTIAL",
                "analysis_attempt_id": "attempt-b",
            },
        }
        self.job.progress = {
            "step": "DONE",
            "pct": 100,
            "analysis_attempt_id": "attempt-b",
        }
        self.job.result = {
            "analysis_attempt_id": "attempt-b",
            "assets": {
                "clips": [{"url": "https://example.test/clip.mp4"}],
            },
        }
        self.job.preview_frames = [
            {
                "time_sec": 10.0,
                "key": "jobs/job-456/frames/frame_0001.jpg",
                "tracks": [
                    {
                        "track_id": 9,
                        "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                    }
                ],
            }
        ]

        for name, invoke in (
            (
                "select-track",
                lambda: api.select_track(
                    self.job.id,
                    self.request,
                    {"track_id": 9},
                    self.session,
                ),
            ),
            (
                "pick-player",
                lambda: api.pick_player(
                    self.job.id,
                    self.request,
                    {
                        "frame_key": "jobs/job-456/frames/frame_0001.jpg",
                        "track_id": 9,
                    },
                    self.session,
                ),
            ),
            (
                "analyze-player",
                lambda: api.analyze_player(
                    self.job.id,
                    self.request,
                    {
                        "frame_key": "jobs/job-456/frames/frame_0001.jpg",
                        "track_id": 9,
                    },
                    self.session,
                ),
            ),
            (
                "selection",
                lambda: api.save_selection(
                    self.job.id,
                    selection,
                    self.request,
                    self.session,
                ),
            ),
            (
                "select-target",
                lambda: api.select_target(
                    self.job.id,
                    self.request,
                    target,
                    self.session,
                ),
            ),
            (
                "target",
                lambda: api.save_target(
                    self.job.id,
                    self.request,
                    target,
                    self.session,
                ),
            ),
            (
                "player-ref",
                lambda: asyncio.run(
                    api.save_player_ref(
                        self.job.id,
                        self.request,
                        {
                            "frame_time_sec": 10.0,
                            "bbox_xywh": {
                                "x": 0.1,
                                "y": 0.1,
                                "w": 0.2,
                                "h": 0.2,
                            },
                        },
                        self.session,
                    )
                ),
            ),
            (
                "enqueue",
                lambda: api.enqueue_job(
                    self.job.id,
                    self.request,
                    None,
                    self.session,
                ),
            ),
            (
                "confirm-selection",
                lambda: api.confirm_selection(
                    self.job.id,
                    self.request,
                    self.session,
                ),
            ),
            (
                "report",
                lambda: api.enqueue_job_report(
                    self.job.id,
                    self.request,
                    1,
                    self.session,
                ),
            ),
            (
                "ai-report",
                lambda: api.job_ai_report(
                    self.job.id,
                    self.request,
                    0,
                    self.session,
                ),
            ),
        ):
            with self.subTest(endpoint=name):
                self.request = self.request_for_attempt("attempt-a")
                self.request._body = b""
                self.session.committed = False
                self.session.refreshed = False
                before = copy.deepcopy(self.job.__dict__)

                with self.assertRaises(HTTPException) as raised:
                    invoke()

                self.assertEqual(raised.exception.status_code, 409)
                self.assertEqual(
                    raised.exception.detail["error"]["code"],
                    "ANALYSIS_ATTEMPT_MISMATCH",
                )
                self.assertEqual(self.job.__dict__, before)
                self.assertFalse(self.session.committed)
                self.assertFalse(self.session.refreshed)

    def test_terminal_selection_fences_attempt_and_legacy_workers_before_enqueue(self):
        from app.workers.pipeline import _validate_task_analysis_attempt

        attempt_a = "attempt-a"
        self.job.status = "PARTIAL"
        self.job.player_ref = {"track_id": 8}
        self.job.target = {
            "confirmed": True,
            "analysis_attempt_id": attempt_a,
            "tracking": {
                "status": "PARTIAL",
                "analysis_attempt_id": attempt_a,
                "tracking_success": False,
            },
        }
        self.job.progress = {
            "step": "DONE",
            "pct": 100,
            "analysis_attempt_id": attempt_a,
        }
        self.job.result = {
            "analysis_attempt_id": attempt_a,
            "tracking": {"tracking_success": False},
            "assets": {
                "input_video": {
                    "bucket": "fnh",
                    "key": "jobs/job-456/input.mp4",
                },
                "clips": [{"key": "stale.mp4"}],
            },
        }
        self.job.report = {"summary": "stale"}
        self.job.ai_report = {"summary": "stale"}
        self.job.report_status = "DONE"
        self.request = self.request_for_attempt(attempt_a)
        selection = SelectionPayload.model_validate(
            {
                "selections": [
                    {
                        "frame_time_sec": 10.0,
                        "x": 0.1,
                        "y": 0.1,
                        "w": 0.2,
                        "h": 0.2,
                    }
                ]
            }
        )

        response = api.save_selection(
            self.job.id,
            selection,
            self.request,
            self.session,
        )

        selection_revision = self.job.target["analysis_attempt_id"]
        self.assertNotEqual(selection_revision, attempt_a)
        self.assertEqual(self.job.status, "READY_TO_ENQUEUE")
        self.assertEqual(
            self.job.target["tracking"]["analysis_attempt_id"],
            selection_revision,
        )
        self.assertEqual(
            self.job.progress["analysis_attempt_id"],
            selection_revision,
        )
        self.assertEqual(
            self.job.result,
            {
                "analysis_attempt_id": selection_revision,
                "assets": {
                    "input_video": {
                        "bucket": "fnh",
                        "key": "jobs/job-456/input.mp4",
                    }
                },
            },
        )
        self.assertIsNone(self.job.report)
        self.assertIsNone(self.job.ai_report)
        self.assertEqual(self.job.report_status, "PENDING")
        self.assertTrue(response["ok"])

        with self.assertRaises(RuntimeError):
            _validate_task_analysis_attempt(self.job.target, attempt_a)
        with self.assertRaises(RuntimeError):
            _validate_task_analysis_attempt(self.job.target, None)

        delay_calls = []
        dummy_module = types.SimpleNamespace(
            run_analysis=types.SimpleNamespace(
                delay=lambda *args, **kwargs: delay_calls.append((args, kwargs))
            )
        )
        with patch.dict("sys.modules", {"app.workers.pipeline": dummy_module}):
            self.request = self.request_for_attempt(selection_revision)
            enqueued = api.enqueue_job(
                self.job.id,
                self.request,
                None,
                self.session,
            )

        attempt_b = enqueued["data"]["analysis_attempt_id"]
        self.assertNotEqual(attempt_b, selection_revision)
        self.assertEqual(self.job.target["analysis_attempt_id"], attempt_b)
        self.assertEqual(delay_calls, [((self.job.id, attempt_b), {})])

    def test_analysis_task_claim_rejects_duplicate_and_accepts_newer_retry(self):
        from app.workers.pipeline import _claim_analysis_task

        self.job.status = "QUEUED"
        self.job.target = {"analysis_attempt_id": "attempt-a"}

        claimed, attempt_id = _claim_analysis_task(
            self.session,
            self.job.id,
            "attempt-a",
            task_id="celery-task-a",
            retry_number=0,
        )

        self.assertIs(claimed, self.job)
        self.assertEqual(attempt_id, "attempt-a")
        self.assertEqual(self.job.status, "RUNNING")
        self.assertEqual(self.job.progress["analysis_task_id"], "celery-task-a")
        self.assertEqual(self.job.progress["analysis_task_retry"], 0)

        with self.assertRaises(RuntimeError):
            _claim_analysis_task(
                self.session,
                self.job.id,
                "attempt-a",
                task_id="celery-task-a",
                retry_number=0,
            )
        with self.assertRaises(RuntimeError):
            _claim_analysis_task(
                self.session,
                self.job.id,
                "attempt-a",
                task_id="duplicate-task",
                retry_number=1,
            )

        retried, _ = _claim_analysis_task(
            self.session,
            self.job.id,
            "attempt-a",
            task_id="celery-task-a",
            retry_number=1,
        )
        self.assertIs(retried, self.job)
        self.assertEqual(self.job.progress["analysis_task_retry"], 1)

        with self.assertRaises(RuntimeError):
            _claim_analysis_task(
                self.session,
                self.job.id,
                "attempt-a",
                task_id="celery-task-a",
                retry_number=1,
            )

    def test_analyze_player_clears_stale_report_for_direct_enqueue(self):
        self.job.status = "PARTIAL"
        self.job.player_ref = {"track_id": 8}
        self.job.preview_frames = [
            {
                "time_sec": 10.0,
                "key": "jobs/job-456/frames/frame_0001.jpg",
                "tracks": [
                    {
                        "track_id": 9,
                        "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                    }
                ],
            }
        ]
        self.job.target = {
            "confirmed": True,
            "analysis_attempt_id": "attempt-a",
            "full_match_mode": True,
            "player": {"player_name": "Mario Rossi", "team": "Blue"},
            "metadata": {"source": "full-match-create"},
        }
        self.job.result = {
            "analysis_attempt_id": "attempt-a",
            "tracking": {"tracking_success": True},
            "assets": {
                "input_video": {
                    "bucket": "fnh",
                    "key": "jobs/job-456/input.mp4",
                }
            },
        }
        self.job.warnings = ["STALE"]
        self.job.failure_reason = "STALE"
        self.job.ai_report = {"summary": "stale"}
        self.job.report = {"summary": "stale"}
        self.job.report_status = "DONE"
        self.job.report_error = "stale"
        self.request = self.request_for_attempt("attempt-a")
        delay_calls = []
        dummy_module = types.SimpleNamespace(
            run_analysis=types.SimpleNamespace(
                delay=lambda *args, **kwargs: delay_calls.append((args, kwargs))
            )
        )

        with patch.dict("sys.modules", {"app.workers.pipeline": dummy_module}):
            response = api.analyze_player(
                self.job.id,
                self.request,
                {
                    "frame_key": "jobs/job-456/frames/frame_0001.jpg",
                    "track_id": 9,
                },
                self.session,
            )

        attempt_b = response["data"]["analysis_attempt_id"]
        self.assertEqual(self.job.status, "QUEUED")
        self.assertEqual(self.job.target["analysis_attempt_id"], attempt_b)
        self.assertTrue(self.job.target["full_match_mode"])
        self.assertEqual(
            self.job.target["player"],
            {"player_name": "Mario Rossi", "team": "Blue"},
        )
        self.assertEqual(
            self.job.target["metadata"],
            {"source": "full-match-create"},
        )
        self.assertEqual(self.job.result["analysis_attempt_id"], attempt_b)
        self.assertNotIn("tracking", self.job.result)
        self.assertEqual(self.job.warnings, [])
        self.assertIsNone(self.job.failure_reason)
        self.assertIsNone(self.job.ai_report)
        self.assertIsNone(self.job.report)
        self.assertEqual(self.job.report_status, "PENDING")
        self.assertIsNone(self.job.report_error)
        self.assertEqual(delay_calls, [((self.job.id, attempt_b), {})])

    def test_analyze_player_reconciles_dispatch_errors(self):
        for claimed in (False, True):
            with self.subTest(claimed=claimed):
                self.job.status = "PARTIAL"
                self.job.player_ref = {"track_id": 8}
                self.job.preview_frames = [
                    {
                        "time_sec": 10.0,
                        "key": "jobs/job-456/frames/frame_0001.jpg",
                        "tracks": [
                            {
                                "track_id": 9,
                                "bbox": {
                                    "x": 0.1,
                                    "y": 0.1,
                                    "w": 0.2,
                                    "h": 0.2,
                                },
                            }
                        ],
                    }
                ]
                self.job.target = {
                    "confirmed": True,
                    "analysis_attempt_id": "attempt-a",
                }
                self.job.result = {
                    "analysis_attempt_id": "attempt-a",
                    "assets": {
                        "input_video": {
                            "bucket": "fnh",
                            "key": "jobs/job-456/input.mp4",
                        }
                    },
                }
                self.job.progress = {
                    "step": "DONE",
                    "pct": 100,
                    "analysis_attempt_id": "attempt-a",
                }
                self.session = DummySession(self.job)
                self.request = self.request_for_attempt("attempt-a")

                def dispatch_then_raise(job_id, analysis_attempt_id):
                    if claimed:
                        self.job.status = "RUNNING"
                        self.job.progress = {
                            **self.job.progress,
                            "analysis_attempt_id": analysis_attempt_id,
                            "analysis_task_id": "celery-task-b",
                        }
                    raise RuntimeError("publish confirmation lost")

                dummy_module = types.SimpleNamespace(
                    run_analysis=types.SimpleNamespace(delay=dispatch_then_raise)
                )
                with patch.dict(
                    "sys.modules",
                    {"app.workers.pipeline": dummy_module},
                ):
                    if claimed:
                        response = api.analyze_player(
                            self.job.id,
                            self.request,
                            {
                                "frame_key": ("jobs/job-456/frames/frame_0001.jpg"),
                                "track_id": 9,
                            },
                            self.session,
                        )
                    else:
                        with self.assertRaises(HTTPException) as raised:
                            api.analyze_player(
                                self.job.id,
                                self.request,
                                {
                                    "frame_key": ("jobs/job-456/frames/frame_0001.jpg"),
                                    "track_id": 9,
                                },
                                self.session,
                            )

                self.assertEqual(self.session.rollbacks, 1)
                self.assertEqual(self.session.expirations, 1)
                if claimed:
                    self.assertTrue(response["data"]["dispatch_ambiguous"])
                    self.assertEqual(self.job.status, "RUNNING")
                else:
                    self.assertEqual(raised.exception.status_code, 503)
                    self.assertEqual(
                        raised.exception.detail["error"]["code"],
                        "ANALYSIS_ENQUEUE_FAILED",
                    )
                    self.assertEqual(self.job.status, "FAILED")

    def test_compute_evaluation_returns_scores(self):
        tracking = {
            "coverage_pct": 72.5,
            "lost_segments": [{"start": 1, "end": 2}],
        }
        evidence_metrics = {
            "distance_covered_m": 320.0,
            "avg_speed_kmh": 11.5,
            "top_speed_kmh": 22.3,
        }

        evaluation = scoring.compute_evaluation(
            "player", {}, tracking, evidence_metrics
        )

        self.assertIsNotNone(evaluation["overall_score"])
        self.assertIsNotNone(evaluation["role_score"])
        self.assertTrue(evaluation["radar"])


if __name__ == "__main__":
    unittest.main()
