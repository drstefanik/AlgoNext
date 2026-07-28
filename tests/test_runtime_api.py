import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Request

from app import runtime_api


class DummyDB:
    def __init__(self, job):
        self.job = job
        self.commits = 0
        self.rollbacks = 0
        self.expirations = 0

    def get(self, _model, _job_id, populate_existing=False):
        return self.job

    def commit(self):
        self.commits += 1

    def refresh(self, _job):
        return None

    def rollback(self):
        self.rollbacks += 1

    def expire_all(self):
        self.expirations += 1


class DelayRecorder:
    def __init__(self):
        self.calls = []

    def delay(self, *args):
        self.calls.append(args)


class RuntimeApiTests(unittest.TestCase):
    def request(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/jobs/job-1/retry",
                "headers": [],
            }
        )
        request.state.request_id = "request-1"
        return request

    def runtime_ready(self):
        return {
            "ready": True,
            "dependencies": {"redis": "ready", "worker": "ready"},
            "worker": {"revision": "sha-1"},
        }

    def job(self, **overrides):
        base = {
            "id": "job-1",
            "status": "FAILED",
            "player_ref": {"track_id": 52},
            "target": {"confirmed": True, "full_match_mode": True},
            "video_key": None,
            "video_bucket": None,
            "result": {
                "candidates": {"candidates": [{"track_id": 52}]},
                "framesProcessed": 32,
                "tracking": {"segments": [1, 2, 3]},
                "assets": {
                    "input_video": {"bucket": "fnh", "key": "jobs/job-1/input.mp4"},
                    "clips": [{"key": "old.mp4"}],
                },
            },
            "progress": {"step": "FAILED", "pct": 100},
            "warnings": ["TRACKING_TIMEOUT"],
            "error": "Tracking timeout exceeded",
            "failure_reason": "TRACKING_TIMEOUT",
            "ai_report": {"old": True},
            "report": {"old": True},
            "report_status": "DONE",
            "report_error": "old",
            "updated_at": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def retry(self, job, *, runtime=None, payload=None):
        db = DummyDB(job)
        recorder = DelayRecorder()
        pipeline = types.ModuleType("app.workers.pipeline")
        pipeline.run_analysis = recorder
        snapshot = runtime if runtime is not None else self.runtime_ready()
        if payload is None:
            current_target = job.target if isinstance(job.target, dict) else {}
            current_attempt_id = current_target.get("analysis_attempt_id")
            payload = (
                {"expected_analysis_attempt_id": current_attempt_id}
                if current_attempt_id
                else {}
            )
        with patch.dict(sys.modules, {"app.workers.pipeline": pipeline}), patch.object(
            runtime_api, "inspect_runtime", return_value=snapshot
        ):
            response = runtime_api.retry_job(
                "job-1",
                self.request(),
                payload=payload,
                db=db,
            )
        return response, db, recorder

    def test_failed_job_retries_without_losing_selection_or_candidates(self):
        job = self.job()
        payload, db, recorder = self.retry(job)

        self.assertTrue(payload["ok"])
        self.assertEqual(job.status, "QUEUED")
        self.assertIsNone(job.error)
        self.assertIsNone(job.failure_reason)
        self.assertEqual(job.warnings, [])
        self.assertEqual(job.target["full_match_mode"], True)
        self.assertEqual(job.target["tracking"]["status"], "PENDING")
        attempt_id = payload["data"]["analysis_attempt_id"]
        self.assertEqual(job.target["analysis_attempt_id"], attempt_id)
        self.assertEqual(
            job.target["tracking"]["analysis_attempt_id"],
            attempt_id,
        )
        self.assertEqual(job.result["analysis_attempt_id"], attempt_id)
        self.assertEqual(job.progress["analysis_attempt_id"], attempt_id)
        self.assertIn("candidates", job.result)
        self.assertNotIn("tracking", job.result)
        self.assertEqual(job.result["assets"].keys(), {"input_video"})
        self.assertEqual(job.result["retry_count"], 1)
        self.assertEqual(len(job.result["retry_history"]), 1)
        self.assertEqual(recorder.calls, [("job-1", attempt_id)])
        self.assertGreaterEqual(db.commits, 1)

    def test_retry_rotates_attempt_and_records_previous_nonce(self):
        job = self.job(
            target={
                "confirmed": True,
                "full_match_mode": True,
                "analysis_attempt_id": "attempt-a",
            },
            result={
                "analysis_attempt_id": "attempt-a",
                "assets": {
                    "input_video": {
                        "bucket": "fnh",
                        "key": "jobs/job-1/input.mp4",
                    }
                },
            },
        )

        payload, _, recorder = self.retry(job)

        attempt_b = payload["data"]["analysis_attempt_id"]
        self.assertNotEqual(attempt_b, "attempt-a")
        self.assertEqual(job.target["analysis_attempt_id"], attempt_b)
        self.assertEqual(job.result["analysis_attempt_id"], attempt_b)
        self.assertEqual(
            job.result["retry_history"][-1]["previous_analysis_attempt_id"],
            "attempt-a",
        )
        self.assertEqual(
            job.result["retry_history"][-1]["analysis_attempt_id"],
            attempt_b,
        )
        self.assertEqual(recorder.calls, [("job-1", attempt_b)])

    def test_retry_uses_stored_input_when_original_url_is_not_stable(self):
        job = self.job(video_key=None, video_bucket=None)
        self.retry(job)
        self.assertEqual(job.video_key, "jobs/job-1/input.mp4")
        self.assertEqual(job.video_bucket, "fnh")

    def test_active_job_retry_is_idempotent_without_worker_check(self):
        job = self.job(status="RUNNING")
        with patch.object(runtime_api, "inspect_runtime") as inspect:
            payload = runtime_api.retry_job(
                "job-1", self.request(), payload={}, db=DummyDB(job)
            )
        self.assertTrue(payload["data"]["already_active"])
        self.assertEqual(job.status, "RUNNING")
        inspect.assert_not_called()

    def test_force_alone_does_not_supersede_an_active_job(self):
        job = self.job(
            status="RUNNING",
            target={
                "confirmed": True,
                "analysis_attempt_id": "attempt-a",
            },
        )
        with patch.object(runtime_api, "inspect_runtime") as inspect:
            payload = runtime_api.retry_job(
                "job-1",
                self.request(),
                payload={
                    "force": True,
                    "expected_analysis_attempt_id": "attempt-a",
                },
                db=DummyDB(job),
            )
        self.assertTrue(payload["data"]["already_active"])
        self.assertEqual(job.target["analysis_attempt_id"], "attempt-a")
        inspect.assert_not_called()

    def test_string_flags_do_not_supersede_an_active_job(self):
        job = self.job(
            status="RUNNING",
            target={
                "confirmed": True,
                "analysis_attempt_id": "attempt-a",
            },
        )
        with patch.object(runtime_api, "inspect_runtime") as inspect:
            payload = runtime_api.retry_job(
                "job-1",
                self.request(),
                payload={
                    "force": "false",
                    "supersede_active": "false",
                    "expected_analysis_attempt_id": "attempt-a",
                },
                db=DummyDB(job),
            )
        self.assertTrue(payload["data"]["already_active"])
        self.assertEqual(job.target["analysis_attempt_id"], "attempt-a")
        inspect.assert_not_called()

    def test_legacy_active_job_cannot_be_superseded_without_nonce(self):
        job = self.job(
            status="RUNNING",
            target={"confirmed": True},
            progress={
                "step": "TRACKING",
                "updated_at": "2026-07-28T20:00:00+00:00",
            },
            updated_at=datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc),
        )
        with patch.object(runtime_api, "inspect_runtime") as inspect:
            with self.assertRaises(HTTPException) as context:
                runtime_api.retry_job(
                    "job-1",
                    self.request(),
                    payload={
                        "force": True,
                        "supersede_active": True,
                    },
                    db=DummyDB(job),
                )
        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(
            context.exception.detail["code"],
            "ACTIVE_RETRY_ATTEMPT_REQUIRED",
        )
        self.assertEqual(job.status, "RUNNING")
        inspect.assert_not_called()

    def test_active_retry_stale_threshold_cannot_drop_below_fifteen_minutes(self):
        with patch.dict(
            "os.environ",
            {"ACTIVE_RETRY_STALE_AFTER_SECONDS": "60"},
            clear=False,
        ):
            self.assertEqual(
                runtime_api._active_retry_stale_after_seconds(),
                900.0,
            )

    def test_recent_active_job_cannot_be_superseded(self):
        now = datetime(2026, 7, 28, 21, 45, tzinfo=timezone.utc)
        job = self.job(
            status="RUNNING",
            target={
                "confirmed": True,
                "analysis_attempt_id": "attempt-a",
            },
            progress={
                "step": "TRACKING",
                "updated_at": (now - timedelta(seconds=30)).isoformat(),
            },
            updated_at=now - timedelta(seconds=30),
        )
        with patch.object(runtime_api, "_now", return_value=now), patch.object(
            runtime_api, "inspect_runtime"
        ) as inspect:
            with self.assertRaises(HTTPException) as context:
                runtime_api.retry_job(
                    "job-1",
                    self.request(),
                    payload={
                        "force": True,
                        "supersede_active": True,
                        "expected_analysis_attempt_id": "attempt-a",
                    },
                    db=DummyDB(job),
                )
        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(
            context.exception.detail["code"],
            "ACTIVE_RETRY_NOT_STALE",
        )
        self.assertEqual(job.status, "RUNNING")
        self.assertEqual(job.target["analysis_attempt_id"], "attempt-a")
        inspect.assert_not_called()

    def test_stale_active_job_can_be_superseded_with_current_nonce(self):
        now = datetime(2026, 7, 28, 21, 45, tzinfo=timezone.utc)
        job = self.job(
            status="RUNNING",
            target={
                "confirmed": True,
                "full_match_mode": True,
                "analysis_attempt_id": "attempt-a",
            },
            progress={
                "step": "TRACKING_CANDIDATES",
                "updated_at": (now - timedelta(minutes=30)).isoformat(),
                "analysis_attempt_id": "attempt-a",
            },
            updated_at=now - timedelta(minutes=30),
        )
        db = DummyDB(job)
        recorder = DelayRecorder()
        pipeline = types.ModuleType("app.workers.pipeline")
        pipeline.run_analysis = recorder
        with patch.dict(sys.modules, {"app.workers.pipeline": pipeline}), patch.object(
            runtime_api, "_now", return_value=now
        ), patch.object(
            runtime_api,
            "inspect_runtime",
            return_value=self.runtime_ready(),
        ):
            payload = runtime_api.retry_job(
                "job-1",
                self.request(),
                payload={
                    "force": True,
                    "supersede_active": True,
                    "expected_analysis_attempt_id": "attempt-a",
                },
                db=db,
            )

        attempt_b = payload["data"]["analysis_attempt_id"]
        self.assertNotEqual(attempt_b, "attempt-a")
        self.assertEqual(job.status, "QUEUED")
        self.assertEqual(job.target["analysis_attempt_id"], attempt_b)
        self.assertEqual(
            job.result["retry_history"][-1]["previous_status"],
            "RUNNING",
        )
        self.assertEqual(recorder.calls, [("job-1", attempt_b)])

    def test_stale_retry_cannot_rotate_newer_terminal_attempt(self):
        for force in (False, True):
            with self.subTest(force=force):
                job = self.job(
                    status="COMPLETED",
                    target={
                        "confirmed": True,
                        "analysis_attempt_id": "attempt-b",
                        "tracking": {"analysis_attempt_id": "attempt-b"},
                    },
                    result={
                        "analysis_attempt_id": "attempt-b",
                        "tracking": {
                            "analysis_attempt_id": "attempt-b",
                            "tracking_success": True,
                        },
                    },
                )
                original_target = dict(job.target)
                original_result = dict(job.result)

                with patch.object(runtime_api, "inspect_runtime") as inspect:
                    with self.assertRaises(HTTPException) as context:
                        runtime_api.retry_job(
                            "job-1",
                            self.request(),
                            payload={
                                "force": force,
                                "expected_analysis_attempt_id": "attempt-a",
                            },
                            db=DummyDB(job),
                        )

                self.assertEqual(context.exception.status_code, 409)
                self.assertEqual(
                    context.exception.detail["code"],
                    "ANALYSIS_ATTEMPT_MISMATCH",
                )
                self.assertEqual(job.status, "COMPLETED")
                self.assertEqual(job.target, original_target)
                self.assertEqual(job.result, original_result)
                inspect.assert_not_called()

    def test_modern_retry_requires_current_attempt_precondition(self):
        for force in (False, True):
            with self.subTest(force=force), patch.object(
                runtime_api, "inspect_runtime"
            ) as inspect:
                job = self.job(
                    target={
                        "confirmed": True,
                        "analysis_attempt_id": "attempt-a",
                    }
                )
                with self.assertRaises(HTTPException) as context:
                    runtime_api.retry_job(
                        "job-1",
                        self.request(),
                        payload={"force": force},
                        db=DummyDB(job),
                    )

                self.assertEqual(context.exception.status_code, 409)
                self.assertEqual(
                    context.exception.detail["code"],
                    "ANALYSIS_ATTEMPT_PRECONDITION_REQUIRED",
                )
                self.assertEqual(job.status, "FAILED")
                inspect.assert_not_called()

    def test_current_forced_retry_rotates_expected_terminal_attempt(self):
        job = self.job(
            status="COMPLETED",
            target={
                "confirmed": True,
                "analysis_attempt_id": "attempt-a",
                "tracking": {"analysis_attempt_id": "attempt-a"},
            },
            result={
                "analysis_attempt_id": "attempt-a",
                "assets": {
                    "input_video": {
                        "bucket": "fnh",
                        "key": "jobs/job-1/input.mp4",
                    }
                },
            },
        )
        db = DummyDB(job)
        recorder = DelayRecorder()
        pipeline = types.ModuleType("app.workers.pipeline")
        pipeline.run_analysis = recorder

        with patch.dict(sys.modules, {"app.workers.pipeline": pipeline}), patch.object(
            runtime_api,
            "inspect_runtime",
            return_value=self.runtime_ready(),
        ):
            payload = runtime_api.retry_job(
                "job-1",
                self.request(),
                payload={
                    "force": True,
                    "expected_analysis_attempt_id": "attempt-a",
                },
                db=db,
            )

        attempt_b = payload["data"]["analysis_attempt_id"]
        self.assertNotEqual(attempt_b, "attempt-a")
        self.assertEqual(job.target["analysis_attempt_id"], attempt_b)
        self.assertEqual(recorder.calls, [("job-1", attempt_b)])

    def test_retry_requires_saved_target(self):
        job = self.job(target={"confirmed": False})
        with self.assertRaises(HTTPException) as context:
            runtime_api.retry_job("job-1", self.request(), payload={}, db=DummyDB(job))
        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(context.exception.detail["code"], "RETRY_NOT_READY")

    def test_worker_must_be_ready_before_mutation(self):
        job = self.job()
        unavailable = {
            "ready": False,
            "dependencies": {"redis": "ready", "worker": "stale"},
            "worker": {"revision": "old"},
            "worker_age_seconds": 120.0,
            "worker_revision_matches_api": False,
        }
        with patch.object(runtime_api, "inspect_runtime", return_value=unavailable):
            with self.assertRaises(HTTPException) as context:
                runtime_api.retry_job(
                    "job-1", self.request(), payload={}, db=DummyDB(job)
                )
        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail["code"], "WORKER_NOT_READY")
        self.assertEqual(job.status, "FAILED")

    def test_delivery_then_raise_preserves_worker_claimed_attempt(self):
        job = self.job()
        db = DummyDB(job)

        class DeliveryThenRaise:
            def delay(self, job_id, analysis_attempt_id):
                self.calls = [(job_id, analysis_attempt_id)]
                job.status = "RUNNING"
                job.progress = {
                    **(job.progress or {}),
                    "step": "STARTING",
                    "analysis_attempt_id": analysis_attempt_id,
                    "analysis_task_id": "celery-task-b",
                    "analysis_task_retry": 0,
                }
                raise RuntimeError("publish confirmation lost")

        dispatch = DeliveryThenRaise()
        pipeline = types.ModuleType("app.workers.pipeline")
        pipeline.run_analysis = dispatch
        with patch.dict(sys.modules, {"app.workers.pipeline": pipeline}), patch.object(
            runtime_api,
            "inspect_runtime",
            return_value=self.runtime_ready(),
        ):
            payload = runtime_api.retry_job(
                "job-1",
                self.request(),
                payload={},
                db=db,
            )

        attempt_id = payload["data"]["analysis_attempt_id"]
        self.assertEqual(dispatch.calls, [("job-1", attempt_id)])
        self.assertTrue(payload["data"]["dispatch_ambiguous"])
        self.assertEqual(payload["data"]["status"], "RUNNING")
        self.assertEqual(job.status, "RUNNING")
        self.assertIsNone(job.failure_reason)
        self.assertIsNone(job.error)
        self.assertEqual(job.progress["analysis_task_id"], "celery-task-b")
        self.assertEqual(db.rollbacks, 1)
        self.assertEqual(db.expirations, 1)
        self.assertEqual(db.commits, 1)

    def test_dispatch_raise_fails_only_still_queued_unclaimed_attempt(self):
        job = self.job()
        db = DummyDB(job)

        def raise_before_delivery(*_args):
            raise RuntimeError("broker unavailable")

        pipeline = types.ModuleType("app.workers.pipeline")
        pipeline.run_analysis = types.SimpleNamespace(delay=raise_before_delivery)
        with patch.dict(sys.modules, {"app.workers.pipeline": pipeline}), patch.object(
            runtime_api,
            "inspect_runtime",
            return_value=self.runtime_ready(),
        ):
            with self.assertRaises(HTTPException) as context:
                runtime_api.retry_job(
                    "job-1",
                    self.request(),
                    payload={},
                    db=db,
                )

        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail["code"], "RETRY_ENQUEUE_FAILED")
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.failure_reason, "RETRY_ENQUEUE_FAILED")
        self.assertEqual(db.rollbacks, 1)
        self.assertEqual(db.expirations, 1)
        self.assertEqual(db.commits, 2)

    def test_retry_history_is_bounded_and_count_is_monotonic(self):
        source = {
            "retry_count": 15,
            "retry_history": [{"attempt": index} for index in range(6, 16)],
        }
        self.assertEqual(runtime_api._retry_count(source), 15)
        preserved = runtime_api._preserve_retry_inputs(source, {"attempt": 16})
        self.assertEqual(len(preserved["retry_history"]), 10)
        self.assertEqual(preserved["retry_history"][-1]["attempt"], 16)
        self.assertEqual(preserved["retry_count"], 16)


if __name__ == "__main__":
    unittest.main()
