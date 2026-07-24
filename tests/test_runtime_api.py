import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Request

from app import runtime_api


class DummyDB:
    def __init__(self, job):
        self.job = job
        self.commits = 0

    def get(self, _model, _job_id):
        return self.job

    def commit(self):
        self.commits += 1

    def refresh(self, _job):
        return None


class DelayRecorder:
    def __init__(self):
        self.calls = []

    def delay(self, job_id):
        self.calls.append(job_id)


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

    def retry(self, job, *, runtime=None):
        db = DummyDB(job)
        recorder = DelayRecorder()
        pipeline = types.ModuleType("app.workers.pipeline")
        pipeline.run_analysis = recorder
        snapshot = runtime if runtime is not None else self.runtime_ready()
        with patch.dict(sys.modules, {"app.workers.pipeline": pipeline}), patch.object(
            runtime_api, "inspect_runtime", return_value=snapshot
        ):
            payload = runtime_api.retry_job(
                "job-1", self.request(), payload={}, db=db
            )
        return payload, db, recorder

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
        self.assertIn("candidates", job.result)
        self.assertNotIn("tracking", job.result)
        self.assertEqual(job.result["assets"].keys(), {"input_video"})
        self.assertEqual(job.result["retry_count"], 1)
        self.assertEqual(len(job.result["retry_history"]), 1)
        self.assertEqual(recorder.calls, ["job-1"])
        self.assertGreaterEqual(db.commits, 1)

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

    def test_retry_requires_saved_target(self):
        job = self.job(target={"confirmed": False})
        with self.assertRaises(HTTPException) as context:
            runtime_api.retry_job(
                "job-1", self.request(), payload={}, db=DummyDB(job)
            )
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
