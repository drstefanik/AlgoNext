import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from app import runtime_api
from app.core.preparation_retry import can_retry_preparation
from tests import test_runtime_api as runtime_tests


class PreparationRetryTests(unittest.TestCase):
    def job(self, **changes):
        values = dict(
            status="FAILED",
            player_ref={},
            target={"confirmed": False},
            result={"assets": {"input_video": {"bucket": "fnh", "key": "cached.mp4"}}},
            progress={"step": "PREVIEWS_FAILED", "pct": 20},
            preview_frames=[],
            failure_reason="preview_generation_failed",
        )
        values.update(changes)
        return runtime_tests.RuntimeApiTests().job(**values)

    def call_retry(self, job, task=None, payload=None):
        pipeline = types.ModuleType("app.workers.pipeline")
        pipeline.extract_preview_frames = Mock()
        pipeline.extract_candidates = Mock()
        pipeline.run_analysis = Mock()
        if task is not None:
            pipeline.extract_preview_frames.delay.side_effect = task
        db = runtime_tests.DummyDB(job)
        with patch.dict(sys.modules, {"app.workers.pipeline": pipeline}), patch.object(
            runtime_api,
            "inspect_runtime",
            return_value={"ready": True, "worker": {"revision": "test"}},
        ):
            response = runtime_api.retry_job(
                "job-1",
                runtime_tests.RuntimeApiTests().request(),
                payload=payload or {},
                db=db,
            )
        return response, pipeline

    def test_failed_preview_restarts_same_job_with_a_fenced_attempt(self):
        job = self.job()
        response, pipeline = self.call_retry(job)
        attempt = response["data"]["analysis_attempt_id"]
        self.assertEqual(job.status, "CREATED")
        self.assertEqual(job.progress["step"], "CREATED")
        self.assertEqual(job.video_key, "cached.mp4")
        self.assertEqual(job.target["analysis_attempt_id"], attempt)
        pipeline.extract_preview_frames.delay.assert_called_once_with("job-1", attempt)
        pipeline.run_analysis.delay.assert_not_called()

    def test_detector_retry_reuses_frames_and_clears_failed_candidate_cache(self):
        job = self.job(
            preview_frames=[{"key": "jobs/job-1/frames/frame_0001.jpg"}],
            result={"candidates": {"autodetection_status": "FAILED"}},
            failure_reason="candidates_generation_failed",
        )
        response, pipeline = self.call_retry(job)
        pipeline.extract_candidates.delay.assert_called_once_with(
            "job-1", response["data"]["analysis_attempt_id"]
        )
        self.assertEqual(job.progress["step"], "PREVIEWS_READY")
        self.assertEqual(len(job.preview_frames), 1)
        self.assertNotIn("candidates", job.result)
        pipeline.extract_preview_frames.delay.assert_not_called()

    def test_dispatch_failure_is_recoverable(self):
        job = self.job()
        with self.assertRaises(HTTPException) as ctx:
            self.call_retry(job, task=ConnectionError("broker disconnected"))
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.failure_reason, "PREPARATION_ENQUEUE_FAILED")
        self.assertTrue(can_retry_preparation(job))

    def test_ambiguous_publish_does_not_overwrite_started_previews(self):
        job = self.job()

        def started_then_connection_dropped(*args):
            job.progress = {**job.progress, "step": "EXTRACTING_PREVIEWS"}
            raise ConnectionError("ack lost")

        response, _ = self.call_retry(job, task=started_then_connection_dropped)
        self.assertTrue(response["data"]["dispatch_ambiguous"])
        self.assertEqual(job.progress["step"], "EXTRACTING_PREVIEWS")
        self.assertNotEqual(job.status, "FAILED")

    def test_prior_analysis_cannot_be_reset_as_preparation(self):
        for changes in (
            {"player_ref": {"track_id": 1}},
            {"target": {"confirmed": True}},
            {"target": {"selections": [{"t": 0}]}},
            {"result": {"tracking": {"tracking_success": False}}},
            {"result": {"analysis_outcome": {"tracking_state": "FAILED"}}},
            {"progress": {"analysis_task_id": "older-analysis"}},
            {"status": "RUNNING"},
            {"failure_reason": "UNKNOWN"},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(can_retry_preparation(self.job(**changes)))

    def test_old_attempt_cannot_retry_new_preparation(self):
        job = self.job(target={"analysis_attempt_id": "current"})
        with self.assertRaises(HTTPException) as ctx:
            self.call_retry(job, payload={"expected_analysis_attempt_id": "old"})
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(job.target["analysis_attempt_id"], "current")


if __name__ == "__main__":
    unittest.main()
