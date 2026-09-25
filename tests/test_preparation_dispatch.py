import ast
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from fastapi import Request
from app import api
from app.schemas import JobCreate


class CreateDB:
    def __init__(self):
        self.job = None

    def add(self, job):
        self.job = job

    def commit(self):
        pass

    def rollback(self):
        pass

    def refresh(self, job):
        pass


class PreparationDispatchTests(unittest.TestCase):
    def create(self, *, started=False):
        db = CreateDB()
        pipeline = types.ModuleType("app.workers.pipeline")

        def fail(*args):
            if started:
                db.job.progress = {"step": "EXTRACTING_PREVIEWS", "pct": 15}
            raise ConnectionError("publish confirmation lost")

        pipeline.kickoff_job = SimpleNamespace(delay=fail)
        with patch.dict(sys.modules, {"app.workers.pipeline": pipeline}), patch.object(
            api, "_load_job_for_update", side_effect=lambda *_: db.job
        ):
            response = api.create_job(
                JobCreate(
                    video_url="https://example.test/video.mp4",
                    role="MF",
                    category="U18",
                ),
                Request({"type": "http", "headers": []}),
                db,
            )
        return response, db.job

    def test_initial_dispatch_failure_returns_saved_job_id_and_retryable_failure(self):
        response, job = self.create()
        self.assertEqual(response["data"]["id"], job.id)
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.failure_reason, "PREPARATION_ENQUEUE_FAILED")

    def test_initial_ambiguous_dispatch_preserves_worker_progress(self):
        response, job = self.create(started=True)
        self.assertEqual(job.status, "CREATED")
        self.assertEqual(job.progress["step"], "EXTRACTING_PREVIEWS")
        self.assertTrue(response["ok"])

    def kickoff(self, retries):
        # Exercise the worker function without loading detector models. Model
        # inference is unrelated to publishing the next preparation task.
        source = Path("app/workers/pipeline.py")
        functions = [
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"kickoff_job", "_validate_task_analysis_attempt"}
        ]
        function = next(node for node in functions if node.name == "kickoff_job")
        function.decorator_list = []
        tree = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *functions,
            ],
            type_ignores=[],
        )
        job = SimpleNamespace(
            id="one", target={}, status="CREATED", progress={"step": "CREATED"}
        )
        db = Mock()
        retry_error = type("TaskRetry", (Exception,), {})
        updates = []

        def update(_db, _id, _attempt, callback, **kwargs):
            updates.append(kwargs)
            callback(job)

        ns = {
            "SessionLocal": lambda: db,
            "_load_job_for_update": lambda *_: job,
            "_validate_preanalysis_task_state": lambda *_: None,
            "safe_commit": lambda *_: None,
            "extract_preview_frames": SimpleNamespace(
                delay=Mock(side_effect=ConnectionError("queue"))
            ),
            "StaleAnalysisAttemptError": type("Stale", (Exception,), {}),
            "Retry": retry_error,
            "logger": logging.getLogger("test"),
            "update_preanalysis_job": update,
            "set_progress": lambda job, step, pct, message: setattr(
                job, "progress", {"step": step}
            ),
        }
        exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
        task = SimpleNamespace(
            request=SimpleNamespace(retries=retries),
            retry=Mock(side_effect=retry_error),
        )
        return ns["kickoff_job"], task, job, updates, retry_error

    def test_kickoff_retries_transient_publish_error(self):
        fn, task, job, updates, retry_error = self.kickoff(0)
        with self.assertRaises(retry_error):
            fn(task, "one")
        task.retry.assert_called_once()
        self.assertEqual(updates, [])
        self.assertEqual(job.status, "CREATED")

    def test_exhausted_kickoff_no_longer_leaves_an_infinite_spinner(self):
        fn, task, job, updates, _ = self.kickoff(2)
        fn(task, "one")
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.failure_reason, "PREPARATION_ENQUEUE_FAILED")
        self.assertEqual(updates[0]["allowed_progress_steps"], frozenset({"CREATED"}))

    def test_delayed_initial_kickoff_cannot_adopt_a_new_preparation_attempt(self):
        fn, task, job, updates, _ = self.kickoff(1)
        job.target = {"analysis_attempt_id": "new-attempt"}
        fn(task, "one")
        fn.__globals__["extract_preview_frames"].delay.assert_not_called()
        task.retry.assert_not_called()
        self.assertEqual(updates, [])
        self.assertEqual(job.status, "CREATED")

    def test_exhausted_candidate_preparation_is_failed_and_retryable(self):
        from app.core.preparation_retry import can_retry_preparation

        source = Path("app/workers/pipeline.py")
        function = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "extract_candidates"
        )
        function.decorator_list = []
        tree = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                function,
            ],
            type_ignores=[],
        )
        for missing_frames in (False, True):
            with self.subTest(
                missing_frames=missing_frames
            ), TemporaryDirectory() as directory:
                job = SimpleNamespace(
                    status="CREATED",
                    target={},
                    player_ref={},
                    result={},
                    progress={"step": "PREVIEWS_READY"},
                    warnings=[],
                    preview_frames=[],
                )
                ns = {
                    "SessionLocal": Mock(),
                    "os": os,
                    "S3_ENDPOINT_URL": "https://s3.example.test",
                    "S3_PUBLIC_ENDPOINT_URL": "https://s3.example.test",
                    "_claim_preanalysis_task": lambda *_: (job, None),
                    "_CANDIDATE_PROGRESS_STEPS": frozenset(
                        {"PREVIEWS_READY", "TRACKING_CANDIDATES", "TRACKING"}
                    ),
                    "StaleAnalysisAttemptError": type("Stale", (Exception,), {}),
                    "Retry": type("TaskRetry", (Exception,), {}),
                    "logger": logging.getLogger("test"),
                    "update_preanalysis_job": lambda _db, _id, _attempt, update, **_: update(
                        job
                    ),
                    "set_progress": lambda job, step, pct, message: setattr(
                        job, "progress", {"step": step}
                    ),
                    "_build_candidates_error_detail": str,
                    "_cleanup_workdir": Mock(),
                    "get_s3_client": Mock(),
                    "ensure_bucket_exists": Mock(),
                    "_safe_namespace_component": lambda value, **_: value,
                    "_attempt_namespace": lambda _: "initial",
                    "Path": lambda _: Path(directory),
                }
                exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), ns)
                task = SimpleNamespace(
                    request=SimpleNamespace(retries=2, id="task"), retry=Mock()
                )
                with patch.dict(
                    os.environ,
                    {
                        "CANDIDATES_TASK_RETRIES": "2",
                        "S3_ACCESS_KEY": "test" if missing_frames else "",
                        "S3_SECRET_KEY": "test",
                        "S3_BUCKET": "test",
                    },
                ):
                    ns["extract_candidates"](task, "one")
                self.assertEqual(job.status, "FAILED")
                self.assertEqual(job.failure_reason, "candidates_generation_failed")
                self.assertTrue(can_retry_preparation(job))
                task.retry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
