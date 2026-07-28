import ast
import copy
import math
import os
import re
import threading
import unittest
from pathlib import Path

from app.workers.multi_anchor import normalize_anchors


class AnalysisAttemptPipelineTests(unittest.TestCase):
    @staticmethod
    def _pipeline_tree():
        pipeline_path = Path(__file__).resolve().parents[1] / "app/workers/pipeline.py"
        return pipeline_path, ast.parse(pipeline_path.read_text(encoding="utf-8"))

    @classmethod
    def _load_helper(cls, helper_name):
        pipeline_path, tree = cls._pipeline_tree()
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == helper_name
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                helper,
            ],
            type_ignores=[],
        )
        namespace = {
            "StaleAnalysisAttemptError": RuntimeError,
            "normalize_anchors": normalize_anchors,
            "utc_now_iso": lambda: "2026-07-28T00:00:00+00:00",
        }
        exec(
            compile(
                ast.fix_missing_locations(module),
                str(pipeline_path),
                "exec",
            ),
            namespace,
        )
        return namespace[helper_name]

    @classmethod
    def _load_attempt_mutator(cls):
        pipeline_path, tree = cls._pipeline_tree()
        helper_names = {
            "safe_commit",
            "_load_job_for_update",
            "_validate_task_analysis_attempt",
            "_validate_task_analysis_state",
            "update_analysis_job",
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *helpers,
            ],
            type_ignores=[],
        )
        namespace = {
            "AnalysisJob": object,
            "StaleAnalysisAttemptError": RuntimeError,
        }
        exec(
            compile(
                ast.fix_missing_locations(module),
                str(pipeline_path),
                "exec",
            ),
            namespace,
        )
        return namespace["update_analysis_job"]

    @classmethod
    def _load_helpers(cls, helper_names, *, namespace=None):
        pipeline_path, tree = cls._pipeline_tree()
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *helpers,
            ],
            type_ignores=[],
        )
        loaded = {
            "AnalysisJob": object,
            "StaleAnalysisAttemptError": RuntimeError,
            "math": math,
            "normalize_anchors": normalize_anchors,
            "os": os,
            "re": re,
        }
        loaded.update(namespace or {})
        exec(
            compile(
                ast.fix_missing_locations(module),
                str(pipeline_path),
                "exec",
            ),
            loaded,
        )
        return loaded

    def test_tracking_payload_is_bound_to_target_attempt(self):
        bind = self._load_helper("_bind_analysis_attempt_id")
        tracking = {"tracking_success": True}
        attempt_id = "f0243750-3488-49a4-ada3-579859961671"

        returned = bind(
            tracking,
            {"analysis_attempt_id": attempt_id},
        )

        self.assertEqual(returned, attempt_id)
        self.assertEqual(tracking["analysis_attempt_id"], attempt_id)

    def test_legacy_target_does_not_fabricate_attempt(self):
        bind = self._load_helper("_bind_analysis_attempt_id")
        tracking = {"tracking_success": True}

        returned = bind(tracking, {"confirmed": True})

        self.assertIsNone(returned)
        self.assertNotIn("analysis_attempt_id", tracking)

    def test_binding_rejects_a_payload_from_another_attempt(self):
        bind = self._load_helper("_bind_analysis_attempt_id")
        tracking = {
            "tracking_success": True,
            "analysis_attempt_id": "1fdaf4b6-3c5c-4923-b80d-c542df602e96",
        }

        with self.assertRaises(RuntimeError):
            bind(
                tracking,
                {"analysis_attempt_id": ("63d748f7-66a4-485d-adca-d3c7a6067cb0")},
            )

    def test_task_rejects_superseded_dispatch_attempt(self):
        validate = self._load_helper("_validate_task_analysis_attempt")
        current_attempt = "63d748f7-66a4-485d-adca-d3c7a6067cb0"
        stale_attempt = "1fdaf4b6-3c5c-4923-b80d-c542df602e96"
        target = {"analysis_attempt_id": current_attempt}

        with self.assertRaises(RuntimeError):
            validate(target, stale_attempt)

        self.assertEqual(target, {"analysis_attempt_id": current_attempt})

    def test_task_accepts_matching_and_truly_legacy_dispatch(self):
        validate = self._load_helper("_validate_task_analysis_attempt")
        attempt_id = "63d748f7-66a4-485d-adca-d3c7a6067cb0"
        target = {"analysis_attempt_id": attempt_id}

        self.assertEqual(validate(target, attempt_id), attempt_id)
        self.assertIsNone(validate({"confirmed": True}, None))
        with self.assertRaises(RuntimeError):
            validate(target, None)

    def test_progress_keeps_the_current_analysis_attempt(self):
        set_progress = self._load_helper("set_progress")
        attempt_id = "63d748f7-66a4-485d-adca-d3c7a6067cb0"
        job = type(
            "Job",
            (),
            {
                "target": {"analysis_attempt_id": attempt_id},
                "progress": {},
            },
        )()

        set_progress(job, "DONE", 100, "Analysis completed")

        self.assertEqual(job.progress["analysis_attempt_id"], attempt_id)
        self.assertEqual(job.progress["step"], "DONE")
        self.assertEqual(job.progress["pct"], 100)

    def test_confirmed_selection_skips_redundant_candidate_tracking(self):
        helpers = self._load_helpers(
            {
                "_confirmed_full_match_selections",
                "_requires_candidate_tracking",
            }
        )
        requires_candidate_tracking = helpers["_requires_candidate_tracking"]
        job_id = "job-confirmed"
        first_selection = {
            "time_sec": 719.003,
            "bbox": {"x": 0.58, "y": 0.46, "w": 0.12, "h": 0.54},
            "frame_key": f"jobs/{job_id}/frames/frame_0001.jpg",
        }
        second_selection = {
            "time_sec": 2157.009,
            "bbox": {"x": 0.27, "y": 0.28, "w": 0.03, "h": 0.15},
            "frame_key": f"jobs/{job_id}/frames/frame_0002.jpg",
        }
        target = {
            "confirmed": True,
            "full_match_mode": True,
            "selections": [first_selection, second_selection],
        }

        self.assertFalse(
            requires_candidate_tracking(
                {},
                target,
                full_match_mode=True,
                player_ref=first_selection,
                duration=5931.775,
            )
        )
        self.assertFalse(
            requires_candidate_tracking(
                {},
                {
                    key: value
                    for key, value in target.items()
                    if key != "full_match_mode"
                },
                full_match_mode=True,
                player_ref=first_selection,
                duration=5931.775,
            )
        )
        self.assertFalse(
            requires_candidate_tracking(
                {"candidates": {"candidates": [{"track_id": 7}]}},
                {},
                full_match_mode=False,
                player_ref=None,
            )
        )
        rejected_cases = (
            (target, False, first_selection, 5931.775),
            ({**target, "selections": []}, True, first_selection, 5931.775),
            ({**target, "selections": [{}]}, True, first_selection, 5931.775),
            ({**target, "confirmed": "true"}, True, first_selection, 5931.775),
            (target, True, None, 5931.775),
            (
                target,
                True,
                {
                    "time_sec": 720.0,
                    "bbox": first_selection["bbox"],
                    "frame_key": first_selection["frame_key"],
                },
                5931.775,
            ),
            (target, True, first_selection, 1000.0),
        )
        for current_target, mode, player_ref, duration in rejected_cases:
            with self.subTest(
                target=current_target,
                mode=mode,
                player_ref=player_ref,
                duration=duration,
            ):
                self.assertTrue(
                    requires_candidate_tracking(
                        {},
                        current_target,
                        full_match_mode=mode,
                        player_ref=player_ref,
                        duration=duration,
                    )
                )

    def test_confirmed_preview_reuse_requires_canonical_persisted_assets(self):
        class FakeClientError(RuntimeError):
            def __init__(self, code, *, status_code=None):
                super().__init__(code)
                self.response = {
                    "Error": {"Code": code},
                    "ResponseMetadata": {"HTTPStatusCode": status_code},
                }

        helpers = self._load_helpers(
            {
                "_confirmed_full_match_selections",
                "_reusable_confirmed_full_match_preview_keys",
                "_preview_objects_exist",
            },
            namespace={"ClientError": FakeClientError},
        )
        reusable_keys = helpers["_reusable_confirmed_full_match_preview_keys"]
        objects_exist = helpers["_preview_objects_exist"]
        job_id = "job-confirmed"
        bucket = "fnh"
        first_key = f"jobs/{job_id}/frames/frame_0001.jpg"
        second_key = f"jobs/{job_id}/frames/frame_0002.jpg"
        first_selection = {
            "time_sec": 719.003,
            "bbox": {"x": 0.58, "y": 0.46, "w": 0.12, "h": 0.54},
            "frame_key": first_key,
        }
        second_selection = {
            "time_sec": 2157.009,
            "bbox": {"x": 0.27, "y": 0.28, "w": 0.03, "h": 0.15},
            "frame_key": second_key,
        }
        target = {
            "confirmed": True,
            "full_match_mode": True,
            "selections": [first_selection, second_selection],
        }
        preview_frames = [
            {
                "time_sec": 719.003,
                "bucket": bucket,
                "key": first_key,
            },
            {
                "time_sec": 2157.009,
                "bucket": bucket,
                "key": second_key,
            },
        ]

        self.assertEqual(
            reusable_keys(
                job_id,
                target,
                first_selection,
                preview_frames,
                full_match_mode=True,
                s3_bucket=bucket,
                duration=5931.775,
            ),
            [first_key, second_key],
        )
        self.assertEqual(
            reusable_keys(
                job_id,
                {
                    key: value
                    for key, value in target.items()
                    if key != "full_match_mode"
                },
                first_selection,
                preview_frames,
                full_match_mode=True,
                s3_bucket=bucket,
                duration=5931.775,
            ),
            [first_key, second_key],
        )

        invalid_cases = (
            ({**target, "confirmed": "true"}, first_selection, preview_frames),
            (
                target,
                first_selection,
                [
                    {
                        **preview_frames[0],
                        "key": f"jobs/other/frames/frame_0001.jpg",
                    },
                    preview_frames[1],
                ],
            ),
            (
                target,
                first_selection,
                [
                    {
                        **preview_frames[0],
                        "key": f"jobs/{job_id}/review/frame_0001.jpg",
                    },
                    preview_frames[1],
                ],
            ),
            (
                target,
                first_selection,
                [{**preview_frames[0], "time_sec": 718.0}, preview_frames[1]],
            ),
            (
                target,
                first_selection,
                [
                    {**preview_frames[0], "time_sec": float("nan")},
                    preview_frames[1],
                ],
            ),
            (
                target,
                first_selection,
                [
                    {**preview_frames[0], "time_sec": float("inf")},
                    preview_frames[1],
                ],
            ),
            (
                target,
                first_selection,
                [{**preview_frames[0], "bucket": "other"}, preview_frames[1]],
            ),
            (
                target,
                {
                    **first_selection,
                    "time_sec": 720.0,
                },
                preview_frames,
            ),
            (
                target,
                first_selection,
                [*preview_frames, dict(preview_frames[0])],
            ),
        )
        for current_target, player_ref, frames in invalid_cases:
            with self.subTest(
                target=current_target,
                player_ref=player_ref,
                frames=frames,
            ):
                self.assertEqual(
                    reusable_keys(
                        job_id,
                        current_target,
                        player_ref,
                        frames,
                        full_match_mode=True,
                        s3_bucket=bucket,
                        duration=5931.775,
                    ),
                    [],
                )

        self.assertEqual(
            reusable_keys(
                job_id,
                target,
                first_selection,
                preview_frames,
                full_match_mode=True,
                s3_bucket=bucket,
                duration=1000.0,
            ),
            [],
        )

        class Store:
            def __init__(self, lengths, error_code=None, status_code=None):
                self.lengths = lengths
                self.error_code = error_code
                self.status_code = status_code

            def head_object(self, *, Bucket, Key):
                self.assert_bucket = Bucket
                if self.error_code:
                    raise FakeClientError(
                        self.error_code,
                        status_code=self.status_code,
                    )
                if Key not in self.lengths:
                    raise FakeClientError("NoSuchKey")
                return {"ContentLength": self.lengths[Key]}

        self.assertTrue(
            objects_exist(
                Store({first_key: 128, second_key: 256}),
                bucket,
                [first_key, second_key],
            )
        )
        self.assertFalse(
            objects_exist(
                Store({first_key: 128, second_key: 0}),
                bucket,
                [first_key, second_key],
            )
        )
        self.assertFalse(objects_exist(Store({}), bucket, [first_key, second_key]))
        self.assertFalse(
            objects_exist(
                Store({}, error_code="NoSuchObject"),
                bucket,
                [first_key, second_key],
            )
        )
        self.assertFalse(
            objects_exist(
                Store({}, error_code="Unknown", status_code=404),
                bucket,
                [first_key, second_key],
            )
        )
        with self.assertRaises(FakeClientError):
            objects_exist(
                Store({}, error_code="AccessDenied"),
                bucket,
                [first_key, second_key],
            )

    def test_missing_selection_preview_requires_reselection(self):
        applied = []

        def apply_outcome(job, payload):
            applied.append(copy.deepcopy(payload))
            job.warnings = ["PLAYER_RESELECTION_REQUIRED"]
            return True

        def set_test_progress(job, step, pct, message):
            job.progress = {"step": step, "pct": pct, "message": message}

        helper = self._load_helpers(
            {"_require_selection_preview_reselection"},
            namespace={
                "_apply_tracking_outcome": apply_outcome,
                "set_progress": set_test_progress,
            },
        )["_require_selection_preview_reselection"]
        job = type(
            "Job",
            (),
            {
                "target": {
                    "analysis_attempt_id": "attempt-a",
                    "selections": [{}, {}],
                },
                "warnings": [],
            },
        )()

        helper(job)

        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["analysis_attempt_id"], "attempt-a")
        self.assertEqual(applied[0]["anchors_total"], 2)
        self.assertEqual(
            applied[0]["tracking_status"],
            "SELECTION_PREVIEW_UNAVAILABLE",
        )
        self.assertEqual(applied[0]["action_required"], "RESELECT_PLAYER")
        self.assertIn("PLAYER_SELECTION_PREVIEW_UNAVAILABLE", job.warnings)
        self.assertEqual(job.progress["step"], "WAITING_FOR_PLAYER")

    def test_run_analysis_uses_confirmed_selection_candidate_policy(self):
        _, tree = self._pipeline_tree()
        candidate_policy = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_requires_candidate_tracking"
        )
        candidate_policy_calls = {
            child.func.id
            for child in ast.walk(candidate_policy)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        self.assertIn(
            "_confirmed_full_match_selections",
            candidate_policy_calls,
        )
        self.assertNotIn(
            "_reusable_confirmed_full_match_preview_keys",
            candidate_policy_calls,
        )
        self.assertNotIn("_preview_objects_exist", candidate_policy_calls)

        task = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_analysis"
        )
        policy_guards = [
            node
            for node in ast.walk(task)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_requires_candidate_tracking"
                for child in ast.walk(node.test)
            )
        ]
        self.assertEqual(len(policy_guards), 1)

        guarded_tracking_calls = [
            node
            for node in ast.walk(
                ast.Module(body=policy_guards[0].body, type_ignores=[])
            )
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "track_all_players"
        ]
        all_tracking_calls = [
            node
            for node in ast.walk(task)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "track_all_players"
        ]

        self.assertEqual(len(guarded_tracking_calls), 1)
        self.assertEqual(
            {node.lineno for node in all_tracking_calls},
            {node.lineno for node in guarded_tracking_calls},
        )

        preview_reuse_guards = [
            node
            for node in ast.walk(task)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "reuse_persisted_previews"
        ]
        self.assertEqual(len(preview_reuse_guards), 1)
        regeneration_branch = ast.Module(
            body=preview_reuse_guards[0].orelse,
            type_ignores=[],
        )
        guarded_timestamp_calls = [
            node
            for node in ast.walk(regeneration_branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_build_preview_timestamps"
        ]
        all_timestamp_calls = [
            node
            for node in ast.walk(task)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_build_preview_timestamps"
        ]
        guarded_preview_writes = [
            node
            for node in ast.walk(regeneration_branch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "preview_frames"
        ]
        self.assertEqual(len(guarded_timestamp_calls), 1)
        self.assertEqual(
            {node.lineno for node in all_timestamp_calls},
            {node.lineno for node in guarded_timestamp_calls},
        )
        self.assertEqual(len(guarded_preview_writes), 1)

        reselection_guards = [
            node
            for node in ast.walk(task)
            if isinstance(node, ast.If)
            and any(
                isinstance(statement, ast.Expr)
                and any(
                    isinstance(child, ast.Name)
                    and child.id == "_require_selection_preview_reselection"
                    for child in ast.walk(statement)
                )
                for statement in node.body
            )
        ]
        self.assertEqual(len(reselection_guards), 2)
        for guard in reselection_guards:
            self.assertTrue(
                any(isinstance(child, ast.Return) for child in ast.walk(guard))
            )

    def test_task_attempt_gate_precedes_every_job_update(self):
        _, tree = self._pipeline_tree()
        task = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_analysis"
        )
        helper_lines = [
            node.lineno
            for node in ast.walk(task)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validate_task_analysis_attempt"
        ]
        guarded_update_lines = [
            node.lineno
            for node in ast.walk(task)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "update_current_attempt"
        ]
        unguarded_update_lines = [
            node.lineno
            for node in ast.walk(task)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "update_job"
        ]
        claim_lines = [
            node.lineno
            for node in ast.walk(task)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_claim_analysis_task"
        ]

        self.assertGreaterEqual(len(helper_lines), 1)
        self.assertEqual(len(claim_lines), 1)
        self.assertTrue(guarded_update_lines)
        self.assertEqual(unguarded_update_lines, [])
        self.assertLess(claim_lines[0], min(guarded_update_lines))

    def test_stale_attempt_cannot_write_preview_features_or_finalize(self):
        update = self._load_attempt_mutator()
        attempt_a = "1fdaf4b6-3c5c-4923-b80d-c542df602e96"
        attempt_b = "63d748f7-66a4-485d-adca-d3c7a6067cb0"

        class Job:
            id = "job-race"
            target = {"analysis_attempt_id": attempt_b}
            preview_frames = [{"owner": "attempt-b"}]
            result = {
                "analysis_attempt_id": attempt_b,
                "raw_video_features": {"owner": "attempt-b"},
            }
            status = "QUEUED"

        class DB:
            def __init__(self, job):
                self.job = job
                self.commits = 0
                self.rollbacks = 0

            def get(self, _model, _job_id):
                return self.job

            def commit(self):
                self.commits += 1

            def rollback(self):
                self.rollbacks += 1

        job = Job()
        db = DB(job)
        before = copy.deepcopy(job.__dict__)
        stale_writes = (
            lambda current: setattr(
                current,
                "preview_frames",
                [{"owner": "attempt-a"}],
            ),
            lambda current: setattr(
                current,
                "result",
                {"raw_video_features": {"owner": "attempt-a"}},
            ),
            lambda current: (
                setattr(current, "status", "COMPLETED"),
                setattr(current, "result", {"analysis_attempt_id": attempt_a}),
            ),
        )

        for updater in stale_writes:
            with self.assertRaises(RuntimeError):
                update(db, job.id, attempt_a, updater)

        self.assertEqual(job.__dict__, before)
        self.assertEqual(db.commits, 0)

    def test_matching_attempt_mutator_commits_under_row_lock_helper(self):
        update = self._load_attempt_mutator()
        attempt_id = "63d748f7-66a4-485d-adca-d3c7a6067cb0"

        class Job:
            id = "job-current"
            target = {"analysis_attempt_id": attempt_id}
            status = "RUNNING"

        class DB:
            def __init__(self, job):
                self.job = job
                self.commits = 0

            def get(self, _model, _job_id):
                return self.job

            def commit(self):
                self.commits += 1

            def rollback(self):
                return None

        job = Job()
        db = DB(job)
        self.assertTrue(
            update(
                db,
                job.id,
                attempt_id,
                lambda current: setattr(current, "status", "COMPLETED"),
            )
        )
        self.assertEqual(job.status, "COMPLETED")
        self.assertEqual(db.commits, 1)

    def test_matching_attempt_cannot_reopen_terminal_or_selection_state(self):
        update = self._load_attempt_mutator()
        attempt_id = "63d748f7-66a4-485d-adca-d3c7a6067cb0"

        class Job:
            id = "job-delayed"
            target = {"analysis_attempt_id": attempt_id}
            status = "COMPLETED"

        class DB:
            def __init__(self, job):
                self.job = job
                self.commits = 0

            def get(self, _model, _job_id, **_kwargs):
                return self.job

            def commit(self):
                self.commits += 1

            def rollback(self):
                return None

        for status in (
            "COMPLETED",
            "PARTIAL",
            "FAILED",
            "READY_TO_ENQUEUE",
            "WAITING_FOR_TARGET",
            "WAITING_FOR_SELECTION",
        ):
            with self.subTest(status=status):
                job = Job()
                job.status = status
                db = DB(job)
                with self.assertRaises(RuntimeError):
                    update(
                        db,
                        job.id,
                        attempt_id,
                        lambda current: setattr(current, "status", "RUNNING"),
                    )
                self.assertEqual(job.status, status)
                self.assertEqual(db.commits, 0)

    def test_preanalysis_writer_rejects_legacy_stale_and_completed_phase(self):
        helpers = self._load_helpers(
            {
                "safe_commit",
                "_load_job_for_update",
                "_validate_task_analysis_attempt",
                "_validate_preanalysis_task_state",
                "update_preanalysis_job",
            },
            namespace={
                "_PREANALYSIS_MUTABLE_STATUSES": frozenset(
                    {"CREATED", "WAITING_FOR_SELECTION"}
                )
            },
        )
        update = helpers["update_preanalysis_job"]
        attempt_a = "attempt-a"
        attempt_b = "attempt-b"

        class Job:
            id = "job-preanalysis-race"
            target = {"analysis_attempt_id": attempt_b}
            status = "CREATED"
            progress = {"step": "TRACKING_CANDIDATES"}
            result = {"analysis_attempt_id": attempt_b}

        class DB:
            def __init__(self, job):
                self.job = job
                self.commits = 0

            def get(self, _model, _job_id, **_kwargs):
                return self.job

            def commit(self):
                self.commits += 1

            def rollback(self):
                return None

        for stale_attempt in (None, attempt_a):
            job = Job()
            db = DB(job)
            before = copy.deepcopy(job.__dict__)
            with self.assertRaises(RuntimeError):
                update(
                    db,
                    job.id,
                    stale_attempt,
                    lambda current: setattr(
                        current,
                        "result",
                        {"analysis_attempt_id": stale_attempt},
                    ),
                    allowed_progress_steps=frozenset({"TRACKING_CANDIDATES"}),
                )
            self.assertEqual(job.__dict__, before)
            self.assertEqual(db.commits, 0)

        job = Job()
        job.progress = {"step": "CANDIDATES_READY"}
        db = DB(job)
        with self.assertRaises(RuntimeError):
            update(
                db,
                job.id,
                attempt_b,
                lambda current: setattr(current, "status", "WAITING_FOR_SELECTION"),
                allowed_progress_steps=frozenset(
                    {"PREVIEWS_READY", "TRACKING_CANDIDATES", "TRACKING"}
                ),
            )
        self.assertEqual(job.progress["step"], "CANDIDATES_READY")
        self.assertEqual(db.commits, 0)

    def test_preanalysis_tasks_propagate_retry_and_dispatch_attempt(self):
        _, tree = self._pipeline_tree()
        tasks = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"extract_preview_frames", "extract_candidates"}
        }

        for task_name, task in tasks.items():
            with self.subTest(task=task_name):
                outer_try = next(
                    node for node in task.body if isinstance(node, ast.Try)
                )
                handler_names = [
                    handler.type.id
                    for handler in outer_try.handlers
                    if isinstance(handler.type, ast.Name)
                ]
                self.assertIn("Retry", handler_names)
                self.assertLess(
                    handler_names.index("Retry"),
                    handler_names.index("Exception"),
                )
                retry_handler = next(
                    handler
                    for handler in outer_try.handlers
                    if isinstance(handler.type, ast.Name) and handler.type.id == "Retry"
                )
                self.assertTrue(
                    any(isinstance(node, ast.Raise) for node in retry_handler.body)
                )
                generic_handler = next(
                    handler
                    for handler in outer_try.handlers
                    if isinstance(handler.type, ast.Name)
                    and handler.type.id == "Exception"
                )
                retry_calls = [
                    node
                    for node in ast.walk(
                        ast.Module(
                            body=generic_handler.body,
                            type_ignores=[],
                        )
                    )
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "retry"
                ]
                self.assertEqual(len(retry_calls), 1)

        preview_task = tasks["extract_preview_frames"]
        candidate_dispatches = [
            node
            for node in ast.walk(preview_task)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "delay"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "extract_candidates"
        ]
        self.assertEqual(len(candidate_dispatches), 1)
        self.assertEqual(len(candidate_dispatches[0].args), 2)

    def test_stale_attempt_cannot_overwrite_canonical_tracking_artifact(self):
        helpers = self._load_helpers(
            {
                "safe_commit",
                "_load_job_for_update",
                "_validate_task_analysis_attempt",
                "_validate_task_analysis_state",
                "_safe_namespace_component",
                "_attempt_namespace",
                "_publish_tracking_artifact",
            },
            namespace={
                "presign_get_object": (
                    lambda bucket, key, _expires: f"https://assets/{bucket}/{key}"
                )
            },
        )
        publish = helpers["_publish_tracking_artifact"]
        attempt_a = "attempt-a"
        attempt_b = "attempt-b"
        job_id = "job-artifact-race"

        class Job:
            id = job_id
            target = {"analysis_attempt_id": attempt_b}
            status = "RUNNING"

        class DB:
            def __init__(self, job):
                self.job = job
                self.commits = 0

            def get(self, _model, _job_id, **_kwargs):
                return self.job

            def commit(self):
                self.commits += 1

            def rollback(self):
                return None

        class ObjectStore:
            def __init__(self):
                self.objects = {
                    (f"jobs/{job_id}/attempts/{attempt_a}/" "tracking/tracking.json"): {
                        "analysis_attempt_id": attempt_a
                    },
                    (f"jobs/{job_id}/attempts/{attempt_b}/" "tracking/tracking.json"): {
                        "analysis_attempt_id": attempt_b
                    },
                }
                self.copies = []

            def copy_object(self, *, Bucket, CopySource, Key, **_kwargs):
                self.copies.append((CopySource["Key"], Key))
                self.objects[Key] = copy.deepcopy(self.objects[CopySource["Key"]])

        job = Job()
        db = DB(job)
        store = ObjectStore()
        output_b = {
            "analysis_attempt_id": attempt_b,
            "tracking_key": (
                f"jobs/{job_id}/attempts/{attempt_b}/tracking/tracking.json"
            ),
        }
        canonical_key = publish(
            db,
            job_id=job_id,
            expected_analysis_attempt_id=attempt_b,
            s3_internal=store,
            s3_bucket="fnh",
            tracking_output=output_b,
        )

        self.assertEqual(canonical_key, f"jobs/{job_id}/tracking/tracking.json")
        self.assertEqual(output_b["tracking_key"], canonical_key)
        self.assertIn(canonical_key, output_b["tracking_url"])
        self.assertEqual(
            store.objects[canonical_key]["analysis_attempt_id"],
            attempt_b,
        )
        self.assertEqual(job.target["analysis_attempt_id"], attempt_b)

        release_stale = threading.Barrier(2)
        stale_errors = []

        def delayed_attempt_a():
            release_stale.wait()
            try:
                publish(
                    db,
                    job_id=job_id,
                    expected_analysis_attempt_id=attempt_a,
                    s3_internal=store,
                    s3_bucket="fnh",
                    tracking_output={
                        "analysis_attempt_id": attempt_a,
                        "tracking_key": (
                            f"jobs/{job_id}/attempts/{attempt_a}/"
                            "tracking/tracking.json"
                        ),
                    },
                )
            except RuntimeError as exc:
                stale_errors.append(exc)

        stale_thread = threading.Thread(target=delayed_attempt_a)
        stale_thread.start()
        release_stale.wait()
        stale_thread.join(timeout=2)

        self.assertFalse(stale_thread.is_alive())
        self.assertEqual(len(stale_errors), 1)
        self.assertEqual(len(store.copies), 1)
        self.assertEqual(
            store.objects[canonical_key]["analysis_attempt_id"],
            attempt_b,
        )

    def test_stale_task_rolls_back_and_closes_without_generic_failure(self):
        _, tree = self._pipeline_tree()
        task = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_analysis"
        )
        outer_try = next(node for node in task.body if isinstance(node, ast.Try))
        handler_names = [
            handler.type.id
            for handler in outer_try.handlers
            if isinstance(handler.type, ast.Name)
        ]

        self.assertEqual(handler_names[0], "StaleAnalysisAttemptError")
        self.assertLess(
            handler_names.index("StaleAnalysisAttemptError"),
            handler_names.index("Exception"),
        )
        stale_handler = outer_try.handlers[0]
        stale_calls = {
            (node.func.value.id, node.func.attr)
            for node in ast.walk(stale_handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        }
        self.assertIn(("db", "rollback"), stale_calls)
        self.assertTrue(
            any(isinstance(node, ast.Return) for node in stale_handler.body)
        )
        final_calls = {
            (node.func.value.id, node.func.attr)
            for node in ast.walk(ast.Module(body=outer_try.finalbody, type_ignores=[]))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        }
        self.assertIn(("db", "close"), final_calls)

    def test_retry_exception_is_raised_outside_failure_write_guard(self):
        _, tree = self._pipeline_tree()
        task = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_analysis"
        )
        outer_try = next(node for node in task.body if isinstance(node, ast.Try))
        generic_handler = next(
            handler
            for handler in outer_try.handlers
            if isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        )

        retry_if = next(
            node
            for node in generic_handler.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "retry"
                for child in ast.walk(node)
            )
        )
        self.assertIsInstance(retry_if.body[0], ast.Raise)
        self.assertFalse(
            any(
                isinstance(parent, ast.Try) and retry_if in parent.body
                for parent in generic_handler.body
            )
        )
        exhausted_try = next(
            node for node in generic_handler.body if isinstance(node, ast.Try)
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "update_current_attempt"
                for node in ast.walk(exhausted_try)
            )
        )


if __name__ == "__main__":
    unittest.main()
