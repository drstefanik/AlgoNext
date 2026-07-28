import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from app.core.tracking_outcome import StaleAnalysisAttemptError
from app.reid.full_match_runtime import (
    mark_partial_timeout,
    persist_canonical_tracking_artifact,
    persist_fail_closed_legacy_fallback,
    select_full_match_profile,
)
from app.reid.runtime import install_windowed_reid


class ReIDRuntimeTests(unittest.TestCase):
    def test_partial_timeout_writer_rejects_same_attempt_after_completion(self):
        class Column:
            def __eq__(self, other):
                return ("id", other)

        class AnalysisJob:
            id = Column()

        class Statement:
            def __init__(self):
                self.locked = False
                self.populate_existing = False

            def where(self, _condition):
                return self

            def with_for_update(self):
                self.locked = True
                return self

            def execution_options(self, *, populate_existing):
                self.populate_existing = populate_existing
                return self

        job = SimpleNamespace(
            target={"analysis_attempt_id": "attempt-a"},
            warnings=[],
            status="COMPLETED",
            error=None,
            failure_reason=None,
            progress={"analysis_attempt_id": "attempt-a"},
        )

        class Session:
            committed = False
            rolled_back = False
            statement = None

            def execute(self, statement):
                self.statement = statement
                return SimpleNamespace(scalar_one_or_none=lambda: job)

            def commit(self):
                self.committed = True

            def rollback(self):
                self.rolled_back = True

            def close(self):
                pass

        session = Session()
        db_module = ModuleType("app.core.db")
        db_module.SessionLocal = lambda: session
        models_module = ModuleType("app.core.models")
        models_module.AnalysisJob = AnalysisJob
        normalizers_module = ModuleType("app.core.normalizers")
        normalizers_module.normalize_failure_reason = lambda value: value
        sqlalchemy_module = ModuleType("sqlalchemy")
        sqlalchemy_module.select = lambda _model: Statement()

        with patch.dict(
            sys.modules,
            {
                "app.core.db": db_module,
                "app.core.models": models_module,
                "app.core.normalizers": normalizers_module,
                "sqlalchemy": sqlalchemy_module,
            },
        ):
            with self.assertRaises(StaleAnalysisAttemptError):
                mark_partial_timeout(
                    "job-1",
                    None,
                    analysis_attempt_id="attempt-a",
                )

        self.assertTrue(session.statement.locked)
        self.assertTrue(session.statement.populate_existing)
        self.assertTrue(session.rolled_back)
        self.assertFalse(session.committed)
        self.assertEqual(job.progress["analysis_attempt_id"], "attempt-a")

    def test_stale_attempt_never_falls_back_to_legacy(self):
        calls = []

        def original(*_args, **_kwargs):
            calls.append("legacy")
            return {"mode": "legacy"}

        def implementation(*_args, fallback, **_kwargs):
            self.assertIs(fallback, original)
            raise StaleAnalysisAttemptError("attempt changed")

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_ENABLED": "1",
                "PLAYER_REID_FAIL_OPEN": "1",
            },
            clear=False,
        ):
            install_windowed_reid(module, implementation)
            with self.assertRaises(StaleAnalysisAttemptError):
                module.track_player_windowed(
                    "job-stale",
                    analysis_attempt_id="attempt-a",
                )

        self.assertEqual(calls, [])

    def test_disabled_flag_returns_budgeted_fail_closed_output_without_legacy(self):
        calls = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("disabled Player ReID must not run legacy tracking")

        module = SimpleNamespace(track_player_windowed=original)
        environment = {
            "PLAYER_REID_ENABLED": "0",
            "FULL_MATCH_TARGET_SAMPLES": "6000",
            "FULL_MATCH_MIN_FPS": "1",
            "FULL_MATCH_MAX_FPS": "2",
            "FULL_MATCH_WINDOW_SEC": "60",
            "FULL_MATCH_OVERLAP_SEC": "5",
            "FULL_MATCH_DETECTOR_MODEL": "yolo11n.pt",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertFalse(install_windowed_reid(module, lambda: "reid"))
            self.assertIsNot(module.track_player_windowed, original)
            output = module.track_player_windowed(
                "job-disabled",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                video_duration_sec=6000.0,
                fps=5,
                window_sec=45.0,
                overlap_sec=10.0,
            )

        self.assertEqual(calls, [])
        self.assertEqual(output["reid_summary"]["status"], "DISABLED")
        self.assertEqual(output["runtime_profile"]["fps"], 1)
        self.assertEqual(output["runtime_profile"]["window_sec"], 60.0)
        self.assertEqual(output["runtime_profile"]["overlap_sec"], 5.0)
        self.assertEqual(output["runtime_profile"]["detector_model"], "yolo11n.pt")
        self.assertEqual(output["fps"], 1)
        self.assertEqual(output["window_sec"], 60.0)
        self.assertEqual(output["overlap_sec"], 5.0)
        self.assertFalse(output["tracking_success"])
        self.assertEqual(
            output["tracking_status"],
            "PLAYER_REID_DISABLED_UNVERIFIED",
        )
        self.assertEqual(output["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(output["segments"], [])
        self.assertEqual(output["segments_total"], 0)
        self.assertEqual(output["bboxes"], [])
        self.assertEqual(output["anchors_matched"], 0)
        self.assertEqual(output["anchor_matches"], [])
        self.assertEqual(output["anchors_used"], {})
        self.assertNotIn("tracking_key", output)
        self.assertNotIn("tracking_url", output)

    def test_enabled_flag_installs_wrapper_and_passes_fallback(self):
        calls = []

        def original(value):
            calls.append("legacy")
            return {"mode": "legacy", "value": value}

        def implementation(value, *, fallback):
            self.assertIs(fallback, original)
            calls.append("reid")
            return {"mode": "reid", "value": value}

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "1"}, clear=False):
            self.assertTrue(install_windowed_reid(module, implementation))
            self.assertEqual(module.track_player_windowed(7)["mode"], "reid")
            self.assertTrue(install_windowed_reid(module, implementation))
        self.assertEqual(calls, ["reid"])

    def test_enabled_wrapper_forwards_every_selection_unchanged(self):
        captured = {}
        selections = [
            {
                "frame_key": "frame-0.jpg",
                "frame_time_sec": 0.0,
                "x": 0.10,
                "y": 0.20,
                "w": 0.10,
                "h": 0.25,
            },
            {
                "frame_key": "frame-90.jpg",
                "frame_time_sec": 90.0,
                "x": 0.40,
                "y": 0.20,
                "w": 0.10,
                "h": 0.25,
            },
            {
                "frame_key": "frame-180.jpg",
                "frame_time_sec": 180.0,
                "x": 0.70,
                "y": 0.20,
                "w": 0.10,
                "h": 0.25,
            },
        ]

        def original(*args, **kwargs):
            return {"mode": "legacy"}

        def implementation(*args, fallback, **kwargs):
            self.assertIs(fallback, original)
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {
                "mode": "reid",
                "anchors_used": {"selections": args[3]},
            }

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "1"}, clear=False):
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(
                "job-multi",
                "/tmp/input.mp4",
                {"t": 0.0},
                selections,
                video_duration_sec=240.0,
                fps=5,
            )

        self.assertIs(captured["args"][3], selections)
        self.assertEqual(captured["args"][3], selections)
        self.assertEqual(
            output["anchors_used"]["selections"],
            selections,
        )

    def test_runtime_failure_falls_back_when_enabled(self):
        def original(value):
            return {
                "mode": "full_match_windowed",
                "value": value,
                "segments": [
                    {
                        "selected_track_id": 7,
                        "selected_track_ids": [7, 8],
                        "identity_id": "legacy-identity-secret",
                        "identity_status": "ACCEPTED",
                        "anchor_matches": [{"anchor_id": 1, "local_track_id": 7}],
                        "reacquire_score": 0.99,
                        "reacquire_source": "manual_anchor",
                        "reacquire_metrics": {"candidate_id": 7},
                        "bboxes": [{"t": 1.0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.2}],
                        "reid": {
                            "status": "ACCEPTED",
                            "selected_candidate_id": "7",
                            "autonomous_bboxes_count": 1,
                        },
                    }
                ],
                "track_id": 7,
                "segments_with_player": 1,
                "anchors_total": 1,
                "anchors_matched": 1,
                "anchor_matches": [{"anchor_id": 1, "local_track_id": 7}],
                "anchors_used": {"player_ref": {"track_id": 7}},
                "anchor_reacquisitions": 1,
                "coverage_pct": 25.0,
                "largest_gap_sec": 10.0,
            }

        def implementation(value, *, fallback):
            raise RuntimeError("boom")

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_ENABLED": "1",
                "PLAYER_REID_FAIL_OPEN": "1",
            },
            clear=False,
        ):
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(9)
        self.assertEqual(output["mode"], "full_match_windowed")
        self.assertEqual(output["reid_summary"]["status"], "FALLBACK_LEGACY")
        self.assertFalse(output["tracking_success"])
        self.assertEqual(
            output["tracking_status"],
            "REID_FALLBACK_LEGACY_UNVERIFIED",
        )
        self.assertEqual(output["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(output["segments_with_player"], 0)
        self.assertEqual(output["tracking_scope_status"], "EMPTY")
        self.assertEqual(output["segments"], [])
        self.assertEqual(output["segments_total"], 0)
        self.assertEqual(output["bboxes"], [])
        self.assertEqual(output["bboxes_count"], 0)
        self.assertEqual(output["anchors_total"], 0)
        self.assertEqual(output["anchors_matched"], 0)
        self.assertEqual(output["anchor_matches"], [])
        self.assertEqual(output["anchors_used"], {})
        self.assertEqual(output["anchor_reacquisitions"], 0)
        self.assertIsNone(output["largest_gap_sec"])
        self.assertNotIn("track_id", output)
        self.assertNotIn("selected_track_id", output)
        self.assertNotIn("selected_track_ids", output)
        self.assertNotIn("reacquire_score", output)
        self.assertNotIn("reacquire_source", output)
        self.assertNotIn("reacquire_metrics", output)

    def test_runtime_failure_rewrites_persisted_legacy_asset_fail_closed(self):
        uploaded = {}

        def original(*_args, **_kwargs):
            return {
                "mode": "full_match_windowed",
                "method": "yolo+bytetrack",
                "tracking_key": "jobs/another-job/tracking/tracking.json",
                "tracking_url": "https://unsafe.example/raw-tracking.json",
                "segments": [
                    {
                        "selected_track_id": 7,
                        "selected_track_ids": [7, 8],
                        "identity_id": "legacy-identity-secret",
                        "identity_status": "ACCEPTED",
                        "anchor_matches": [{"anchor_id": 1, "local_track_id": 7}],
                        "reacquire_score": 0.99,
                        "reacquire_source": "manual_anchor",
                        "reacquire_metrics": {"candidate_id": 7},
                        "bboxes": [
                            {
                                "t": 1.0,
                                "x": 0.1,
                                "y": 0.2,
                                "w": 0.1,
                                "h": 0.2,
                            }
                        ],
                    }
                ],
                "segments_total": 1,
                "segments_with_player": 1,
                "anchors_total": 1,
                "anchors_matched": 1,
                "anchor_matches": [{"anchor_id": 1, "local_track_id": 7}],
                "anchors_used": {"player_ref": {"track_id": 7}},
                "coverage_pct": 25.0,
                "largest_gap_sec": 10.0,
            }

        def implementation(*_args, fallback, **_kwargs):
            self.assertIs(fallback, original)
            raise RuntimeError("boom")

        def upload_file(_client, bucket, path, key, content_type):
            uploaded.update(
                {
                    "bucket": bucket,
                    "key": key,
                    "content_type": content_type,
                    "payload": json.loads(Path(path).read_text()),
                }
            )

        module = SimpleNamespace(
            track_player_windowed=original,
            S3_ENDPOINT_URL="http://s3.internal",
            _get_s3_client=lambda _endpoint: object(),
            _ensure_bucket_exists=lambda _client, _bucket: None,
            _upload_file=upload_file,
            _presign_get_object=(
                lambda bucket, key, _expires: f"https://safe.example/{bucket}/{key}"
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_root, patch.dict(
            os.environ,
            {
                "PLAYER_REID_ENABLED": "1",
                "PLAYER_REID_FAIL_OPEN": "1",
                "S3_BUCKET": "tracking",
            },
            clear=False,
        ), patch(
            "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
            Path(temporary_root),
        ):
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(
                "job-runtime-asset",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                analysis_attempt_id="attempt-a",
            )

        self.assertEqual(uploaded["bucket"], "tracking")
        self.assertEqual(
            uploaded["key"],
            "jobs/job-runtime-asset/attempts/attempt-a/tracking/tracking.json",
        )
        self.assertEqual(uploaded["content_type"], "application/json")
        persisted = uploaded["payload"]
        self.assertFalse(persisted["tracking_success"])
        self.assertEqual(
            persisted["tracking_status"],
            "REID_FALLBACK_LEGACY_UNVERIFIED",
        )
        self.assertEqual(persisted["segments"], [])
        self.assertEqual(persisted["segments_total"], 0)
        self.assertEqual(persisted["bboxes"], [])
        self.assertEqual(persisted["anchors_matched"], 0)
        self.assertEqual(persisted["anchor_matches"], [])
        self.assertEqual(persisted["anchors_used"], {})
        persisted_json = json.dumps(persisted, sort_keys=True)
        self.assertNotIn("legacy-identity-secret", persisted_json)
        self.assertNotIn("local_track_id", persisted_json)
        self.assertNotIn("candidate_id", persisted_json)
        self.assertNotIn("selected_track_id", persisted_json)
        self.assertNotIn("selected_track_ids", persisted_json)
        self.assertNotIn("reacquire_score", persisted_json)
        self.assertNotIn("reacquire_source", persisted_json)
        self.assertNotIn("reacquire_metrics", persisted_json)
        self.assertNotIn("tracking_key", persisted)
        self.assertNotIn("tracking_url", persisted)
        self.assertEqual(output["tracking_key"], uploaded["key"])
        self.assertEqual(
            output["tracking_url"],
            "https://safe.example/tracking/"
            "jobs/job-runtime-asset/attempts/attempt-a/tracking/tracking.json",
        )
        self.assertEqual(persisted["analysis_attempt_id"], "attempt-a")
        self.assertEqual(output["analysis_attempt_id"], "attempt-a")
        self.assertNotEqual(
            output["tracking_url"],
            "https://unsafe.example/raw-tracking.json",
        )

    def test_source_tracking_keys_cannot_override_attempt_key(self):
        module = SimpleNamespace(
            S3_ENDPOINT_URL="http://s3.internal",
            _get_s3_client=lambda _endpoint: object(),
            _ensure_bucket_exists=lambda _client, _bucket: None,
            _presign_get_object=(
                lambda bucket, key, _expires: f"https://safe.example/{bucket}/{key}"
            ),
        )
        job_id = "550e8400-e29b-41d4-a716-446655440000"
        attempt_id = "attempt-a"
        attempt_key = f"jobs/{job_id}/attempts/{attempt_id}/tracking/tracking.json"

        with tempfile.TemporaryDirectory() as temporary_root:
            source_keys = (
                "jobs/another-job/tracking/tracking.json",
                "../../victim/tracking.json",
                str(Path(temporary_root).parent / "absolute-victim.json"),
                r"..\victim\tracking.json",
            )
            for source_key in source_keys:
                with self.subTest(source_key=source_key):
                    uploads = []

                    def upload_file(
                        _client,
                        bucket,
                        path,
                        key,
                        content_type,
                    ):
                        uploads.append(
                            {
                                "bucket": bucket,
                                "key": key,
                                "content_type": content_type,
                                "payload": json.loads(Path(path).read_text()),
                            }
                        )

                    module._upload_file = upload_file
                    with patch.dict(
                        os.environ,
                        {"S3_BUCKET": "tracking"},
                        clear=False,
                    ), patch(
                        "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
                        Path(temporary_root),
                    ):
                        output = persist_canonical_tracking_artifact(
                            {
                                "tracking_key": source_key,
                                "tracking_url": "https://unsafe.example/raw.json",
                                "segments": [],
                                "tracking_success": False,
                            },
                            job_id=job_id,
                            tracking_module=module,
                            analysis_attempt_id=attempt_id,
                        )

                    self.assertEqual(len(uploads), 1)
                    self.assertEqual(uploads[0]["key"], attempt_key)
                    self.assertEqual(output["tracking_key"], attempt_key)
                    self.assertNotEqual(output["tracking_key"], source_key)
                    self.assertEqual(output["analysis_attempt_id"], attempt_id)
                    self.assertEqual(
                        sorted(
                            str(path.relative_to(temporary_root))
                            for path in Path(temporary_root).rglob("*")
                            if path.is_file()
                        ),
                        [f"{job_id}/attempts/{attempt_id}/" "tracking/tracking.json"],
                    )

    def test_tracking_artifacts_are_isolated_between_attempts(self):
        uploads = []
        module = SimpleNamespace(
            S3_ENDPOINT_URL="http://s3.internal",
            _get_s3_client=lambda _endpoint: object(),
            _ensure_bucket_exists=lambda _client, _bucket: None,
            _upload_file=lambda _client, _bucket, path, key, _content_type: (
                uploads.append(
                    {
                        "key": key,
                        "payload": json.loads(Path(path).read_text()),
                    }
                )
            ),
            _presign_get_object=lambda _bucket, key, _expires: f"https://safe/{key}",
        )

        with tempfile.TemporaryDirectory() as temporary_root, patch.dict(
            os.environ,
            {"S3_BUCKET": "tracking"},
            clear=False,
        ), patch(
            "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
            Path(temporary_root),
        ):
            output_a = persist_canonical_tracking_artifact(
                {"segments": [{"attempt": "a"}]},
                job_id="job-1",
                tracking_module=module,
                analysis_attempt_id="attempt-a",
            )
            output_b = persist_canonical_tracking_artifact(
                {"segments": [{"attempt": "b"}]},
                job_id="job-1",
                tracking_module=module,
                analysis_attempt_id="attempt-b",
            )

            local_a = json.loads(
                (
                    Path(temporary_root)
                    / "job-1"
                    / "attempts"
                    / "attempt-a"
                    / "tracking"
                    / "tracking.json"
                ).read_text()
            )
            local_b = json.loads(
                (
                    Path(temporary_root)
                    / "job-1"
                    / "attempts"
                    / "attempt-b"
                    / "tracking"
                    / "tracking.json"
                ).read_text()
            )

        self.assertNotEqual(output_a["tracking_key"], output_b["tracking_key"])
        self.assertEqual(local_a["segments"], [{"attempt": "a"}])
        self.assertEqual(local_b["segments"], [{"attempt": "b"}])
        self.assertEqual(
            [item["payload"]["analysis_attempt_id"] for item in uploads],
            ["attempt-a", "attempt-b"],
        )

    def test_malicious_job_ids_never_write_or_upload(self):
        uploads = []
        module = SimpleNamespace(
            S3_ENDPOINT_URL="http://s3.internal",
            _get_s3_client=lambda _endpoint: object(),
            _ensure_bucket_exists=lambda _client, _bucket: None,
            _upload_file=lambda *_args, **_kwargs: uploads.append(True),
            _presign_get_object=lambda *_args, **_kwargs: "https://safe.example",
        )
        malicious_job_ids = (
            None,
            "",
            ".",
            "..",
            "../victim",
            "job/child",
            r"job\child",
            "/absolute/job",
            r"C:\absolute\job",
            "job..victim",
            " job",
            "job ",
            "job\nchild",
            "job\x00child",
            "nul",
        )

        for malicious_job_id in malicious_job_ids:
            with self.subTest(
                job_id=repr(malicious_job_id)
            ), tempfile.TemporaryDirectory() as temporary_root, patch.dict(
                os.environ,
                {"S3_BUCKET": "tracking"},
                clear=False,
            ), patch(
                "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
                Path(temporary_root),
            ):
                uploads.clear()
                output = persist_canonical_tracking_artifact(
                    {
                        "tracking_key": "jobs/victim/tracking/tracking.json",
                        "tracking_url": "https://unsafe.example/raw.json",
                        "segments": [],
                    },
                    job_id=malicious_job_id,
                    tracking_module=module,
                )

                self.assertNotIn("tracking_key", output)
                self.assertNotIn("tracking_url", output)
                self.assertEqual(uploads, [])
                self.assertEqual(list(Path(temporary_root).iterdir()), [])

    def test_failed_asset_rewrite_drops_raw_references(self):
        def upload_file(*_args, **_kwargs):
            raise RuntimeError("upload failed")

        module = SimpleNamespace(
            S3_ENDPOINT_URL="http://s3.internal",
            _get_s3_client=lambda _endpoint: object(),
            _ensure_bucket_exists=lambda _client, _bucket: None,
            _upload_file=upload_file,
            _presign_get_object=(
                lambda bucket, key, _expires: f"https://safe.example/{bucket}/{key}"
            ),
        )
        raw = {
            "mode": "full_match_windowed",
            "tracking_key": "jobs/job-rewrite-failed/tracking/tracking.json",
            "tracking_url": "https://unsafe.example/raw-tracking.json",
            "segments": [
                {
                    "selected_track_id": 7,
                    "identity_id": "legacy-identity-secret",
                    "identity_status": "ACCEPTED",
                    "bboxes": [{"t": 1.0}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_root, patch.dict(
            os.environ,
            {"S3_BUCKET": "tracking"},
            clear=False,
        ), patch(
            "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
            Path(temporary_root),
        ), patch(
            "app.reid.full_match_runtime.logger.exception"
        ) as log_exception:
            output = persist_fail_closed_legacy_fallback(
                raw,
                reason_code="REID_RUNTIME_EXCEPTION",
                job_id="job-rewrite-failed",
                tracking_module=module,
                analysis_attempt_id="attempt-a",
            )
            local_payload = json.loads(
                (
                    Path(temporary_root)
                    / "job-rewrite-failed"
                    / "attempts"
                    / "attempt-a"
                    / "tracking"
                    / "tracking.json"
                ).read_text()
            )

        self.assertNotIn("tracking_key", output)
        self.assertNotIn("tracking_url", output)
        self.assertFalse(output["tracking_success"])
        self.assertEqual(local_payload["segments"], [])
        self.assertNotIn("legacy-identity-secret", json.dumps(local_payload))
        log_exception.assert_called_once()

    def test_runtime_failure_can_fail_closed(self):
        def original():
            return {"mode": "legacy"}

        def implementation(*, fallback):
            raise RuntimeError("boom")

        module = SimpleNamespace(track_player_windowed=original)
        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_ENABLED": "1",
                "PLAYER_REID_FAIL_OPEN": "0",
            },
            clear=False,
        ):
            install_windowed_reid(module, implementation)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                module.track_player_windowed()

    def test_tracking_timeout_returns_partial_without_restarting_legacy(self):
        class TrackingTimeoutError(RuntimeError):
            pass

        calls = []

        def original(*args, **kwargs):
            calls.append("legacy")
            return {"mode": "legacy"}

        def implementation(*args, fallback, **kwargs):
            raise TrackingTimeoutError("timeout")

        module = SimpleNamespace(
            track_player_windowed=original,
            TrackingTimeoutError=TrackingTimeoutError,
        )
        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_ENABLED": "1",
                "PLAYER_REID_FAIL_OPEN": "1",
            },
            clear=False,
        ), patch("app.reid.runtime.mark_partial_timeout") as mark_partial:
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(
                "job-1",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                video_duration_sec=6000.0,
                fps=5,
            )

        self.assertEqual(calls, [])
        self.assertTrue(output["partial"])
        self.assertEqual(output["partial_reason"], "TRACKING_TIMEOUT")
        self.assertEqual(output["reid_summary"]["status"], "PARTIAL_TIMEOUT")
        self.assertEqual(output["runtime_profile"]["fps"], 2)
        self.assertEqual(output["autonomous_segments_with_player"], 0)
        self.assertEqual(output["autonomous_bboxes_count"], 0)
        self.assertEqual(output["tracking_scope_status"], "EMPTY")
        self.assertEqual(output["windows_processed"], 0)
        self.assertEqual(
            output["reid_summary"]["autonomous_segments_with_player"],
            0,
        )
        self.assertEqual(output["reid_summary"]["autonomous_bboxes_count"], 0)
        self.assertEqual(output["reid_summary"]["tracking_scope_status"], "EMPTY")
        self.assertEqual(output["reid_summary"]["windows_processed"], 0)
        mark_partial.assert_called_once()

    def test_disabled_path_does_not_enter_legacy_timeout(self):
        class TrackingTimeoutError(RuntimeError):
            pass

        calls = []

        def original(*args, **kwargs):
            calls.append((args, kwargs))
            raise TrackingTimeoutError("timeout")

        module = SimpleNamespace(
            track_player_windowed=original,
            TrackingTimeoutError=TrackingTimeoutError,
        )
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "0"}, clear=False), patch(
            "app.reid.runtime.mark_partial_timeout"
        ) as mark_partial:
            install_windowed_reid(module)
            output = module.track_player_windowed(
                "job-disabled-timeout",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                video_duration_sec=6000.0,
                fps=5,
            )
        self.assertEqual(calls, [])
        self.assertFalse(output["partial"])
        self.assertEqual(output["identity_mode"], "disabled")
        self.assertEqual(output["reid_summary"]["status"], "DISABLED")
        self.assertFalse(output["tracking_success"])
        self.assertEqual(
            output["tracking_status"],
            "PLAYER_REID_DISABLED_UNVERIFIED",
        )
        self.assertEqual(output["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(output["autonomous_segments_with_player"], 0)
        self.assertEqual(output["autonomous_bboxes_count"], 0)
        self.assertEqual(output["tracking_scope_status"], "EMPTY")
        self.assertEqual(output["windows_processed"], 0)
        self.assertEqual(
            output["reid_summary"]["autonomous_segments_with_player"],
            0,
        )
        self.assertEqual(output["reid_summary"]["autonomous_bboxes_count"], 0)
        self.assertEqual(output["reid_summary"]["tracking_scope_status"], "EMPTY")
        self.assertEqual(output["reid_summary"]["windows_processed"], 0)
        self.assertNotIn(
            "TRACKING_BUDGET_EXHAUSTED", output["reid_summary"]["reason_codes"]
        )
        mark_partial.assert_not_called()

    def test_long_full_match_uses_cpu_budget_profile(self):
        captured = {}

        def original(*args, **kwargs):
            return {"mode": "legacy"}

        def implementation(*args, fallback, **kwargs):
            captured.update(kwargs)
            return {"mode": "reid"}

        module = SimpleNamespace(track_player_windowed=original)
        environment = {
            "PLAYER_REID_ENABLED": "1",
            "FULL_MATCH_TARGET_SAMPLES": "6000",
            "FULL_MATCH_MIN_FPS": "1",
            "FULL_MATCH_MAX_FPS": "2",
            "FULL_MATCH_WINDOW_SEC": "60",
            "FULL_MATCH_OVERLAP_SEC": "5",
            "FULL_MATCH_DETECTOR_MODEL": "yolo11n.pt",
        }
        with patch.dict(os.environ, environment, clear=True):
            install_windowed_reid(module, implementation)
            output = module.track_player_windowed(
                "job-2",
                "/tmp/input.mp4",
                {"t": 10.0},
                [],
                video_duration_sec=6000.0,
                fps=5,
                window_sec=45.0,
                overlap_sec=10.0,
            )

        self.assertEqual(captured["fps"], 1)
        self.assertEqual(captured["window_sec"], 60.0)
        self.assertEqual(captured["overlap_sec"], 5.0)
        self.assertEqual(captured["detector_model"], "yolo11n.pt")
        self.assertEqual(output["runtime_profile"]["fps"], 1)
        self.assertLess(output["runtime_profile"]["estimated_samples"], 7000)

    def test_long_match_defaults_use_two_fps_small_model_profile(self):
        with patch.dict(os.environ, {}, clear=True):
            profile = select_full_match_profile(
                video_duration_sec=5931.775,
                requested_fps=5,
                requested_window_sec=45.0,
                requested_overlap_sec=10.0,
                requested_detector_model="yolo11s.pt",
            )

        self.assertEqual(profile.fps, 2)
        self.assertEqual(profile.window_sec, 60.0)
        self.assertEqual(profile.overlap_sec, 5.0)
        self.assertEqual(profile.detector_model, "yolo11s.pt")
        self.assertEqual(profile.target_samples, 12000)
        self.assertEqual(profile.estimated_samples, 12943)

    def test_long_match_explicit_quality_budget_overrides_remain_stable(self):
        environment = {
            "FULL_MATCH_TARGET_SAMPLES": "6000",
            "FULL_MATCH_MIN_FPS": "1",
            "FULL_MATCH_MAX_FPS": "2",
            "FULL_MATCH_WINDOW_SEC": "60",
            "FULL_MATCH_OVERLAP_SEC": "5",
            "FULL_MATCH_DETECTOR_MODEL": "yolo11n.pt",
        }
        with patch.dict(os.environ, environment, clear=True):
            profile = select_full_match_profile(
                video_duration_sec=5931.775,
                requested_fps=5,
                requested_window_sec=45.0,
                requested_overlap_sec=10.0,
                requested_detector_model="yolo11s.pt",
            )

        self.assertEqual(profile.fps, 1)
        self.assertEqual(profile.window_sec, 60.0)
        self.assertEqual(profile.overlap_sec, 5.0)
        self.assertEqual(profile.detector_model, "yolo11n.pt")
        self.assertEqual(profile.target_samples, 6000)
        self.assertEqual(profile.estimated_samples, 6472)

    def test_forced_fps_override_remains_bounded_by_configured_max(self):
        environment = {
            "FULL_MATCH_TARGET_SAMPLES": "50000",
            "FULL_MATCH_MIN_FPS": "1",
            "FULL_MATCH_MAX_FPS": "2",
            "FULL_MATCH_TRACKING_FPS": "9",
            "FULL_MATCH_DETECTOR_MODEL": "custom-detector.pt",
        }
        with patch.dict(os.environ, environment, clear=True):
            profile = select_full_match_profile(
                video_duration_sec=5931.775,
                requested_fps=10,
            )

        self.assertEqual(profile.fps, 2)
        self.assertEqual(profile.detector_model, "custom-detector.pt")
        self.assertEqual(profile.target_samples, 50000)

    def test_short_video_preserves_requested_quality_profile(self):
        with patch.dict(os.environ, {}, clear=True):
            profile = select_full_match_profile(
                video_duration_sec=600.0,
                requested_fps=5,
                requested_window_sec=45.0,
                requested_overlap_sec=10.0,
                requested_detector_model="yolo11s.pt",
            )

        self.assertEqual(profile.fps, 5)
        self.assertEqual(profile.window_sec, 45.0)
        self.assertEqual(profile.overlap_sec, 10.0)
        self.assertEqual(profile.detector_model, "yolo11s.pt")

    def test_progress_adapter_maps_window_stage_to_visible_range(self):
        progress_calls = []

        def original_tracker():
            return {"mode": "legacy"}

        def implementation(*, fallback):
            return {"mode": "reid"}

        def update_progress(
            job_id,
            pct,
            message,
            *,
            analysis_attempt_id=None,
        ):
            progress_calls.append((job_id, pct, message, analysis_attempt_id))

        module = SimpleNamespace(
            track_player_windowed=original_tracker,
            _update_tracking_progress=update_progress,
        )
        with patch.dict(os.environ, {"PLAYER_REID_ENABLED": "1"}, clear=False):
            install_windowed_reid(module, implementation)
            module._update_tracking_progress(
                "job-3",
                25,
                "Tracking player with experimental ReID",
                analysis_attempt_id="attempt-a",
            )

        self.assertEqual(progress_calls[0][0], "job-3")
        self.assertEqual(progress_calls[0][1], 53)
        self.assertIn("50% finestre", progress_calls[0][2])
        self.assertEqual(progress_calls[0][3], "attempt-a")


if __name__ == "__main__":
    unittest.main()
