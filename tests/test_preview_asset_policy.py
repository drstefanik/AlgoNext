import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.core.preview_asset_policy import (
    install_preview_asset_policy,
    install_worker_preview_asset_policy,
)
from app.core.tracking_outcome import apply_tracking_outcome


class ExistingObjectClient:
    def head_object(self, *, Bucket, Key):
        return {"Bucket": Bucket, "Key": Key}


class MissingObjectError(RuntimeError):
    def __init__(self):
        super().__init__("not found")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class MissingObjectClient:
    def head_object(self, *, Bucket, Key):
        raise MissingObjectError()


class PreviewAssetPolicyTests(unittest.TestCase):
    def pipeline(self, jobs, uploads, commits):
        def legacy_generator(**_kwargs):
            raise AssertionError("legacy preview generator must be replaced")

        def update_job(db, job_id, updater):
            job = db.get(job_id)
            if job is None:
                return False
            updater(job)
            commits.append(
                {
                    "preview_frames": copy.deepcopy(
                        list(getattr(job, "preview_frames", None) or [])
                    ),
                    "result": copy.deepcopy(getattr(job, "result", None) or {}),
                }
            )
            return True

        def reload_job(db, job_id):
            return db.get(job_id)

        def safe_commit(_db):
            raise AssertionError("preview policy must not perform a second commit")

        def run(command):
            output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"frame")
            return ""

        def upload_file(_client, bucket, path, key, content_type):
            uploads.append(
                {
                    "bucket": bucket,
                    "path": str(path),
                    "key": key,
                    "content_type": content_type,
                }
            )

        return SimpleNamespace(
            _generate_tracking_preview_frames=legacy_generator,
            update_job=update_job,
            reload_job=reload_job,
            safe_commit=safe_commit,
            _run=run,
            probe_image_dimensions=lambda _path: (1920, 1080),
            upload_file=upload_file,
        )

    def test_tracking_frames_use_separate_namespace_and_do_not_replace_selection(self):
        job_id = "job-123"
        selection_frames = [
            {
                "time_sec": 179.751,
                "bucket": "fnh",
                "key": f"jobs/{job_id}/frames/frame_0001.jpg",
                "width": 1920,
                "height": 1080,
                "tracks": [{"track_id": 52}],
            }
        ]
        job = SimpleNamespace(
            preview_frames=list(selection_frames),
            result={"assets": {"input_video": {"key": "input.mp4"}}},
        )
        jobs = {job_id: job}
        uploads = []
        commits = []
        pipeline = self.pipeline(jobs, uploads, commits)

        self.assertTrue(install_preview_asset_policy(pipeline))
        with tempfile.TemporaryDirectory() as tmpdir:
            generated = pipeline._generate_tracking_preview_frames(
                job_id=job_id,
                input_path=Path(tmpdir) / "input.mp4",
                frames_dir=Path(tmpdir) / "tracking_frames",
                s3_internal=object(),
                s3_bucket="fnh",
                candidates=[
                    {
                        "time_sec": 59.751,
                        "has_player": True,
                        "is_target": False,
                    }
                ],
            )

        self.assertEqual(len(generated), 1)
        self.assertEqual(
            generated[0]["key"],
            f"jobs/{job_id}/tracking_frames/tracking_frame_0001.jpg",
        )
        self.assertEqual(uploads[0]["key"], generated[0]["key"])
        self.assertNotEqual(generated[0]["key"], selection_frames[0]["key"])

        pipeline.update_job(
            jobs,
            job_id,
            lambda current_job: setattr(current_job, "preview_frames", generated),
        )

        self.assertEqual(job.preview_frames, selection_frames)
        self.assertEqual(commits[0]["preview_frames"], selection_frames)
        self.assertEqual(
            job.result["tracking_review_frames"][0]["key"], generated[0]["key"]
        )
        self.assertEqual(
            job.result["assets"]["tracking_review_frames"][0]["key"],
            generated[0]["key"],
        )
        self.assertTrue(
            job.result["preview_asset_integrity"]["selection_frames_immutable"]
        )
        self.assertEqual(len(commits), 1)

    def test_analysis_refresh_cannot_overwrite_selection_object_or_metadata(self):
        job_id = "job-456"
        key = f"jobs/{job_id}/frames/frame_0001.jpg"
        selection_frames = [
            {
                "time_sec": 179.751,
                "bucket": "fnh",
                "key": key,
                "width": 1920,
                "height": 1080,
                "tracks": [{"track_id": 52}],
            }
        ]
        attempted_refresh = [
            {
                "time_sec": 59.751,
                "bucket": "fnh",
                "key": key,
                "width": 1920,
                "height": 1080,
            }
        ]
        job = SimpleNamespace(preview_frames=list(selection_frames), result={})
        jobs = {job_id: job}
        uploads = []
        commits = []
        pipeline = self.pipeline(jobs, uploads, commits)
        install_preview_asset_policy(pipeline)

        pipeline.upload_file(
            ExistingObjectClient(),
            "fnh",
            Path("/tmp/replacement.jpg"),
            key,
            "image/jpeg",
        )
        self.assertEqual(uploads, [])

        pipeline.update_job(
            jobs,
            job_id,
            lambda current_job: setattr(
                current_job,
                "preview_frames",
                attempted_refresh,
            ),
        )

        self.assertEqual(job.preview_frames, selection_frames)
        self.assertEqual(commits[0]["preview_frames"], selection_frames)
        self.assertTrue(
            job.result["preview_asset_integrity"]["selection_refresh_suppressed"]
        )
        self.assertEqual(len(commits), 1)

    def test_fail_closed_recovery_retains_selectable_preview_tracks(self):
        job_id = "job-reselection"
        key = f"jobs/{job_id}/frames/frame_0001.jpg"
        selection_frames = [
            {
                "time_sec": 179.751,
                "bucket": "fnh",
                "key": key,
                "tracks": [
                    {
                        "track_id": 10,
                        "bbox": {"x": 0.4, "y": 0.2, "w": 0.02, "h": 0.08},
                    }
                ],
            }
        ]
        job = SimpleNamespace(
            status="RUNNING",
            preview_frames=copy.deepcopy(selection_frames),
            result={"candidates": {"candidates": [{"track_id": 10}]}},
            target={
                "confirmed": True,
                "full_match_mode": True,
                "selection": {"frame_key": key},
                "selections": [{"frame_key": key}],
            },
            player_ref={"track_id": 10},
            anchor={"t": 179.751},
            warnings=[],
            error=None,
            failure_reason=None,
            progress={"step": "TRACKING", "pct": 35},
        )
        jobs = {job_id: job}
        pipeline = self.pipeline(jobs, [], [])
        install_preview_asset_policy(pipeline)

        pipeline.update_job(
            jobs,
            job_id,
            lambda current_job: setattr(
                current_job,
                "preview_frames",
                [
                    {
                        "time_sec": 59.751,
                        "bucket": "fnh",
                        "key": key,
                        "tracks": [],
                    }
                ],
            ),
        )
        pipeline.update_job(
            jobs,
            job_id,
            lambda current_job: apply_tracking_outcome(
                current_job,
                {
                    "tracking_success": False,
                    "tracking_status": "ANCHOR_NOT_FOUND",
                    "action_required": "RESELECT_PLAYER",
                    "bboxes_count": 0,
                    "segments_total": 108,
                    "segments_with_player": 0,
                    "windows_processed": 1,
                    "anchors_total": 1,
                    "anchors_matched": 0,
                    "reid_summary": {
                        "reason_codes": ["REID_ANCHORS_NOT_FOUND"],
                    },
                },
                set_progress=lambda current, step, pct, message: setattr(
                    current,
                    "progress",
                    {"step": step, "pct": pct, "message": message},
                ),
            ),
        )

        self.assertEqual(job.status, "WAITING_FOR_PLAYER")
        self.assertEqual(job.preview_frames, selection_frames)
        self.assertEqual(job.preview_frames[0]["tracks"][0]["track_id"], 10)

    def test_first_selection_frame_upload_is_allowed(self):
        job_id = "job-new"
        key = f"jobs/{job_id}/frames/frame_0001.jpg"
        uploads = []
        pipeline = self.pipeline({}, uploads, [])
        install_preview_asset_policy(pipeline)

        pipeline.upload_file(
            MissingObjectClient(),
            "fnh",
            Path("/tmp/frame.jpg"),
            key,
            "image/jpeg",
        )

        self.assertEqual([item["key"] for item in uploads], [key])

    def test_selection_track_enrichment_is_preserved(self):
        job_id = "job-tracks"
        key = f"jobs/{job_id}/frames/frame_0001.jpg"
        job = SimpleNamespace(
            preview_frames=[{"time_sec": 10.0, "key": key, "tracks": []}],
            result={},
        )
        jobs = {job_id: job}
        commits = []
        pipeline = self.pipeline(jobs, [], commits)
        install_preview_asset_policy(pipeline)

        enriched = [
            {
                "time_sec": 10.0,
                "key": key,
                "tracks": [{"track_id": 9}],
            }
        ]
        pipeline.update_job(
            jobs,
            job_id,
            lambda current_job: setattr(current_job, "preview_frames", enriched),
        )

        self.assertEqual(job.preview_frames[0]["tracks"], [{"track_id": 9}])
        self.assertEqual(len(commits), 1)

    def test_unrelated_updates_are_not_intercepted(self):
        job = SimpleNamespace(preview_frames=[], result={}, status="RUNNING")
        jobs = {"job": job}
        commits = []
        pipeline = self.pipeline(jobs, [], commits)
        install_preview_asset_policy(pipeline)

        pipeline.update_job(
            jobs,
            "job",
            lambda current_job: setattr(current_job, "status", "PARTIAL"),
        )

        self.assertEqual(job.status, "PARTIAL")
        self.assertEqual(job.preview_frames, [])
        self.assertNotIn("tracking_review_frames", job.result)
        self.assertEqual(len(commits), 1)

    def test_install_is_idempotent(self):
        pipeline = self.pipeline({}, [], [])
        self.assertTrue(install_preview_asset_policy(pipeline))
        self.assertFalse(install_preview_asset_policy(pipeline))

    def test_worker_loader_imports_pipeline_only_when_called(self):
        pipeline = self.pipeline({}, [], [])
        imported = []

        def loader(module_name):
            imported.append(module_name)
            return pipeline

        self.assertEqual(imported, [])
        self.assertTrue(install_worker_preview_asset_policy(loader))
        self.assertEqual(imported, ["app.workers.pipeline"])


if __name__ == "__main__":
    unittest.main()
