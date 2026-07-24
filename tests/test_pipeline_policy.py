import unittest
from types import SimpleNamespace

from app.core.pipeline_policy import (
    install_pipeline_policy,
    install_worker_pipeline_policy,
)


class PipelinePolicyTests(unittest.TestCase):
    def pipeline(self):
        def set_progress(job, step, pct, message=""):
            job.progress = {
                "step": step,
                "pct": int(pct),
                "message": message,
                "updated_at": "now",
            }

        def forbidden(*_args, **_kwargs):
            raise AssertionError("legacy scoring path must not run")

        return SimpleNamespace(
            set_progress=set_progress,
            extract_video_features=forbidden,
            compute_skill_scores=forbidden,
            _update_candidate_score_fields=forbidden,
            _compute_evidence_metrics=forbidden,
            compute_evaluation=forbidden,
            compute_match_rating=forbidden,
            keys_required_for_role=forbidden,
            _compute_performance_score=forbidden,
            _build_explain_text=lambda *_args: "legacy",
            SKILLS_ORDER=["Finishing", "Heading"],
        )

    def test_analysis_progress_is_monotonic_and_preserves_runtime_stats(self):
        pipeline = self.pipeline()
        install_pipeline_policy(pipeline)
        job = SimpleNamespace(
            status="RUNNING",
            target={"confirmed": True},
            progress={
                "step": "TRACKING",
                "pct": 67,
                "stats": {"windows_completed": 97, "windows_total": 108},
                "runtime_profile": {"fps": 1},
            },
        )

        pipeline.set_progress(job, "EXTRACTING_FEATURES", 50, "features")
        self.assertEqual(job.progress["pct"], 72)
        self.assertEqual(job.progress["phase"], "FEATURES")
        self.assertEqual(job.progress["stats"]["windows_completed"], 97)
        self.assertEqual(job.progress["runtime_profile"]["fps"], 1)

        pipeline.set_progress(job, "EXTRACTING", 55, "clips")
        self.assertEqual(job.progress["pct"], 78)
        pipeline.set_progress(job, "UPLOADING_CLIPS", 75, "upload")
        self.assertEqual(job.progress["pct"], 84)
        pipeline.set_progress(job, "ANALYZING", 85, "analysis")
        self.assertEqual(job.progress["pct"], 90)
        pipeline.set_progress(job, "FINALIZING", 95, "final")
        self.assertEqual(job.progress["pct"], 96)
        pipeline.set_progress(job, "DONE", 100, "done")
        self.assertEqual(job.progress["pct"], 100)

    def test_preselection_progress_keeps_existing_scale(self):
        pipeline = self.pipeline()
        install_pipeline_policy(pipeline)
        job = SimpleNamespace(
            status="CREATED",
            target={"confirmed": False},
            progress={"pct": 0},
        )
        pipeline.set_progress(job, "EXTRACTING_PREVIEWS", 15, "preview")
        self.assertEqual(job.progress["pct"], 15)

    def test_tracking_only_features_do_not_scan_the_video(self):
        pipeline = self.pipeline()
        install_pipeline_policy(pipeline)
        features = pipeline.extract_video_features(
            "/tmp/input.mp4",
            {
                "format": {"duration": "120.5"},
                "streams": [
                    {
                        "codec_type": "video",
                        "avg_frame_rate": "25/1",
                        "nb_frames": "3012",
                    }
                ],
            },
        )
        self.assertEqual(features["feature_mode"], "tracking_only")
        self.assertFalse(features["validated"])
        self.assertEqual(features["frame_count"], 3012)
        self.assertEqual(features["fps"], 25.0)
        self.assertNotIn("scene_change_count", features)
        self.assertNotIn("scene_change_rate", features)
        self.assertIn(
            "SCENE_BASED_PLAYER_FEATURES_DISABLED", features["reason_codes"]
        )

    def test_skill_scoring_abstains(self):
        pipeline = self.pipeline()
        install_pipeline_policy(pipeline)
        computed, missing = pipeline.compute_skill_scores({})
        self.assertEqual(computed, {})
        self.assertEqual(missing, ["Finishing", "Heading"])
        self.assertIn("si astiene", pipeline._build_explain_text("x", {}, {}, None))

    def test_candidate_stage_persists_evidence_without_scores(self):
        pipeline = self.pipeline()
        install_pipeline_policy(pipeline)
        job = SimpleNamespace(result={"candidates": {"candidates": []}})

        pipeline._update_candidate_score_fields(
            job,
            82.0,
            {"visibility": 80.0},
            {"candidate_metrics": {"coveragePct": 0.125}},
            "legacy score explanation",
        )

        self.assertIsNone(job.result["overall_score"])
        self.assertIsNone(job.result["role_score"])
        self.assertEqual(job.result["radar"], {})
        self.assertEqual(job.result["breakdown"], {})
        self.assertEqual(job.result["evaluation_status"], "TRACKING_ONLY")
        self.assertFalse(job.result["player_evaluation_available"])
        self.assertTrue(job.result["legacy_scores_suppressed"])
        self.assertEqual(
            job.result["evidence_metrics"]["candidate_metrics"]["coveragePct"],
            0.125,
        )

    def test_physical_metrics_and_player_scores_are_never_generated(self):
        pipeline = self.pipeline()
        install_pipeline_policy(pipeline)
        tracking = {
            "bboxes": [
                {"t": 0.0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.2},
                {"t": 1.0, "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2},
            ]
        }

        evidence = pipeline._compute_evidence_metrics(tracking)
        self.assertEqual(evidence["metric_space"], "image_plane_normalized")
        self.assertFalse(evidence["validated"])
        self.assertEqual(evidence["image_motion"]["observed_samples"], 2)
        for forbidden_key in (
            "distance_covered_m",
            "avg_speed_kmh",
            "top_speed_kmh",
            "top_speed_kmh_clamped",
            "sprints_count",
        ):
            self.assertNotIn(forbidden_key, evidence)

        evaluation = pipeline.compute_evaluation("Midfielder", {}, tracking, evidence)
        self.assertIsNone(evaluation["overall_score"])
        self.assertIsNone(evaluation["role_score"])
        self.assertEqual(evaluation["radar"], {})
        self.assertFalse(evaluation["player_evaluation_available"])

        rating = pipeline.compute_match_rating(SimpleNamespace())
        self.assertIsNone(rating["match_rating_10"])
        self.assertIsNone(rating["impact_100"])
        self.assertEqual(pipeline.keys_required_for_role("Midfielder"), [])
        self.assertEqual(pipeline._compute_performance_score(evidence), (0.0, {}))

    def test_install_is_idempotent(self):
        pipeline = self.pipeline()
        self.assertTrue(install_pipeline_policy(pipeline))
        self.assertFalse(install_pipeline_policy(pipeline))

    def test_worker_loader_imports_pipeline_only_when_called(self):
        pipeline = self.pipeline()
        imported = []

        def loader(module_name):
            imported.append(module_name)
            return pipeline

        self.assertEqual(imported, [])
        self.assertTrue(install_worker_pipeline_policy(loader))
        self.assertEqual(imported, ["app.workers.pipeline"])
        self.assertTrue(
            getattr(pipeline.set_progress, "__algonext_pipeline_policy__", False)
        )


if __name__ == "__main__":
    unittest.main()
