import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from app.core.tracking_outcome import StaleAnalysisAttemptError
from app.reid import progress_reporting


class ProgressReportingTests(unittest.TestCase):
    def setUp(self):
        progress_reporting._profiles.clear()

    def profile(self):
        return SimpleNamespace(
            duration_sec=6000.0,
            fps=1,
            window_sec=60.0,
            overlap_sec=5.0,
            detector_model="yolo11n.pt",
            target_samples=6000,
            estimated_samples=6546,
            to_payload=lambda: {
                "duration_sec": 6000.0,
                "fps": 1,
                "window_sec": 60.0,
                "overlap_sec": 5.0,
                "detector_model": "yolo11n.pt",
                "target_samples": 6000,
                "estimated_samples": 6546,
            },
        )

    def test_adapter_reports_window_count(self):
        calls = []
        persisted = []

        def current(job_id, pct, message, *, analysis_attempt_id=None):
            calls.append((job_id, pct, message, analysis_attempt_id))

        tracking = SimpleNamespace(_update_tracking_progress=current)
        profile = self.profile()
        progress_reporting._profiles[("job-1", "attempt-a")] = profile
        progress_reporting.install_progress_stats_adapter(tracking)

        with patch.object(
            progress_reporting,
            "_persist_progress",
            side_effect=lambda job_id, **kwargs: persisted.append((job_id, kwargs)),
        ):
            tracking._update_tracking_progress(
                "job-1",
                25,
                "Tracking player with experimental ReID",
                analysis_attempt_id="attempt-a",
            )

        self.assertEqual(calls[0][1], 25)
        self.assertEqual(calls[0][3], "attempt-a")
        self.assertEqual(
            persisted[0][1]["analysis_attempt_id"],
            "attempt-a",
        )
        self.assertEqual(persisted[0][1]["windows_total"], 109)
        self.assertEqual(persisted[0][1]["windows_completed"], 54)
        self.assertEqual(persisted[0][1]["window_progress_pct"], 50.0)
        self.assertIn("54/109", persisted[0][1]["message"])

    def test_begin_and_end_manage_profile_context(self):
        profile = self.profile()
        with patch.object(progress_reporting, "_persist_progress") as persist:
            progress_reporting.begin_full_match_progress(
                "job-1",
                profile,
                analysis_attempt_id="attempt-a",
            )
        self.assertIs(
            progress_reporting._profiles[("job-1", "attempt-a")],
            profile,
        )
        self.assertEqual(persist.call_args.kwargs["windows_total"], 109)
        self.assertEqual(
            persist.call_args.kwargs["analysis_attempt_id"],
            "attempt-a",
        )
        progress_reporting.end_full_match_progress(
            "job-1",
            analysis_attempt_id="attempt-a",
        )
        self.assertNotIn(("job-1", "attempt-a"), progress_reporting._profiles)

    def test_profile_cleanup_is_isolated_by_attempt(self):
        profile_a = self.profile()
        profile_b = self.profile()
        with patch.object(progress_reporting, "_persist_progress"):
            progress_reporting.begin_full_match_progress(
                "job-1",
                profile_a,
                analysis_attempt_id="attempt-a",
            )
            progress_reporting.begin_full_match_progress(
                "job-1",
                profile_b,
                analysis_attempt_id="attempt-b",
            )

        progress_reporting.end_full_match_progress(
            "job-1",
            analysis_attempt_id="attempt-a",
        )

        self.assertNotIn(("job-1", "attempt-a"), progress_reporting._profiles)
        self.assertIs(
            progress_reporting._profiles[("job-1", "attempt-b")],
            profile_b,
        )

    def test_stale_attempt_is_locked_and_raises_without_commit(self):
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
            target={"analysis_attempt_id": "attempt-b"},
            progress={"analysis_attempt_id": "attempt-b"},
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
        sqlalchemy_module = ModuleType("sqlalchemy")
        sqlalchemy_module.select = lambda _model: Statement()

        with patch.dict(
            sys.modules,
            {
                "app.core.db": db_module,
                "app.core.models": models_module,
                "sqlalchemy": sqlalchemy_module,
            },
        ):
            with self.assertRaises(StaleAnalysisAttemptError):
                progress_reporting._persist_progress(
                    "job-1",
                    analysis_attempt_id="attempt-a",
                    profile=self.profile(),
                    windows_completed=1,
                    windows_total=10,
                    window_progress_pct=10.0,
                )

        self.assertTrue(session.statement.locked)
        self.assertTrue(session.statement.populate_existing)
        self.assertTrue(session.rolled_back)
        self.assertFalse(session.committed)
        self.assertEqual(job.progress["analysis_attempt_id"], "attempt-b")


if __name__ == "__main__":
    unittest.main()
