import copy
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core import job_recovery

NOW = datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc)


def interrupted_job(job_id="job-recovery-1"):
    attempt_id = "attempt-a"
    return SimpleNamespace(
        id=job_id,
        status="RUNNING",
        target={"analysis_attempt_id": attempt_id},
        error=None,
        failure_reason=None,
        warnings=["OLD_WARNING"],
        progress={
            "step": "TRACKING",
            "pct": 52,
            "updated_at": (NOW - timedelta(hours=7)).isoformat(),
            "analysis_attempt_id": attempt_id,
            "analysis_task_id": "celery-task-a",
            "analysis_task_retry": 0,
        },
        updated_at=NOW - timedelta(hours=7),
    )


class ScanResult:
    def __init__(self, job_ids):
        self.job_ids = job_ids

    def scalars(self):
        return self

    def all(self):
        return self.job_ids


class LockResult:
    def __init__(self, job):
        self.job = job

    def scalar_one_or_none(self):
        return self.job


class DummyDB:
    def __init__(self, job, *, scan_ids=None, on_lock=None):
        self.job = job
        self.scan_ids = list(scan_ids or [job.id])
        self.on_lock = on_lock
        self.execute_calls = 0
        self.commits = 0
        self.rollbacks = 0
        self.adds = 0
        self.closed = False
        self.lock_options = []
        self.locked = []

    def execute(self, statement):
        self.execute_calls += 1
        if self.execute_calls == 1:
            return ScanResult(self.scan_ids)
        self.lock_options.append(dict(statement.get_execution_options()))
        self.locked.append(getattr(statement, "_for_update_arg", None) is not None)
        if self.on_lock is not None:
            self.on_lock(self.job)
        return LockResult(self.job)

    def add(self, _job):
        self.adds += 1

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class JobRecoveryTests(unittest.TestCase):
    def recovery_env(self):
        return patch.dict(
            os.environ,
            {
                "RECOVER_INTERRUPTED_JOBS_ON_WORKER_START": "1",
                "APP_GIT_SHA": "revision-a",
                "INTERRUPTED_JOB_STALE_AFTER_SECONDS": "300",
                "INTERRUPTED_JOB_PROBE_GRACE_SECONDS": "60",
            },
        )

    def run_recovery(self, db, *, now=NOW):
        return job_recovery.recover_interrupted_jobs(
            lambda: db,
            recovery_owner="worker-a:revision-a",
            recovery_revision="revision-a",
            now=now,
        )

    def test_recovery_is_destructive_only_after_two_unchanged_stale_probes(self):
        job = interrupted_job()
        first_db = DummyDB(job)

        with self.recovery_env():
            first_count = self.run_recovery(first_db)

        self.assertEqual(first_count, 0)
        self.assertEqual(job.status, "RUNNING")
        probe = copy.deepcopy(job.progress["recovery_probe"])
        self.assertTrue(probe["token"])
        self.assertEqual(probe["analysis_attempt_id"], "attempt-a")
        self.assertEqual(probe["analysis_task_id"], "celery-task-a")
        self.assertEqual(probe["recovery_revision"], "revision-a")
        self.assertEqual(first_db.commits, 1)
        self.assertTrue(all(first_db.locked))
        self.assertTrue(
            all(options.get("populate_existing") for options in first_db.lock_options)
        )

        second_db = DummyDB(job)
        with self.recovery_env():
            second_count = self.run_recovery(
                second_db,
                now=NOW + timedelta(seconds=61),
            )

        self.assertEqual(second_count, 1)
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.failure_reason, "WORKER_RESTARTED")
        self.assertIn("WORKER_RESTARTED", job.warnings)
        self.assertEqual(job.progress["step"], "FAILED")
        self.assertEqual(job.progress["interrupted_progress"]["pct"], 52)
        self.assertEqual(job.progress["recovery"]["token"], probe["token"])
        self.assertEqual(job.progress["recovery"]["revision"], "revision-a")
        self.assertEqual(second_db.commits, 1)
        self.assertTrue(second_db.closed)

    def test_two_recovery_sessions_cannot_recover_the_same_job_twice(self):
        job = interrupted_job()
        first_probe_db = DummyDB(job)
        with self.recovery_env():
            self.run_recovery(first_probe_db)

        recovery_time = NOW + timedelta(seconds=61)
        winner_db = DummyDB(job)
        loser_db = DummyDB(job, scan_ids=[job.id])
        with self.recovery_env():
            winner_count = self.run_recovery(winner_db, now=recovery_time)
            loser_count = self.run_recovery(loser_db, now=recovery_time)

        self.assertEqual(winner_count, 1)
        self.assertEqual(loser_count, 0)
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(winner_db.commits, 1)
        self.assertEqual(loser_db.commits, 0)

    def test_two_probe_sessions_share_one_recovery_token(self):
        job = interrupted_job()
        first_db = DummyDB(job)
        second_db = DummyDB(job)

        with self.recovery_env():
            first_count = self.run_recovery(first_db)
            first_token = job.progress["recovery_probe"]["token"]
            second_count = self.run_recovery(second_db)

        self.assertEqual(first_count, 0)
        self.assertEqual(second_count, 0)
        self.assertEqual(job.status, "RUNNING")
        self.assertEqual(job.progress["recovery_probe"]["token"], first_token)
        self.assertEqual(first_db.commits, 1)
        self.assertEqual(second_db.commits, 0)

    def test_retry_attempt_rotation_between_scan_and_lock_is_a_barrier(self):
        job = interrupted_job()

        def rotate_to_retry_b(current):
            current.status = "QUEUED"
            current.target = {"analysis_attempt_id": "attempt-b"}
            current.progress = {
                "step": "QUEUED",
                "pct": 20,
                "updated_at": NOW.isoformat(),
                "analysis_attempt_id": "attempt-b",
            }
            current.error = None
            current.failure_reason = None
            current.warnings = []

        db = DummyDB(job, on_lock=rotate_to_retry_b)
        with self.recovery_env():
            count = self.run_recovery(db)

        self.assertEqual(count, 0)
        self.assertEqual(job.status, "QUEUED")
        self.assertEqual(job.target["analysis_attempt_id"], "attempt-b")
        self.assertEqual(job.progress["analysis_attempt_id"], "attempt-b")
        self.assertNotIn("recovery_probe", job.progress)
        self.assertEqual(db.commits, 0)
        self.assertTrue(all(db.locked))
        self.assertTrue(
            all(options.get("populate_existing") for options in db.lock_options)
        )

    def test_missing_task_ownership_fails_closed_without_probe(self):
        job = interrupted_job()
        job.progress.pop("analysis_task_id")
        db = DummyDB(job)

        with self.recovery_env():
            count = self.run_recovery(db)

        self.assertEqual(count, 0)
        self.assertEqual(job.status, "RUNNING")
        self.assertNotIn("recovery_probe", job.progress)
        self.assertEqual(db.commits, 0)

    def test_fresh_task_heartbeat_is_never_probed(self):
        job = interrupted_job()
        job.progress["updated_at"] = (NOW - timedelta(seconds=30)).isoformat()
        db = DummyDB(job)

        with self.recovery_env():
            count = self.run_recovery(db)

        self.assertEqual(count, 0)
        self.assertEqual(job.status, "RUNNING")
        self.assertNotIn("recovery_probe", job.progress)
        self.assertEqual(db.commits, 0)

    def test_unknown_revision_fails_closed_before_opening_session(self):
        called = False

        def factory():
            nonlocal called
            called = True
            return DummyDB(interrupted_job())

        with self.recovery_env():
            count = job_recovery.recover_interrupted_jobs(
                factory,
                recovery_owner="worker-a",
                recovery_revision="unknown",
                now=NOW,
            )

        self.assertEqual(count, 0)
        self.assertFalse(called)

    def test_recovery_is_disabled_by_default(self):
        called = False

        def factory():
            nonlocal called
            called = True
            return DummyDB(interrupted_job())

        with patch.dict(
            os.environ,
            {"RECOVER_INTERRUPTED_JOBS_ON_WORKER_START": "0"},
        ):
            self.assertEqual(job_recovery.recover_interrupted_jobs(factory), 0)
        self.assertFalse(called)

        compose = Path("docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn(
            "RECOVER_INTERRUPTED_JOBS_ON_WORKER_START:-0",
            compose,
        )


if __name__ == "__main__":
    unittest.main()
