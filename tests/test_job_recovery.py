import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core import job_recovery


class ScalarResult:
    def __init__(self, jobs):
        self.jobs = jobs

    def scalars(self):
        return self

    def all(self):
        return self.jobs


class DummyDB:
    def __init__(self, jobs):
        self.jobs = jobs
        self.committed = False
        self.closed = False

    def execute(self, _statement):
        return ScalarResult(self.jobs)

    def commit(self):
        self.committed = True

    def rollback(self):
        return None

    def close(self):
        self.closed = True


class JobRecoveryTests(unittest.TestCase):
    def test_running_jobs_become_retryable_failures(self):
        job = SimpleNamespace(
            status="RUNNING",
            error=None,
            failure_reason=None,
            warnings=["OLD_WARNING"],
            progress={"step": "TRACKING", "pct": 52},
            updated_at=None,
        )
        db = DummyDB([job])
        statement = SimpleNamespace(where=lambda _condition: "statement")
        with patch.dict(
            "os.environ", {"RECOVER_INTERRUPTED_JOBS_ON_WORKER_START": "1"}
        ), patch.object(job_recovery, "select", return_value=statement):
            count = job_recovery.recover_interrupted_jobs(lambda: db)
        self.assertEqual(count, 1)
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(job.failure_reason, "WORKER_RESTARTED")
        self.assertIn("WORKER_RESTARTED", job.warnings)
        self.assertEqual(job.progress["step"], "FAILED")
        self.assertEqual(job.progress["interrupted_progress"]["pct"], 52)
        self.assertTrue(db.committed)
        self.assertTrue(db.closed)

    def test_recovery_can_be_disabled(self):
        called = False

        def factory():
            nonlocal called
            called = True
            return DummyDB([])

        with patch.dict(
            "os.environ", {"RECOVER_INTERRUPTED_JOBS_ON_WORKER_START": "0"}
        ):
            self.assertEqual(job_recovery.recover_interrupted_jobs(factory), 0)
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
