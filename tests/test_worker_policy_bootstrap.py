import unittest
from pathlib import Path


class WorkerPolicyBootstrapTests(unittest.TestCase):
    def test_pipeline_policy_is_installed_after_celery_exists_and_before_worker_ready(self):
        source = Path("app/workers/celery_app.py").read_text(encoding="utf-8")

        celery_created = source.index("celery = Celery(")
        eager_policy = source.index(
            "PIPELINE_POLICY_INSTALLED_EAGERLY = install_worker_pipeline_policy()"
        )
        worker_ready_handler = source.index("@worker_ready.connect")

        self.assertLess(celery_created, eager_policy)
        self.assertLess(eager_policy, worker_ready_handler)
        self.assertIn(
            "do not accept tasks before the policy is present",
            source,
        )


if __name__ == "__main__":
    unittest.main()
