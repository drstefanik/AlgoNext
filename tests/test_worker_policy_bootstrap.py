import ast
import unittest
from pathlib import Path


class WorkerPolicyBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("app/workers/celery_app.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_pipeline_policy_is_installed_from_worker_init_before_worker_ready(self):
        celery_created = self.source.index("celery = Celery(")
        worker_init_handler = self.source.index("@worker_init.connect")
        worker_ready_handler = self.source.index("@worker_ready.connect")

        self.assertLess(celery_created, worker_init_handler)
        self.assertLess(worker_init_handler, worker_ready_handler)

        init_block = self.source[worker_init_handler:worker_ready_handler]
        self.assertIn("install_worker_pipeline_policy()", init_block)
        self.assertIn("before the worker begins", init_block)

    def test_pipeline_policy_is_not_installed_at_module_import_time(self):
        self.assertNotIn("PIPELINE_POLICY_INSTALLED_EAGERLY", self.source)

        direct_top_level_calls = []
        for statement in self.tree.body:
            value = None
            if isinstance(statement, ast.Expr):
                value = statement.value
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value

            if not isinstance(value, ast.Call):
                continue
            if isinstance(value.func, ast.Name):
                direct_top_level_calls.append(value.func.id)

        self.assertNotIn("install_worker_pipeline_policy", direct_top_level_calls)

    def test_worker_ready_keeps_an_idempotent_policy_safety_net(self):
        worker_ready_handler = self.source.index("@worker_ready.connect")
        worker_shutdown_handler = self.source.index("@worker_shutdown.connect")
        ready_block = self.source[worker_ready_handler:worker_shutdown_handler]

        self.assertIn("install_worker_pipeline_policy()", ready_block)
        self.assertIn("recover_interrupted_jobs()", ready_block)
        self.assertIn("start_worker_heartbeat", ready_block)


if __name__ == "__main__":
    unittest.main()
