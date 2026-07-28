import ast
import copy
import os
import socket
import unittest
from pathlib import Path
from types import SimpleNamespace


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
        self.assertIn("recover_interrupted_jobs(", ready_block)
        self.assertIn("start_worker_heartbeat", ready_block)
        self.assertIn("inspect_runtime()", ready_block)
        self.assertIn("if heartbeat_confirmed:", ready_block)
        self.assertIn("recovery_revision=APP_GIT_SHA", ready_block)
        self.assertLess(
            ready_block.index("start_worker_heartbeat"),
            ready_block.index("recover_interrupted_jobs("),
        )

    def _load_worker_ready(self, *, runtime_snapshot):
        handler = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_on_worker_ready"
        )
        handler = copy.deepcopy(handler)
        handler.decorator_list = []
        module = ast.Module(body=[handler], type_ignores=[])
        events = []
        namespace = {
            "APP_GIT_SHA": "revision-a",
            "install_worker_pipeline_policy": lambda: True,
            "install_worker_preview_asset_policy": lambda: True,
            "start_worker_heartbeat": (
                lambda worker_name: events.append(("heartbeat", worker_name))
            ),
            "inspect_runtime": lambda: runtime_snapshot,
            "recover_interrupted_jobs": (
                lambda **kwargs: events.append(("recover", kwargs))
            ),
            "logger": SimpleNamespace(
                info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None
            ),
            "os": os,
            "socket": socket,
        }
        exec(
            compile(
                ast.fix_missing_locations(module),
                "app/workers/celery_app.py",
                "exec",
            ),
            namespace,
        )
        return namespace["_on_worker_ready"], events

    def test_worker_ready_recovers_only_after_its_own_heartbeat_is_confirmed(self):
        worker_name = "celery@worker-a"
        confirmed = {
            "dependencies": {"worker": "ready"},
            "worker": {
                "worker_name": worker_name,
                "revision": "revision-a",
                "pid": os.getpid(),
            },
        }
        handler, events = self._load_worker_ready(runtime_snapshot=confirmed)

        handler(sender=SimpleNamespace(hostname=worker_name))

        self.assertEqual(events[0], ("heartbeat", worker_name))
        self.assertEqual(
            events[1],
            (
                "recover",
                {
                    "recovery_owner": f"{worker_name}:revision-a",
                    "recovery_revision": "revision-a",
                },
            ),
        )

        unconfirmed = {
            "dependencies": {"worker": "stale"},
            "worker": {
                "worker_name": worker_name,
                "revision": "revision-a",
                "pid": os.getpid(),
            },
        }
        handler, events = self._load_worker_ready(runtime_snapshot=unconfirmed)

        handler(sender=SimpleNamespace(hostname=worker_name))

        self.assertEqual(events, [("heartbeat", worker_name)])


if __name__ == "__main__":
    unittest.main()
