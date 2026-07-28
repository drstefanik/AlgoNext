import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class TrackingAttemptFencingTests(unittest.TestCase):
    @classmethod
    def _load_attempt_guard(cls):
        tracking_path = Path(__file__).resolve().parents[1] / "app/workers/tracking.py"
        tree = ast.parse(tracking_path.read_text(encoding="utf-8"))
        selected = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "_TRACKING_MUTABLE_STATUSES"
                    for target in node.targets
                )
            )
            or (
                isinstance(node, ast.FunctionDef)
                and node.name
                in {
                    "_normalized_analysis_attempt_id",
                    "_require_analysis_attempt",
                }
            )
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *selected,
            ],
            type_ignores=[],
        )
        namespace = {"StaleAnalysisAttemptError": RuntimeError}
        exec(
            compile(
                ast.fix_missing_locations(module),
                str(tracking_path),
                "exec",
            ),
            namespace,
        )
        return namespace["_require_analysis_attempt"]

    def test_same_attempt_cannot_write_after_terminal_transition(self):
        guard = self._load_attempt_guard()
        job = SimpleNamespace(
            target={"analysis_attempt_id": "attempt-a"},
            status="COMPLETED",
        )

        with self.assertRaises(RuntimeError):
            guard(job, "attempt-a")

    def test_missing_attempt_only_matches_legacy_target(self):
        guard = self._load_attempt_guard()
        legacy_job = SimpleNamespace(target={}, status="CREATED")
        current_job = SimpleNamespace(
            target={"analysis_attempt_id": "attempt-a"},
            status="RUNNING",
        )

        self.assertIsNone(guard(legacy_job, None))
        with self.assertRaises(RuntimeError):
            guard(current_job, None)


if __name__ == "__main__":
    unittest.main()
