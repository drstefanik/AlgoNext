import json
import unittest
from datetime import timedelta
from unittest.mock import patch

from app.core import runtime_health


class FakeRedis:
    def __init__(self, raw=None):
        self.raw = raw
        self.set_calls = []
        self.delete_calls = []

    def ping(self):
        return True

    def get(self, _key):
        return self.raw

    def set(self, key, value, ex=None):
        self.set_calls.append((key, value, ex))
        self.raw = value
        return True

    def delete(self, key):
        self.delete_calls.append(key)
        self.raw = None
        return 1


class RuntimeHealthTests(unittest.TestCase):
    def heartbeat(self, *, revision="sha-1", seconds_ago=0, state="ready"):
        last_seen = (runtime_health.utc_now() - timedelta(seconds=seconds_ago)).isoformat()
        return json.dumps(
            {
                "service": "algonext-worker",
                "revision": revision,
                "build_time": "build",
                "state": state,
                "worker_name": "worker@node",
                "hostname": "node",
                "pid": 42,
                "started_at": last_seen,
                "last_seen": last_seen,
            }
        )

    def test_matching_fresh_worker_is_ready(self):
        fake = FakeRedis(self.heartbeat())
        with patch.object(runtime_health, "APP_GIT_SHA", "sha-1"), patch.object(
            runtime_health, "_redis_client", return_value=fake
        ):
            snapshot = runtime_health.inspect_runtime()
        self.assertTrue(snapshot["ready"])
        self.assertEqual(snapshot["dependencies"]["redis"], "ready")
        self.assertEqual(snapshot["dependencies"]["worker"], "ready")
        self.assertTrue(snapshot["worker_revision_matches_api"])

    def test_stale_worker_is_not_ready(self):
        fake = FakeRedis(
            self.heartbeat(seconds_ago=runtime_health.WORKER_HEARTBEAT_MAX_AGE_SECONDS + 5)
        )
        with patch.object(runtime_health, "APP_GIT_SHA", "sha-1"), patch.object(
            runtime_health, "_redis_client", return_value=fake
        ):
            snapshot = runtime_health.inspect_runtime()
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["dependencies"]["worker"], "stale")

    def test_revision_mismatch_is_not_ready(self):
        fake = FakeRedis(self.heartbeat(revision="old-sha"))
        with patch.object(runtime_health, "APP_GIT_SHA", "new-sha"), patch.object(
            runtime_health, "_redis_client", return_value=fake
        ):
            snapshot = runtime_health.inspect_runtime()
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["dependencies"]["worker"], "revision_mismatch")
        self.assertFalse(snapshot["worker_revision_matches_api"])

    def test_write_heartbeat_uses_ttl(self):
        fake = FakeRedis()
        with patch.object(runtime_health, "APP_GIT_SHA", "sha-1"), patch.object(
            runtime_health, "_redis_client", return_value=fake
        ):
            payload = runtime_health.write_worker_heartbeat(worker_name="worker@test")
        self.assertEqual(payload["worker_name"], "worker@test")
        self.assertEqual(payload["revision"], "sha-1")
        self.assertEqual(fake.set_calls[0][2], runtime_health.WORKER_HEARTBEAT_TTL_SECONDS)

    def test_stop_removes_heartbeat(self):
        fake = FakeRedis(self.heartbeat())
        with patch.object(runtime_health, "_redis_client", return_value=fake):
            runtime_health.stop_worker_heartbeat()
        self.assertEqual(fake.delete_calls, [runtime_health.WORKER_HEARTBEAT_KEY])


if __name__ == "__main__":
    unittest.main()
