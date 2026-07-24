from __future__ import annotations

import json
import logging
import os
import socket
import threading
from datetime import datetime, timezone
from typing import Any

import redis

logger = logging.getLogger(__name__)

APP_GIT_SHA = (os.getenv("APP_GIT_SHA") or "unknown").strip() or "unknown"
APP_BUILD_TIME = (os.getenv("APP_BUILD_TIME") or "unknown").strip() or "unknown"
REDIS_URL = (
    os.getenv("REDIS_URL")
    or os.getenv("CELERY_BROKER_URL")
    or "redis://redis:6379/0"
)
WORKER_HEARTBEAT_KEY = (
    os.getenv("WORKER_HEARTBEAT_KEY") or "algonext:worker:heartbeat:v1"
).strip()
WORKER_HEARTBEAT_INTERVAL_SECONDS = max(
    2.0, float(os.getenv("WORKER_HEARTBEAT_INTERVAL_SECONDS", "15") or 15)
)
WORKER_HEARTBEAT_TTL_SECONDS = max(
    int(WORKER_HEARTBEAT_INTERVAL_SECONDS * 3),
    int(os.getenv("WORKER_HEARTBEAT_TTL_SECONDS", "90") or 90),
)
WORKER_HEARTBEAT_MAX_AGE_SECONDS = max(
    WORKER_HEARTBEAT_INTERVAL_SECONDS * 2,
    float(os.getenv("WORKER_HEARTBEAT_MAX_AGE_SECONDS", "60") or 60),
)
CHECK_WORKER_READINESS = (
    (os.getenv("CHECK_WORKER_READINESS") or "1").strip().lower()
    not in {"0", "false", "no", "off"}
)

_stop_event = threading.Event()
_thread_lock = threading.Lock()
_heartbeat_thread: threading.Thread | None = None
_started_at: str | None = None
_worker_name: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_metadata(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "revision": APP_GIT_SHA,
        "build_time": APP_BUILD_TIME,
    }


def _redis_client() -> redis.Redis:
    return redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=1.5,
        socket_timeout=1.5,
        health_check_interval=15,
    )


def _worker_payload(*, state: str, worker_name: str | None = None) -> dict[str, Any]:
    global _started_at
    if _started_at is None:
        _started_at = utc_now().isoformat()
    now = utc_now().isoformat()
    return {
        **build_metadata("algonext-worker"),
        "state": state,
        "worker_name": worker_name or _worker_name or socket.gethostname(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": _started_at,
        "last_seen": now,
    }


def write_worker_heartbeat(
    *, state: str = "ready", worker_name: str | None = None
) -> dict[str, Any]:
    payload = _worker_payload(state=state, worker_name=worker_name)
    _redis_client().set(
        WORKER_HEARTBEAT_KEY,
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        ex=WORKER_HEARTBEAT_TTL_SECONDS,
    )
    return payload


def _heartbeat_loop() -> None:
    while not _stop_event.is_set():
        try:
            write_worker_heartbeat(state="ready")
        except Exception:
            logger.exception("Unable to publish worker heartbeat")
        _stop_event.wait(WORKER_HEARTBEAT_INTERVAL_SECONDS)


def start_worker_heartbeat(worker_name: str | None = None) -> None:
    global _heartbeat_thread, _worker_name, _started_at
    with _thread_lock:
        if _heartbeat_thread and _heartbeat_thread.is_alive():
            return
        _worker_name = worker_name or socket.gethostname()
        _started_at = utc_now().isoformat()
        _stop_event.clear()
        try:
            write_worker_heartbeat(state="ready", worker_name=_worker_name)
        except Exception:
            logger.exception("Unable to publish initial worker heartbeat")
        _heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name="algonext-worker-heartbeat",
            daemon=True,
        )
        _heartbeat_thread.start()
        logger.info(
            "Worker heartbeat started key=%s revision=%s",
            WORKER_HEARTBEAT_KEY,
            APP_GIT_SHA,
        )


def stop_worker_heartbeat() -> None:
    global _heartbeat_thread
    with _thread_lock:
        _stop_event.set()
        thread = _heartbeat_thread
        _heartbeat_thread = None
    if thread and thread.is_alive():
        thread.join(timeout=min(2.0, WORKER_HEARTBEAT_INTERVAL_SECONDS))
    try:
        write_worker_heartbeat(state="stopping")
    except Exception:
        logger.debug("Unable to publish final worker heartbeat", exc_info=True)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def inspect_runtime() -> dict[str, Any]:
    dependencies: dict[str, Any] = {
        "redis": "unknown",
        "worker": "unknown",
    }
    worker_payload: dict[str, Any] | None = None
    worker_age_seconds: float | None = None
    revision_match: bool | None = None
    error: str | None = None

    try:
        client = _redis_client()
        client.ping()
        dependencies["redis"] = "ready"
        raw = client.get(WORKER_HEARTBEAT_KEY)
    except Exception as exc:
        raw = None
        error = str(exc)
        dependencies["redis"] = "unavailable"
        dependencies["worker"] = "unknown"
    else:
        if not raw:
            dependencies["worker"] = "missing"
        else:
            try:
                parsed = json.loads(raw)
                worker_payload = parsed if isinstance(parsed, dict) else None
            except (TypeError, json.JSONDecodeError) as exc:
                error = f"Invalid worker heartbeat: {exc}"
                dependencies["worker"] = "invalid"

            if worker_payload is not None:
                last_seen = _parse_timestamp(worker_payload.get("last_seen"))
                if last_seen is not None:
                    worker_age_seconds = max(
                        0.0, (utc_now() - last_seen).total_seconds()
                    )
                worker_revision = str(worker_payload.get("revision") or "unknown")
                revision_match = (
                    APP_GIT_SHA == "unknown" or worker_revision == APP_GIT_SHA
                )
                state = str(worker_payload.get("state") or "unknown").lower()
                if worker_age_seconds is None:
                    dependencies["worker"] = "invalid"
                elif worker_age_seconds > WORKER_HEARTBEAT_MAX_AGE_SECONDS:
                    dependencies["worker"] = "stale"
                elif not revision_match:
                    dependencies["worker"] = "revision_mismatch"
                elif state != "ready":
                    dependencies["worker"] = state
                else:
                    dependencies["worker"] = "ready"

    worker_ready = dependencies["worker"] == "ready"
    ready = (not CHECK_WORKER_READINESS) or (
        dependencies["redis"] == "ready" and worker_ready
    )
    sanitized_worker = None
    if worker_payload is not None:
        sanitized_worker = {
            key: worker_payload.get(key)
            for key in (
                "service",
                "revision",
                "build_time",
                "state",
                "worker_name",
                "hostname",
                "pid",
                "started_at",
                "last_seen",
            )
        }

    return {
        "ready": ready,
        "required": CHECK_WORKER_READINESS,
        "dependencies": dependencies,
        "worker": sanitized_worker,
        "worker_age_seconds": (
            round(worker_age_seconds, 3)
            if worker_age_seconds is not None
            else None
        ),
        "worker_revision_matches_api": revision_match,
        "max_worker_age_seconds": WORKER_HEARTBEAT_MAX_AGE_SECONDS,
        "error": error,
    }
