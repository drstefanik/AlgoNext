from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


PREPARATION_FAILURES = frozenset(
    {
        "preview_generation_failed",
        "candidates_generation_failed",
        "PREPARATION_ENQUEUE_FAILED",
    }
)


def can_retry_preparation(job: Any, *, now: datetime | None = None) -> bool:
    """Recover failed or unstarted preparation without resetting analysis truth."""
    target = job.target if isinstance(job.target, dict) else {}
    result = job.result if isinstance(job.result, dict) else {}
    progress = job.progress if isinstance(job.progress, dict) else {}
    status = str(job.status or "").upper()
    recoverable = status == "FAILED" and job.failure_reason in PREPARATION_FAILURES
    if (
        status == "CREATED"
        and progress.get("step") == "CREATED"
        and progress.get("pct") == 0
    ):
        stamp = progress.get("updated_at") or getattr(job, "created_at", None)
        try:
            started = (
                stamp
                if isinstance(stamp, datetime)
                else datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            )
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            recoverable = (
                (now or datetime.now(timezone.utc)) - started
            ).total_seconds() >= 600
        except (TypeError, ValueError, OverflowError):
            recoverable = False
    return bool(
        recoverable
        and not job.player_ref
        and not target.get("confirmed")
        and not target.get("selections")
        and not progress.get("analysis_task_id")
        and not result.get("analysis_outcome")
        and not result.get("tracking")
    )
