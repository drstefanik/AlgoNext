from __future__ import annotations

from typing import Any


PREPARATION_FAILURES = frozenset(
    {
        "preview_generation_failed",
        "candidates_generation_failed",
        "PREPARATION_ENQUEUE_FAILED",
    }
)


def can_retry_preparation(job: Any) -> bool:
    """Only retry an explicitly failed preparation, never reset analysis truth."""
    target = job.target if isinstance(job.target, dict) else {}
    result = job.result if isinstance(job.result, dict) else {}
    progress = job.progress if isinstance(job.progress, dict) else {}
    return bool(
        str(job.status or "").upper() == "FAILED"
        and job.failure_reason in PREPARATION_FAILURES
        and not job.player_ref
        and not target.get("confirmed")
        and not target.get("selections")
        and not progress.get("analysis_task_id")
        and not result.get("analysis_outcome")
        and not result.get("tracking")
    )
