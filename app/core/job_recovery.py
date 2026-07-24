from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.core.db import SessionLocal
from app.core.models import AnalysisJob
from app.core.normalizers import normalize_failure_reason

logger = logging.getLogger(__name__)

INTERRUPTED_STATUSES = {"RUNNING", "PROCESSING"}


def recovery_enabled() -> bool:
    value = (os.getenv("RECOVER_INTERRUPTED_JOBS_ON_WORKER_START") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def recover_interrupted_jobs(session_factory: Any = SessionLocal) -> int:
    """Mark runs owned by a previous single worker as retryable failures."""

    if not recovery_enabled():
        return 0

    db = session_factory()
    recovered = 0
    try:
        jobs = list(
            db.execute(
                select(AnalysisJob).where(AnalysisJob.status.in_(INTERRUPTED_STATUSES))
            )
            .scalars()
            .all()
        )
        now = datetime.now(timezone.utc)
        for job in jobs:
            warnings = [
                code
                for code in list(job.warnings or [])
                if code not in {"TRACKING_TIMEOUT", "WORKER_RESTARTED"}
            ]
            warnings.append("WORKER_RESTARTED")
            previous_progress = dict(job.progress or {})
            job.status = "FAILED"
            job.error = "Analysis interrupted by a worker restart"
            job.failure_reason = normalize_failure_reason("WORKER_RESTARTED")
            job.warnings = warnings
            job.progress = {
                **previous_progress,
                "step": "FAILED",
                "phase": previous_progress.get("phase") or "WORKER",
                "pct": 100,
                "message": "Analysis interrupted by worker restart",
                "updated_at": now.isoformat(),
                "interrupted_progress": previous_progress,
            }
            job.updated_at = now
            recovered += 1
        if recovered:
            db.commit()
            logger.warning("Marked %s interrupted jobs as retryable", recovered)
        return recovered
    except Exception:
        db.rollback()
        logger.exception("Unable to recover interrupted analysis jobs")
        return 0
    finally:
        db.close()
