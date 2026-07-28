from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from sqlalchemy import select

from app.core.db import SessionLocal
from app.core.models import AnalysisJob
from app.core.normalizers import normalize_failure_reason

logger = logging.getLogger(__name__)

INTERRUPTED_STATUSES = frozenset({"RUNNING", "PROCESSING"})
_RECOVERY_PROBE_KEY = "recovery_probe"


def recovery_enabled() -> bool:
    # Recovery mutates analysis truth, so an explicit production opt-in is
    # required. A newly started worker is not proof that another worker died.
    value = (os.getenv("RECOVER_INTERRUPTED_JOBS_ON_WORKER_START") or "0").strip()
    return value.lower() not in {"0", "false", "no", "off"}


def _env_seconds(name: str, default: float, minimum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def recovery_stale_after_seconds() -> float:
    return _env_seconds("INTERRUPTED_JOB_STALE_AFTER_SECONDS", 21600.0, 300.0)


def recovery_probe_grace_seconds() -> float:
    return _env_seconds("INTERRUPTED_JOB_PROBE_GRACE_SECONDS", 900.0, 60.0)


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _target_attempt_id(job: AnalysisJob) -> str | None:
    target = job.target if isinstance(job.target, dict) else {}
    return str(target.get("analysis_attempt_id") or "").strip() or None


def _activity_progress(progress: Any) -> dict[str, Any]:
    payload = dict(progress) if isinstance(progress, Mapping) else {}
    payload.pop(_RECOVERY_PROBE_KEY, None)
    return payload


def _activity_fingerprint(
    *,
    status: str,
    analysis_attempt_id: str,
    task_id: str,
    task_retry: int,
    progress: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        {
            "status": status,
            "analysis_attempt_id": analysis_attempt_id,
            "analysis_task_id": task_id,
            "analysis_task_retry": task_retry,
            "progress": dict(progress),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_job_for_update(db, job_id: str) -> AnalysisJob | None:
    statement = (
        select(AnalysisJob)
        .where(AnalysisJob.id == job_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    return db.execute(statement).scalar_one_or_none()


def _candidate_job_ids(db) -> list[str]:
    rows = (
        db.execute(
            select(AnalysisJob.id).where(AnalysisJob.status.in_(INTERRUPTED_STATUSES))
        )
        .scalars()
        .all()
    )
    job_ids: list[str] = []
    for row in rows:
        raw_id = getattr(row, "id", row)
        job_id = str(raw_id or "").strip()
        if job_id and job_id not in job_ids:
            job_ids.append(job_id)
    return job_ids


def _owned_stale_activity(
    job: AnalysisJob,
    *,
    now: datetime,
) -> tuple[str, str, int, dict[str, Any], str] | None:
    status = str(job.status or "").upper()
    if status not in INTERRUPTED_STATUSES:
        return None
    attempt_id = _target_attempt_id(job)
    progress = dict(job.progress or {})
    progress_attempt_id = str(progress.get("analysis_attempt_id") or "").strip() or None
    task_id = str(progress.get("analysis_task_id") or "").strip() or None
    try:
        task_retry = max(0, int(progress.get("analysis_task_retry") or 0))
    except (TypeError, ValueError):
        return None
    activity_updated_at = _parse_timestamp(progress.get("updated_at"))
    if (
        attempt_id is None
        or progress_attempt_id != attempt_id
        or task_id is None
        or activity_updated_at is None
    ):
        return None
    age_seconds = max(0.0, (now - activity_updated_at).total_seconds())
    if age_seconds < recovery_stale_after_seconds():
        return None
    activity_progress = _activity_progress(progress)
    fingerprint = _activity_fingerprint(
        status=status,
        analysis_attempt_id=attempt_id,
        task_id=task_id,
        task_retry=task_retry,
        progress=activity_progress,
    )
    return attempt_id, task_id, task_retry, activity_progress, fingerprint


def _matching_probe_observed_at(
    probe: Any,
    *,
    recovery_revision: str,
    attempt_id: str,
    status: str,
    task_id: str,
    task_retry: int,
    fingerprint: str,
) -> datetime | None:
    if not isinstance(probe, Mapping):
        return None
    observed_at = _parse_timestamp(probe.get("observed_at"))
    if observed_at is None:
        return None
    try:
        probe_retry = int(probe.get("analysis_task_retry") or 0)
    except (TypeError, ValueError):
        return None
    matches = bool(
        str(probe.get("token") or "").strip()
        and str(probe.get("recovery_revision") or "").strip() == recovery_revision
        and str(probe.get("analysis_attempt_id") or "").strip() == attempt_id
        and str(probe.get("status") or "").upper() == status
        and str(probe.get("analysis_task_id") or "").strip() == task_id
        and probe_retry == task_retry
        and str(probe.get("activity_fingerprint") or "").strip() == fingerprint
    )
    return observed_at if matches else None


def _install_recovery_probe(
    job: AnalysisJob,
    *,
    now: datetime,
    recovery_owner: str,
    recovery_revision: str,
    attempt_id: str,
    status: str,
    task_id: str,
    task_retry: int,
    activity_progress: Mapping[str, Any],
    fingerprint: str,
) -> None:
    job.progress = {
        **dict(activity_progress),
        _RECOVERY_PROBE_KEY: {
            "token": str(uuid4()),
            "recovery_owner": recovery_owner,
            "recovery_revision": recovery_revision,
            "analysis_attempt_id": attempt_id,
            "status": status,
            "analysis_task_id": task_id,
            "analysis_task_retry": task_retry,
            "activity_fingerprint": fingerprint,
            "activity_updated_at": activity_progress.get("updated_at"),
            "observed_at": now.isoformat(),
        },
    }


def _mark_interrupted(
    job: AnalysisJob,
    *,
    now: datetime,
    recovery_owner: str,
    recovery_revision: str,
    recovery_token: str,
    activity_progress: Mapping[str, Any],
) -> None:
    warnings = [
        code
        for code in list(job.warnings or [])
        if code not in {"TRACKING_TIMEOUT", "WORKER_RESTARTED"}
    ]
    warnings.append("WORKER_RESTARTED")
    job.status = "FAILED"
    job.error = "Analysis ownership remained stale across recovery probes"
    job.failure_reason = normalize_failure_reason("WORKER_RESTARTED")
    job.warnings = warnings
    job.progress = {
        **dict(activity_progress),
        "step": "FAILED",
        "phase": activity_progress.get("phase") or "WORKER",
        "pct": 100,
        "message": "Analysis interrupted after stale ownership was confirmed",
        "updated_at": now.isoformat(),
        "interrupted_progress": dict(activity_progress),
        "recovery": {
            "token": recovery_token,
            "owner": recovery_owner,
            "revision": recovery_revision,
            "confirmed_at": now.isoformat(),
        },
    }
    job.updated_at = now


def recover_interrupted_jobs(
    session_factory: Any = SessionLocal,
    *,
    recovery_owner: str | None = None,
    recovery_revision: str | None = None,
    now: datetime | None = None,
) -> int:
    """Recover only ownership that stayed unchanged across two stale probes.

    Every mutation is performed under a row lock and revalidates status,
    analysis attempt, task ownership, retry generation, activity heartbeat and
    recovery revision. Missing ownership evidence always results in a no-op.
    """

    if not recovery_enabled():
        return 0

    observed_at = now or datetime.now(timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    else:
        observed_at = observed_at.astimezone(timezone.utc)
    owner = str(recovery_owner or f"{socket.gethostname()}:{os.getpid()}").strip()
    revision = str(recovery_revision or os.getenv("APP_GIT_SHA") or "").strip()
    if not owner or not revision or revision == "unknown":
        logger.warning(
            "Interrupted-job recovery skipped: owner/revision evidence missing"
        )
        return 0

    db = session_factory()
    recovered = 0
    probed = 0
    try:
        job_ids = _candidate_job_ids(db)
        db.rollback()
        for job_id in job_ids:
            try:
                job = _load_job_for_update(db, job_id)
                if job is None:
                    db.rollback()
                    continue
                owned = _owned_stale_activity(job, now=observed_at)
                if owned is None:
                    db.rollback()
                    continue
                (
                    attempt_id,
                    task_id,
                    task_retry,
                    activity_progress,
                    fingerprint,
                ) = owned
                status = str(job.status or "").upper()
                probe = (job.progress or {}).get(_RECOVERY_PROBE_KEY)
                probe_observed_at = _matching_probe_observed_at(
                    probe,
                    recovery_revision=revision,
                    attempt_id=attempt_id,
                    status=status,
                    task_id=task_id,
                    task_retry=task_retry,
                    fingerprint=fingerprint,
                )
                if probe_observed_at is None:
                    _install_recovery_probe(
                        job,
                        now=observed_at,
                        recovery_owner=owner,
                        recovery_revision=revision,
                        attempt_id=attempt_id,
                        status=status,
                        task_id=task_id,
                        task_retry=task_retry,
                        activity_progress=activity_progress,
                        fingerprint=fingerprint,
                    )
                    db.add(job)
                    db.commit()
                    probed += 1
                    continue
                probe_age_seconds = max(
                    0.0,
                    (observed_at - probe_observed_at).total_seconds(),
                )
                if probe_age_seconds < recovery_probe_grace_seconds():
                    db.rollback()
                    continue

                recovery_token = str(probe.get("token"))
                _mark_interrupted(
                    job,
                    now=observed_at,
                    recovery_owner=owner,
                    recovery_revision=revision,
                    recovery_token=recovery_token,
                    activity_progress=activity_progress,
                )
                db.add(job)
                db.commit()
                recovered += 1
            except Exception:
                db.rollback()
                logger.exception(
                    "Unable to recover interrupted analysis job_id=%s",
                    job_id,
                )
        if probed:
            logger.warning(
                "Installed %s interrupted-job recovery probes; no immediate failure",
                probed,
            )
        if recovered:
            logger.warning(
                "Marked %s proven-stale interrupted jobs as retryable",
                recovered,
            )
        return recovered
    except Exception:
        db.rollback()
        logger.exception("Unable to inspect interrupted analysis jobs")
        return 0
    finally:
        db.close()
