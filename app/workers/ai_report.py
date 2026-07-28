import logging
import os

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import _build_ai_report_payload
from app.core.ai_report import generate_ai_report
from app.core.db import SessionLocal
from app.core.models import AnalysisJob
from app.core.tracking_outcome import StaleAnalysisAttemptError
from app.workers.celery_app import celery

logger = logging.getLogger(__name__)


UNAVAILABLE_MESSAGE = (
    "Player evaluation is unavailable until player ReID, pitch calibration, "
    "ball events, and the scoring model are validated."
)


def _is_job_ready(job: AnalysisJob) -> bool:
    if job.status in {"DONE", "COMPLETED", "PARTIAL"} and job.result:
        return True
    progress_step = (job.progress or {}).get("step")
    return progress_step == "DONE" and bool(job.result)


def _player_evaluation_available(job: AnalysisJob) -> bool:
    result = job.result or {}
    if not isinstance(result, dict):
        return False
    provenance = result.get("score_provenance") or {}
    if not isinstance(provenance, dict):
        return False
    return bool(
        result.get("player_evaluation_available") is True
        and provenance.get("kind") == "player_evaluation"
        and provenance.get("validated_player_score") is True
    )


def _job_attempt_id(job: AnalysisJob) -> str | None:
    target = job.target if isinstance(job.target, dict) else {}
    return str(target.get("analysis_attempt_id") or "").strip() or None


def _validate_report_attempt(
    job: AnalysisJob,
    expected_analysis_attempt_id: str | None,
) -> str | None:
    current_attempt_id = _job_attempt_id(job)
    expected_attempt_id = (
        str(expected_analysis_attempt_id).strip()
        if expected_analysis_attempt_id is not None
        else None
    ) or None
    if expected_attempt_id is None:
        if current_attempt_id is not None:
            raise StaleAnalysisAttemptError(
                "Report task is missing the current analysis attempt"
            )
        return None
    if expected_attempt_id != current_attempt_id:
        raise StaleAnalysisAttemptError(
            "Report task attempt differs from the current job target: "
            f"task={expected_attempt_id} target={current_attempt_id or '<missing>'}"
        )
    return current_attempt_id


def _reload_job(db: Session, job_id: str) -> AnalysisJob | None:
    try:
        return db.get(AnalysisJob, job_id, populate_existing=True)
    except TypeError:
        return db.get(AnalysisJob, job_id)


def _load_job_for_update(db: Session, job_id: str) -> AnalysisJob | None:
    execute = getattr(db, "execute", None)
    if callable(execute):
        statement = (
            select(AnalysisJob)
            .where(AnalysisJob.id == job_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return execute(statement).scalar_one_or_none()
    try:
        return db.get(AnalysisJob, job_id, populate_existing=True)
    except TypeError:
        return db.get(AnalysisJob, job_id)


def _update_report_job(
    db: Session,
    job_id: str,
    expected_analysis_attempt_id: str | None,
    updater,
) -> AnalysisJob | None:
    job = _load_job_for_update(db, job_id)
    if job is None:
        return None
    _validate_report_attempt(job, expected_analysis_attempt_id)
    updater(job)
    db.add(job)
    db.commit()
    return job


def _save_report_failure(
    db: Session,
    job_id: str,
    expected_analysis_attempt_id: str | None,
    error: str,
) -> None:
    def update(job: AnalysisJob) -> None:
        job.report_status = "FAILED"
        job.report_error = error
        job.report = None
        # backward compatibility for old endpoint clients
        job.ai_report = {"error": error}

    _update_report_job(db, job_id, expected_analysis_attempt_id, update)


def _save_report_unavailable(
    db: Session,
    job_id: str,
    expected_analysis_attempt_id: str | None,
) -> None:
    def update(job: AnalysisJob) -> None:
        job.report_status = "UNAVAILABLE"
        job.report_error = UNAVAILABLE_MESSAGE
        job.report = {
            "summary": "Valutazione del giocatore non disponibile.",
            "strengths": [],
            "risks": [],
            "key_moments": [],
            "training_plan_14_days": [],
            "limitations": list(
                (job.result or {}).get("limitations") or [UNAVAILABLE_MESSAGE]
            ),
            "confidence": 0.0,
        }
        # backward compatibility for old endpoint clients
        job.ai_report = job.report

    _update_report_job(db, job_id, expected_analysis_attempt_id, update)


def _generate_report_impl(
    job_id: str,
    expected_analysis_attempt_id: str | None = None,
    force: bool = False,
) -> None:
    db: Session = SessionLocal()
    try:
        job = _reload_job(db, job_id)
        if not job:
            logger.warning("AI_REPORT_FAIL job_id=%s error=job_not_found", job_id)
            return
        _validate_report_attempt(job, expected_analysis_attempt_id)
        if not _is_job_ready(job):
            logger.info("AI_REPORT_SKIP job_id=%s reason=job_not_ready", job_id)
            _save_report_failure(
                db,
                job_id,
                expected_analysis_attempt_id,
                "Job not completed yet",
            )
            return
        if not _player_evaluation_available(job):
            logger.info(
                "AI_REPORT_SKIP job_id=%s reason=player_evaluation_unavailable", job_id
            )
            _save_report_unavailable(
                db,
                job_id,
                expected_analysis_attempt_id,
            )
            return
        if job.report and not force:
            logger.info("AI_REPORT_SKIP job_id=%s reason=already_exists", job_id)
            _update_report_job(
                db,
                job_id,
                expected_analysis_attempt_id,
                lambda current: (
                    setattr(current, "report_status", "DONE"),
                    setattr(current, "report_error", None),
                ),
            )
            return

        ai_payload = _build_ai_report_payload(job)
        if not ai_payload.get("clips"):
            logger.info("AI_REPORT_SKIP job_id=%s reason=missing_clips", job_id)
            _save_report_failure(
                db,
                job_id,
                expected_analysis_attempt_id,
                "Job clips are missing",
            )
            return

        current = _update_report_job(
            db,
            job_id,
            expected_analysis_attempt_id,
            lambda current: (
                setattr(current, "report_status", "RUNNING"),
                setattr(current, "report_error", None),
            ),
        )
        if current is None:
            return

        model = (os.environ.get("OPENAI_MODEL") or "gpt-5.2").strip()
        logger.info("AI_REPORT_START job_id=%s model=%s", job_id, model)
        try:
            ai_report, usage = generate_ai_report(ai_payload)
        except Exception as exc:
            logger.error("AI_REPORT_FAIL job_id=%s error=%s", job_id, exc)
            _save_report_failure(
                db,
                job_id,
                expected_analysis_attempt_id,
                str(exc),
            )
            return
        if usage is not None:
            logger.info("AI_REPORT_OK job_id=%s usage=%s", job_id, usage)
        else:
            logger.info("AI_REPORT_OK job_id=%s", job_id)

        def save(current: AnalysisJob) -> None:
            current.report = ai_report
            current.report_status = "DONE"
            current.report_error = None
            # backward compatibility for old endpoint clients
            current.ai_report = ai_report

        _update_report_job(
            db,
            job_id,
            expected_analysis_attempt_id,
            save,
        )
    except StaleAnalysisAttemptError as exc:
        db.rollback()
        logger.warning(
            "AI_REPORT_STALE job_id=%s reason=%s",
            job_id,
            exc,
        )
        return
    finally:
        db.close()


@celery.task(name="app.workers.ai_report.generate_report", bind=True)
def generate_report(
    self,
    job_id: str,
    expected_analysis_attempt_id: str | None = None,
    force: bool = False,
) -> None:
    _generate_report_impl(
        job_id,
        expected_analysis_attempt_id=expected_analysis_attempt_id,
        force=force,
    )


@celery.task(name="app.workers.ai_report.generate_ai_report_task", bind=True)
def generate_ai_report_task(
    self,
    job_id: str,
    expected_analysis_attempt_id: str | None = None,
    force: bool = False,
) -> None:
    _generate_report_impl(
        job_id,
        expected_analysis_attempt_id=expected_analysis_attempt_id,
        force=force,
    )
