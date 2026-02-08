import logging
import os

from sqlalchemy.orm import Session

from app.api import _build_ai_report_payload
from app.core.ai_report import generate_ai_report
from app.core.db import SessionLocal
from app.core.models import AnalysisJob
from app.workers.celery_app import celery


logger = logging.getLogger(__name__)


def _is_job_ready(job: AnalysisJob) -> bool:
    if job.status in {"DONE", "COMPLETED", "PARTIAL"} and job.result:
        return True
    progress_step = (job.progress or {}).get("step")
    return progress_step == "DONE" and bool(job.result)


@celery.task(name="app.workers.ai_report.generate_ai_report_task", bind=True)
def generate_ai_report_task(self, job_id: str, force: bool = False) -> None:
    db: Session = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        if not job:
            logger.warning("AI_REPORT_FAIL job_id=%s error=job_not_found", job_id)
            return
        if not _is_job_ready(job):
            logger.info("AI_REPORT_SKIP job_id=%s reason=job_not_ready", job_id)
            return
        if job.ai_report and not force:
            logger.info("AI_REPORT_SKIP job_id=%s reason=already_exists", job_id)
            return

        ai_payload = _build_ai_report_payload(job)
        if not ai_payload.get("clips"):
            logger.info("AI_REPORT_SKIP job_id=%s reason=missing_clips", job_id)
            return

        model = (os.environ.get("OPENAI_MODEL") or "gpt-5.2").strip()
        logger.info("AI_REPORT_START job_id=%s model=%s", job.id, model)
        try:
            ai_report, usage = generate_ai_report(ai_payload)
        except Exception as exc:
            logger.error("AI_REPORT_FAIL job_id=%s error=%s", job.id, exc)
            job.ai_report = {"error": str(exc)}
            db.add(job)
            db.commit()
            return

        if usage is not None:
            logger.info("AI_REPORT_OK job_id=%s usage=%s", job.id, usage)
        else:
            logger.info("AI_REPORT_OK job_id=%s", job.id)
        job.ai_report = ai_report
        db.add(job)
        db.commit()
    finally:
        db.close()
