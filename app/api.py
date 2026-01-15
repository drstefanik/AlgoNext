from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.deps import get_db
from app.core.models import AnalysisJob
from app.schemas import JobCreate, JobOut
from app.workers.pipeline import run_analysis

router = APIRouter()


@router.post("/jobs", response_model=JobOut)
def create_job(payload: JobCreate, db: Session = Depends(get_db)):
    job_id = str(uuid4())

    job = AnalysisJob(
        id=job_id,
        status="WAITING_FOR_ANCHOR",
        category=payload.category,
        role=payload.role,
        video_url=payload.video_url,
        target={"shirt_number": payload.shirt_number} if payload.shirt_number is not None else {},
        video_meta={},
        anchor={},
        progress={"step": "CREATED", "pct": 0, "message": "Job created"},
        result={},
        error=None,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    return JobOut(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "status": job.status,
        "category": job.category,
        "role": job.role,
        "video_url": job.video_url,
        "target": job.target,
        "anchor": job.anchor,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


# ✅ endpoint leggero per polling
@router.get("/jobs/{job_id}/status")
def job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


@router.get("/jobs/{job_id}/poll")
def job_poll(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "result": job.result if job.status == "COMPLETED" else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


# ✅ result “pulito” (solo quando pronto)
@router.get("/jobs/{job_id}/result")
def job_result(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status == "FAILED":
        raise HTTPException(status_code=409, detail=job.error or "Job failed")

    if job.status != "COMPLETED":
        raise HTTPException(status_code=409, detail="Job not completed yet")

    return job.result or {}


@router.post("/jobs/{job_id}/enqueue", response_model=JobOut)
def enqueue_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # idempotente: se già avanzato, non reinvio
    if job.status in ["QUEUED", "RUNNING", "COMPLETED", "FAILED"]:
        return JobOut(job_id=job.id, status=job.status)

    if job.status != "WAITING_FOR_ANCHOR":
        raise HTTPException(status_code=400, detail="Job not enqueueable")

    job.status = "QUEUED"
    job.error = None
    db.commit()
    db.refresh(job)

    run_analysis.delay(job.id)

    return JobOut(job_id=job.id, status=job.status)
