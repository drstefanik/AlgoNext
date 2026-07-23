from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.db import SessionLocal
from app.core.evaluation_guard import sanitize_analysis_job
from app.core.models import AnalysisJob

REPORT_UNAVAILABLE_MESSAGE = (
    "Player evaluation is unavailable until player ReID, pitch calibration, "
    "ball events, and the scoring model are validated."
)
_REPORT_PATH = re.compile(r"^/jobs/([^/]+)/(report|ai-report)$")


def validated_player_evaluation_available(result: Mapping[str, Any] | None) -> bool:
    if not isinstance(result, Mapping):
        return False
    provenance = result.get("score_provenance")
    if not isinstance(provenance, Mapping):
        return False
    return bool(
        result.get("player_evaluation_available") is True
        and provenance.get("kind") == "player_evaluation"
        and provenance.get("validated_player_score") is True
    )


def build_unavailable_report(result: Mapping[str, Any] | None) -> dict[str, Any]:
    limitations = []
    if isinstance(result, Mapping) and isinstance(result.get("limitations"), list):
        limitations = [
            item
            for item in result.get("limitations") or []
            if isinstance(item, str) and item.strip()
        ]
    if not limitations:
        limitations = [REPORT_UNAVAILABLE_MESSAGE]
    return {
        "summary": "Valutazione del giocatore non disponibile.",
        "strengths": [],
        "risks": [],
        "key_moments": [],
        "training_plan_14_days": [],
        "limitations": limitations,
        "confidence": 0.0,
    }


def _response_meta(request: Request, request_id: str) -> dict[str, str]:
    request.state.request_id = request_id
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class EvaluationReportGuardMiddleware(BaseHTTPMiddleware):
    """Prevent every report endpoint from bypassing evaluation abstention."""

    async def dispatch(self, request: Request, call_next):
        match = _REPORT_PATH.fullmatch(request.url.path)
        if not match or request.method not in {"GET", "POST"}:
            return await call_next(request)

        job_id, endpoint = match.groups()
        db = SessionLocal()
        try:
            job = db.get(AnalysisJob, job_id)
            if job is None:
                return await call_next(request)

            sanitize_analysis_job(job)
            if validated_player_evaluation_available(job.result):
                return await call_next(request)

            report = (
                job.report
                if job.report_status == "UNAVAILABLE" and isinstance(job.report, dict)
                else build_unavailable_report(job.result)
            )
            if request.method == "POST":
                job.report_status = "UNAVAILABLE"
                job.report_error = REPORT_UNAVAILABLE_MESSAGE
                job.report = report
                job.ai_report = report
                db.add(job)
                db.commit()

            request_id = (
                getattr(request.state, "request_id", None)
                or request.headers.get("x-request-id")
                or str(uuid4())
            )
            if endpoint == "ai-report":
                data = {
                    "status": "UNAVAILABLE",
                    "ai_report": report,
                    "reason": REPORT_UNAVAILABLE_MESSAGE,
                }
            else:
                data = {
                    "status": "UNAVAILABLE",
                    "report": report,
                    "reason": REPORT_UNAVAILABLE_MESSAGE,
                }
            return JSONResponse(
                status_code=200,
                content={
                    "ok": True,
                    "data": data,
                    "meta": _response_meta(request, request_id),
                },
                headers={"cache-control": "no-store", "x-request-id": request_id},
            )
        finally:
            db.close()
