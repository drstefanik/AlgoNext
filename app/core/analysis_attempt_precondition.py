from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fastapi import HTTPException, Request

ErrorDetailFactory = Callable[[str, str, dict[str, Any] | None], Any]


def _normalized_ids(*values: Any) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if value is None:
            continue
        candidate = str(value).strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if lowered not in normalized:
            normalized.append(lowered)
    return normalized


def _error_detail(
    code: str,
    message: str,
    details: dict[str, Any],
    factory: ErrorDetailFactory | None,
) -> Any:
    if factory is not None:
        return factory(code, message, details)
    return {"code": code, "message": message, "details": details}


def require_analysis_attempt(
    job: Any,
    request: Request,
    payload: Mapping[str, Any] | None = None,
    *,
    mutation: str,
    error_detail_factory: ErrorDetailFactory | None = None,
) -> str | None:
    """Fence a job mutation to the authoritative analysis attempt.

    Initial and legacy jobs without an attempt nonce remain writable without a
    precondition. Once the server has issued a nonce, every mutating request
    must echo it in ``X-Analysis-Attempt-Id`` (or the body compatibility alias).
    """

    target = job.target if isinstance(getattr(job, "target", None), dict) else {}
    current_ids = _normalized_ids(
        target.get("analysis_attempt_id"),
        target.get("analysisAttemptId"),
    )
    request_payload = payload if isinstance(payload, Mapping) else {}
    expected_ids = _normalized_ids(
        request.headers.get("x-analysis-attempt-id"),
        request_payload.get("expected_analysis_attempt_id"),
        request_payload.get("expectedAnalysisAttemptId"),
    )
    status = str(getattr(job, "status", "") or "").upper()

    if len(current_ids) > 1:
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "ANALYSIS_ATTEMPT_MISMATCH",
                "The job contains conflicting analysis attempt identifiers.",
                {
                    "mutation": mutation,
                    "status": status,
                    "current_analysis_attempt_ids": current_ids,
                },
                error_detail_factory,
            ),
        )

    if len(expected_ids) > 1:
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "ANALYSIS_ATTEMPT_MISMATCH",
                "The request contains conflicting analysis attempt identifiers.",
                {
                    "mutation": mutation,
                    "status": status,
                    "expected_analysis_attempt_ids": expected_ids,
                    "current_analysis_attempt_id": (
                        current_ids[0] if current_ids else None
                    ),
                },
                error_detail_factory,
            ),
        )

    current_id = current_ids[0] if current_ids else None
    expected_id = expected_ids[0] if expected_ids else None
    if current_id is None:
        if expected_id is None:
            return None
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "ANALYSIS_ATTEMPT_MISMATCH",
                "The job does not have the analysis attempt named by the request.",
                {
                    "mutation": mutation,
                    "status": status,
                    "expected_analysis_attempt_id": expected_id,
                    "current_analysis_attempt_id": None,
                },
                error_detail_factory,
            ),
        )

    if expected_id is None:
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "ANALYSIS_ATTEMPT_PRECONDITION_REQUIRED",
                "Reload the job and retry with its current analysis attempt id.",
                {
                    "mutation": mutation,
                    "status": status,
                    "current_analysis_attempt_id": current_id,
                },
                error_detail_factory,
            ),
        )

    if expected_id != current_id:
        raise HTTPException(
            status_code=409,
            detail=_error_detail(
                "ANALYSIS_ATTEMPT_MISMATCH",
                "The job advanced to a different analysis attempt.",
                {
                    "mutation": mutation,
                    "status": status,
                    "expected_analysis_attempt_id": expected_id,
                    "current_analysis_attempt_id": current_id,
                },
                error_detail_factory,
            ),
        )
    return current_id
