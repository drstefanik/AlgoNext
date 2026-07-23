from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_DEFAULT_CODE = "HTTP_ERROR"
_DEFAULT_MESSAGE = "Request failed"
_PASSTHROUGH_FIELDS = ("missing", "allow_force", "allowForce")


def _non_empty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def normalize_http_exception_detail(detail: Any) -> dict[str, Any]:
    """Convert every FastAPI ``HTTPException.detail`` shape to our public contract.

    Older endpoints wrap errors as ``{"error": {...}}`` while newer ones use a
    flat mapping. Keeping this normalization at the application boundary avoids
    leaking those historical differences to clients.
    """

    if isinstance(detail, Mapping):
        nested_error = detail.get("error")
        source: Mapping[str, Any] = (
            nested_error if isinstance(nested_error, Mapping) else detail
        )

        code = _non_empty_text(source.get("code")) or _DEFAULT_CODE
        message = (
            _non_empty_text(source.get("message"))
            or _non_empty_text(detail.get("message"))
            or _DEFAULT_MESSAGE
        )

        error: dict[str, Any] = {"code": code, "message": message}

        details = source.get("details")
        if isinstance(details, Mapping):
            error["details"] = dict(details)
        elif details is not None:
            error["details"] = details

        for field in _PASSTHROUGH_FIELDS:
            if field in source:
                error[field] = source[field]
            elif field in detail:
                error[field] = detail[field]

        return error

    if isinstance(detail, list):
        return {
            "code": _DEFAULT_CODE,
            "message": "Request validation failed",
            "details": {"errors": detail},
        }

    message = _non_empty_text(detail) if isinstance(detail, str) else None
    if message is None and detail is not None:
        message = str(detail).strip() or None

    return {
        "code": _DEFAULT_CODE,
        "message": message or _DEFAULT_MESSAGE,
    }
