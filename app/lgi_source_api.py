from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.integrations.lgi_readonly import (
    LgiMatchNotFound,
    LgiSourceError,
    assert_read_only_connection,
    get_match,
    list_matches,
)

router = APIRouter(prefix="/integrations/lgi", tags=["LGI read-only"])


def _service_unavailable(exc: LgiSourceError) -> HTTPException:
    if isinstance(exc, LgiMatchNotFound):
        return HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc), "read_only": True},
        )
    return HTTPException(
        status_code=503,
        detail={
            "code": exc.code,
            "message": str(exc),
            "read_only": True,
        },
    )


@router.get("/health")
def lgi_health():
    try:
        health = assert_read_only_connection()
    except LgiSourceError as exc:
        raise _service_unavailable(exc) from exc
    return {**health, "source": "LGI Channel"}


@router.get("/matches")
def lgi_matches(
    query: str = Query(default="", max_length=120),
    limit: int = Query(default=30, ge=1, le=100),
    pilot_only: bool = Query(default=False),
):
    try:
        items = list_matches(query=query, limit=limit, pilot_only=pilot_only)
    except LgiSourceError as exc:
        raise _service_unavailable(exc) from exc
    return {
        "items": [item.as_dict(include_lineup=False) for item in items],
        "count": len(items),
        "read_only": True,
    }


@router.get("/matches/{match_id}")
def lgi_match(match_id: str):
    try:
        match = get_match(match_id)
    except (LgiSourceError, ValueError) as exc:
        if isinstance(exc, LgiSourceError):
            raise _service_unavailable(exc) from exc
        raise HTTPException(
            status_code=400,
            detail={"code": "LGI_MATCH_ID_INVALID", "message": "Invalid LGI match ID."},
        ) from exc
    return {**match.as_dict(), "read_only": True}
