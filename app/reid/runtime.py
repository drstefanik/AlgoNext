from __future__ import annotations

import logging
import os
from typing import Any, Callable

from app.reid.full_match_runtime import (
    budget_full_match_kwargs,
    install_progress_adapter,
    mark_partial_timeout,
    partial_timeout_output,
)
from app.reid.progress_reporting import (
    begin_full_match_progress,
    end_full_match_progress,
    install_progress_stats_adapter,
)

logger = logging.getLogger(__name__)


def reid_enabled() -> bool:
    value = (os.environ.get("PLAYER_REID_ENABLED") or "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def fail_open_enabled() -> bool:
    value = (os.environ.get("PLAYER_REID_FAIL_OPEN") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _decorate_output(output: Any, profile: Any | None) -> Any:
    if not isinstance(output, dict):
        return output
    decorated = dict(output)
    if profile is not None:
        decorated.setdefault("runtime_profile", profile.to_payload())
    return decorated


def _partial_timeout_result(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    profile: Any | None,
    *,
    reid_was_active: bool,
) -> dict[str, Any]:
    output = partial_timeout_output(args, kwargs, profile)
    if reid_was_active:
        return output
    output = dict(output)
    output["identity_mode"] = "disabled"
    output["method"] = "yolo+bytetrack"
    output["reid_summary"] = {
        "status": "DISABLED",
        "validated": False,
        "reason_codes": [
            "PLAYER_REID_DISABLED",
            "TRACKING_BUDGET_EXHAUSTED",
        ],
    }
    return output


def install_windowed_reid(
    tracking_module: Any | None = None,
    implementation: Callable[..., Any] | None = None,
) -> bool:
    if tracking_module is None:
        from app.workers import tracking as tracking_module

    install_progress_adapter(tracking_module)
    install_progress_stats_adapter(tracking_module)

    current = getattr(tracking_module, "track_player_windowed", None)
    if current is None:
        raise RuntimeError("tracking module does not expose track_player_windowed")
    if getattr(current, "__algonext_reid_wrapper__", False):
        return reid_enabled()

    if implementation is None and reid_enabled():
        from app.reid import windowed_tracking
        from app.reid.benchmark_evidence import install_candidate_evidence
        from app.reid.team_color_guard import guard_windowed_reid

        install_candidate_evidence(windowed_tracking)
        implementation = guard_windowed_reid(
            windowed_tracking.track_player_windowed_reid
        )

    original = current
    timeout_error = getattr(tracking_module, "TrackingTimeoutError", None)

    def patched(*args: Any, **kwargs: Any) -> Any:
        effective_kwargs, profile = budget_full_match_kwargs(kwargs)
        job_id = str(args[0]) if args else None
        reid_was_active = reid_enabled() and implementation is not None
        begin_full_match_progress(job_id, profile)
        try:
            if not reid_was_active:
                output = original(*args, **effective_kwargs)
                output = _decorate_output(output, profile)
                if isinstance(output, dict):
                    output.setdefault(
                        "reid_summary",
                        {
                            "status": "DISABLED",
                            "validated": False,
                            "reason_codes": ["PLAYER_REID_DISABLED"],
                        },
                    )
                return output

            try:
                output = implementation(
                    *args,
                    fallback=original,
                    **effective_kwargs,
                )
                return _decorate_output(output, profile)
            except Exception as exc:
                if isinstance(timeout_error, type) and isinstance(exc, timeout_error):
                    logger.warning(
                        "Full-match tracking reached its runtime budget job_id=%s",
                        job_id,
                    )
                    mark_partial_timeout(job_id, profile)
                    return _partial_timeout_result(
                        args,
                        effective_kwargs,
                        profile,
                        reid_was_active=True,
                    )

                logger.exception("Experimental Player ReID failed")
                if not fail_open_enabled():
                    raise
                output = _decorate_output(
                    original(*args, **effective_kwargs), profile
                )
                if isinstance(output, dict):
                    output["reid_summary"] = {
                        "status": "FALLBACK_LEGACY",
                        "validated": False,
                        "reason_codes": ["REID_RUNTIME_EXCEPTION"],
                    }
                return output
        except Exception as exc:
            if isinstance(timeout_error, type) and isinstance(exc, timeout_error):
                logger.warning(
                    "Legacy full-match tracking reached its runtime budget job_id=%s",
                    job_id,
                )
                mark_partial_timeout(job_id, profile)
                return _partial_timeout_result(
                    args,
                    effective_kwargs,
                    profile,
                    reid_was_active=reid_was_active,
                )
            raise
        finally:
            end_full_match_progress(job_id)

    patched.__name__ = "track_player_windowed_budgeted"
    patched.__doc__ = (
        "Budget-aware full-match tracker with optional conservative Player ReID."
    )
    setattr(patched, "__algonext_reid_wrapper__", True)
    setattr(patched, "__algonext_legacy_tracker__", original)
    tracking_module.track_player_windowed = patched
    logger.info(
        "Budget-aware windowed tracker installed reid_enabled=%s",
        reid_enabled(),
    )
    return reid_enabled()
