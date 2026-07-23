from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def reid_enabled() -> bool:
    value = (os.environ.get("PLAYER_REID_ENABLED") or "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def fail_open_enabled() -> bool:
    value = (os.environ.get("PLAYER_REID_FAIL_OPEN") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def install_windowed_reid(
    tracking_module: Any | None = None,
    implementation: Callable[..., Any] | None = None,
) -> bool:
    if not reid_enabled():
        logger.info("Player ReID feature flag is disabled")
        return False
    if tracking_module is None:
        from app.workers import tracking as tracking_module
    current = getattr(tracking_module, "track_player_windowed", None)
    if current is None:
        raise RuntimeError("tracking module does not expose track_player_windowed")
    if getattr(current, "__algonext_reid_wrapper__", False):
        return True
    if implementation is None:
        from app.reid.windowed_tracking import track_player_windowed_reid

        implementation = track_player_windowed_reid
    original = current
    timeout_error = getattr(tracking_module, "TrackingTimeoutError", None)

    def patched(*args: Any, **kwargs: Any) -> Any:
        try:
            return implementation(*args, fallback=original, **kwargs)
        except Exception as exc:
            if isinstance(timeout_error, type) and isinstance(exc, timeout_error):
                raise
            logger.exception("Experimental Player ReID failed")
            if not fail_open_enabled():
                raise
            output = original(*args, **kwargs)
            if isinstance(output, dict):
                output = dict(output)
                output["reid_summary"] = {
                    "status": "FALLBACK_LEGACY",
                    "validated": False,
                    "reason_codes": ["REID_RUNTIME_EXCEPTION"],
                }
            return output

    patched.__name__ = "track_player_windowed_reid_feature_flag"
    patched.__doc__ = "Feature-flagged conservative Player ReID wrapper."
    setattr(patched, "__algonext_reid_wrapper__", True)
    setattr(patched, "__algonext_legacy_tracker__", original)
    tracking_module.track_player_windowed = patched
    logger.info("Player ReID windowed tracker installed")
    return True
