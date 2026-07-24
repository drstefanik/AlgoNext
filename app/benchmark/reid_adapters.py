from __future__ import annotations

import math
from typing import Any, Mapping

from app.benchmark.reid_schema import ReIDSequencePrediction


def _finite(value: Any, field: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def _optional_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def prediction_from_algonext_reid(
    tracking: Mapping[str, Any], *, video_id: str
) -> ReIDSequencePrediction:
    """Convert AlgoNext full-match segment decisions into benchmark predictions."""

    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must be a non-empty string")
    segments = tracking.get("segments")
    if not isinstance(segments, list):
        raise ValueError("tracking.segments must be an array")

    windows: list[dict[str, Any]] = []
    for window_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"tracking.segments[{window_index}] must be an object")
        reid = segment.get("reid")
        reid = reid if isinstance(reid, Mapping) else {}
        reason_codes = [
            str(code).strip()
            for code in (reid.get("reason_codes") or [])
            if isinstance(code, str) and code.strip()
        ]
        candidates = [
            item for item in (reid.get("candidates") or []) if isinstance(item, Mapping)
        ]
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                _finite(item.get("combined_score"), "candidate.combined_score", 0.0),
                _optional_id(item.get("candidate_id")) or "",
            ),
            reverse=True,
        )
        candidate_ids = [
            candidate_id
            for candidate_id in (
                _optional_id(item.get("candidate_id")) for item in ranked_candidates
            )
            if candidate_id is not None
        ]
        selected_candidate_id = _optional_id(reid.get("selected_candidate_id"))
        raw_status = str(
            reid.get("status") or segment.get("identity_status") or "ABSTAINED"
        ).strip().upper()
        failed = "WINDOW_PROCESSING_FAILED" in reason_codes or raw_status == "FAILED"
        if raw_status == "ACCEPTED" and selected_candidate_id is not None:
            decision = "ACCEPTED"
        elif failed:
            decision = "FAILED"
            selected_candidate_id = None
        else:
            decision = "ABSTAINED"
            selected_candidate_id = None

        best_candidate_id = candidate_ids[0] if candidate_ids else selected_candidate_id
        if selected_candidate_id is not None and selected_candidate_id not in candidate_ids:
            candidate_ids.insert(0, selected_candidate_id)
        windows.append(
            {
                "window_index": window_index,
                "window_start": _finite(
                    segment.get("window_start"),
                    f"tracking.segments[{window_index}].window_start",
                ),
                "window_end": _finite(
                    segment.get("window_end"),
                    f"tracking.segments[{window_index}].window_end",
                ),
                "decision": decision,
                "selected_candidate_id": selected_candidate_id,
                "best_candidate_id": best_candidate_id,
                "best_score": max(
                    0.0,
                    min(
                        1.0,
                        _finite(
                            reid.get("best_score", segment.get("reacquire_score")),
                            f"tracking.segments[{window_index}].reid.best_score",
                            0.0,
                        ),
                    ),
                ),
                "margin": max(
                    0.0,
                    min(
                        1.0,
                        _finite(
                            reid.get("margin"),
                            f"tracking.segments[{window_index}].reid.margin",
                            0.0,
                        ),
                    ),
                ),
                "candidate_ids": candidate_ids,
                "reason_codes": reason_codes,
            }
        )

    return ReIDSequencePrediction.from_payload(
        {
            "schema_version": "reid-window-prediction-v1",
            "video_id": video_id.strip(),
            "windows": windows,
        }
    )
