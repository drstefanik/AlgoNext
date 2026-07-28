from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any, Mapping, Sequence

from app.reid.association import CandidateProfile
from app.reid.window_logic import choose_descriptor_detections


def _sample_limit() -> int:
    try:
        value = int(os.environ.get("PLAYER_REID_BENCHMARK_EVIDENCE_SAMPLES", "3"))
    except (TypeError, ValueError):
        value = 3
    return max(1, min(6, value))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _bbox_payload(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    x = max(0.0, min(1.0, _finite(value.get("x"))))
    y = max(0.0, min(1.0, _finite(value.get("y"))))
    w = max(0.0, min(1.0 - x, _finite(value.get("w"))))
    h = max(0.0, min(1.0 - y, _finite(value.get("h"))))
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def candidate_evidence(
    detections: Sequence[Mapping[str, Any]],
    *,
    window_start: float,
    fps: float,
) -> list[dict[str, Any]]:
    """Persist a few bbox samples so candidate IDs can be human-reviewed later."""

    evidence: list[dict[str, Any]] = []
    for detection in choose_descriptor_detections(list(detections), _sample_limit()):
        bbox = _bbox_payload(detection.get("bbox"))
        if bbox is None:
            continue
        absolute_time = max(0.0, window_start + _finite(detection.get("t")))
        item: dict[str, Any] = {
            "time_sec": round(absolute_time, 6),
            "frame_index": int(round(absolute_time * max(1e-9, fps))),
            "bbox": bbox,
            "confidence": round(max(0.0, min(1.0, _finite(detection.get("conf")))), 6),
        }
        evidence.append(item)
    return evidence


class _DecisionWithEvidence:
    def __init__(self, decision: Any, candidates: Sequence[CandidateProfile]):
        self._decision = decision
        self._evidence = {
            candidate.candidate_id: list(
                (candidate.metadata or {}).get("benchmark_evidence") or []
            )
            for candidate in candidates
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._decision, name)

    def to_payload(self) -> dict[str, Any]:
        payload = self._decision.to_payload()
        for candidate in payload.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id") or "")
            evidence = self._evidence.get(candidate_id) or []
            if evidence:
                candidate["evidence"] = evidence
        return payload


def install_candidate_evidence(windowed_tracking_module: Any) -> bool:
    """Patch the experimental runtime without changing its association semantics."""

    current_builder = getattr(
        windowed_tracking_module, "_build_candidate_profiles", None
    )
    current_associate = getattr(windowed_tracking_module, "associate_identity", None)
    if not callable(current_builder) or not callable(current_associate):
        raise RuntimeError("windowed tracking module lacks ReID candidate hooks")
    if getattr(current_builder, "__algonext_benchmark_evidence__", False):
        return False

    def build_with_evidence(*args: Any, **kwargs: Any):
        profiles, id_lookup, descriptor_lookup = current_builder(*args, **kwargs)
        track_map = args[1] if len(args) > 1 else kwargs.get("track_map")
        track_map = track_map if isinstance(track_map, Mapping) else {}
        window_start = _finite(kwargs.get("window_start"))
        fps = max(1e-9, _finite(kwargs.get("fps"), 1.0))
        enriched: list[CandidateProfile] = []
        for profile in profiles:
            metadata = dict(profile.metadata or {})
            local_track_id = metadata.get("local_track_id")
            raw_detections = track_map.get(local_track_id) or []
            if "tracklet_detections" in metadata:
                raw_detections = list(metadata.get("tracklet_detections") or ())
            elif "tracklet_sample_indices" in metadata:
                tracklet_sample_indices = {
                    int(value)
                    for value in (metadata.get("tracklet_sample_indices") or ())
                }
                raw_detections = [
                    item
                    for item in raw_detections
                    if item.get("sample_index") is not None
                    and int(item.get("sample_index")) in tracklet_sample_indices
                ]
            metadata["benchmark_evidence"] = candidate_evidence(
                [item for item in raw_detections if isinstance(item, Mapping)],
                window_start=window_start,
                fps=fps,
            )
            enriched.append(replace(profile, metadata=metadata))
        return enriched, id_lookup, descriptor_lookup

    def associate_with_evidence(identity: Any, candidates: Any, **kwargs: Any):
        candidate_list = tuple(candidates)
        decision = current_associate(identity, candidate_list, **kwargs)
        return _DecisionWithEvidence(decision, candidate_list)

    setattr(build_with_evidence, "__algonext_benchmark_evidence__", True)
    setattr(build_with_evidence, "__algonext_original__", current_builder)
    setattr(associate_with_evidence, "__algonext_benchmark_evidence__", True)
    setattr(associate_with_evidence, "__algonext_original__", current_associate)
    windowed_tracking_module._build_candidate_profiles = build_with_evidence
    windowed_tracking_module.associate_identity = associate_with_evidence
    return True
