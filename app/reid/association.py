from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

DESCRIPTOR_VERSION = "hsv-torso-v1"
ASSOCIATION_VERSION = "reid-association-v1"


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    parsed = tuple(_finite(value, "descriptor.vector") for value in vector)
    if not parsed:
        raise ValueError("descriptor.vector must not be empty")
    norm = math.sqrt(sum(value * value for value in parsed))
    if norm <= 1e-12:
        raise ValueError("descriptor.vector must have non-zero norm")
    return tuple(value / norm for value in parsed)


@dataclass(frozen=True)
class AppearanceDescriptor:
    vector: tuple[float, ...]
    sample_count: int
    quality: float
    version: str = DESCRIPTOR_VERSION

    def __post_init__(self) -> None:
        normalized = _normalize(self.vector)
        if self.sample_count < 1:
            raise ValueError("descriptor.sample_count must be >= 1")
        quality = _finite(self.quality, "descriptor.quality")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("descriptor.quality must be in [0, 1]")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("descriptor.version must be a non-empty string")
        object.__setattr__(self, "vector", normalized)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "version", self.version.strip())

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "vector": list(self.vector),
            "sample_count": self.sample_count,
            "quality": round(self.quality, 6),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "AppearanceDescriptor":
        vector = payload.get("vector")
        if not isinstance(vector, list):
            raise ValueError("descriptor.vector must be an array")
        sample_count = payload.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise ValueError("descriptor.sample_count must be an integer")
        return cls(
            vector=tuple(vector),
            sample_count=sample_count,
            quality=_finite(payload.get("quality"), "descriptor.quality"),
            version=str(payload.get("version") or DESCRIPTOR_VERSION),
        )


@dataclass(frozen=True)
class IdentityProfile:
    identity_id: str
    descriptor: AppearanceDescriptor
    accepted_segments: int = 1
    source: str = "manual_anchor"

    def __post_init__(self) -> None:
        if not isinstance(self.identity_id, str) or not self.identity_id.strip():
            raise ValueError("identity_id must be a non-empty string")
        if self.accepted_segments < 1:
            raise ValueError("accepted_segments must be >= 1")
        object.__setattr__(self, "identity_id", self.identity_id.strip())

    def to_payload(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "descriptor": self.descriptor.to_payload(),
            "accepted_segments": self.accepted_segments,
            "source": self.source,
        }


@dataclass(frozen=True)
class CandidateProfile:
    candidate_id: str
    descriptor: AppearanceDescriptor | None
    overlap_score: float | None = None
    geometry_score: float | None = None
    detection_count: int = 0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if self.detection_count < 0:
            raise ValueError("detection_count must be >= 0")
        for field_name in ("overlap_score", "geometry_score"):
            value = getattr(self, field_name)
            if value is None:
                continue
            parsed = _finite(value, field_name)
            if not 0.0 <= parsed <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
            object.__setattr__(self, field_name, parsed)
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())


@dataclass(frozen=True)
class AssociationThresholds:
    min_combined_score: float = 0.76
    min_appearance_similarity: float = 0.78
    strong_overlap_score: float = 0.65
    min_margin: float = 0.07
    min_descriptor_quality: float = 0.30
    min_descriptor_samples: int = 2

    def __post_init__(self) -> None:
        for field_name in (
            "min_combined_score",
            "min_appearance_similarity",
            "strong_overlap_score",
            "min_margin",
            "min_descriptor_quality",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
            object.__setattr__(self, field_name, value)
        if self.min_descriptor_samples < 1:
            raise ValueError("min_descriptor_samples must be >= 1")


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    combined_score: float
    appearance_similarity: float | None
    overlap_score: float | None
    geometry_score: float | None
    descriptor_quality: float | None
    descriptor_samples: int
    reason_codes: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "combined_score": round(self.combined_score, 6),
            "appearance_similarity": (
                round(self.appearance_similarity, 6)
                if self.appearance_similarity is not None
                else None
            ),
            "overlap_score": (
                round(self.overlap_score, 6)
                if self.overlap_score is not None
                else None
            ),
            "geometry_score": (
                round(self.geometry_score, 6)
                if self.geometry_score is not None
                else None
            ),
            "descriptor_quality": (
                round(self.descriptor_quality, 6)
                if self.descriptor_quality is not None
                else None
            ),
            "descriptor_samples": self.descriptor_samples,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class AssociationDecision:
    status: str
    selected_candidate_id: str | None
    best_score: float
    margin: float
    reason_codes: tuple[str, ...]
    candidates: tuple[CandidateScore, ...]
    version: str = ASSOCIATION_VERSION
    validated: bool = False

    @property
    def accepted(self) -> bool:
        return self.status == "ACCEPTED" and self.selected_candidate_id is not None

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "validated": self.validated,
            "status": self.status,
            "selected_candidate_id": self.selected_candidate_id,
            "best_score": round(self.best_score, 6),
            "margin": round(self.margin, 6),
            "reason_codes": list(self.reason_codes),
            "candidates": [candidate.to_payload() for candidate in self.candidates],
        }


def cosine_similarity(
    first: AppearanceDescriptor,
    second: AppearanceDescriptor,
) -> float:
    if first.version != second.version:
        raise ValueError(
            f"descriptor version mismatch: {first.version!r} != {second.version!r}"
        )
    if len(first.vector) != len(second.vector):
        raise ValueError("descriptor vectors must have the same dimension")
    raw = sum(left * right for left, right in zip(first.vector, second.vector))
    return _clamp(raw)


def merge_descriptors(
    first: AppearanceDescriptor,
    second: AppearanceDescriptor,
) -> AppearanceDescriptor:
    if first.version != second.version:
        raise ValueError("cannot merge descriptors with different versions")
    if len(first.vector) != len(second.vector):
        raise ValueError("cannot merge descriptors with different dimensions")
    first_weight = first.sample_count * max(0.05, first.quality)
    second_weight = second.sample_count * max(0.05, second.quality)
    total_weight = first_weight + second_weight
    vector = tuple(
        (left * first_weight + right * second_weight) / total_weight
        for left, right in zip(first.vector, second.vector)
    )
    total_samples = first.sample_count + second.sample_count
    quality = (
        first.quality * first.sample_count + second.quality * second.sample_count
    ) / total_samples
    return AppearanceDescriptor(
        vector=vector,
        sample_count=total_samples,
        quality=_clamp(quality),
        version=first.version,
    )


def update_identity_profile(
    identity: IdentityProfile,
    descriptor: AppearanceDescriptor,
    *,
    source: str = "accepted_association",
) -> IdentityProfile:
    return replace(
        identity,
        descriptor=merge_descriptors(identity.descriptor, descriptor),
        accepted_segments=identity.accepted_segments + 1,
        source=source,
    )


def _score_candidate(
    identity: IdentityProfile,
    candidate: CandidateProfile,
    thresholds: AssociationThresholds,
) -> CandidateScore:
    reasons: list[str] = []
    descriptor = candidate.descriptor
    appearance: float | None = None
    descriptor_quality: float | None = None
    descriptor_samples = 0

    if descriptor is None:
        reasons.append("MISSING_APPEARANCE_DESCRIPTOR")
    else:
        descriptor_quality = descriptor.quality
        descriptor_samples = descriptor.sample_count
        if descriptor.version != identity.descriptor.version:
            reasons.append("DESCRIPTOR_VERSION_MISMATCH")
        else:
            appearance = cosine_similarity(identity.descriptor, descriptor)
        if descriptor.quality < thresholds.min_descriptor_quality:
            reasons.append("LOW_DESCRIPTOR_QUALITY")
        if descriptor.sample_count < thresholds.min_descriptor_samples:
            reasons.append("INSUFFICIENT_DESCRIPTOR_SAMPLES")

    weighted_components: list[tuple[float, float]] = []
    if appearance is not None:
        weighted_components.append((appearance, 0.65))
    if candidate.overlap_score is not None:
        weighted_components.append((candidate.overlap_score, 0.25))
    if candidate.geometry_score is not None:
        weighted_components.append((candidate.geometry_score, 0.10))
    weight_sum = sum(weight for _, weight in weighted_components)
    combined = (
        sum(value * weight for value, weight in weighted_components) / weight_sum
        if weight_sum > 0
        else 0.0
    )

    if appearance is not None and appearance < thresholds.min_appearance_similarity:
        if (candidate.overlap_score or 0.0) < thresholds.strong_overlap_score:
            reasons.append("LOW_APPEARANCE_SIMILARITY")
    if combined < thresholds.min_combined_score:
        reasons.append("LOW_COMBINED_SCORE")

    return CandidateScore(
        candidate_id=candidate.candidate_id,
        combined_score=_clamp(combined),
        appearance_similarity=appearance,
        overlap_score=candidate.overlap_score,
        geometry_score=candidate.geometry_score,
        descriptor_quality=descriptor_quality,
        descriptor_samples=descriptor_samples,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def associate_identity(
    identity: IdentityProfile,
    candidates: Iterable[CandidateProfile],
    *,
    thresholds: AssociationThresholds | None = None,
) -> AssociationDecision:
    thresholds = thresholds or AssociationThresholds()
    scored = tuple(
        sorted(
            (
                _score_candidate(identity, candidate, thresholds)
                for candidate in candidates
            ),
            key=lambda candidate: (
                candidate.combined_score,
                candidate.appearance_similarity or 0.0,
                candidate.candidate_id,
            ),
            reverse=True,
        )
    )
    if not scored:
        return AssociationDecision(
            status="ABSTAINED",
            selected_candidate_id=None,
            best_score=0.0,
            margin=0.0,
            reason_codes=("NO_CANDIDATES",),
            candidates=(),
        )

    best = scored[0]
    runner_up_score = scored[1].combined_score if len(scored) > 1 else 0.0
    margin = max(0.0, best.combined_score - runner_up_score)
    reasons = list(best.reason_codes)
    if len(scored) > 1 and margin < thresholds.min_margin:
        reasons.append("AMBIGUOUS_CANDIDATE_MARGIN")

    hard_failures = {
        "MISSING_APPEARANCE_DESCRIPTOR",
        "DESCRIPTOR_VERSION_MISMATCH",
        "LOW_DESCRIPTOR_QUALITY",
        "INSUFFICIENT_DESCRIPTOR_SAMPLES",
        "LOW_APPEARANCE_SIMILARITY",
        "LOW_COMBINED_SCORE",
        "AMBIGUOUS_CANDIDATE_MARGIN",
    }
    accepted = not hard_failures.intersection(reasons)
    if accepted:
        return AssociationDecision(
            status="ACCEPTED",
            selected_candidate_id=best.candidate_id,
            best_score=best.combined_score,
            margin=margin,
            reason_codes=("ASSOCIATION_ACCEPTED",),
            candidates=scored,
        )
    return AssociationDecision(
        status="ABSTAINED",
        selected_candidate_id=None,
        best_score=best.combined_score,
        margin=margin,
        reason_codes=tuple(dict.fromkeys(reasons or ["ASSOCIATION_UNCERTAIN"])),
        candidates=scored,
    )
