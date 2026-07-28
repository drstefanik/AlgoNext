import unittest

from app.reid.association import (
    AppearanceDescriptor,
    AssociationThresholds,
    CandidateProfile,
    IdentityProfile,
    associate_identity,
    cosine_similarity,
    update_identity_profile,
)


def descriptor(values, *, samples=3, quality=0.8):
    return AppearanceDescriptor(
        vector=tuple(values),
        sample_count=samples,
        quality=quality,
    )


class ReIdAssociationTests(unittest.TestCase):
    def setUp(self):
        self.identity = IdentityProfile(
            identity_id="selected-player",
            descriptor=descriptor(
                [1.0, 0.1, 0.0, 0.0],
                samples=4,
                quality=0.9,
            ),
        )

    def test_clear_candidate_is_accepted(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    descriptor([0.99, 0.12, 0.0, 0.0], quality=0.85),
                    overlap_score=0.72,
                    geometry_score=0.75,
                    detection_count=50,
                ),
                CandidateProfile(
                    "track-9",
                    descriptor([0.2, 0.9, 0.0, 0.0], quality=0.85),
                    overlap_score=0.15,
                    geometry_score=0.55,
                    detection_count=45,
                ),
            ],
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.selected_candidate_id, "track-6")
        self.assertGreaterEqual(decision.margin, 0.07)
        self.assertFalse(decision.validated)

    def test_similar_kits_with_small_margin_abstain(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    descriptor([0.98, 0.15, 0.0, 0.0]),
                    overlap_score=0.35,
                    geometry_score=0.60,
                ),
                CandidateProfile(
                    "track-8",
                    descriptor([0.97, 0.17, 0.0, 0.0]),
                    overlap_score=0.34,
                    geometry_score=0.61,
                ),
            ],
        )

        self.assertFalse(decision.accepted)
        self.assertIn(
            "AMBIGUOUS_CANDIDATE_MARGIN",
            decision.reason_codes,
        )

    def test_unique_strong_overlap_resolves_a_tiny_appearance_margin(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-overlap",
                    descriptor([0.99, 0.12, 0.0, 0.0]),
                    overlap_score=0.78,
                    geometry_score=0.60,
                    metadata={"strong_overlap_unique": True},
                ),
                CandidateProfile(
                    "track-lookalike",
                    descriptor([0.98, 0.15, 0.0, 0.0]),
                    overlap_score=0.64,
                    geometry_score=0.60,
                ),
            ],
            thresholds=AssociationThresholds(
                require_strong_overlap=True,
            ),
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.selected_candidate_id, "track-overlap")
        self.assertLess(decision.margin, 0.07)
        self.assertIn("STRONG_TEMPORAL_OVERLAP", decision.reason_codes)

    def test_unverified_unique_overlap_does_not_bypass_margin(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-overlap",
                    descriptor([0.99, 0.12, 0.0, 0.0]),
                    overlap_score=0.78,
                    geometry_score=0.60,
                ),
                CandidateProfile(
                    "track-lookalike",
                    descriptor([0.98, 0.15, 0.0, 0.0]),
                    overlap_score=0.64,
                    geometry_score=0.60,
                ),
            ],
            thresholds=AssociationThresholds(
                require_strong_overlap=True,
            ),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("AMBIGUOUS_CANDIDATE_MARGIN", decision.reason_codes)

    def test_two_strong_overlap_candidates_keep_the_margin_gate(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    descriptor([0.99, 0.12, 0.0, 0.0]),
                    overlap_score=0.78,
                    geometry_score=0.60,
                ),
                CandidateProfile(
                    "track-8",
                    descriptor([0.98, 0.15, 0.0, 0.0]),
                    overlap_score=0.76,
                    geometry_score=0.60,
                ),
            ],
            thresholds=AssociationThresholds(
                require_strong_overlap=True,
            ),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("AMBIGUOUS_CANDIDATE_MARGIN", decision.reason_codes)

    def test_non_unique_strong_overlap_abstains_even_with_large_appearance_margin(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    descriptor([0.99, 0.12, 0.0, 0.0]),
                    overlap_score=0.78,
                    geometry_score=0.80,
                    metadata={"strong_overlap_unique": False},
                ),
                CandidateProfile(
                    "track-8",
                    descriptor([0.50, 0.86, 0.0, 0.0]),
                    overlap_score=0.70,
                    geometry_score=0.75,
                    metadata={"strong_overlap_unique": False},
                ),
            ],
            thresholds=AssociationThresholds(
                require_strong_overlap=True,
            ),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("AMBIGUOUS_STRONG_OVERLAP", decision.reason_codes)

    def test_appearance_only_candidate_abstains_when_overlap_is_required(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-lookalike",
                    descriptor([0.99, 0.12, 0.0, 0.0]),
                    geometry_score=0.90,
                )
            ],
            thresholds=AssociationThresholds(
                require_strong_overlap=True,
            ),
        )

        self.assertFalse(decision.accepted)
        self.assertIn(
            "STRONG_TEMPORAL_OVERLAP_REQUIRED",
            decision.reason_codes,
        )

    def test_missing_appearance_never_accepts_geometry_alone(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    None,
                    overlap_score=1.0,
                    geometry_score=1.0,
                )
            ],
        )

        self.assertFalse(decision.accepted)
        self.assertIn(
            "MISSING_APPEARANCE_DESCRIPTOR",
            decision.reason_codes,
        )

    def test_low_quality_descriptor_abstains(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    descriptor(
                        [1.0, 0.1, 0.0, 0.0],
                        samples=3,
                        quality=0.1,
                    ),
                    overlap_score=0.9,
                    geometry_score=0.9,
                )
            ],
        )

        self.assertFalse(decision.accepted)
        self.assertIn("LOW_DESCRIPTOR_QUALITY", decision.reason_codes)

    def test_single_descriptor_sample_abstains(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    descriptor(
                        [1.0, 0.1, 0.0, 0.0],
                        samples=1,
                        quality=0.9,
                    ),
                    overlap_score=0.9,
                    geometry_score=0.9,
                )
            ],
        )

        self.assertFalse(decision.accepted)
        self.assertIn(
            "INSUFFICIENT_DESCRIPTOR_SAMPLES",
            decision.reason_codes,
        )

    def test_no_candidates_abstains(self):
        decision = associate_identity(self.identity, [])

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason_codes, ("NO_CANDIDATES",))

    def test_low_appearance_cannot_be_hidden_by_geometry(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    descriptor([0.0, 1.0, 0.0, 0.0], quality=0.9),
                    overlap_score=0.55,
                    geometry_score=1.0,
                )
            ],
        )

        self.assertFalse(decision.accepted)
        self.assertIn(
            "LOW_APPEARANCE_SIMILARITY",
            decision.reason_codes,
        )

    def test_profile_updates_only_after_explicit_acceptance(self):
        candidate_descriptor = descriptor(
            [0.95, 0.2, 0.0, 0.0],
            samples=5,
            quality=0.75,
        )
        updated = update_identity_profile(
            self.identity,
            candidate_descriptor,
        )

        self.assertEqual(updated.accepted_segments, 2)
        self.assertEqual(updated.descriptor.sample_count, 9)
        self.assertGreater(
            cosine_similarity(
                updated.descriptor,
                self.identity.descriptor,
            ),
            0.95,
        )

    def test_descriptor_version_mismatch_is_rejected(self):
        foreign = AppearanceDescriptor(
            vector=(1.0, 0.0, 0.0, 0.0),
            sample_count=3,
            quality=0.9,
            version="foreign-v1",
        )
        decision = associate_identity(
            self.identity,
            [CandidateProfile("track-6", foreign, overlap_score=0.9)],
        )

        self.assertFalse(decision.accepted)
        self.assertIn(
            "DESCRIPTOR_VERSION_MISMATCH",
            decision.reason_codes,
        )

    def test_thresholds_can_be_tightened_for_validation(self):
        decision = associate_identity(
            self.identity,
            [
                CandidateProfile(
                    "track-6",
                    descriptor([0.95, 0.2, 0.0, 0.0]),
                    overlap_score=0.7,
                    geometry_score=0.7,
                )
            ],
            thresholds=AssociationThresholds(min_combined_score=0.99),
        )

        self.assertFalse(decision.accepted)
        self.assertIn("LOW_COMBINED_SCORE", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
