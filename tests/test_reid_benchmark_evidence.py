import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.reid.association import (
    AppearanceDescriptor,
    AssociationDecision,
    CandidateProfile,
    CandidateScore,
)
from app.reid.benchmark_evidence import (
    candidate_evidence,
    install_candidate_evidence,
)


class ReIDBenchmarkEvidenceTests(unittest.TestCase):
    def test_candidate_evidence_uses_absolute_time_and_normalized_bbox(self):
        detections = [
            {
                "t": float(index),
                "sample_index": index,
                "conf": 0.8,
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.2},
            }
            for index in range(5)
        ]
        with patch.dict(os.environ, {"PLAYER_REID_BENCHMARK_EVIDENCE_SAMPLES": "3"}):
            evidence = candidate_evidence(detections, window_start=10.0, fps=2.0)

        self.assertEqual(len(evidence), 3)
        self.assertEqual(evidence[0]["time_sec"], 10.0)
        self.assertEqual(evidence[0]["frame_index"], 20)
        self.assertEqual(evidence[0]["bbox"]["x"], 0.1)

    def test_installer_enriches_candidate_payload_without_changing_decision(self):
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0), sample_count=3, quality=0.9
        )

        def build(_segment, track_map, **_kwargs):
            return (
                [
                    CandidateProfile(
                        candidate_id="7",
                        descriptor=descriptor,
                        detection_count=3,
                        metadata={"local_track_id": 7},
                    )
                ],
                {"7": 7},
                {"7": descriptor},
            )

        decision = AssociationDecision(
            status="ACCEPTED",
            selected_candidate_id="7",
            best_score=0.9,
            margin=0.2,
            reason_codes=("ASSOCIATION_ACCEPTED",),
            candidates=(
                CandidateScore(
                    candidate_id="7",
                    combined_score=0.9,
                    appearance_similarity=0.95,
                    overlap_score=None,
                    geometry_score=0.5,
                    descriptor_quality=0.9,
                    descriptor_samples=3,
                    reason_codes=(),
                ),
            ),
        )

        module = SimpleNamespace(
            _build_candidate_profiles=build,
            associate_identity=lambda _identity, _candidates, **_kwargs: decision,
        )
        self.assertTrue(install_candidate_evidence(module))
        self.assertFalse(install_candidate_evidence(module))

        profiles, _, _ = module._build_candidate_profiles(
            None,
            {
                7: [
                    {
                        "t": 1.0,
                        "sample_index": 0,
                        "conf": 0.9,
                        "bbox": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2},
                    }
                ]
            },
            window_start=20.0,
            fps=1,
        )
        wrapped = module.associate_identity(object(), profiles)
        payload = wrapped.to_payload()

        self.assertTrue(wrapped.accepted)
        self.assertEqual(payload["selected_candidate_id"], "7")
        self.assertEqual(payload["candidates"][0]["evidence"][0]["time_sec"], 21.0)

    def test_empty_scoped_tracklet_does_not_reopen_raw_id_evidence(self):
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0), sample_count=3, quality=0.9
        )

        def build(_segment, _track_map, **_kwargs):
            return (
                [
                    CandidateProfile(
                        candidate_id="7",
                        descriptor=None,
                        detection_count=0,
                        metadata={
                            "local_track_id": 7,
                            "tracklet_sample_indices": (),
                            "tracklet_detections": (),
                        },
                    )
                ],
                {"7": 7},
                {"7": descriptor},
            )

        module = SimpleNamespace(
            _build_candidate_profiles=build,
            associate_identity=lambda *_args, **_kwargs: None,
        )
        self.assertTrue(install_candidate_evidence(module))
        profiles, _, _ = module._build_candidate_profiles(
            None,
            {
                7: [
                    {
                        "t": 1.0,
                        "sample_index": 0,
                        "conf": 0.9,
                        "bbox": {
                            "x": 0.2,
                            "y": 0.2,
                            "w": 0.1,
                            "h": 0.2,
                        },
                    }
                ]
            },
            window_start=20.0,
            fps=1,
        )

        self.assertEqual(profiles[0].metadata["benchmark_evidence"], [])


if __name__ == "__main__":
    unittest.main()
