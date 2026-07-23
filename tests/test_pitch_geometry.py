import unittest

import numpy as np

from app.vision.pitch_geometry import (
    PitchGeometryThresholds,
    detect_pitch_geometry,
    estimate_pitch_evidence,
)
from tests.synthetic_pitch import make_non_pitch_frame, make_pitch_frame


class PitchGeometryTests(unittest.TestCase):
    def test_synthetic_pitch_produces_geometry_candidates(self):
        result = detect_pitch_geometry(make_pitch_frame())

        self.assertEqual(result.status, "CANDIDATE")
        self.assertEqual(result.evidence.classification, "PITCH_CANDIDATE")
        self.assertGreaterEqual(result.evidence.line_count, 4)
        self.assertGreaterEqual(result.evidence.orientation_family_count, 2)
        self.assertGreaterEqual(result.evidence.intersection_count, 2)
        self.assertGreater(result.evidence.pitch_probability, 0.7)
        self.assertFalse(result.semantic_landmarks_available)
        self.assertFalse(result.calibration_ready)
        self.assertIn("SEMANTIC_LANDMARKS_NOT_ASSIGNED", result.reason_codes)
        for point in result.keypoints:
            self.assertGreaterEqual(point.x, 0.0)
            self.assertLessEqual(point.x, 1.0)
            self.assertGreaterEqual(point.y, 0.0)
            self.assertLessEqual(point.y, 1.0)

    def test_non_pitch_frame_is_rejected(self):
        result = detect_pitch_geometry(make_non_pitch_frame())

        self.assertEqual(result.status, "INSUFFICIENT")
        self.assertEqual(result.evidence.classification, "NON_PITCH")
        self.assertIn("FRAME_NOT_CONFIDENT_PITCH_VIEW", result.reason_codes)

    def test_black_frame_does_not_crash_or_claim_pitch(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        evidence = estimate_pitch_evidence(frame)

        self.assertEqual(evidence.classification, "NON_PITCH")
        self.assertEqual(evidence.pitch_probability, 0.0)

    def test_threshold_validation(self):
        with self.assertRaisesRegex(ValueError, "maximum_lines"):
            PitchGeometryThresholds(maximum_lines=1, minimum_line_count=3)


if __name__ == "__main__":
    unittest.main()
