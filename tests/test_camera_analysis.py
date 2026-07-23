import unittest

from app.vision.camera_analysis import analyze_camera_sequence
from tests.synthetic_pitch import make_non_pitch_frame, make_pitch_frame


class CameraAnalysisTests(unittest.TestCase):
    def test_pitch_shot_reaches_geometry_candidate_but_not_auto_calibration(self):
        samples = [(index * 0.5, make_pitch_frame()) for index in range(8)]
        result = analyze_camera_sequence(samples, source_duration_sec=4.0)

        self.assertEqual(len(result.segments), 1)
        segment = result.segments[0]
        self.assertEqual(segment.status, "GEOMETRY_CANDIDATE")
        self.assertGreaterEqual(segment.geometry_candidate_count, 1)
        self.assertFalse(segment.automatic_calibration_available)
        self.assertFalse(result.automatic_calibration_available)
        self.assertIn(
            "SEMANTIC_PITCH_KEYPOINT_MODEL_REQUIRED",
            result.reason_codes,
        )

    def test_non_pitch_shot_is_excluded_before_geometry(self):
        samples = [(index * 0.5, make_non_pitch_frame()) for index in range(8)]
        result = analyze_camera_sequence(samples, source_duration_sec=4.0)

        segment = result.segments[0]
        self.assertEqual(segment.status, "EXCLUDED")
        self.assertEqual(segment.geometry_frames, ())
        self.assertTrue(segment.shot.exclude_from_calibration)

    def test_mixed_sequence_only_analyzes_pitch_segments(self):
        samples = []
        for index in range(6):
            samples.append((index * 0.5, make_pitch_frame()))
        for index in range(4):
            samples.append(((index + 6) * 0.5, make_non_pitch_frame()))
        result = analyze_camera_sequence(samples, source_duration_sec=5.0)

        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].status, "GEOMETRY_CANDIDATE")
        self.assertEqual(result.segments[1].status, "EXCLUDED")


if __name__ == "__main__":
    unittest.main()
