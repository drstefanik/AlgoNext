import unittest

from app.vision.shot_segmentation import (
    ShotSegmentationThresholds,
    analyze_frame_sequence,
    frame_distance,
)
from tests.synthetic_pitch import make_non_pitch_frame, make_pitch_frame


class ShotSegmentationTests(unittest.TestCase):
    def test_brightness_drift_does_not_create_a_cut(self):
        frames = [
            (index * 0.5, make_pitch_frame(brightness=1.0 - index * 0.025))
            for index in range(8)
        ]
        result = analyze_frame_sequence(frames)

        self.assertEqual(result.boundaries, ())
        self.assertEqual(len(result.shots), 1)
        self.assertEqual(result.shots[0].classification, "PITCH_CANDIDATE")

    def test_two_hard_cuts_create_three_contiguous_shots(self):
        samples = []
        for index in range(4):
            samples.append((index * 0.5, make_pitch_frame(brightness=1.0 - index * 0.02)))
        for index in range(4):
            samples.append(((index + 4) * 0.5, make_non_pitch_frame()))
        for index in range(4):
            samples.append(((index + 8) * 0.5, make_pitch_frame(brightness=0.95)))

        result = analyze_frame_sequence(samples)

        self.assertEqual(len(result.boundaries), 2)
        self.assertEqual(len(result.shots), 3)
        self.assertEqual(
            [shot.classification for shot in result.shots],
            ["PITCH_CANDIDATE", "NON_PITCH", "PITCH_CANDIDATE"],
        )
        self.assertAlmostEqual(result.shots[0].end_sec, result.shots[1].start_sec)
        self.assertAlmostEqual(result.shots[1].end_sec, result.shots[2].start_sec)
        self.assertTrue(result.shots[1].exclude_from_calibration)

    def test_frame_distance_separates_cut_from_brightness_change(self):
        pitch = make_pitch_frame()
        cut_distance = frame_distance(pitch, make_non_pitch_frame())
        drift_distance = frame_distance(pitch, make_pitch_frame(brightness=0.90))

        self.assertGreater(cut_distance, 0.5)
        self.assertLess(drift_distance, 0.15)
        self.assertGreater(cut_distance, drift_distance * 4.0)

    def test_duplicate_timestamps_are_rejected(self):
        frame = make_pitch_frame()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            analyze_frame_sequence([(0.0, frame), (0.0, frame)])

    def test_thresholds_require_calibration_duration_not_below_shot_duration(self):
        with self.assertRaisesRegex(ValueError, "calibration_minimum_duration_sec"):
            ShotSegmentationThresholds(
                minimum_shot_duration_sec=3.0,
                calibration_minimum_duration_sec=2.0,
            )


if __name__ == "__main__":
    unittest.main()
