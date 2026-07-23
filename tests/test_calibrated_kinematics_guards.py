import unittest

from app.calibration.homography import CalibrationThresholds, PitchCalibration
from app.calibration.kinematics import (
    CalibratedTrackPoint,
    MotionThresholds,
    _smooth_points,
    calculate_calibrated_motion,
    project_tracking_footpoints,
)
from app.calibration.model import PitchDimensions


def calibration(segment_id, start_sec, end_sec):
    return PitchCalibration(
        camera_segment_id=segment_id,
        status="VALIDATED",
        validated=True,
        matrix_image_to_field=(
            (105.0, 0.0, 0.0),
            (0.0, 68.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        matrix_field_to_image=(
            (1.0 / 105.0, 0.0, 0.0),
            (0.0, 1.0 / 68.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        pitch=PitchDimensions(),
        source="unit-test",
        start_sec=start_sec,
        end_sec=end_sec,
        total_correspondences=6,
        inlier_count=6,
        inlier_mask=(True, True, True, True, True, True),
        inlier_ratio=1.0,
        rmse_m=0.0,
        median_error_m=0.0,
        p95_error_m=0.0,
        maximum_error_m=0.0,
        image_hull_area_ratio=1.0,
        field_hull_area_ratio=1.0,
        condition_number=1.0,
        minimum_projective_denominator=1.0,
        reason_codes=(),
        thresholds=CalibrationThresholds(),
    )


def bbox(time_sec, x_m):
    width = 0.02
    height = 0.10
    return {
        "t": time_sec,
        "x": x_m / 105.0 - width / 2.0,
        "y": 0.5 - height,
        "w": width,
        "h": height,
        "conf": 0.9,
    }


class CalibratedKinematicsGuardTests(unittest.TestCase):
    def test_camera_change_never_creates_a_motion_transition(self):
        first = calibration("camera-a", 0.0, 1.0)
        second = calibration("camera-b", 1.0, 2.0)
        result = calculate_calibrated_motion(
            {
                "bboxes": [
                    bbox(0.9, 1.0),
                    bbox(1.0, 100.0),
                ]
            },
            [first, second],
            thresholds=MotionThresholds(
                smoothing_window=1,
                maximum_gap_sec=0.5,
                minimum_projected_points=2,
            ),
        )

        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["observed_path_length_m"], 0.0)
        self.assertEqual(result["quality"]["accepted_transitions"], 0)
        self.assertEqual(result["quality"]["rejected_camera_changes"], 1)

    def test_smoothing_never_uses_a_point_beyond_maximum_gap(self):
        points = [
            CalibratedTrackPoint(0.0, 0.0, 0.0, "camera-a"),
            CalibratedTrackPoint(0.1, 0.1, 0.0, "camera-a"),
            CalibratedTrackPoint(1.0, 100.0, 0.0, "camera-a"),
        ]
        smoothed = _smooth_points(
            points,
            window=3,
            maximum_gap_sec=0.2,
        )
        self.assertAlmostEqual(smoothed[2].x_m, 100.0, places=6)

    def test_negative_timestamp_is_rejected_before_projection(self):
        points, counters = project_tracking_footpoints(
            [bbox(-1.0, 10.0)],
            [calibration("camera-a", 0.0, 2.0)],
            thresholds=MotionThresholds(minimum_projected_points=2),
        )
        self.assertEqual(points, [])
        self.assertEqual(counters["invalid_bbox"], 1)

    def test_motion_thresholds_reject_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            MotionThresholds(maximum_gap_sec=float("nan"))

    def test_sprint_threshold_cannot_exceed_speed_filter(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            MotionThresholds(
                maximum_speed_mps=6.0,
                sprint_threshold_mps=7.0,
            )

    def test_calibrated_point_rejects_non_finite_coordinates(self):
        with self.assertRaisesRegex(ValueError, "x_m must be finite"):
            CalibratedTrackPoint(
                time_sec=1.0,
                x_m=float("inf"),
                y_m=10.0,
                calibration_id="camera-a",
            )


if __name__ == "__main__":
    unittest.main()
