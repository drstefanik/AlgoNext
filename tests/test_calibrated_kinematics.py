import unittest

import numpy as np

from app.calibration.homography import PitchCalibration, fit_pitch_calibration
from app.calibration.kinematics import (
    MotionThresholds,
    calculate_calibrated_motion,
    collect_tracking_bboxes,
    project_tracking_footpoints,
)
from app.calibration.schema import CalibrationRequest


def exact_calibration(
    *,
    segment_id="camera-1",
    start_sec=0.0,
    end_sec=30.0,
):
    image = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    field = np.array(
        [
            [0.0, 0.0],
            [105.0, 0.0],
            [0.0, 68.0],
            [105.0, 68.0],
            [52.5, 0.0],
            [52.5, 68.0],
            [0.0, 34.0],
            [105.0, 34.0],
            [52.5, 34.0],
        ],
        dtype=np.float64,
    )
    request = CalibrationRequest.from_payload(
        {
            "schema_version": "pitch-calibration-request-v1",
            "camera_segment_id": segment_id,
            "source": "unit-test",
            "start_sec": start_sec,
            "end_sec": end_sec,
            "correspondences": [
                {
                    "image": {"x": float(image_point[0]), "y": float(image_point[1])},
                    "field": {
                        "x_m": float(field_point[0]),
                        "y_m": float(field_point[1]),
                    },
                }
                for image_point, field_point in zip(image, field)
            ],
        }
    )
    calibration = fit_pitch_calibration(request)
    assert calibration.validated
    return calibration


def bbox_for_field_point(time_sec, x_m, y_m=34.0, confidence=0.9):
    width = 0.02
    height = 0.10
    foot_x = x_m / 105.0
    foot_y = y_m / 68.0
    return {
        "t": float(time_sec),
        "x": foot_x - width / 2.0,
        "y": foot_y - height,
        "w": width,
        "h": height,
        "conf": confidence,
    }


class CalibratedKinematicsTests(unittest.TestCase):
    def test_constant_one_metre_per_second_path(self):
        calibration = exact_calibration()
        bboxes = [
            bbox_for_field_point(index * 0.2, index * 0.2)
            for index in range(26)
        ]
        tracking = {"bboxes": bboxes}
        result = calculate_calibrated_motion(
            tracking,
            [calibration],
            thresholds=MotionThresholds(
                smoothing_window=1,
                maximum_gap_sec=0.3,
                minimum_projected_points=10,
            ),
        )

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertTrue(result["pitch_calibration_validated"])
        self.assertFalse(result["athletic_metric_validated"])
        self.assertAlmostEqual(result["observed_path_length_m"], 5.0, places=2)
        self.assertAlmostEqual(result["average_observed_speed_kmh"], 3.6, places=2)
        self.assertAlmostEqual(result["p95_observed_speed_kmh"], 3.6, places=2)
        self.assertEqual(result["sprint_bouts_proxy"], 0)

    def test_unvalidated_calibration_makes_metrics_unavailable(self):
        calibration = exact_calibration()
        payload = calibration.to_payload()
        payload["status"] = "REJECTED"
        payload["validated"] = False
        payload["reason_codes"] = ["TEST_REJECTION"]
        rejected = PitchCalibration.from_payload(payload)

        result = calculate_calibrated_motion(
            {"bboxes": [bbox_for_field_point(0.0, 1.0)]},
            [rejected],
        )
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertIn(
            "NO_VALIDATED_PITCH_CALIBRATION",
            result["reason_codes"],
        )

    def test_camera_segments_are_selected_by_time(self):
        first = exact_calibration(
            segment_id="camera-a", start_sec=0.0, end_sec=2.0
        )
        second = exact_calibration(
            segment_id="camera-b", start_sec=2.0, end_sec=5.0
        )
        bboxes = [
            bbox_for_field_point(1.0, 10.0),
            bbox_for_field_point(2.5, 20.0),
        ]
        points, counters = project_tracking_footpoints(
            bboxes,
            [first, second],
            thresholds=MotionThresholds(minimum_projected_points=2),
        )
        self.assertEqual(counters["projected_points"], 2)
        self.assertEqual(
            [point.calibration_id for point in points],
            ["camera-a", "camera-b"],
        )

    def test_missing_camera_segment_does_not_reuse_wrong_homography(self):
        calibration = exact_calibration(
            segment_id="camera-a", start_sec=0.0, end_sec=2.0
        )
        points, counters = project_tracking_footpoints(
            [
                bbox_for_field_point(1.0, 10.0),
                bbox_for_field_point(3.0, 20.0),
            ],
            [calibration],
            thresholds=MotionThresholds(minimum_projected_points=2),
        )
        self.assertEqual(len(points), 1)
        self.assertEqual(counters["missing_calibration"], 1)

    def test_implausible_jump_is_removed(self):
        calibration = exact_calibration()
        bboxes = [
            bbox_for_field_point(0.0, 0.0),
            bbox_for_field_point(0.2, 0.2),
            bbox_for_field_point(0.4, 80.0),
            bbox_for_field_point(0.6, 0.6),
            bbox_for_field_point(0.8, 0.8),
        ]
        result = calculate_calibrated_motion(
            {"bboxes": bboxes},
            [calibration],
            thresholds=MotionThresholds(
                smoothing_window=1,
                maximum_gap_sec=0.3,
                minimum_projected_points=2,
            ),
        )
        self.assertGreaterEqual(
            result["quality"]["rejected_speed_outliers"],
            2,
        )
        self.assertLess(result["observed_path_length_m"], 2.0)

    def test_sprint_requires_duration_not_single_frame_spike(self):
        calibration = exact_calibration()
        bboxes = []
        for index in range(16):
            time_sec = index * 0.1
            x_m = index * 0.8
            bboxes.append(bbox_for_field_point(time_sec, x_m))
        result = calculate_calibrated_motion(
            {"bboxes": bboxes},
            [calibration],
            thresholds=MotionThresholds(
                smoothing_window=1,
                maximum_gap_sec=0.2,
                sprint_threshold_mps=7.0,
                minimum_sprint_duration_sec=1.0,
                minimum_projected_points=10,
            ),
        )
        self.assertEqual(result["sprint_bouts_proxy"], 1)
        self.assertGreaterEqual(result["sprint_duration_sec_proxy"], 1.0)

    def test_abstained_reid_segments_are_not_used(self):
        tracking = {
            "segments": [
                {
                    "identity_status": "ACCEPTED",
                    "bboxes": [bbox_for_field_point(0.0, 1.0)],
                },
                {
                    "identity_status": "ABSTAINED",
                    "bboxes": [bbox_for_field_point(0.2, 50.0)],
                },
            ]
        }
        bboxes = collect_tracking_bboxes(tracking)
        self.assertEqual(len(bboxes), 1)
        self.assertAlmostEqual(float(bboxes[0]["t"]), 0.0)

    def test_duplicate_overlap_sample_keeps_higher_confidence(self):
        calibration = exact_calibration()
        points, counters = project_tracking_footpoints(
            [
                bbox_for_field_point(1.0, 10.0, confidence=0.2),
                bbox_for_field_point(1.0001, 20.0, confidence=0.9),
            ],
            [calibration],
            thresholds=MotionThresholds(minimum_projected_points=2),
        )
        self.assertEqual(counters["projected_points"], 1)
        self.assertAlmostEqual(points[0].x_m, 20.0, places=4)


if __name__ == "__main__":
    unittest.main()
