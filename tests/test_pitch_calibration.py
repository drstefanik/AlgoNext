import unittest

import cv2
import numpy as np

from app.calibration.homography import (
    CalibrationFitError,
    CalibrationThresholds,
    PitchCalibration,
    fit_pitch_calibration,
)
from app.calibration.schema import (
    CalibrationRequest,
    CalibrationValidationError,
)


def request_from_pairs(
    image_points,
    field_points,
    *,
    segment_id="camera-1",
    start_sec=0.0,
    end_sec=30.0,
):
    return CalibrationRequest.from_payload(
        {
            "schema_version": "pitch-calibration-request-v1",
            "camera_segment_id": segment_id,
            "source": "unit-test",
            "start_sec": start_sec,
            "end_sec": end_sec,
            "pitch": {"length_m": 105.0, "width_m": 68.0},
            "correspondences": [
                {
                    "image": {"x": float(image[0]), "y": float(image[1])},
                    "field": {
                        "x_m": float(field[0]),
                        "y_m": float(field[1]),
                    },
                }
                for image, field in zip(image_points, field_points)
            ],
        }
    )


def full_pitch_pairs():
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
    return image, field


class PitchCalibrationTests(unittest.TestCase):
    def test_exact_full_pitch_calibration_is_validated(self):
        image, field = full_pitch_pairs()
        calibration = fit_pitch_calibration(request_from_pairs(image, field))

        self.assertTrue(calibration.validated)
        self.assertEqual(calibration.status, "VALIDATED")
        self.assertEqual(calibration.reason_codes, ())
        self.assertEqual(calibration.inlier_count, 9)
        self.assertAlmostEqual(calibration.rmse_m, 0.0, places=6)
        self.assertAlmostEqual(calibration.field_hull_area_ratio, 1.0, places=6)

        x_m, y_m = calibration.project_image_point(0.25, 0.75)
        self.assertAlmostEqual(x_m, 26.25, places=5)
        self.assertAlmostEqual(y_m, 51.0, places=5)
        image_x, image_y = calibration.project_field_point(x_m, y_m)
        self.assertAlmostEqual(image_x, 0.25, places=6)
        self.assertAlmostEqual(image_y, 0.75, places=6)

    def test_perspective_trapezoid_round_trip(self):
        field_corners = np.array(
            [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]],
            dtype=np.float32,
        )
        image_corners = np.array(
            [[0.12, 0.18], [0.88, 0.18], [0.98, 0.88], [0.02, 0.88]],
            dtype=np.float32,
        )
        field_to_image = cv2.getPerspectiveTransform(field_corners, image_corners)
        field = np.array(
            [
                [0.0, 0.0],
                [105.0, 0.0],
                [105.0, 68.0],
                [0.0, 68.0],
                [52.5, 0.0],
                [52.5, 68.0],
                [16.5, 13.84],
                [88.5, 54.16],
                [52.5, 34.0],
            ],
            dtype=np.float64,
        )
        image = cv2.perspectiveTransform(
            field.reshape(-1, 1, 2), field_to_image
        ).reshape(-1, 2)
        calibration = fit_pitch_calibration(request_from_pairs(image, field))

        self.assertTrue(calibration.validated)
        projected = calibration.project_image_point(
            float(image[-1][0]), float(image[-1][1])
        )
        self.assertAlmostEqual(projected[0], 52.5, places=4)
        self.assertAlmostEqual(projected[1], 34.0, places=4)

    def test_ransac_rejects_one_outlier(self):
        image, field = full_pitch_pairs()
        image = image.copy()
        image[-1] = [0.9, 0.9]
        calibration = fit_pitch_calibration(request_from_pairs(image, field))

        self.assertTrue(calibration.validated)
        self.assertEqual(calibration.inlier_count, 8)
        self.assertFalse(calibration.inlier_mask[-1])
        self.assertGreater(calibration.inlier_ratio, 0.8)

    def test_four_points_fit_but_fail_validation_gate(self):
        image, field = full_pitch_pairs()
        calibration = fit_pitch_calibration(
            request_from_pairs(image[:4], field[:4])
        )

        self.assertFalse(calibration.validated)
        self.assertIn(
            "INSUFFICIENT_CALIBRATION_POINTS",
            calibration.reason_codes,
        )

    def test_clustered_points_fail_coverage_gate(self):
        image = np.array(
            [
                [0.45, 0.45],
                [0.55, 0.45],
                [0.45, 0.55],
                [0.55, 0.55],
                [0.50, 0.45],
                [0.50, 0.55],
            ],
            dtype=np.float64,
        )
        field = np.array(
            [
                [47.5, 29.0],
                [57.5, 29.0],
                [47.5, 39.0],
                [57.5, 39.0],
                [52.5, 29.0],
                [52.5, 39.0],
            ],
            dtype=np.float64,
        )
        calibration = fit_pitch_calibration(request_from_pairs(image, field))

        self.assertFalse(calibration.validated)
        self.assertIn("LOW_IMAGE_POINT_COVERAGE", calibration.reason_codes)
        self.assertIn("LOW_FIELD_POINT_COVERAGE", calibration.reason_codes)

    def test_collinear_points_cannot_fit(self):
        image = np.array(
            [[0.1, 0.2], [0.2, 0.2], [0.3, 0.2], [0.4, 0.2], [0.5, 0.2], [0.6, 0.2]]
        )
        field = np.array(
            [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0], [40.0, 20.0], [50.0, 20.0], [60.0, 20.0]]
        )
        with self.assertRaisesRegex(CalibrationFitError, "collinear"):
            fit_pitch_calibration(request_from_pairs(image, field))

    def test_landmark_payload_and_serialization_round_trip(self):
        image, _ = full_pitch_pairs()
        landmarks = [
            "corner_left_top",
            "corner_right_top",
            "corner_left_bottom",
            "corner_right_bottom",
            "halfway_top",
            "halfway_bottom",
        ]
        request = CalibrationRequest.from_payload(
            {
                "schema_version": "pitch-calibration-request-v1",
                "camera_segment_id": "camera-landmarks",
                "correspondences": [
                    {
                        "image": {"x": float(point[0]), "y": float(point[1])},
                        "landmark": landmark,
                    }
                    for point, landmark in zip(image[:6], landmarks)
                ],
            }
        )
        calibration = fit_pitch_calibration(request)
        restored = PitchCalibration.from_payload(calibration.to_payload())

        self.assertEqual(restored.status, calibration.status)
        self.assertEqual(restored.camera_segment_id, "camera-landmarks")
        self.assertAlmostEqual(restored.rmse_m, calibration.rmse_m, places=8)

    def test_duplicate_points_are_rejected_by_schema(self):
        with self.assertRaisesRegex(CalibrationValidationError, "duplicate image"):
            CalibrationRequest.from_payload(
                {
                    "schema_version": "pitch-calibration-request-v1",
                    "camera_segment_id": "bad",
                    "correspondences": [
                        {
                            "image": {"x": 0.1, "y": 0.1},
                            "field": {"x_m": 0.0, "y_m": 0.0},
                        },
                        {
                            "image": {"x": 0.1, "y": 0.1},
                            "field": {"x_m": 105.0, "y_m": 0.0},
                        },
                        {
                            "image": {"x": 0.2, "y": 0.8},
                            "field": {"x_m": 0.0, "y_m": 68.0},
                        },
                        {
                            "image": {"x": 0.8, "y": 0.8},
                            "field": {"x_m": 105.0, "y_m": 68.0},
                        },
                    ],
                }
            )

    def test_thresholds_can_be_tightened(self):
        image, field = full_pitch_pairs()
        field = field.copy()
        field[-1] += [0.3, 0.0]
        calibration = fit_pitch_calibration(
            request_from_pairs(image, field),
            thresholds=CalibrationThresholds(maximum_rmse_m=0.01),
        )
        self.assertFalse(calibration.validated)
        self.assertIn("HIGH_CALIBRATION_RMSE", calibration.reason_codes)


if __name__ == "__main__":
    unittest.main()
