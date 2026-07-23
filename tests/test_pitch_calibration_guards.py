import unittest
from unittest.mock import patch

import numpy as np

from app.calibration.homography import (
    CalibrationFitError,
    CalibrationThresholds,
    PitchCalibration,
    _minimum_denominator,
    fit_pitch_calibration,
)
from app.calibration.schema import CalibrationRequest


def request_with_six_points():
    return CalibrationRequest.from_payload(
        {
            "schema_version": "pitch-calibration-request-v1",
            "camera_segment_id": "guard-test",
            "correspondences": [
                {
                    "image": {"x": 0.0, "y": 0.0},
                    "field": {"x_m": 0.0, "y_m": 0.0},
                },
                {
                    "image": {"x": 1.0, "y": 0.0},
                    "field": {"x_m": 105.0, "y_m": 0.0},
                },
                {
                    "image": {"x": 0.0, "y": 1.0},
                    "field": {"x_m": 0.0, "y_m": 68.0},
                },
                {
                    "image": {"x": 1.0, "y": 1.0},
                    "field": {"x_m": 105.0, "y_m": 68.0},
                },
                {
                    "image": {"x": 0.5, "y": 0.0},
                    "field": {"x_m": 52.5, "y_m": 0.0},
                },
                {
                    "image": {"x": 0.5, "y": 1.0},
                    "field": {"x_m": 52.5, "y_m": 68.0},
                },
            ],
        }
    )


class PitchCalibrationGuardTests(unittest.TestCase):
    def test_fewer_than_four_ransac_inliers_is_a_fit_error(self):
        matrix = np.eye(3, dtype=np.float64)
        mask = np.array([[1], [1], [1], [0], [0], [0]], dtype=np.uint8)
        with patch(
            "app.calibration.homography.cv2.findHomography",
            return_value=(matrix, mask),
        ):
            with self.assertRaisesRegex(CalibrationFitError, "fewer than four"):
                fit_pitch_calibration(request_with_six_points())

    def test_invalid_homography_scale_is_a_fit_error(self):
        matrix = np.eye(3, dtype=np.float64)
        matrix[2, 2] = 0.0
        mask = np.ones((6, 1), dtype=np.uint8)
        with patch(
            "app.calibration.homography.cv2.findHomography",
            return_value=(matrix, mask),
        ):
            with self.assertRaisesRegex(CalibrationFitError, "invalid scale"):
                fit_pitch_calibration(request_with_six_points())

    def test_non_finite_projection_is_a_fit_error(self):
        matrix = np.eye(3, dtype=np.float64)
        mask = np.ones((6, 1), dtype=np.uint8)
        projection = np.full((6, 2), np.nan, dtype=np.float64)
        with patch(
            "app.calibration.homography.cv2.findHomography",
            return_value=(matrix, mask),
        ), patch(
            "app.calibration.homography._project",
            return_value=projection,
        ):
            with self.assertRaisesRegex(
                CalibrationFitError,
                "non-finite projected",
            ):
                fit_pitch_calibration(request_with_six_points())

    def test_projective_horizon_crossing_is_detected_between_samples(self):
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-5.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.assertEqual(_minimum_denominator(matrix), 0.0)

    def test_result_payload_rejects_non_finite_quality_values(self):
        calibration = fit_pitch_calibration(request_with_six_points())
        payload = calibration.to_payload()
        payload["quality"]["condition_number"] = float("inf")
        with self.assertRaisesRegex(ValueError, "condition_number must be finite"):
            PitchCalibration.from_payload(payload)

    def test_rejected_result_requires_reason_codes(self):
        calibration = fit_pitch_calibration(request_with_six_points())
        payload = calibration.to_payload()
        payload["status"] = "REJECTED"
        payload["validated"] = False
        payload["reason_codes"] = []
        with self.assertRaisesRegex(ValueError, "must contain reason codes"):
            PitchCalibration.from_payload(payload)

    def test_result_parser_rejects_wrong_schema_version(self):
        calibration = fit_pitch_calibration(request_with_six_points())
        payload = calibration.to_payload()
        payload["schema_version"] = "pitch-calibration-result-v0"
        with self.assertRaisesRegex(ValueError, "schema_version must equal"):
            PitchCalibration.from_payload(payload)

    def test_result_parser_requires_boolean_validated_flag(self):
        calibration = fit_pitch_calibration(request_with_six_points())
        payload = calibration.to_payload()
        payload["validated"] = "true"
        with self.assertRaisesRegex(ValueError, "validated must be a boolean"):
            PitchCalibration.from_payload(payload)

    def test_result_parser_requires_boolean_inlier_mask(self):
        calibration = fit_pitch_calibration(request_with_six_points())
        payload = calibration.to_payload()
        payload["quality"]["inlier_mask"][0] = 1
        with self.assertRaisesRegex(ValueError, "array of booleans"):
            PitchCalibration.from_payload(payload)

    def test_thresholds_reject_nan(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            CalibrationThresholds(maximum_rmse_m=float("nan"))


if __name__ == "__main__":
    unittest.main()
