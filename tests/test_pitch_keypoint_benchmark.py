import copy
import unittest

from app.benchmark.pitch_keypoints import (
    AnnotationDataset,
    KeypointSchemaError,
    PredictionDataset,
    PredictionFrame,
    PredictedKeypoint,
    build_calibration_request,
    evaluate_keypoint_dataset,
    evaluate_quality_gate,
    load_annotations,
    load_json,
    load_predictions,
)
from app.calibration.schema import CalibrationRequest

ANNOTATIONS = "tests/fixtures/pitch_keypoints/annotations/demo.json"
PREDICTIONS = "tests/fixtures/pitch_keypoints/predictions/demo.json"


def payloads():
    return copy.deepcopy(load_json(ANNOTATIONS)), copy.deepcopy(load_json(PREDICTIONS))


class PitchKeypointBenchmarkTests(unittest.TestCase):
    def test_perfect_fixture_passes_metric_and_calibration_gate(self):
        report = evaluate_keypoint_dataset(
            load_annotations(ANNOTATIONS),
            load_predictions(PREDICTIONS),
        )
        gate = evaluate_quality_gate(report)

        self.assertEqual(report["metrics"]["semantic_f1"], 1.0)
        self.assertEqual(report["metrics"]["pck_02"], 1.0)
        self.assertEqual(report["metrics"]["non_pitch_false_positive_rate"], 0.0)
        self.assertEqual(report["metrics"]["calibration_validated_rate"], 1.0)
        self.assertTrue(report["per_frame"][0]["calibration"]["validated"])
        self.assertTrue(gate["passed"])

    def test_multiple_semantic_swaps_fail_scoring_and_calibration(self):
        annotations_payload, predictions_payload = payloads()
        points = predictions_payload["frames"][0]["keypoints"]
        by_label = {point["landmark"]: point for point in points}
        swaps = [
            ("corner_left_top", "corner_right_bottom"),
            ("corner_left_bottom", "corner_right_top"),
        ]
        for first, second in swaps:
            first_xy = (by_label[first]["x"], by_label[first]["y"])
            second_xy = (by_label[second]["x"], by_label[second]["y"])
            by_label[first]["x"], by_label[first]["y"] = second_xy
            by_label[second]["x"], by_label[second]["y"] = first_xy

        report = evaluate_keypoint_dataset(
            AnnotationDataset.from_payload(annotations_payload),
            PredictionDataset.from_payload(predictions_payload),
        )
        gate = evaluate_quality_gate(report)

        self.assertLess(report["metrics"]["semantic_f1"], 0.75)
        self.assertFalse(report["per_frame"][0]["calibration"]["validated"])
        self.assertFalse(gate["passed"])

    def test_low_confidence_points_are_not_counted(self):
        annotations_payload, predictions_payload = payloads()
        for point in predictions_payload["frames"][0]["keypoints"]:
            point["confidence"] = 0.1
        report = evaluate_keypoint_dataset(
            AnnotationDataset.from_payload(annotations_payload),
            PredictionDataset.from_payload(predictions_payload),
        )

        self.assertEqual(report["metrics"]["semantic_recall"], 0.0)
        self.assertEqual(report["metrics"]["pitch_frame_prediction_coverage"], 0.0)
        self.assertEqual(report["counts"]["calibration_attempted_frames"], 0)

    def test_non_pitch_prediction_is_measured_as_false_positive(self):
        annotations_payload, predictions_payload = payloads()
        negative = predictions_payload["frames"][1]
        negative["abstained"] = False
        negative["reason_codes"] = []
        negative["keypoints"] = [
            {
                "landmark": "centre_spot",
                "x": 0.5,
                "y": 0.5,
                "confidence": 0.99,
            }
        ]
        report = evaluate_keypoint_dataset(
            AnnotationDataset.from_payload(annotations_payload),
            PredictionDataset.from_payload(predictions_payload),
        )

        self.assertEqual(report["metrics"]["non_pitch_false_positive_rate"], 1.0)
        self.assertFalse(evaluate_quality_gate(report)["passed"])

    def test_missing_prediction_frame_is_treated_as_abstention(self):
        annotations_payload, predictions_payload = payloads()
        predictions_payload["frames"] = predictions_payload["frames"][1:]
        report = evaluate_keypoint_dataset(
            AnnotationDataset.from_payload(annotations_payload),
            PredictionDataset.from_payload(predictions_payload),
        )

        self.assertEqual(report["metrics"]["semantic_recall"], 0.0)
        self.assertEqual(report["counts"]["abstained_frames"], 2)

    def test_calibration_request_is_strict_and_auditable(self):
        prediction = load_predictions(PREDICTIONS).frames[0]
        request = build_calibration_request(prediction)

        self.assertIsNotNone(request)
        self.assertEqual(request["source"], "semantic_keypoint_model")
        self.assertGreaterEqual(len(request["correspondences"]), 6)
        self.assertNotIn("provenance", request)
        CalibrationRequest.from_payload(request)

    def test_calibration_request_abstains_for_clustered_points(self):
        labels = [
            "corner_left_top",
            "corner_left_bottom",
            "corner_right_top",
            "corner_right_bottom",
            "halfway_top",
            "halfway_bottom",
        ]
        prediction = PredictionFrame(
            frame_id="clustered",
            video_id="video",
            shot_id="shot",
            time_sec=1.0,
            abstained=False,
            keypoints=tuple(
                PredictedKeypoint(label, 0.50 + index * 0.001, 0.50, 0.99)
                for index, label in enumerate(labels)
            ),
            model_version="model-v1",
            configuration_hash="sha256:clustered",
        )

        self.assertIsNone(build_calibration_request(prediction))

    def test_duplicate_landmark_is_rejected(self):
        _, predictions_payload = payloads()
        predictions_payload["frames"][0]["keypoints"].append(
            copy.deepcopy(predictions_payload["frames"][0]["keypoints"][0])
        )
        with self.assertRaisesRegex(KeypointSchemaError, "appears more than once"):
            PredictionDataset.from_payload(predictions_payload)

    def test_unknown_fields_and_wrong_schema_are_rejected(self):
        annotations_payload, predictions_payload = payloads()
        annotations_payload["unexpected"] = True
        with self.assertRaisesRegex(KeypointSchemaError, "unknown fields"):
            AnnotationDataset.from_payload(annotations_payload)
        predictions_payload["schema_version"] = "pitch-keypoint-prediction-v0"
        with self.assertRaisesRegex(KeypointSchemaError, "expected"):
            PredictionDataset.from_payload(predictions_payload)


if __name__ == "__main__":
    unittest.main()
