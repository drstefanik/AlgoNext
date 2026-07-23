import unittest

from app.benchmark.schema import (
    BoundingBox,
    SchemaValidationError,
    SequenceAnnotation,
    SequencePrediction,
)
from app.benchmark.tracking_metrics import (
    GateThresholds,
    bbox_iou,
    evaluate_dataset,
    evaluate_quality_gate,
    evaluate_sequence,
)


def annotation(video_id, frames):
    return SequenceAnnotation.from_payload(
        {
            "schema_version": "tracking-annotation-v1",
            "video_id": video_id,
            "fps": 5,
            "frames": frames,
        }
    )


def prediction(video_id, frames):
    return SequencePrediction.from_payload(
        {
            "schema_version": "tracking-prediction-v1",
            "video_id": video_id,
            "frames": frames,
        }
    )


def gt(identity, x, *, ignore=False):
    return {
        "identity": identity,
        "ignore": ignore,
        "bbox": {"x": x, "y": 0.1, "w": 0.1, "h": 0.2},
    }


def pred(track_id, x, confidence=0.9):
    return {
        "track_id": track_id,
        "confidence": confidence,
        "bbox": {"x": x, "y": 0.1, "w": 0.1, "h": 0.2},
    }


class TrackingBenchmarkTests(unittest.TestCase):
    def test_bbox_iou(self):
        first = BoundingBox(0.0, 0.0, 0.2, 0.2)
        second = BoundingBox(0.1, 0.0, 0.2, 0.2)

        self.assertAlmostEqual(bbox_iou(first, second), 1 / 3)

    def test_perfect_sequence(self):
        annotations = annotation(
            "perfect",
            [
                {
                    "frame_index": index,
                    "objects": [gt("player-8", 0.1 + index * 0.01)],
                }
                for index in range(4)
            ],
        )
        predictions = prediction(
            "perfect",
            [
                {
                    "frame_index": index,
                    "tracks": [pred("track-a", 0.1 + index * 0.01)],
                }
                for index in range(4)
            ],
        )

        report = evaluate_sequence(annotations, predictions)

        self.assertEqual(report["counts"]["true_positives"], 4)
        self.assertEqual(report["metrics"]["detection_f1"], 1.0)
        self.assertEqual(report["metrics"]["idf1"], 1.0)
        self.assertEqual(report["counts"]["id_switches"], 0)
        self.assertEqual(report["metrics"]["track_coverage"], 1.0)
        self.assertEqual(report["metrics"]["mostly_tracked_ratio"], 1.0)

    def test_identity_switch_reduces_idf1_even_with_perfect_detection(self):
        annotations = annotation(
            "switch",
            [
                {"frame_index": index, "objects": [gt("player-8", 0.2)]}
                for index in range(4)
            ],
        )
        predictions = prediction(
            "switch",
            [
                {
                    "frame_index": index,
                    "tracks": [
                        pred("track-a" if index < 2 else "track-b", 0.2)
                    ],
                }
                for index in range(4)
            ],
        )

        report = evaluate_sequence(annotations, predictions)

        self.assertEqual(report["metrics"]["detection_f1"], 1.0)
        self.assertEqual(report["metrics"]["idf1"], 0.5)
        self.assertAlmostEqual(
            report["metrics"]["association_accuracy"], 0.5, places=6
        )
        self.assertAlmostEqual(
            report["metrics"]["hota_style_at_threshold"], 2**-0.5, places=6
        )
        self.assertEqual(report["counts"]["id_switches"], 1)
        self.assertEqual(report["metrics"]["id_switches_per_100_matches"], 25.0)

    def test_missing_detection_and_false_positive(self):
        annotations = annotation(
            "errors",
            [
                {"frame_index": 0, "objects": [gt("player-8", 0.2)]},
                {"frame_index": 1, "objects": [gt("player-8", 0.2)]},
            ],
        )
        predictions = prediction(
            "errors",
            [
                {
                    "frame_index": 0,
                    "tracks": [pred("track-a", 0.2), pred("ghost", 0.7)],
                },
                {"frame_index": 1, "tracks": []},
            ],
        )

        report = evaluate_sequence(annotations, predictions)

        self.assertEqual(report["counts"]["true_positives"], 1)
        self.assertEqual(report["counts"]["false_positives"], 1)
        self.assertEqual(report["counts"]["false_negatives"], 1)
        self.assertEqual(report["metrics"]["detection_precision"], 0.5)
        self.assertEqual(report["metrics"]["detection_recall"], 0.5)
        self.assertEqual(report["metrics"]["detection_f1"], 0.5)

    def test_prediction_on_ignore_region_does_not_count_as_false_positive(self):
        annotations = annotation(
            "ignore",
            [
                {
                    "frame_index": 0,
                    "objects": [
                        gt("player-8", 0.2),
                        gt("ignore-referee", 0.7, ignore=True),
                    ],
                }
            ],
        )
        predictions = prediction(
            "ignore",
            [
                {
                    "frame_index": 0,
                    "tracks": [
                        pred("track-a", 0.2),
                        pred("referee-track", 0.7),
                    ],
                }
            ],
        )

        report = evaluate_sequence(annotations, predictions)

        self.assertEqual(report["counts"]["true_positives"], 1)
        self.assertEqual(report["counts"]["ignored_predictions"], 1)
        self.assertEqual(report["counts"]["false_positives"], 0)

    def test_fragmentation_is_distinct_from_identity_switch(self):
        annotations = annotation(
            "fragment",
            [
                {"frame_index": index, "objects": [gt("player-8", 0.2)]}
                for index in range(4)
            ],
        )
        predictions = prediction(
            "fragment",
            [
                {"frame_index": 0, "tracks": [pred("track-a", 0.2)]},
                {"frame_index": 1, "tracks": []},
                {"frame_index": 2, "tracks": []},
                {"frame_index": 3, "tracks": [pred("track-a", 0.2)]},
            ],
        )

        report = evaluate_sequence(annotations, predictions)
        identity = report["per_identity"][0]

        self.assertEqual(identity["fragmentations"], 1)
        self.assertEqual(identity["longest_gap_frames"], 2)
        self.assertEqual(identity["id_switches"], 0)

    def test_prediction_frames_outside_annotation_set_are_unscored(self):
        annotations = annotation(
            "sparse",
            [{"frame_index": 10, "objects": [gt("player-8", 0.2)]}],
        )
        predictions = prediction(
            "sparse",
            [
                {"frame_index": 9, "tracks": [pred("outside", 0.2)]},
                {"frame_index": 10, "tracks": [pred("track-a", 0.2)]},
            ],
        )

        report = evaluate_sequence(annotations, predictions)

        self.assertEqual(report["counts"]["prediction_frames_unscored"], 1)
        self.assertEqual(report["counts"]["false_positives"], 0)

    def test_dataset_aggregate_and_gate(self):
        pairs = []
        for video_id in ("a", "b"):
            annotations = annotation(
                video_id,
                [
                    {"frame_index": index, "objects": [gt("player-8", 0.2)]}
                    for index in range(3)
                ],
            )
            predictions = prediction(
                video_id,
                [
                    {"frame_index": index, "tracks": [pred("track-a", 0.2)]}
                    for index in range(3)
                ],
            )
            pairs.append((annotations, predictions))

        report = evaluate_dataset(pairs)
        gate = evaluate_quality_gate(report)

        self.assertEqual(report["sequence_count"], 2)
        self.assertEqual(report["aggregate"]["counts"]["true_positives"], 6)
        self.assertTrue(gate["passed"])

    def test_gate_fails_identity_switching(self):
        annotations = annotation(
            "gate-fail",
            [
                {"frame_index": index, "objects": [gt("player-8", 0.2)]}
                for index in range(4)
            ],
        )
        predictions = prediction(
            "gate-fail",
            [
                {
                    "frame_index": index,
                    "tracks": [pred(f"track-{index}", 0.2)],
                }
                for index in range(4)
            ],
        )
        report = evaluate_dataset([(annotations, predictions)])
        gate = evaluate_quality_gate(
            report,
            GateThresholds(
                detection_f1_min=0.9,
                idf1_min=0.8,
                track_coverage_min=0.9,
                id_switches_per_100_max=5.0,
                hota_style_min=0.8,
            ),
        )

        self.assertFalse(gate["passed"])
        failed_metrics = {
            check["metric"] for check in gate["checks"] if not check["passed"]
        }
        self.assertIn("idf1", failed_metrics)
        self.assertIn("id_switches_per_100_matches", failed_metrics)

    def test_invalid_duplicate_identity_in_frame(self):
        with self.assertRaises(SchemaValidationError):
            annotation(
                "bad",
                [
                    {
                        "frame_index": 0,
                        "objects": [
                            gt("player-8", 0.2),
                            gt("player-8", 0.5),
                        ],
                    }
                ],
            )

    def test_invalid_bbox_outside_normalized_frame(self):
        with self.assertRaises(SchemaValidationError):
            annotation(
                "bad-box",
                [
                    {
                        "frame_index": 0,
                        "objects": [
                            {
                                "identity": "player-8",
                                "bbox": {
                                    "x": 0.95,
                                    "y": 0.1,
                                    "w": 0.2,
                                    "h": 0.2,
                                },
                            }
                        ],
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
