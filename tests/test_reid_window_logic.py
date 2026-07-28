import unittest

from app.reid.window_logic import (
    autonomous_tracking_evidence,
    choose_descriptor_detections,
    geometry_similarity,
    largest_tracking_gap_sec,
    processing_order,
    temporal_overlap_score,
    tracking_coverage_pct,
)


class ReIDWindowLogicTests(unittest.TestCase):
    def test_processing_starts_at_anchor_and_expands_both_directions(self):
        windows = [(0, 45), (35, 80), (70, 115), (105, 150)]
        anchor_index, forward, backward = processing_order(windows, 78)
        self.assertEqual(anchor_index, 2)
        self.assertEqual(forward, (3,))
        self.assertEqual(backward, (1, 0))

    def test_descriptor_sampling_spreads_time_and_keeps_best_crop(self):
        detections = [
            {
                "t": float(index),
                "sample_index": index,
                "conf": 0.5,
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.02, "h": 0.04},
            }
            for index in range(10)
        ]
        detections[6] = {
            **detections[6],
            "conf": 0.99,
            "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.4},
        }
        selected = choose_descriptor_detections(detections, 4)
        selected_indices = [item["sample_index"] for item in selected]
        self.assertEqual(len(selected), 4)
        self.assertIn(6, selected_indices)
        self.assertLessEqual(min(selected_indices), 1)
        self.assertGreaterEqual(max(selected_indices), 8)

    def test_temporal_overlap_distinguishes_consistent_candidate(self):
        previous = [
            {"t": 35.0, "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2},
            {"t": 36.0, "x": 0.21, "y": 0.2, "w": 0.1, "h": 0.2},
        ]
        matching = [
            {"t": 0.0, "bbox": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2}},
            {"t": 1.0, "bbox": {"x": 0.21, "y": 0.2, "w": 0.1, "h": 0.2}},
        ]
        wrong = [
            {"t": 0.0, "bbox": {"x": 0.75, "y": 0.2, "w": 0.1, "h": 0.2}},
            {"t": 1.0, "bbox": {"x": 0.75, "y": 0.2, "w": 0.1, "h": 0.2}},
        ]
        matching_score = temporal_overlap_score(
            previous,
            matching,
            time_offset=35.0,
            tolerance_sec=0.2,
        )
        wrong_score = temporal_overlap_score(
            previous,
            wrong,
            time_offset=35.0,
            tolerance_sec=0.2,
        )
        self.assertIsNotNone(matching_score)
        self.assertGreater(matching_score, 0.8)
        self.assertEqual(wrong_score, 0.0)

    def test_geometry_similarity_is_bounded_and_prefers_close_box(self):
        reference = {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2}
        close = {"x": 0.21, "y": 0.2, "w": 0.1, "h": 0.2}
        far = {"x": 0.8, "y": 0.7, "w": 0.05, "h": 0.1}
        self.assertGreater(geometry_similarity(reference, close), 0.8)
        self.assertLess(geometry_similarity(reference, far), 0.4)

    def test_tracking_coverage_deduplicates_overlap(self):
        segments = [
            {"bboxes": [{"t": 1.0}, {"t": 1.2}, {"t": 1.4}]},
            {"bboxes": [{"t": 1.2}, {"t": 1.4}, {"t": 1.6}]},
        ]
        self.assertAlmostEqual(
            tracking_coverage_pct(segments, duration_sec=2.0, fps=5),
            40.0,
        )

    def test_largest_gap_uses_actual_observations(self):
        segments = [{"bboxes": [{"t": 1.0}, {"t": 2.0}, {"t": 8.0}]}]
        self.assertEqual(
            largest_tracking_gap_sec(segments, duration_sec=10.0),
            6.0,
        )

    def test_guarded_autonomous_chain_stops_at_abstention(self):
        accepted = {
            "identity_status": "ACCEPTED",
            "reid": {"status": "ACCEPTED"},
        }
        segments = [
            {
                **accepted,
                "direction": "anchor",
                "window_start": 0.0,
                "window_end": 2.0,
                "bboxes": [{"t": 1.0}],
            },
            {
                "direction": "forward",
                "window_start": 1.5,
                "window_end": 3.5,
                "identity_status": "ABSTAINED",
                "reid": {"status": "ABSTAINED"},
                "bboxes": [],
            },
            {
                **accepted,
                "direction": "forward",
                "window_start": 3.0,
                "window_end": 5.0,
                "bboxes": [{"t": 4.0}, {"t": 4.5}],
            },
        ]

        evidence = autonomous_tracking_evidence(
            segments,
            fps=4.0,
            require_retained_chain=True,
        )

        self.assertFalse(evidence["proven"])
        self.assertEqual(evidence["segments_with_player"], 0)
        self.assertEqual(evidence["bboxes_count"], 0)
        self.assertEqual(evidence["segment_counts"], {})

    def test_later_manual_anchor_reseeds_guarded_autonomous_chain(self):
        accepted = {
            "identity_status": "ACCEPTED",
            "reid": {"status": "ACCEPTED"},
        }
        segments = [
            {
                **accepted,
                "direction": "anchor",
                "window_start": 0.0,
                "window_end": 2.0,
                "bboxes": [{"t": 1.0}],
            },
            {
                "direction": "forward",
                "window_start": 1.5,
                "window_end": 3.5,
                "identity_status": "ABSTAINED",
                "reid": {"status": "ABSTAINED"},
                "bboxes": [],
            },
            {
                **accepted,
                "direction": "anchor",
                "processing_direction": "forward",
                "window_start": 3.0,
                "window_end": 5.0,
                "bboxes": [{"t": 4.0}],
            },
            {
                **accepted,
                "direction": "forward",
                "window_start": 4.5,
                "window_end": 6.5,
                "bboxes": [{"t": 5.5}, {"t": 6.0}],
            },
        ]

        evidence = autonomous_tracking_evidence(
            segments,
            fps=4.0,
            require_retained_chain=True,
        )

        self.assertTrue(evidence["proven"])
        self.assertEqual(evidence["segments_with_player"], 1)
        self.assertEqual(evidence["bboxes_count"], 2)
        self.assertEqual(evidence["segment_counts"], {3: 2})


if __name__ == "__main__":
    unittest.main()
