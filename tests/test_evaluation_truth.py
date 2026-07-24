import unittest

from app.core.evaluation_truth import (
    apply_evaluation_truth_gate,
    build_tracking_evaluation,
    compute_image_motion_metrics,
)


class EvaluationTruthTests(unittest.TestCase):
    def test_low_coverage_candidate_is_not_inflated_into_player_score(self):
        candidate_metrics = {
            "coveragePct": 0.125,
            "stabilityScore": 0.333,
            "sampleFramesCount": 4,
            "tierLabel": "PRIMARY",
        }

        evaluation = build_tracking_evaluation(candidate_metrics=candidate_metrics)

        self.assertEqual(evaluation["status"], "TRACKING_ONLY")
        self.assertFalse(evaluation["player_evaluation_available"])
        self.assertEqual(evaluation["tracking_confidence"], "low")
        self.assertAlmostEqual(evaluation["tracking_quality_index"], 17.6, places=1)
        self.assertAlmostEqual(evaluation["signals"]["coverage_pct"], 12.5)
        self.assertAlmostEqual(evaluation["signals"]["coverage_ratio"], 0.125)
        self.assertIn("LOW_TRACKING_COVERAGE", evaluation["reason_codes"])
        self.assertIn("PLAYER_SCORING_NOT_VALIDATED", evaluation["reason_codes"])

    def test_tracking_percentage_points_below_one_are_not_inflated(self):
        evaluation = build_tracking_evaluation(
            candidate_metrics={"stabilityScore": 0.0},
            tracking={
                "coverage_pct_total": 0.49,
                "coverage_pct": 0.49,
                "bboxes_count": 29,
                "segments_total": 108,
                "segments_with_player": 7,
                "largest_gap_sec": 2031.94,
            },
        )

        self.assertAlmostEqual(evaluation["signals"]["coverage_pct"], 0.49)
        self.assertAlmostEqual(evaluation["signals"]["coverage_ratio"], 0.0049)
        self.assertAlmostEqual(evaluation["tracking_quality_index"], 9.9, places=1)
        self.assertIn("LOW_TRACKING_COVERAGE", evaluation["reason_codes"])
        self.assertIn("LONG_TRACKING_GAPS", evaluation["reason_codes"])

    def test_explicit_tracking_ratio_takes_precedence_over_legacy_percent_field(self):
        evaluation = build_tracking_evaluation(
            tracking={
                "coverage_ratio": 0.0049,
                "coverage_pct": 49.0,
                "bboxes_count": 60,
                "stability_score": 0.8,
            }
        )

        self.assertAlmostEqual(evaluation["signals"]["coverage_pct"], 0.49)
        self.assertAlmostEqual(evaluation["signals"]["coverage_ratio"], 0.0049)

    def test_no_evidence_means_zero_quality_not_free_continuity_points(self):
        evaluation = build_tracking_evaluation()

        self.assertEqual(evaluation["tracking_quality_index"], 0.0)
        self.assertEqual(
            evaluation["signals"]["tracklet_continuity_source"],
            "unavailable",
        )
        self.assertIn("CONTINUITY_NOT_MEASURED", evaluation["reason_codes"])
        self.assertIn("INSUFFICIENT_TRACKING_SAMPLES", evaluation["reason_codes"])

    def test_truth_gate_removes_legacy_player_scores(self):
        legacy_result = {
            "match_rating_10": 7.4,
            "impact_100": 63,
            "overall_score": 88.0,
            "role_score": 88.0,
            "radar": {"Finishing": 91.0},
            "report": {"scores": {"overall_score": 88.0}},
            "evidence_metrics": {
                "distance_covered_m": 1234.0,
                "avg_speed_kmh": 12.2,
                "candidate_metrics": {
                    "coveragePct": 0.5,
                    "stabilityScore": 0.8,
                    "sampleFramesCount": 60,
                },
            },
        }

        result = apply_evaluation_truth_gate(
            legacy_result,
            candidate_metrics=legacy_result["evidence_metrics"]["candidate_metrics"],
        )

        self.assertIsNone(result["match_rating_10"])
        self.assertIsNone(result["impact_100"])
        self.assertIsNone(result["overall_score"])
        self.assertIsNone(result["role_score"])
        self.assertEqual(result["radar"], {})
        self.assertNotIn("report", result)
        self.assertNotIn("distance_covered_m", result["evidence_metrics"])
        self.assertNotIn("avg_speed_kmh", result["evidence_metrics"])
        self.assertEqual(result["score_kind"], "tracking_quality")
        self.assertFalse(result["player_evaluation_available"])
        self.assertTrue(result["legacy_scores_suppressed"])

    def test_image_motion_uses_explicit_normalized_coordinate_space(self):
        tracking = {
            "bboxes": [
                {"t": 0.0, "x": 0.10, "y": 0.20, "w": 0.10, "h": 0.20},
                {"t": 1.0, "x": 0.20, "y": 0.20, "w": 0.10, "h": 0.20},
                {"t": 2.0, "x": 0.30, "y": 0.20, "w": 0.10, "h": 0.20},
            ]
        }

        metrics = compute_image_motion_metrics(tracking)

        self.assertEqual(metrics["metric_space"], "image_plane_normalized")
        self.assertFalse(metrics["camera_motion_compensated"])
        self.assertFalse(metrics["pitch_calibrated"])
        self.assertEqual(metrics["observed_samples"], 3)
        self.assertAlmostEqual(metrics["normalized_path_length"], 0.2, places=6)
        self.assertAlmostEqual(metrics["avg_center_speed_norm_per_sec"], 0.1, places=6)

    def test_even_strong_tracking_does_not_claim_player_evaluation(self):
        evaluation = build_tracking_evaluation(
            candidate_metrics={
                "coveragePct": 0.92,
                "stabilityScore": 0.95,
                "sampleFramesCount": 180,
            },
            tracking={
                "coverage_pct": 92.0,
                "bboxes_count": 180,
                "lost_segments": [],
            },
        )

        self.assertEqual(evaluation["tracking_confidence"], "medium")
        self.assertGreater(evaluation["tracking_quality_index"], 90.0)
        self.assertFalse(evaluation["player_evaluation_available"])
        self.assertFalse(
            evaluation["capabilities"]["cross_shot_player_reidentification"]
        )
        self.assertFalse(evaluation["capabilities"]["technical_tactical_scoring"])


if __name__ == "__main__":
    unittest.main()
