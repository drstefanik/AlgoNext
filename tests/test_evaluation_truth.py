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

    def test_failed_selected_tracking_never_inherits_preview_candidate_metrics(self):
        candidate_metrics = {
            "coveragePct": 0.125,
            "stabilityScore": 0.333,
            "sampleFramesCount": 4,
        }
        tracking = {
            "tracking_success": False,
            "tracking_status": "ANCHOR_NOT_FOUND",
            "action_required": "RESELECT_PLAYER",
            "coverage_pct_total": 0.0,
            "bboxes_count": 0,
            "segments_total": 108,
            "segments_with_player": 0,
            "largest_gap_sec": 5931.775,
            "anchors_total": 1,
            "anchors_matched": 0,
            "reid_summary": {
                "status": "ANCHOR_NOT_FOUND",
                "reason_codes": ["REID_ANCHORS_NOT_FOUND"],
            },
        }

        evaluation = build_tracking_evaluation(
            candidate_metrics=candidate_metrics,
            tracking=tracking,
        )

        self.assertEqual(evaluation["status"], "TRACKING_FAILED")
        self.assertIsNone(evaluation["tracking_quality_index"])
        self.assertEqual(evaluation["signals"]["coverage_pct"], 0.0)
        self.assertEqual(evaluation["signals"]["tracklet_continuity_pct"], 0.0)
        self.assertEqual(evaluation["signals"]["samples_used"], 0)
        self.assertIsNone(evaluation["signals"]["largest_gap_sec"])
        self.assertIn("REID_ANCHORS_NOT_FOUND", evaluation["reason_codes"])
        self.assertIn("PLAYER_RESELECTION_REQUIRED", evaluation["reason_codes"])

        result = apply_evaluation_truth_gate(
            {"evidence_metrics": {"candidate_metrics": candidate_metrics}},
            candidate_metrics=candidate_metrics,
            tracking=tracking,
        )
        self.assertNotIn("candidate_metrics", result["evidence_metrics"])
        self.assertEqual(
            result["evidence_metrics"]["preview_candidate_metrics"],
            candidate_metrics,
        )

    def test_anchor_acquisition_error_requests_retry_not_reselection(self):
        tracking = {
            "tracking_success": False,
            "tracking_status": "ANCHOR_ACQUISITION_ERROR",
            "action_required": "RETRY_ANALYSIS",
            "bboxes_count": 0,
            "segments_total": 108,
            "segments_with_player": 0,
            "reid_summary": {
                "status": "ANCHOR_ACQUISITION_ERROR",
                "reason_codes": ["REID_ANCHOR_ACQUISITION_ERROR"],
            },
        }

        result = apply_evaluation_truth_gate({}, tracking=tracking)

        self.assertEqual(result["evaluation_status"], "TRACKING_FAILED")
        self.assertIn("Riprova l'analisi", result["explain"])
        self.assertNotIn("Seleziona un riferimento", result["explain"])
        self.assertNotIn(
            "PLAYER_RESELECTION_REQUIRED",
            result["reason_codes"],
        )

    def test_partial_timeout_is_incomplete_not_anchor_failure(self):
        tracking = {
            "partial": True,
            "partial_reason": "TRACKING_TIMEOUT",
            "segments_total": 108,
            "segments_with_player": 0,
            "bboxes_count": 0,
            "largest_gap_sec": 5931.775,
            "reid_summary": {
                "status": "PARTIAL_TIMEOUT",
                "reason_codes": ["TRACKING_BUDGET_EXHAUSTED"],
            },
        }

        evaluation = build_tracking_evaluation(tracking=tracking)

        self.assertEqual(evaluation["status"], "TRACKING_INCOMPLETE")
        self.assertIsNone(evaluation["tracking_quality_index"])
        self.assertEqual(evaluation["signals"]["samples_used"], 0)
        self.assertIsNone(evaluation["signals"]["largest_gap_sec"])
        self.assertIn(
            "TRACKING_BUDGET_EXHAUSTED",
            evaluation["reason_codes"],
        )
        self.assertNotIn(
            "PLAYER_RESELECTION_REQUIRED",
            evaluation["reason_codes"],
        )

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

    def test_operational_observability_is_exposed_as_experimental(self):
        tracking = {
            "identity_mode": "appearance_reid_v1",
            "tracking_scope_status": "CROSS_WINDOW_EVIDENCE",
            "coverage_pct_total": 8.0,
            "segments_total": 5,
            "segments_with_player": 3,
            "segments": [
                {
                    "bboxes": [
                        {"t": 0.0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.3},
                        {"t": 1.0, "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.3},
                    ],
                    "camera_motion": {
                        "player_motion": {
                            "available": True,
                            "compensated_path_length": 0.03,
                        }
                    },
                }
            ],
            "reid_summary": {
                "validated": False,
                "accepted_associations": 2,
            },
            "camera_motion": {
                "available": True,
                "validated": False,
                "method": "multi-person-median-displacement-v1",
            },
            "ball_tracking": {
                "available": True,
                "validated": False,
                "method": "yolo-coco-sports-ball+bytetrack-v1",
            },
            "event_detection": {
                "available": True,
                "validated": False,
                "method": "selected-player-ball-proximity-v1",
            },
        }

        evaluation = build_tracking_evaluation(tracking=tracking)

        self.assertFalse(
            evaluation["capabilities"]["cross_shot_player_reidentification"]
        )
        self.assertTrue(evaluation["capabilities"]["camera_motion_compensation"])
        self.assertTrue(evaluation["capabilities"]["ball_tracking"])
        self.assertTrue(evaluation["capabilities"]["event_detection"])
        self.assertEqual(
            evaluation["capability_details"][
                "cross_shot_player_reidentification"
            ]["status"],
            "experimental",
        )
        self.assertEqual(
            evaluation["capability_details"]["pitch_calibration"]["status"],
            "foundation",
        )
        self.assertIn(
            "BALL_AND_EVENTS_EXPERIMENTAL_NOT_VALIDATED",
            evaluation["reason_codes"],
        )
        self.assertNotIn(
            "BALL_AND_EVENTS_NOT_MODELLED",
            evaluation["reason_codes"],
        )
        self.assertTrue(
            evaluation["signals"]["image_motion"]["camera_motion_compensated"]
        )


if __name__ == "__main__":
    unittest.main()
