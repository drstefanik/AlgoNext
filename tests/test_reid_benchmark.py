import unittest

from app.benchmark.reid_metrics import (
    ReIDGateThresholds,
    evaluate_reid_quality_gate,
    evaluate_reid_sequence,
)
from app.benchmark.reid_schema import (
    ReIDSchemaValidationError,
    ReIDSequenceAnnotation,
    ReIDSequencePrediction,
)


def annotation(windows):
    return ReIDSequenceAnnotation.from_payload(
        {
            "schema_version": "reid-window-annotation-v1",
            "video_id": "job-1",
            "identity": "selected-player",
            "fps": 1,
            "windows": windows,
        }
    )


def prediction(windows):
    return ReIDSequencePrediction.from_payload(
        {
            "schema_version": "reid-window-prediction-v1",
            "video_id": "job-1",
            "windows": windows,
        }
    )


def truth(
    index,
    visibility,
    *,
    candidate_state=None,
    candidate=None,
    selected_correct=None,
):
    payload = {
        "window_index": index,
        "window_start": index * 10.0,
        "window_end": index * 10.0 + 10.0,
        "target_visibility": visibility,
        "candidate_state": candidate_state,
        "target_candidate_id": candidate,
        "selected_track_is_target": selected_correct,
        "evidence_frames": [],
    }
    return payload


def predicted(index, decision, selected=None, *, best="1", score=0.9, reasons=None):
    candidate_ids = [] if best is None else [best]
    if selected is not None and selected not in candidate_ids:
        candidate_ids.append(selected)
    return {
        "window_index": index,
        "window_start": index * 10.0,
        "window_end": index * 10.0 + 10.0,
        "decision": decision,
        "selected_candidate_id": selected,
        "best_candidate_id": best,
        "best_score": score,
        "margin": 0.1,
        "candidate_ids": candidate_ids,
        "reason_codes": reasons or [],
    }


class ReIDBenchmarkTests(unittest.TestCase):
    def test_true_accept_and_correct_nonvisible_abstention(self):
        report = evaluate_reid_sequence(
            annotation(
                [
                    truth(
                        0,
                        "VISIBLE",
                        candidate_state="PRESENT",
                        candidate="7",
                        selected_correct=True,
                    ),
                    truth(1, "NOT_VISIBLE"),
                ]
            ),
            prediction(
                [
                    predicted(0, "ACCEPTED", "7", best="7"),
                    predicted(1, "ABSTAINED", best="4"),
                ]
            ),
        )

        self.assertEqual(report["counts"]["true_accepts"], 1)
        self.assertEqual(report["counts"]["correct_associations"], 1)
        self.assertEqual(report["counts"]["correct_abstentions_not_visible"], 1)
        self.assertEqual(report["metrics"]["accepted_precision"], 1.0)
        self.assertEqual(report["metrics"]["visible_window_recall"], 1.0)
        self.assertEqual(report["metrics"]["nonvisible_abstention_rate"], 1.0)
        self.assertEqual(report["metrics"]["end_to_end_window_success_rate"], 1.0)

    def test_wrong_candidate_is_visible_false_link(self):
        report = evaluate_reid_sequence(
            annotation(
                [
                    truth(
                        0,
                        "VISIBLE",
                        candidate_state="PRESENT",
                        candidate="7",
                        selected_correct=False,
                    )
                ]
            ),
            prediction([predicted(0, "ACCEPTED", "9", best="9")]),
        )
        self.assertEqual(report["counts"]["false_links_visible_wrong_target"], 1)
        self.assertEqual(report["counts"]["wrong_candidate_accepts"], 1)
        self.assertEqual(report["metrics"]["false_link_rate"], 1.0)
        self.assertEqual(report["metrics"]["accepted_precision"], 0.0)

    def test_accept_when_target_not_visible_is_false_accept(self):
        report = evaluate_reid_sequence(
            annotation([truth(0, "NOT_VISIBLE", selected_correct=False)]),
            prediction([predicted(0, "ACCEPTED", "2", best="2")]),
        )
        self.assertEqual(report["counts"]["false_accepts_not_visible"], 1)
        self.assertEqual(report["windows"][0]["outcome"], "FALSE_ACCEPT_NOT_VISIBLE")

    def test_unjudged_accept_is_excluded_from_precision_and_reported(self):
        report = evaluate_reid_sequence(
            annotation(
                [
                    truth(
                        0,
                        "VISIBLE",
                        candidate_state="UNVERIFIABLE",
                    )
                ]
            ),
            prediction([predicted(0, "ACCEPTED", "2", best="2")]),
        )
        self.assertEqual(report["counts"]["accepted_unjudged_windows"], 1)
        self.assertEqual(report["metrics"]["accepted_judgement_coverage"], 0.0)
        self.assertEqual(report["windows"][0]["outcome"], "ACCEPT_UNJUDGED")

    def test_abstention_with_target_candidate_is_missed_association(self):
        report = evaluate_reid_sequence(
            annotation(
                [
                    truth(
                        0,
                        "VISIBLE",
                        candidate_state="PRESENT",
                        candidate="7",
                    )
                ]
            ),
            prediction(
                [
                    predicted(
                        0,
                        "ABSTAINED",
                        best="7",
                        reasons=["AMBIGUOUS_CANDIDATE_MARGIN"],
                    )
                ]
            ),
        )
        self.assertEqual(report["counts"]["missed_associations"], 1)
        self.assertEqual(report["metrics"]["association_recall_given_candidate"], 0.0)
        self.assertEqual(
            report["reason_code_counts"]["AMBIGUOUS_CANDIDATE_MARGIN"], 1
        )

    def test_visible_target_absent_from_candidates_separates_candidate_recall(self):
        report = evaluate_reid_sequence(
            annotation(
                [truth(0, "VISIBLE", candidate_state="ABSENT")]
            ),
            prediction([predicted(0, "ABSTAINED", best="3")]),
        )
        self.assertEqual(report["counts"]["candidate_absent_windows"], 1)
        self.assertEqual(report["metrics"]["candidate_annotation_coverage"], 1.0)
        self.assertEqual(report["metrics"]["candidate_recall_visible"], 0.0)
        self.assertEqual(report["metrics"]["visible_window_recall"], 0.0)
        self.assertEqual(report["metrics"]["reid_decision_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["end_to_end_window_success_rate"], 0.0)

    def test_unverifiable_candidate_is_not_mislabelled_as_candidate_miss(self):
        report = evaluate_reid_sequence(
            annotation(
                [truth(0, "VISIBLE", candidate_state="UNVERIFIABLE")]
            ),
            prediction([predicted(0, "ABSTAINED", best="3")]),
        )
        self.assertEqual(report["counts"]["candidate_unverifiable_windows"], 1)
        self.assertEqual(report["metrics"]["candidate_annotation_coverage"], 0.0)
        self.assertEqual(report["counts"]["candidate_absent_windows"], 0)

    def test_uncertain_window_is_excluded(self):
        report = evaluate_reid_sequence(
            annotation([truth(0, "UNCERTAIN")]),
            prediction([predicted(0, "ACCEPTED", "2", best="2")]),
        )
        self.assertEqual(report["counts"]["scorable_windows"], 0)
        self.assertEqual(report["counts"]["unscored_uncertain_windows"], 1)
        self.assertEqual(report["metrics"]["annotation_coverage"], 0.0)

    def test_missing_prediction_is_processing_failure(self):
        report = evaluate_reid_sequence(
            annotation(
                [
                    truth(
                        0,
                        "VISIBLE",
                        candidate_state="PRESENT",
                        candidate="7",
                    )
                ]
            ),
            prediction([]),
        )
        self.assertEqual(report["counts"]["failed_windows"], 1)
        self.assertEqual(report["counts"]["missed_associations"], 1)
        self.assertEqual(report["metrics"]["processing_failure_rate"], 1.0)

    def test_window_time_mismatch_fails_closed(self):
        bad = predicted(0, "ABSTAINED", best="1")
        bad["window_start"] = 0.2
        with self.assertRaises(ValueError):
            evaluate_reid_sequence(
                annotation([truth(0, "NOT_VISIBLE")]), prediction([bad])
            )

    def test_annotation_requires_candidate_id_for_present_candidate(self):
        with self.assertRaises(ReIDSchemaValidationError):
            annotation(
                [truth(0, "VISIBLE", candidate_state="PRESENT")]
            )

    def test_annotation_rejects_candidate_state_on_not_visible_window(self):
        with self.assertRaises(ReIDSchemaValidationError):
            annotation(
                [truth(0, "NOT_VISIBLE", candidate_state="ABSENT")]
            )

    def test_candidate_absent_cannot_claim_selected_track_is_target(self):
        with self.assertRaises(ReIDSchemaValidationError):
            annotation(
                [
                    truth(
                        0,
                        "VISIBLE",
                        candidate_state="ABSENT",
                        selected_correct=True,
                    )
                ]
            )

    def test_conflicting_selected_correctness_and_candidate_id_fails_closed(self):
        with self.assertRaises(ValueError):
            evaluate_reid_sequence(
                annotation(
                    [
                        truth(
                            0,
                            "VISIBLE",
                            candidate_state="PRESENT",
                            candidate="7",
                            selected_correct=True,
                        )
                    ]
                ),
                prediction([predicted(0, "ACCEPTED", "9", best="9")]),
            )

    def test_quality_gate_uses_judgement_coverage_false_links_and_sample_size(self):
        report = evaluate_reid_sequence(
            annotation(
                [
                    truth(
                        0,
                        "VISIBLE",
                        candidate_state="PRESENT",
                        candidate="7",
                        selected_correct=False,
                    )
                ]
            ),
            prediction([predicted(0, "ACCEPTED", "9", best="9")]),
        )
        gate = evaluate_reid_quality_gate(
            report,
            ReIDGateThresholds(minimum_scorable_windows=1),
        )
        self.assertFalse(gate["passed"])
        failed = {check["metric"] for check in gate["checks"] if not check["passed"]}
        self.assertIn("accepted_precision", failed)
        self.assertIn("false_link_rate", failed)


if __name__ == "__main__":
    unittest.main()
