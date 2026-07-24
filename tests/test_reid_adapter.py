import unittest

from app.benchmark.reid_adapters import prediction_from_algonext_reid


class ReIDAdapterTests(unittest.TestCase):
    def test_converts_accepted_abstained_and_failed_windows(self):
        prediction = prediction_from_algonext_reid(
            {
                "segments": [
                    {
                        "window_start": 0.0,
                        "window_end": 10.0,
                        "reid": {
                            "status": "ACCEPTED",
                            "selected_candidate_id": "7",
                            "best_score": 0.91,
                            "margin": 0.12,
                            "reason_codes": ["ASSOCIATION_ACCEPTED"],
                            "candidates": [
                                {"candidate_id": "7", "combined_score": 0.91},
                                {"candidate_id": "2", "combined_score": 0.79},
                            ],
                        },
                    },
                    {
                        "window_start": 10.0,
                        "window_end": 20.0,
                        "reid": {
                            "status": "ABSTAINED",
                            "best_score": 0.88,
                            "margin": 0.01,
                            "reason_codes": ["AMBIGUOUS_CANDIDATE_MARGIN"],
                            "candidates": [
                                {"candidate_id": "3", "combined_score": 0.88},
                                {"candidate_id": "4", "combined_score": 0.87},
                            ],
                        },
                    },
                    {
                        "window_start": 20.0,
                        "window_end": 30.0,
                        "reid": {
                            "status": "ABSTAINED",
                            "reason_codes": ["WINDOW_PROCESSING_FAILED"],
                            "candidates": [],
                        },
                    },
                ]
            },
            video_id="job-1",
        )

        self.assertEqual(
            [window.decision for window in prediction.windows],
            ["ACCEPTED", "ABSTAINED", "FAILED"],
        )
        self.assertEqual(prediction.windows[0].selected_candidate_id, "7")
        self.assertEqual(prediction.windows[1].best_candidate_id, "3")
        self.assertEqual(prediction.windows[1].candidate_ids, ("3", "4"))

    def test_requires_segment_array(self):
        with self.assertRaises(ValueError):
            prediction_from_algonext_reid({}, video_id="job-1")


if __name__ == "__main__":
    unittest.main()
