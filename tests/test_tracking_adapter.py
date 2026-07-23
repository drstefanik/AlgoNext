import unittest

from app.benchmark.adapters import prediction_from_algonext_tracking


class AlgoNextTrackingAdapterTests(unittest.TestCase):
    def test_single_track_output_uses_stable_track_id(self):
        prediction = prediction_from_algonext_tracking(
            {
                "fps": 5,
                "track_id": 6,
                "bboxes": [
                    {
                        "t": 1.0,
                        "x": 0.1,
                        "y": 0.2,
                        "w": 0.1,
                        "h": 0.2,
                        "conf": 0.8,
                    }
                ],
            },
            video_id="video-1",
        )

        self.assertEqual(prediction.frames[0].frame_index, 5)
        self.assertEqual(prediction.frames[0].tracks[0].track_id, "track-6")

    def test_window_local_ids_are_namespaced_to_expose_reidentification(self):
        prediction = prediction_from_algonext_tracking(
            {
                "fps": 5,
                "segments": [
                    {
                        "selected_track_id": 1,
                        "bboxes": [
                            {
                                "t": 1.0,
                                "x": 0.1,
                                "y": 0.2,
                                "w": 0.1,
                                "h": 0.2,
                                "conf": 0.8,
                            }
                        ],
                    },
                    {
                        "selected_track_id": 1,
                        "bboxes": [
                            {
                                "t": 2.0,
                                "x": 0.2,
                                "y": 0.2,
                                "w": 0.1,
                                "h": 0.2,
                                "conf": 0.9,
                            }
                        ],
                    },
                ],
            },
            video_id="video-1",
        )

        self.assertEqual(
            [frame.tracks[0].track_id for frame in prediction.frames],
            ["segment-0001/track-1", "segment-0002/track-1"],
        )

    def test_overlapping_segments_keep_distinct_tracks_for_duplicate_penalty(self):
        prediction = prediction_from_algonext_tracking(
            {
                "fps": 5,
                "segments": [
                    {
                        "selected_track_id": 1,
                        "bboxes": [
                            {
                                "t": 1.0,
                                "x": 0.1,
                                "y": 0.2,
                                "w": 0.1,
                                "h": 0.2,
                            }
                        ],
                    },
                    {
                        "selected_track_id": 2,
                        "bboxes": [
                            {
                                "t": 1.0,
                                "x": 0.11,
                                "y": 0.2,
                                "w": 0.1,
                                "h": 0.2,
                            }
                        ],
                    },
                ],
            },
            video_id="video-1",
        )

        self.assertEqual(len(prediction.frames), 1)
        self.assertEqual(len(prediction.frames[0].tracks), 2)

    def test_duplicate_sample_for_same_track_keeps_higher_confidence(self):
        prediction = prediction_from_algonext_tracking(
            {
                "fps": 5,
                "track_id": 6,
                "bboxes": [
                    {
                        "t": 1.0,
                        "x": 0.1,
                        "y": 0.2,
                        "w": 0.1,
                        "h": 0.2,
                        "conf": 0.3,
                    },
                    {
                        "t": 1.01,
                        "x": 0.2,
                        "y": 0.2,
                        "w": 0.1,
                        "h": 0.2,
                        "conf": 0.9,
                    },
                ],
            },
            video_id="video-1",
        )

        self.assertEqual(len(prediction.frames), 1)
        self.assertEqual(prediction.frames[0].tracks[0].confidence, 0.9)
        self.assertEqual(prediction.frames[0].tracks[0].bbox.x, 0.2)


if __name__ == "__main__":
    unittest.main()
