import unittest

from app.vision.match_observations import (
    aggregate_segment_observability,
    build_segment_observability,
    estimate_camera_motion,
    summarize_compensated_player_motion,
)


def detection(track_id, x, y=0.2, w=0.08, h=0.3):
    return {
        "track_id": track_id,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "conf": 0.9,
    }


def ball(x, y=0.45):
    return {
        "track_id": 99,
        "bbox": {"x": x, "y": y, "w": 0.02, "h": 0.02},
        "conf": 0.8,
        "class_id": 32,
    }


class MatchObservationTests(unittest.TestCase):
    def test_camera_motion_uses_multi_person_consensus(self):
        samples = [
            {
                "t": float(index),
                "detections": [
                    detection(1, 0.10 + index * 0.05),
                    detection(2, 0.35 + index * 0.05),
                    detection(3, 0.65 + index * 0.05),
                ],
                "ball_detections": [],
            }
            for index in range(4)
        ]

        camera = estimate_camera_motion(samples)
        self.assertTrue(camera["available"])
        self.assertEqual(camera["transitions_compensated"], 3)
        self.assertAlmostEqual(camera["coverage_ratio"], 1.0)

        player = [
            {
                "t": float(index),
                "x": 0.10 + index * 0.05,
                "y": 0.2,
                "w": 0.08,
                "h": 0.3,
            }
            for index in range(4)
        ]
        compensated = summarize_compensated_player_motion(
            player,
            window_start=0.0,
            camera_motion=camera,
        )
        self.assertTrue(compensated["available"])
        self.assertAlmostEqual(compensated["raw_path_length"], 0.15, places=6)
        self.assertAlmostEqual(compensated["compensated_path_length"], 0.0, places=6)

    def test_ball_tracking_and_proximity_events_are_auditable(self):
        samples = [
            {
                "t": float(index),
                "detections": [
                    detection(1, 0.20),
                    detection(2, 0.50),
                    detection(3, 0.75),
                ],
                "ball_detections": [ball(0.23 + index * 0.002)],
            }
            for index in range(4)
        ]
        player = [
            {"t": float(index), "x": 0.20, "y": 0.2, "w": 0.08, "h": 0.3}
            for index in range(4)
        ]

        result = build_segment_observability(
            samples,
            player,
            window_start=0.0,
            fps=2.0,
        )

        self.assertTrue(result["camera_motion"]["available"])
        self.assertTrue(result["ball_tracking"]["available"])
        self.assertEqual(result["ball_tracking"]["observed_samples"], 4)
        self.assertTrue(result["event_detection"]["available"])
        self.assertEqual(result["event_detection"]["event_count"], 1)
        event = result["event_detection"]["events"][0]
        self.assertEqual(event["type"], "BALL_PROXIMITY_SEQUENCE")
        self.assertEqual(event["samples"], 4)
        self.assertFalse(result["event_detection"]["validated"])

    def test_aggregation_deduplicates_overlapping_windows(self):
        segment = {
            "camera_motion": {
                "available": True,
                "transitions_total": 3,
                "transitions_compensated": 3,
            },
            "ball_tracking": {
                "sampled_frames": 4,
                "observations": [
                    {
                        "t": 1.0,
                        "x": 0.2,
                        "y": 0.4,
                        "w": 0.02,
                        "h": 0.02,
                        "conf": 0.7,
                    }
                ],
            },
            "event_detection": {
                "available": True,
                "events": [
                    {
                        "type": "BALL_PROXIMITY_SEQUENCE",
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                    }
                ],
            },
        }

        aggregate = aggregate_segment_observability([segment, segment])

        self.assertTrue(aggregate["camera_motion"]["available"])
        self.assertEqual(aggregate["ball_tracking"]["observed_samples"], 1)
        self.assertEqual(aggregate["event_detection"]["event_count"], 1)


if __name__ == "__main__":
    unittest.main()
