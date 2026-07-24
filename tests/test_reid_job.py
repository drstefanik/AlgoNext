import unittest

from app.benchmark.reid_job import discover_job_artifacts, unwrap_job_payload


class ReIDJobExportTests(unittest.TestCase):
    def test_discovers_signed_tracking_and_video_urls(self):
        job = unwrap_job_payload(
            {
                "ok": True,
                "data": {
                    "job_id": "job-1",
                    "video_url": "https://fallback.example/video.mp4",
                    "result": {
                        "tracking": {
                            "asset": {
                                "signed_url": "https://s3.example/tracking.json"
                            },
                            "reid_summary": {"identity_id": "player-a"},
                        },
                        "assets": {
                            "input_video": {
                                "signed_url": "https://s3.example/input.mp4"
                            }
                        },
                    },
                },
            }
        )
        artifacts = discover_job_artifacts(job)
        self.assertEqual(artifacts["job_id"], "job-1")
        self.assertEqual(artifacts["identity"], "player-a")
        self.assertEqual(
            artifacts["tracking_url"], "https://s3.example/tracking.json"
        )
        self.assertEqual(artifacts["video_url"], "https://s3.example/input.mp4")

    def test_missing_tracking_url_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "tracking_url"):
            discover_job_artifacts(
                {
                    "job_id": "job-1",
                    "video_url": "https://example.com/video.mp4",
                    "result": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
