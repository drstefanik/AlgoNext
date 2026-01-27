import unittest

from app.api import _build_candidate_payload


class DummyS3Public:
    def generate_presigned_url(self, *args, **kwargs) -> str:
        return "https://example.com/signed.jpg"


class CandidatePayloadTests(unittest.TestCase):
    def test_sample_frames_include_time_sec(self):
        candidates_payload = {
            "candidates": [
                {
                    "track_id": 1,
                    "sample_frames": [
                        {
                            "key": "jobs/abc/frame.jpg",
                            "frame_time_sec": 3954.517,
                            "bbox": {"x": 0.27, "y": 0.33, "w": 0.03, "h": 0.12},
                        }
                    ],
                }
            ]
        }
        context = {
            "s3_public": DummyS3Public(),
            "bucket": "bucket",
            "expires_seconds": 3600,
        }

        result = _build_candidate_payload(candidates_payload, context)

        self.assertEqual(result[0]["sampleFrames"][0]["frame_time_sec"], 3954.517)
        self.assertEqual(result[0]["sampleFrames"][0]["time_sec"], 3954.517)


if __name__ == "__main__":
    unittest.main()
