import unittest

from fastapi import HTTPException

from app.api import _parse_track_selection_payload


class SelectTrackPayloadTests(unittest.TestCase):
    def test_accepts_frame_time_sec(self):
        payload = {
            "trackId": 8,
            "selection": {
                "frame_time_sec": 3954.517,
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            },
        }

        normalized = _parse_track_selection_payload(payload)

        self.assertEqual(normalized.track_id, 8)
        self.assertEqual(normalized.selection.frame_time_sec, 3954.517)

    def test_accepts_time_sec(self):
        payload = {
            "track_id": 8,
            "selection": {
                "time_sec": 3954.517,
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            },
        }

        normalized = _parse_track_selection_payload(payload)

        self.assertEqual(normalized.track_id, 8)
        self.assertEqual(normalized.selection.frame_time_sec, 3954.517)

    def test_missing_selection_reports_error(self):
        with self.assertRaises(HTTPException) as context:
            _parse_track_selection_payload({"trackId": 8})

        error = context.exception.detail["error"]
        self.assertEqual(error["code"], "INVALID_PAYLOAD")
        self.assertTrue(error["details"]["errors"])


if __name__ == "__main__":
    unittest.main()
