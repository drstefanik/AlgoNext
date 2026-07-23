import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from app.vision.camera_analysis import analyze_camera_sequence
from tests.synthetic_pitch import make_non_pitch_frame, make_pitch_frame

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs" / "schemas" / "camera-analysis-v1.schema.json"


class CameraAnalysisSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_serialized_payload_matches_schema(self):
        samples = [(index * 0.5, make_pitch_frame()) for index in range(8)]
        payload = analyze_camera_sequence(
            samples,
            source_duration_sec=4.0,
        ).to_payload(include_samples=True)

        self.validator.validate(payload)

    def test_mixed_payload_matches_schema(self):
        samples = [(index * 0.5, make_pitch_frame()) for index in range(6)]
        samples.extend(
            ((index + 6) * 0.5, make_non_pitch_frame()) for index in range(4)
        )
        payload = analyze_camera_sequence(
            samples,
            source_duration_sec=5.0,
        ).to_payload()

        self.validator.validate(payload)


if __name__ == "__main__":
    unittest.main()
