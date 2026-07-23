import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from app.calibration.homography import fit_pitch_calibration
from app.calibration.schema import CalibrationRequest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "schemas"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "pitch_calibration"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class PitchCalibrationJsonSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request_schema = load_json(
            SCHEMA_DIR / "pitch-calibration-request-v1.schema.json"
        )
        cls.result_schema = load_json(
            SCHEMA_DIR / "pitch-calibration-result-v1.schema.json"
        )
        Draft202012Validator.check_schema(cls.request_schema)
        Draft202012Validator.check_schema(cls.result_schema)
        cls.request_validator = Draft202012Validator(cls.request_schema)
        cls.result_validator = Draft202012Validator(cls.result_schema)

    def test_exact_request_fixture_matches_documented_schema(self):
        payload = load_json(FIXTURE_DIR / "exact-full-pitch.json")
        self.request_validator.validate(payload)

    def test_serialized_validated_result_matches_documented_schema(self):
        request_payload = load_json(FIXTURE_DIR / "exact-full-pitch.json")
        calibration = fit_pitch_calibration(
            CalibrationRequest.from_payload(request_payload)
        )
        self.assertTrue(calibration.validated)
        self.result_validator.validate(calibration.to_payload())

    def test_serialized_rejected_result_matches_documented_schema(self):
        request_payload = load_json(FIXTURE_DIR / "exact-full-pitch.json")
        request_payload["correspondences"] = request_payload["correspondences"][:4]
        calibration = fit_pitch_calibration(
            CalibrationRequest.from_payload(request_payload)
        )
        self.assertFalse(calibration.validated)
        self.result_validator.validate(calibration.to_payload())


if __name__ == "__main__":
    unittest.main()
