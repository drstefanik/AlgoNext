import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

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

    def validated_result(self):
        request_payload = load_json(FIXTURE_DIR / "exact-full-pitch.json")
        return fit_pitch_calibration(
            CalibrationRequest.from_payload(request_payload)
        ).to_payload()

    def rejected_result(self):
        request_payload = load_json(FIXTURE_DIR / "exact-full-pitch.json")
        request_payload["correspondences"] = request_payload["correspondences"][:4]
        return fit_pitch_calibration(
            CalibrationRequest.from_payload(request_payload)
        ).to_payload()

    def test_exact_request_fixture_matches_documented_schema(self):
        payload = load_json(FIXTURE_DIR / "exact-full-pitch.json")
        self.request_validator.validate(payload)

    def test_serialized_validated_result_matches_documented_schema(self):
        payload = self.validated_result()
        self.assertTrue(payload["validated"])
        self.result_validator.validate(payload)

    def test_serialized_rejected_result_matches_documented_schema(self):
        payload = self.rejected_result()
        self.assertFalse(payload["validated"])
        self.result_validator.validate(payload)

    def test_schema_rejects_missing_threshold(self):
        payload = self.validated_result()
        payload["thresholds"].pop("maximum_rmse_m")
        with self.assertRaises(ValidationError):
            self.result_validator.validate(payload)

    def test_schema_rejects_unknown_threshold(self):
        payload = self.validated_result()
        payload["thresholds"]["unknown_gate"] = 1.0
        with self.assertRaises(ValidationError):
            self.result_validator.validate(payload)

    def test_schema_requires_validated_status_coherence(self):
        payload = self.validated_result()
        payload["validated"] = False
        with self.assertRaises(ValidationError):
            self.result_validator.validate(payload)

    def test_schema_requires_reason_for_rejected_result(self):
        payload = self.rejected_result()
        payload["reason_codes"] = []
        with self.assertRaises(ValidationError):
            self.result_validator.validate(payload)

    def test_schema_rejects_wrong_ransac_seed(self):
        payload = self.validated_result()
        payload["provenance"]["ransac_seed"] = 42
        with self.assertRaises(ValidationError):
            self.result_validator.validate(payload)

    def test_schema_rejects_additional_quality_fields(self):
        payload = self.validated_result()
        payload["quality"]["silent_metric"] = 1.0
        with self.assertRaises(ValidationError):
            self.result_validator.validate(payload)

    def test_schema_rejects_non_boolean_inlier_mask(self):
        payload = self.validated_result()
        payload["quality"]["inlier_mask"][0] = 1
        with self.assertRaises(ValidationError):
            self.result_validator.validate(payload)

    def test_request_schema_requires_exactly_one_field_target(self):
        payload = load_json(FIXTURE_DIR / "exact-full-pitch.json")
        invalid = copy.deepcopy(payload)
        invalid["correspondences"][0]["field"] = {"x_m": 0.0, "y_m": 0.0}
        with self.assertRaises(ValidationError):
            self.request_validator.validate(invalid)


if __name__ == "__main__":
    unittest.main()
