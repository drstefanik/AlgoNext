import asyncio
import json
import unittest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from unittest.mock import patch

with patch("app.core.db.Base.metadata.create_all"):
    from app.main import validation_exception_handler, http_exception_handler
from app.api import _parse_track_selection_payload
from app.schemas import JobCreate
from fastapi import HTTPException


class ValidationResponseTests(unittest.TestCase):
    def request(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/jobs/one/select-track",
                "headers": [],
            }
        )
        request.state.request_id = "validation-request"
        return request

    def test_model_validator_exception_still_returns_422_json(self):
        try:
            JobCreate(role="MF", category="U18")
        except ValidationError as error:
            response = asyncio.run(
                validation_exception_handler(
                    self.request(), RequestValidationError(error.errors())
                )
            )
        self.assertEqual(response.status_code, 422)
        payload = json.loads(response.body)
        self.assertEqual(payload["error"]["code"], "VALIDATION_ERROR")
        self.assertNotIn("ctx", payload["error"]["details"]["errors"][0])

    def test_nonfinite_box_error_remains_an_actionable_http_response(self):
        with self.assertRaises(HTTPException) as ctx:
            _parse_track_selection_payload(
                {
                    "trackId": 2,
                    "selection": {
                        "time_sec": 0,
                        "bbox": {"x": float("nan"), "y": 0, "w": 0.2, "h": 0.2},
                    },
                }
            )
        response = asyncio.run(http_exception_handler(self.request(), ctx.exception))
        self.assertLess(response.status_code, 500)
        self.assertIn("errors", json.loads(response.body)["error"]["details"])


if __name__ == "__main__":
    unittest.main()
