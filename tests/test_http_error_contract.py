import unittest

from app.core.http_errors import normalize_http_exception_detail


class HttpErrorContractTests(unittest.TestCase):
    def test_unwraps_legacy_nested_error(self):
        error = normalize_http_exception_detail(
            {
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": "Job not found",
                    "details": {"job_id": "job-1"},
                }
            }
        )

        self.assertEqual(error["code"], "JOB_NOT_FOUND")
        self.assertEqual(error["message"], "Job not found")
        self.assertEqual(error["details"], {"job_id": "job-1"})

    def test_preserves_enqueue_missing_fields(self):
        error = normalize_http_exception_detail(
            {
                "code": "NOT_READY",
                "message": "Missing required selections for enqueue",
                "missing": ["player_ref", "target"],
            }
        )

        self.assertEqual(error["missing"], ["player_ref", "target"])

    def test_plain_string_detail_remains_readable(self):
        error = normalize_http_exception_detail("Invalid bbox")

        self.assertEqual(error["code"], "HTTP_ERROR")
        self.assertEqual(error["message"], "Invalid bbox")


if __name__ == "__main__":
    unittest.main()
