import unittest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import Response
from app import api
from app.core.deps import get_db


class FrameRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(api.router)
        app.dependency_overrides[get_db] = lambda: object()
        self.client = TestClient(app)

    def test_list_dispatches_to_frame_list_instead_of_s3_filename(self):
        with patch.object(
            api,
            "get_frames",
            return_value={"ok": True, "data": {"items": [{"key": "frame_0001.jpg"}]}},
        ) as frames, patch.object(api, "_stream_s3_image") as stream:
            result = self.client.get("/jobs/job-1/frames/list?count=2")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(len(result.json()["data"]["items"]), 1)
        self.assertEqual(frames.call_args.kwargs["count"], 2)
        stream.assert_not_called()

    def test_overlay_preserves_documented_deprecation(self):
        with patch.object(api, "_stream_s3_image") as stream:
            self.assertEqual(
                self.client.get("/jobs/job-1/frames/overlay").status_code, 410
            )
        stream.assert_not_called()

    def test_actual_image_still_streams(self):
        with patch.object(
            api,
            "_stream_s3_image",
            return_value=Response(b"jpeg", media_type="image/jpeg"),
        ) as stream:
            response = self.client.get("/jobs/job-1/frames/frame_0001.jpg")
        self.assertEqual(response.content, b"jpeg")
        stream.assert_called_once_with("jobs/job-1/frames/frame_0001.jpg")


if __name__ == "__main__":
    unittest.main()
