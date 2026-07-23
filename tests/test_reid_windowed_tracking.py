import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app.reid.association import AppearanceDescriptor


def _bbox():
    return {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2}


class ReIDWindowedTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ultralytics = types.ModuleType("ultralytics")

        class FakeYOLO:
            def __init__(self, _model):
                self.predictor = None

        ultralytics.YOLO = FakeYOLO
        sys.modules.setdefault("ultralytics", ultralytics)

        import app.workers

        tracking = types.ModuleType("app.workers.tracking")

        class TrackingTimeoutError(RuntimeError):
            pass

        tracking.TrackingTimeoutError = TrackingTimeoutError
        tracking.S3_ENDPOINT_URL = "http://s3"
        tracking.S3_PUBLIC_ENDPOINT_URL = "https://s3.example"
        tracking._normalize_player_ref = lambda value: dict(value)
        tracking.iter_windows = lambda *args, **kwargs: [
            (0.0, 45.0),
            (35.0, 80.0),
            (70.0, 115.0),
        ]
        tracking._extract_segment = lambda *args, **kwargs: None
        tracking._update_tracking_progress = lambda *args, **kwargs: None
        tracking._get_s3_client = lambda _endpoint: object()
        tracking._ensure_bucket_exists = lambda *args, **kwargs: None
        tracking._upload_file = lambda *args, **kwargs: None
        tracking._presign_get_object = (
            lambda bucket, key, expires: f"https://s3.example/{bucket}/{key}"
        )

        def iou(first, second):
            from app.reid.window_logic import bbox_iou

            return bbox_iou(first, second)

        tracking._bbox_iou = iou
        cls.track_maps = {}

        def collect(segment_path, **_kwargs):
            name = Path(segment_path).stem
            window_number = int(name.split("_")[-1])
            track_map = cls.track_maps[window_number]
            samples = [
                {
                    "t": float(index),
                    "detections": [
                        {
                            "track_id": track_id,
                            "bbox": dict(items[index]["bbox"]),
                            "conf": items[index]["conf"],
                        }
                        for track_id, items in track_map.items()
                    ],
                }
                for index in range(3)
            ]
            if window_number == 2:
                for index, sample in enumerate(samples):
                    sample["t"] = 15.0 + index
                    for items in track_map.values():
                        items[index]["t"] = 15.0 + index
            return samples, track_map

        tracking._collect_window_samples = collect

        def build_bboxes(samples, selected_track_id, *, fps, time_offset):
            bboxes = []
            for sample in samples:
                selected = next(
                    (
                        item
                        for item in sample["detections"]
                        if item["track_id"] == selected_track_id
                    ),
                    None,
                )
                if selected:
                    bboxes.append(
                        {
                            "t": float(sample["t"]) + time_offset,
                            **selected["bbox"],
                            "conf": selected["conf"],
                        }
                    )
            return bboxes, [], bboxes[-1] if bboxes else None

        tracking._build_window_bboxes = build_bboxes
        sys.modules["app.workers.tracking"] = tracking
        app.workers.tracking = tracking
        sys.modules.pop("app.reid.windowed_tracking", None)
        cls.module = importlib.import_module("app.reid.windowed_tracking")

    def _track(self, track_ids):
        return {
            track_id: [
                {
                    "t": float(index),
                    "bbox": _bbox(),
                    "conf": 0.95,
                    "sample_index": index,
                }
                for index in range(3)
            ]
            for track_id in track_ids
        }

    def test_clear_identity_is_linked_across_both_directions(self):
        type(self).track_maps = {
            1: self._track([30]),
            2: self._track([10]),
            3: self._track([20]),
        }
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        def descriptors(_path, _track_map, track_ids):
            return {track_id: descriptor for track_id in track_ids}

        with patch.dict(
            os.environ,
            {
                "S3_ACCESS_KEY": "key",
                "S3_SECRET_KEY": "secret",
                "S3_BUCKET": "bucket",
            },
            clear=False,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=descriptors,
        ):
            output = self.module.track_player_windowed_reid(
                "job-1",
                "/tmp/input.mp4",
                {"t": 50.0, **_bbox()},
                [],
                video_duration_sec=115.0,
            )

        self.assertEqual(output["mode"], "full_match_windowed")
        self.assertEqual(output["identity_mode"], "appearance_reid_v1")
        self.assertEqual(output["segments_with_player"], 3)
        self.assertEqual(output["reid_summary"]["accepted_associations"], 2)
        identities = {
            segment["reid"].get("identity_id")
            for segment in output["segments"]
        }
        self.assertEqual(identities, {"job-job-1-selected-player"})
        self.assertTrue(
            all(
                segment["identity_status"] == "ACCEPTED"
                for segment in output["segments"]
            )
        )

    def test_equal_candidates_abstain_instead_of_switching_identity(self):
        type(self).track_maps = {
            1: self._track([30, 31]),
            2: self._track([10]),
            3: self._track([20, 21]),
        }
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        def descriptors(_path, _track_map, track_ids):
            return {track_id: descriptor for track_id in track_ids}

        with patch.dict(
            os.environ,
            {
                "S3_ACCESS_KEY": "key",
                "S3_SECRET_KEY": "secret",
                "S3_BUCKET": "bucket",
            },
            clear=False,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=descriptors,
        ):
            output = self.module.track_player_windowed_reid(
                "job-2",
                "/tmp/input.mp4",
                {"t": 50.0, **_bbox()},
                [],
                video_duration_sec=115.0,
            )

        self.assertEqual(output["segments_with_player"], 1)
        self.assertEqual(output["reid_summary"]["accepted_associations"], 0)
        self.assertEqual(output["reid_summary"]["abstained_associations"], 2)
        non_anchor = [
            segment
            for segment in output["segments"]
            if segment["direction"] != "anchor"
        ]
        self.assertTrue(
            all(segment["selected_track_id"] is None for segment in non_anchor)
        )
        self.assertTrue(
            all(
                "AMBIGUOUS_CANDIDATE_MARGIN"
                in segment["reid"]["reason_codes"]
                for segment in non_anchor
            )
        )


if __name__ == "__main__":
    unittest.main()
