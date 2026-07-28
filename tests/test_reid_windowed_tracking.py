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
        missing = object()
        cls._missing_module = missing
        cls._saved_modules = {
            name: sys.modules.get(name, missing)
            for name in (
                "ultralytics",
                "app.workers.tracking",
                "app.reid.windowed_tracking",
            )
        }
        ultralytics = types.ModuleType("ultralytics")

        class FakeYOLO:
            def __init__(self, _model):
                self.predictor = None

        ultralytics.YOLO = FakeYOLO
        sys.modules["ultralytics"] = ultralytics

        import app.reid
        import app.workers

        cls._reid_module = app.reid
        cls._saved_reid_windowed_tracking = getattr(
            app.reid, "windowed_tracking", missing
        )
        cls._workers_module = app.workers
        cls._saved_workers_tracking = getattr(app.workers, "tracking", missing)
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
        cls.sample_times = {}

        def collect(segment_path, **_kwargs):
            name = Path(segment_path).stem
            window_number = int(name.split("_")[-1])
            track_map = cls.track_maps[window_number]
            default_times = (
                [15.0, 16.0, 17.0]
                if window_number == 2
                else [0.0, 1.0, 2.0]
            )
            sample_times = cls.sample_times.get(window_number, default_times)
            samples = [
                {
                    "t": float(sample_times[index]),
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
            for index, sample in enumerate(samples):
                for items in track_map.values():
                    items[index]["t"] = sample["t"]
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

    @classmethod
    def tearDownClass(cls):
        for name, original in cls._saved_modules.items():
            if original is cls._missing_module:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        if cls._saved_workers_tracking is cls._missing_module:
            cls._workers_module.__dict__.pop("tracking", None)
        else:
            cls._workers_module.tracking = cls._saved_workers_tracking
        if cls._saved_reid_windowed_tracking is cls._missing_module:
            cls._reid_module.__dict__.pop("windowed_tracking", None)
        else:
            cls._reid_module.windowed_tracking = (
                cls._saved_reid_windowed_tracking
            )

    def setUp(self):
        type(self).sample_times = {}

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

    def _track_boxes(self, boxes):
        return {
            track_id: [
                {
                    "t": float(index),
                    "bbox": dict(bbox),
                    "conf": 0.95,
                    "sample_index": index,
                }
                for index in range(3)
            ]
            for track_id, bbox in boxes.items()
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

    def test_later_manual_anchor_reseeds_after_abstained_window(self):
        type(self).track_maps = {
            1: self._track([10]),
            2: self._track([20, 21]),
            3: self._track([30]),
        }
        type(self).sample_times = {3: [8.0, 9.0, 10.0]}
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
                "job-multi-reseed",
                "/tmp/input.mp4",
                {"t": 1.0, **_bbox()},
                [
                    {"frame_time_sec": 1.0, "frame_key": "early", **_bbox()},
                    {"frame_time_sec": 78.0, "frame_key": "late", **_bbox()},
                ],
                video_duration_sec=115.0,
            )

        self.assertEqual(output["segments"][1]["identity_status"], "ABSTAINED")
        self.assertEqual(output["segments"][2]["direction"], "anchor")
        self.assertEqual(output["segments"][2]["selected_track_id"], 30)
        self.assertEqual(output["anchor_reacquisitions"], 1)
        self.assertEqual(output["anchors_total"], 2)
        self.assertEqual(output["anchors_matched"], 2)
        self.assertEqual(
            [item["status"] for item in output["anchor_matches"]],
            ["MATCHED", "MATCHED"],
        )
        self.assertEqual(
            output["reid_summary"]["anchor_reacquisitions"], 1
        )

    def test_two_anchors_in_one_window_stitch_local_track_ids(self):
        left = _bbox()
        right = {"x": 0.7, "y": 0.2, "w": 0.1, "h": 0.2}
        type(self).track_maps = {
            1: self._track([30]),
            2: self._track_boxes({10: left, 11: right}),
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
                "job-same-window",
                "/tmp/input.mp4",
                {"t": 50.0, **left},
                [
                    {"frame_time_sec": 50.0, "frame_key": "left", **left},
                    {"frame_time_sec": 52.0, "frame_key": "right", **right},
                ],
                video_duration_sec=115.0,
            )

        anchor_segment = output["segments"][1]
        self.assertEqual(anchor_segment["selected_track_ids"], [10, 11])
        self.assertEqual(
            anchor_segment["reid"]["reason_codes"],
            ["MANUAL_MULTI_ANCHOR"],
        )
        self.assertEqual(
            {round(item["x"], 1) for item in anchor_segment["bboxes"]},
            {0.2, 0.7},
        )
        self.assertEqual(output["anchors_total"], 2)
        self.assertEqual(output["anchors_matched"], 2)
        self.assertEqual(
            [item["local_track_id"] for item in output["anchor_matches"]],
            [10, 11],
        )

    def test_overlap_anchor_has_one_canonical_match(self):
        type(self).track_maps = {
            1: self._track([10]),
            2: self._track([20]),
            3: self._track([30]),
        }
        type(self).sample_times = {1: [39.0, 40.0, 41.0]}
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
                "job-overlap",
                "/tmp/input.mp4",
                {"t": 40.0, **_bbox()},
                [
                    {
                        "frame_time_sec": 40.0,
                        "frame_key": "overlap",
                        **_bbox(),
                    }
                ],
                video_duration_sec=115.0,
            )

        self.assertEqual(output["anchors_total"], 1)
        self.assertEqual(output["anchors_matched"], 1)
        self.assertEqual(len(output["anchor_matches"]), 1)
        self.assertEqual(output["anchor_matches"][0]["window_index"], 0)

    def test_anchor_match_rejects_spatially_unrelated_same_size_track(self):
        anchor = {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2}
        distant = {"x": 0.75, "y": 0.65, "w": 0.1, "h": 0.2}
        detection = {
            "t": 0.0,
            "track_id": 7,
            "bbox": distant,
            "conf": 0.99,
        }

        selected = self.module._select_anchor_track(
            [{"t": 0.0, "detections": [detection]}],
            {7: [detection]},
            anchor_time_local=0.0,
            anchor_bbox=anchor,
        )

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
