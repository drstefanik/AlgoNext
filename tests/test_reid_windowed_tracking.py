import importlib
import json
import os
import sys
import tempfile
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
                "cv2",
                "ultralytics",
                "app.workers.tracking",
                "app.reid.windowed_tracking",
            )
        }
        sys.modules["cv2"] = types.ModuleType("cv2")
        ultralytics = types.ModuleType("ultralytics")

        class FakeYOLO:
            def __init__(self, model_name):
                self.model_name = model_name
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
        cls.segment_extractions = []

        def extract_segment(*args, **kwargs):
            cls.segment_extractions.append(
                {
                    "start": args[2],
                    "accurate": bool(kwargs.get("accurate")),
                }
            )

        tracking._extract_segment = extract_segment
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
        cls.collection_profiles = []

        def collect(segment_path, **kwargs):
            name = Path(segment_path).stem
            window_number = int(name.split("_")[-1])
            cls.collection_profiles.append(
                {
                    "window_number": window_number,
                    "fps": kwargs.get("fps"),
                    "model": getattr(kwargs.get("model"), "model_name", None),
                }
            )
            track_map = cls.track_maps[window_number]
            default_times = (
                [15.0, 16.0, 17.0] if window_number == 2 else [0.0, 1.0, 2.0]
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
                for index in range(len(sample_times))
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
            cls._reid_module.windowed_tracking = cls._saved_reid_windowed_tracking

    def setUp(self):
        type(self).sample_times = {}
        type(self).collection_profiles = []
        type(self).segment_extractions = []

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

    def test_early_fallback_rewrites_persisted_legacy_asset_fail_closed(self):
        uploaded = {}

        def fallback(*_args, **_kwargs):
            return {
                "mode": "full_match_windowed",
                "tracking_key": "jobs/job-early-fallback/tracking/tracking.json",
                "tracking_url": "https://unsafe.example/raw-tracking.json",
                "segments": [
                    {
                        "selected_track_id": 7,
                        "selected_track_ids": [7, 8],
                        "identity_id": "legacy-identity-secret",
                        "identity_status": "ACCEPTED",
                        "anchor_matches": [{"anchor_id": 1, "local_track_id": 7}],
                        "reacquire_score": 0.99,
                        "reacquire_source": "manual_anchor",
                        "reacquire_metrics": {"candidate_id": 7},
                        "bboxes": [{"t": 1.0, **_bbox()}],
                    }
                ],
                "segments_total": 1,
                "segments_with_player": 1,
                "anchors_total": 1,
                "anchors_matched": 1,
                "anchor_matches": [{"anchor_id": 1, "local_track_id": 7}],
                "anchors_used": {"player_ref": {"track_id": 7}},
                "coverage_pct": 25.0,
                "largest_gap_sec": 10.0,
            }

        def upload_file(_client, bucket, path, key, content_type):
            uploaded.update(
                {
                    "bucket": bucket,
                    "key": key,
                    "content_type": content_type,
                    "payload": json.loads(Path(path).read_text()),
                }
            )

        with tempfile.TemporaryDirectory() as temporary_root, patch.dict(
            os.environ,
            {"S3_BUCKET": "tracking"},
            clear=False,
        ), patch(
            "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
            Path(temporary_root),
        ), patch.object(
            self.module.legacy,
            "_upload_file",
            side_effect=upload_file,
        ):
            output = self.module._fallback(
                fallback,
                "REID_ANCHOR_NORMALIZATION_FAILED",
                "job-early-fallback",
                analysis_attempt_id="attempt-a",
            )

        persisted = uploaded["payload"]
        self.assertFalse(persisted["tracking_success"])
        self.assertEqual(persisted["segments"], [])
        self.assertEqual(persisted["segments_total"], 0)
        self.assertEqual(persisted["bboxes"], [])
        self.assertEqual(persisted["anchors_matched"], 0)
        self.assertEqual(persisted["anchor_matches"], [])
        self.assertEqual(persisted["anchors_used"], {})
        persisted_json = json.dumps(persisted, sort_keys=True)
        self.assertNotIn("legacy-identity-secret", persisted_json)
        self.assertNotIn("local_track_id", persisted_json)
        self.assertNotIn("candidate_id", persisted_json)
        self.assertNotIn("selected_track_id", persisted_json)
        self.assertNotIn("selected_track_ids", persisted_json)
        self.assertNotIn("reacquire_score", persisted_json)
        self.assertNotIn("reacquire_source", persisted_json)
        self.assertNotIn("reacquire_metrics", persisted_json)
        self.assertEqual(
            output["tracking_key"],
            "jobs/job-early-fallback/attempts/attempt-a/tracking/tracking.json",
        )
        self.assertEqual(persisted["analysis_attempt_id"], "attempt-a")
        self.assertEqual(output["analysis_attempt_id"], "attempt-a")
        self.assertNotEqual(
            output["tracking_url"],
            "https://unsafe.example/raw-tracking.json",
        )

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

    def test_overlap_tracklet_excludes_same_raw_id_outside_the_boundary(self):
        linked = self.module._overlap_linked_detections(
            [
                {"t": 715.0, **_bbox()},
                {"t": 720.0, **_bbox()},
            ],
            [
                {
                    "t": 43.0,
                    "sample_index": 0,
                    "bbox": _bbox(),
                },
                {
                    "t": 55.0,
                    "sample_index": 1,
                    "bbox": _bbox(),
                },
                {
                    "t": 58.0,
                    "sample_index": 2,
                    "bbox": {
                        "x": 0.75,
                        "y": 0.20,
                        "w": 0.10,
                        "h": 0.20,
                    },
                },
                {
                    "t": 60.0,
                    "sample_index": 3,
                    "bbox": _bbox(),
                },
            ],
            window_start=660.0,
            tolerance_sec=0.6,
        )

        self.assertEqual(
            [item["sample_index"] for item in linked],
            [1, 3],
        )

    def test_overlap_tracklet_deduplicates_frames_and_excludes_weak_iou(self):
        weak = {
            "x": 0.25,
            "y": 0.20,
            "w": 0.10,
            "h": 0.20,
        }
        linked = self.module._overlap_linked_detections(
            [
                {"t": 715.0, **_bbox()},
                {"t": 720.0, **_bbox()},
            ],
            [
                {
                    "t": 55.0,
                    "sample_index": 1,
                    "bbox": weak,
                    "conf": 0.99,
                },
                {
                    "t": 55.0,
                    "sample_index": 1,
                    "bbox": _bbox(),
                    "conf": 0.90,
                },
                {
                    "t": 60.0,
                    "sample_index": 2,
                    "bbox": _bbox(),
                    "conf": 0.90,
                },
            ],
            window_start=660.0,
            tolerance_sec=0.6,
        )

        self.assertEqual(
            [item["sample_index"] for item in linked],
            [1, 2],
        )
        self.assertEqual(linked[0]["bbox"], _bbox())

    def test_manual_anchor_emits_only_its_verified_local_tracklet(self):
        samples = [
            {
                "t": time_sec,
                "detections": [
                    {
                        "track_id": 7,
                        "bbox": _bbox(),
                        "conf": 0.95,
                    }
                ],
            }
            for time_sec in (0.0, 5.0, 10.0)
        ]
        bboxes, link_bboxes, track_ids = self.module._stitch_manual_anchor_bboxes(
            [
                {
                    "anchor": {"anchor_id": 1, "t": 10.0, **_bbox()},
                    "track_id": 7,
                    "descriptor": AppearanceDescriptor(
                        vector=(1.0, 0.0),
                        sample_count=3,
                        quality=0.9,
                    ),
                }
            ],
            samples,
            fps=1,
            window_start=0.0,
            radius_sec=1.0,
        )

        self.assertEqual(track_ids, [7])
        self.assertEqual([item["t"] for item in bboxes], [10.0])
        self.assertEqual([item["t"] for item in link_bboxes], [10.0])

    def test_manual_anchor_tracklet_stops_before_spatial_id_switch(self):
        distant = {"x": 0.72, "y": 0.20, "w": 0.10, "h": 0.20}
        tracklet = self.module._anchor_tracklet_detections(
            [
                {
                    "t": 9.5,
                    "sample_index": 0,
                    "bbox": _bbox(),
                    "conf": 0.95,
                },
                {
                    "t": 10.0,
                    "sample_index": 1,
                    "bbox": _bbox(),
                    "conf": 0.95,
                },
                {
                    "t": 10.5,
                    "sample_index": 2,
                    "bbox": distant,
                    "conf": 0.95,
                },
            ],
            anchor_time_local=10.0,
            anchor_bbox=_bbox(),
            radius_sec=2.0,
        )

        self.assertEqual(
            [item["sample_index"] for item in tracklet],
            [0, 1],
        )

    def test_same_raw_id_anchor_tracklets_remain_separate(self):
        distant = {"x": 0.72, "y": 0.20, "w": 0.10, "h": 0.20}
        raw = [
            {
                "t": 10.0,
                "sample_index": 0,
                "bbox": _bbox(),
                "conf": 0.95,
            },
            {
                "t": 10.5,
                "sample_index": 1,
                "bbox": _bbox(),
                "conf": 0.95,
            },
            {
                "t": 20.0,
                "sample_index": 2,
                "bbox": distant,
                "conf": 0.95,
            },
            {
                "t": 20.5,
                "sample_index": 3,
                "bbox": distant,
                "conf": 0.95,
            },
        ]

        first = self.module._anchor_tracklet_detections(
            raw,
            anchor_time_local=10.0,
            anchor_bbox=_bbox(),
            radius_sec=2.0,
        )
        second = self.module._anchor_tracklet_detections(
            raw,
            anchor_time_local=20.0,
            anchor_bbox=distant,
            radius_sec=2.0,
        )

        self.assertEqual(
            [item["sample_index"] for item in first],
            [0, 1],
        )
        self.assertEqual(
            [item["sample_index"] for item in second],
            [2, 3],
        )

    def test_strong_overlap_without_two_linked_samples_fails_closed(self):
        raw = [
            {
                "t": 55.0,
                "sample_index": 0,
                "bbox": _bbox(),
                "conf": 0.95,
            },
            {
                "t": 30.0,
                "sample_index": 1,
                "bbox": _bbox(),
                "conf": 0.95,
            },
            {
                "t": 20.0,
                "sample_index": 2,
                "bbox": _bbox(),
                "conf": 0.95,
            },
        ]
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        with patch.dict(
            os.environ,
            {"PLAYER_REID_MIN_OVERLAP_LINK_SAMPLES": "2"},
            clear=False,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=lambda _path, track_map, track_ids: {
                track_id: descriptor if track_map.get(track_id) else None
                for track_id in track_ids
            },
        ):
            profiles, _ids, descriptors = self.module._build_candidate_profiles(
                Path("/tmp/window.mp4"),
                {326: raw},
                previous_bboxes=[{"t": 715.0, **_bbox()}],
                window_start=660.0,
                direction="backward",
                fps=2,
                strong_overlap_score=0.65,
            )

        self.assertEqual(len(profiles), 1)
        self.assertIsNone(profiles[0].overlap_score)
        self.assertEqual(profiles[0].detection_count, 0)
        self.assertIsNone(descriptors["326"])
        self.assertEqual(
            profiles[0].metadata["tracklet_scope"],
            "STRONG_OVERLAP_UNRESOLVED",
        )
        self.assertFalse(profiles[0].metadata["strong_overlap_unique"])
        self.assertEqual(profiles[0].metadata["overlap_previous_samples"], 1)

    def test_verified_strong_runner_is_not_hidden_by_candidate_cap(self):
        def raw_track(track_id, confidence):
            return [
                {
                    "t": local_time,
                    "sample_index": sample_index,
                    "track_id": track_id,
                    "bbox": _bbox(),
                    "conf": confidence,
                }
                for sample_index, local_time in enumerate((55.0, 56.0, 30.0))
            ]

        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=2,
            quality=0.9,
        )
        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_MAX_CANDIDATES": "1",
                "PLAYER_REID_MIN_OVERLAP_LINK_SAMPLES": "2",
            },
            clear=False,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=lambda _path, _track_map, track_ids: {
                track_id: descriptor for track_id in track_ids
            },
        ):
            profiles, _ids, _descriptors = self.module._build_candidate_profiles(
                Path("/tmp/window.mp4"),
                {
                    326: raw_track(326, 0.99),
                    927: raw_track(927, 0.50),
                },
                previous_bboxes=[
                    {"t": 715.0, **_bbox()},
                    {"t": 716.0, **_bbox()},
                ],
                window_start=660.0,
                direction="backward",
                fps=2,
                strong_overlap_score=0.65,
            )

        self.assertEqual({item.candidate_id for item in profiles}, {"326", "927"})
        self.assertTrue(
            all(item.metadata["strong_overlap_unique"] is False for item in profiles)
        )

    def test_two_hit_overlap_runner_blocks_physical_uniqueness(self):
        main = [
            {
                "t": local_time,
                "sample_index": sample_index,
                "track_id": 326,
                "bbox": _bbox(),
                "conf": 0.95,
            }
            for sample_index, local_time in enumerate((55.0, 56.0, 30.0))
        ]
        two_hit_runner = [
            {
                "t": local_time,
                "sample_index": sample_index,
                "track_id": 927,
                "bbox": _bbox(),
                "conf": 0.80,
            }
            for sample_index, local_time in enumerate((55.0, 56.0))
        ]
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=2,
            quality=0.9,
        )

        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_MIN_TRACK_HITS": "3",
                "PLAYER_REID_MIN_OVERLAP_LINK_SAMPLES": "2",
            },
            clear=False,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=lambda _path, _track_map, track_ids: {
                track_id: descriptor for track_id in track_ids
            },
        ):
            profiles, _ids, _descriptors = self.module._build_candidate_profiles(
                Path("/tmp/window.mp4"),
                {326: main, 927: two_hit_runner},
                previous_bboxes=[
                    {"t": 715.0, **_bbox()},
                    {"t": 716.0, **_bbox()},
                ],
                window_start=660.0,
                direction="backward",
                fps=2,
                strong_overlap_score=0.65,
            )

        self.assertEqual(
            {item.candidate_id for item in profiles},
            {"326", "927"},
        )
        self.assertTrue(
            all(
                item.metadata["strong_overlap_unique"] is False
                for item in profiles
            )
        )

    def test_unique_two_hit_physical_track_is_surfaceable(self):
        two_hit_track = [
            {
                "t": local_time,
                "sample_index": sample_index,
                "track_id": 927,
                "bbox": _bbox(),
                "conf": 0.80,
            }
            for sample_index, local_time in enumerate((55.0, 56.0))
        ]

        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_MIN_TRACK_HITS": "3",
                "PLAYER_REID_MIN_OVERLAP_LINK_SAMPLES": "2",
            },
            clear=False,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            return_value={927: None},
        ):
            profiles, ids, descriptors = self.module._build_candidate_profiles(
                Path("/tmp/window.mp4"),
                {927: two_hit_track},
                previous_bboxes=[
                    {"t": 715.0, **_bbox()},
                    {"t": 716.0, **_bbox()},
                ],
                window_start=660.0,
                direction="backward",
                fps=2,
                strong_overlap_score=0.65,
            )

        self.assertEqual([item.candidate_id for item in profiles], ["927"])
        self.assertEqual(ids, {"927": 927})
        self.assertEqual(descriptors, {"927": None})
        self.assertEqual(profiles[0].detection_count, 2)
        self.assertTrue(profiles[0].metadata["strong_overlap_unique"])
        self.assertEqual(
            profiles[0].metadata["tracklet_scope"],
            "MOTION_CONTINUOUS_STRONG_OVERLAP",
        )

    def test_raw_link_boxes_avoid_small_player_ema_lag_rejection(self):
        def small_box(x):
            return {"x": x, "y": 0.28, "w": 0.03125, "h": 0.15}

        raw_previous = [
            {"t": 715.0, **small_box(0.2780)},
            {"t": 715.5, **small_box(0.2830)},
            {"t": 716.0, **small_box(0.2880)},
        ]
        ema_lagged_previous = [
            {"t": 715.0, **small_box(0.2680)},
            {"t": 715.5, **small_box(0.2730)},
            {"t": 716.0, **small_box(0.2780)},
        ]
        current = [
            {
                "t": local_time,
                "sample_index": sample_index,
                "track_id": 326,
                "bbox": small_box(x),
                "conf": 0.95,
            }
            for sample_index, (local_time, x) in enumerate(
                ((55.25, 0.2805), (55.75, 0.2855), (56.25, 0.2905))
            )
        ]
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        with patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            return_value={326: descriptor},
        ):
            raw_profiles, _ids, _descriptors = self.module._build_candidate_profiles(
                Path("/tmp/window.mp4"),
                {326: current},
                previous_bboxes=raw_previous,
                window_start=660.0,
                direction="backward",
                fps=2,
                strong_overlap_score=0.65,
            )
            lagged_profiles, _ids, _descriptors = (
                self.module._build_candidate_profiles(
                    Path("/tmp/window.mp4"),
                    {326: current},
                    previous_bboxes=ema_lagged_previous,
                    window_start=660.0,
                    direction="backward",
                    fps=2,
                    strong_overlap_score=0.65,
                )
            )

        self.assertTrue(raw_profiles[0].metadata["strong_overlap_unique"])
        self.assertEqual(
            raw_profiles[0].metadata["tracklet_scope"],
            "MOTION_CONTINUOUS_STRONG_OVERLAP",
        )
        self.assertFalse(lagged_profiles[0].metadata["strong_overlap_unique"])

    def test_manual_anchor_display_keeps_raw_small_player_guard_sample(self):
        anchor_bbox = {
            "x": 0.278125,
            "y": 0.2875,
            "w": 0.03125,
            "h": 0.151388889,
        }
        window_start = 2145.0
        anchor_time = 2157.009
        samples = [
            {
                "t": local_time,
                "detections": [
                    {
                        "track_id": 7,
                        "bbox": {**anchor_bbox, "x": x},
                        "conf": 0.95,
                    }
                ],
            }
            for local_time, x in (
                (11.8, 0.276),
                (12.0, 0.278125),
                (12.2, 0.280),
            )
        ]
        lagged = [
            {
                "t": window_start + float(sample["t"]),
                **anchor_bbox,
                "x": 0.245,
                "conf": 0.95,
            }
            for sample in samples
        ]

        with patch.object(
            self.module.legacy,
            "_build_window_bboxes",
            return_value=(lagged, [], lagged[-1]),
        ):
            display, raw_links, track_ids = (
                self.module._stitch_manual_anchor_bboxes(
                    [
                        {
                            "anchor": {
                                "anchor_id": 1,
                                "t": anchor_time,
                                **anchor_bbox,
                            },
                            "track_id": 7,
                        }
                    ],
                    samples,
                    fps=5,
                    window_start=window_start,
                    radius_sec=2.0,
                )
            )

        nearest = min(
            display,
            key=lambda bbox: abs(float(bbox["t"]) - anchor_time),
        )
        self.assertAlmostEqual(nearest["x"], anchor_bbox["x"], places=6)
        self.assertNotAlmostEqual(nearest["x"], lagged[0]["x"], places=3)
        self.assertEqual(track_ids, [7])
        self.assertGreaterEqual(len(raw_links), 2)

    def test_near_threshold_overlap_runner_blocks_unique_margin_override(self):
        first = [
            {
                "t": local_time,
                "sample_index": sample_index,
                "track_id": 326,
                "bbox": _bbox(),
                "conf": 0.95,
            }
            for sample_index, local_time in enumerate((55.0, 56.0, 30.0))
        ]
        runner_bbox = {
            "x": 0.2214,
            "y": 0.20,
            "w": 0.10,
            "h": 0.20,
        }
        second = [
            {
                "t": local_time,
                "sample_index": sample_index,
                "track_id": 927,
                "bbox": runner_bbox,
                "conf": 0.95,
            }
            for sample_index, local_time in enumerate((55.0, 56.0, 30.0))
        ]
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=2,
            quality=0.9,
        )

        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_MAX_CANDIDATES": "1",
                "PLAYER_REID_MIN_OVERLAP_LINK_SAMPLES": "2",
                "PLAYER_REID_OVERLAP_UNIQUENESS_MARGIN": "0.05",
            },
            clear=False,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=lambda _path, _track_map, track_ids: {
                track_id: descriptor for track_id in track_ids
            },
        ):
            profiles, _ids, _descriptors = self.module._build_candidate_profiles(
                Path("/tmp/window.mp4"),
                {326: first, 927: second},
                previous_bboxes=[
                    {"t": 715.0, **_bbox()},
                    {"t": 716.0, **_bbox()},
                ],
                window_start=660.0,
                direction="backward",
                fps=2,
                strong_overlap_score=0.65,
            )

        strongest = max(
            profiles,
            key=lambda item: float(item.overlap_score or 0.0),
        )
        self.assertGreaterEqual(float(strongest.overlap_score or 0.0), 0.65)
        self.assertTrue(
            any(
                0.60 <= float(item.metadata["raw_overlap_score"] or 0.0) < 0.65
                for item in profiles
            )
        )
        self.assertFalse(strongest.metadata["strong_overlap_unique"])
        self.assertEqual({item.candidate_id for item in profiles}, {"326", "927"})

    def test_overlap_tracklet_extends_only_through_consecutive_motion(self):
        raw = [
            {
                "t": float(index),
                "sample_index": index,
                "track_id": 7,
                "bbox": _bbox(),
                "conf": 0.9,
            }
            for index in range(5)
        ]
        extended = self.module._tracklet_detections_from_overlap(
            raw,
            raw[:2],
            direction="forward",
            fps=1,
        )
        self.assertEqual(
            [item["sample_index"] for item in extended],
            [0, 1, 2, 3, 4],
        )

        missing_sample = [raw[0], raw[1], {**raw[2], "sample_index": 4}]
        stopped = self.module._tracklet_detections_from_overlap(
            missing_sample,
            missing_sample[:2],
            direction="forward",
            fps=1,
        )
        self.assertEqual(
            [item["sample_index"] for item in stopped],
            [0, 1],
        )

        jumped = [
            raw[0],
            raw[1],
            {
                **raw[2],
                "bbox": {"x": 0.75, "y": 0.65, "w": 0.1, "h": 0.2},
            },
        ]
        stopped = self.module._tracklet_detections_from_overlap(
            jumped,
            jumped[:2],
            direction="forward",
            fps=1,
        )
        self.assertEqual(
            [item["sample_index"] for item in stopped],
            [0, 1],
        )

    def test_disconnected_overlap_seeds_fail_closed(self):
        distant = {"x": 0.75, "y": 0.65, "w": 0.1, "h": 0.2}
        raw = [
            {
                "t": float(index),
                "sample_index": index,
                "track_id": 7,
                "bbox": _bbox() if index < 2 else distant,
                "conf": 0.9,
            }
            for index in range(4)
        ]

        extended = self.module._tracklet_detections_from_overlap(
            raw,
            [raw[0], raw[3]],
            direction="forward",
            fps=1,
        )

        self.assertEqual(extended, [])

    def test_manual_anchor_radius_finds_seed_without_truncating_component(self):
        raw = [
            {
                "t": float(index),
                "sample_index": index,
                "track_id": 7,
                "bbox": _bbox(),
                "conf": 0.9,
            }
            for index in range(6)
        ]

        component = self.module._anchor_tracklet_detections(
            raw,
            anchor_time_local=2.0,
            anchor_bbox=_bbox(),
            radius_sec=0.1,
        )

        self.assertEqual(
            [item["sample_index"] for item in component],
            [0, 1, 2, 3, 4, 5],
        )
        descriptor_component = self.module._anchor_descriptor_detections(
            raw,
            anchor_time_local=2.0,
            anchor_bbox=_bbox(),
            radius_sec=0.1,
        )
        self.assertEqual(
            [item["sample_index"] for item in descriptor_component],
            [2],
        )

    def test_continuous_tracklet_propagates_across_two_window_hops(self):
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        def raw_track(start, end):
            return [
                {
                    "t": float(local_time),
                    "sample_index": sample_index,
                    "track_id": 7,
                    "bbox": _bbox(),
                    "conf": 0.9,
                }
                for sample_index, local_time in enumerate(range(start, end + 1))
            ]

        with patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=lambda _path, _track_map, track_ids: {
                track_id: descriptor for track_id in track_ids
            },
        ):
            first, _ids, _descriptors = self.module._build_candidate_profiles(
                Path("/tmp/window-1.mp4"),
                {7: raw_track(4, 45)},
                previous_bboxes=[
                    {"t": float(time_sec), **_bbox()} for time_sec in (39, 40, 41)
                ],
                window_start=35.0,
                direction="forward",
                fps=1,
                strong_overlap_score=0.65,
            )
            first_tracklet = first[0].metadata["tracklet_detections"]
            first_bboxes = [
                {
                    "t": 35.0 + float(item["t"]),
                    **dict(item["bbox"]),
                }
                for item in first_tracklet
            ]
            second, _ids, _descriptors = self.module._build_candidate_profiles(
                Path("/tmp/window-2.mp4"),
                {7: raw_track(0, 45)},
                previous_bboxes=first_bboxes,
                window_start=70.0,
                direction="forward",
                fps=1,
                strong_overlap_score=0.65,
            )

        self.assertEqual(
            first[0].metadata["tracklet_scope"],
            "MOTION_CONTINUOUS_STRONG_OVERLAP",
        )
        self.assertEqual(
            second[0].metadata["tracklet_scope"],
            "MOTION_CONTINUOUS_STRONG_OVERLAP",
        )
        self.assertEqual(
            max(item["t"] for item in second[0].metadata["tracklet_detections"]),
            45.0,
        )

    def test_autonomous_proof_requires_two_samples_outside_all_anchor_windows(self):
        anchor = {
            "direction": "anchor",
            "window_start": 0.0,
            "window_end": 45.0,
            "identity_status": "ACCEPTED",
            "bboxes": [{"t": 39.0, **_bbox()}, {"t": 40.0, **_bbox()}],
        }
        adjacent = {
            "direction": "forward",
            "window_start": 35.0,
            "window_end": 80.0,
            "identity_status": "ACCEPTED",
            "bboxes": [{"t": 39.0, **_bbox()}, {"t": 40.0, **_bbox()}],
        }

        overlap_only = self.module._autonomous_tracking_evidence(
            [anchor, adjacent],
            fps=1,
        )
        one_new_sample = self.module._autonomous_tracking_evidence(
            [
                anchor,
                {**adjacent, "bboxes": [*adjacent["bboxes"], {"t": 47.0, **_bbox()}]},
            ],
            fps=1,
        )
        proven = self.module._autonomous_tracking_evidence(
            [
                anchor,
                {
                    **adjacent,
                    "bboxes": [
                        *adjacent["bboxes"],
                        {"t": 47.0, **_bbox()},
                        {"t": 48.0, **_bbox()},
                    ],
                },
            ],
            fps=1,
        )

        self.assertFalse(overlap_only["proven"])
        self.assertEqual(overlap_only["bboxes_count"], 0)
        self.assertFalse(one_new_sample["proven"])
        self.assertEqual(one_new_sample["bboxes_count"], 1)
        self.assertTrue(proven["proven"])
        self.assertEqual(proven["bboxes_count"], 2)

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
                "PLAYER_REID_REQUIRE_STRONG_OVERLAP": "0",
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
            segment["reid"].get("identity_id") for segment in output["segments"]
        }
        self.assertEqual(identities, {"job-job-1-selected-player"})
        self.assertTrue(
            all(
                segment["identity_status"] == "ACCEPTED"
                for segment in output["segments"]
            )
        )
        self.assertEqual(
            [segment["window_index"] for segment in output["segments"]],
            [0, 1, 2],
        )
        self.assertEqual(
            [segment["parent_window_index"] for segment in output["segments"]],
            [1, None, 1],
        )
        self.assertEqual(
            [segment["processing_direction"] for segment in output["segments"]],
            ["backward", "anchor", "forward"],
        )

    def test_physical_continuity_without_autonomous_descriptor_is_retained(self):
        type(self).track_maps = {
            1: {
                30: [
                    {
                        "t": time_sec,
                        "bbox": _bbox(),
                        "conf": 0.95,
                        "sample_index": index,
                    }
                    for index, time_sec in enumerate((33.0, 34.0, 35.0, 36.0))
                ]
            },
            2: {
                10: [
                    {
                        "t": time_sec,
                        "bbox": _bbox(),
                        "conf": 0.95,
                        "sample_index": index,
                    }
                    for index, time_sec in enumerate(
                        tuple(float(value) for value in range(16))
                    )
                ]
            },
            3: self._track([20]),
        }
        type(self).sample_times = {
            1: [33.0, 34.0, 35.0, 36.0],
            2: [float(value) for value in range(16)],
        }
        manual_descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        def descriptors(path, _track_map, track_ids):
            if Path(path).stem == "window_0002":
                return {track_id: manual_descriptor for track_id in track_ids}
            return {track_id: None for track_id in track_ids}

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
                "job-physical-only",
                "/tmp/input.mp4",
                {"t": 50.0, **_bbox()},
                [],
                video_duration_sec=115.0,
                fps=2,
            )

        self.assertTrue(output["tracking_success"])
        self.assertEqual(output["reid_summary"]["accepted_associations"], 1)
        self.assertEqual(output["reid_summary"]["profile_samples"], 3)
        self.assertEqual(output["autonomous_bboxes_count"], 2)
        backward = output["segments"][0]
        self.assertEqual(backward["identity_status"], "ACCEPTED")
        self.assertEqual(backward["reid"]["descriptor"]["sample_count"], 0)
        self.assertIn(
            "PHYSICAL_CONTINUITY_ONLY",
            backward["reid"]["reason_codes"],
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
                "AMBIGUOUS_CANDIDATE_MARGIN" in segment["reid"]["reason_codes"]
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
        self.assertEqual(
            [
                (
                    segment["window_index"],
                    segment["parent_window_index"],
                    segment["processing_direction"],
                )
                for segment in output["segments"]
            ],
            [
                (0, None, "anchor"),
                (1, 0, "forward"),
                (2, None, "anchor"),
            ],
        )
        self.assertEqual(output["anchor_reacquisitions"], 1)
        self.assertEqual(output["anchors_total"], 2)
        self.assertEqual(output["anchors_matched"], 2)
        self.assertEqual(
            [item["status"] for item in output["anchor_matches"]],
            ["MATCHED", "MATCHED"],
        )
        self.assertEqual(output["reid_summary"]["anchor_reacquisitions"], 1)

    def test_late_anchor_propagates_backward_on_production_window_schedule(self):
        duration = 5931.775
        windows = []
        start = 0.0
        while start < duration:
            windows.append((round(start, 3), round(min(duration, start + 60.0), 3)))
            start += 55.0
        manual_descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        def collect(segment_path, **_kwargs):
            window_number = int(Path(segment_path).stem.split("_")[-1])
            window_index = window_number - 1
            if window_index == 13:
                local_times = [float(value) for value in range(13)]
            elif window_index == 39:
                local_times = [float(value) for value in range(14)]
            elif window_index == 38:
                local_times = [53.0, 54.0, 55.0, 56.0]
            else:
                local_times = [0.0, 1.0, 2.0]
            track_id = 1000 + window_index
            detections = [
                {
                    "t": time_sec,
                    "sample_index": sample_index,
                    "track_id": track_id,
                    "bbox": _bbox(),
                    "conf": 0.95,
                }
                for sample_index, time_sec in enumerate(local_times)
            ]
            samples = [
                {
                    "t": item["t"],
                    "detections": [
                        {
                            "track_id": track_id,
                            "bbox": dict(item["bbox"]),
                            "conf": item["conf"],
                        }
                    ],
                }
                for item in detections
            ]
            return samples, {track_id: detections}

        def descriptors(path, _track_map, track_ids):
            window_number = int(Path(path).stem.split("_")[-1])
            if window_number in {14, 40}:
                return {track_id: manual_descriptor for track_id in track_ids}
            return {track_id: None for track_id in track_ids}

        with patch.dict(
            os.environ,
            {
                "S3_ACCESS_KEY": "key",
                "S3_SECRET_KEY": "secret",
                "S3_BUCKET": "bucket",
            },
            clear=False,
        ), patch.object(
            self.module.legacy,
            "iter_windows",
            return_value=windows,
        ), patch.object(
            self.module.legacy,
            "_collect_window_samples",
            side_effect=collect,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=descriptors,
        ):
            output = self.module.track_player_windowed_reid(
                "job-production-anchor-schedule",
                "/tmp/input.mp4",
                {"t": 719.003, **_bbox()},
                [
                    {
                        "frame_time_sec": 719.003,
                        "frame_key": "frame_0004.jpg",
                        **_bbox(),
                    },
                    {
                        "frame_time_sec": 2157.009,
                        "frame_key": "frame_0012.jpg",
                        **_bbox(),
                    },
                ],
                video_duration_sec=duration,
                window_sec=60.0,
                overlap_sec=5.0,
                fps=2,
            )

        late_root = output["segments"][39]
        backward = output["segments"][38]
        self.assertEqual(late_root["direction"], "anchor")
        self.assertIsNone(late_root["parent_window_index"])
        self.assertEqual(backward["identity_status"], "ACCEPTED")
        self.assertEqual(backward["processing_direction"], "backward")
        self.assertEqual(backward["parent_window_index"], 39)
        self.assertEqual(output["anchors_matched"], 2)
        self.assertEqual(output["windows_processed"], len(windows))
        self.assertGreaterEqual(output["autonomous_bboxes_count"], 2)
        self.assertNotIn(
            "unclaimed",
            {
                str(segment.get("processing_direction"))
                for segment in output["segments"]
            },
        )
        for segment in output["segments"]:
            if segment["direction"] == "anchor":
                self.assertIsNone(segment["parent_window_index"])
                continue
            self.assertIn(segment["direction"], {"forward", "backward"})
            self.assertEqual(
                segment["processing_direction"],
                segment["direction"],
            )
            expected_parent = segment["window_index"] + (
                -1 if segment["direction"] == "forward" else 1
            )
            self.assertEqual(segment["parent_window_index"], expected_parent)

    def test_midpoint_roots_require_the_same_continuous_tracklet_component(self):
        windows = [
            (0.0, 10.0),
            (8.0, 18.0),
            (16.0, 26.0),
            (24.0, 34.0),
            (32.0, 42.0),
        ]
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        def run(right_indices):
            midpoint_detections = [
                {
                    "t": float(index),
                    "bbox": _bbox(),
                    "conf": 0.95,
                    "sample_index": index,
                }
                for index in range(6)
            ]
            type(self).track_maps = {
                1: self._track([1]),
                2: self._track([10]),
                3: {7: midpoint_detections},
                4: self._track([30]),
                5: self._track([5]),
            }
            type(self).sample_times = {
                2: [4.0, 5.0, 6.0],
                3: [float(index) for index in range(6)],
                4: [4.0, 5.0, 6.0],
            }

            def descriptors(_path, _track_map, track_ids):
                return {track_id: descriptor for track_id in track_ids}

            def profiles(
                segment_path,
                track_map,
                *,
                direction,
                **_kwargs,
            ):
                window_number = int(Path(segment_path).stem.split("_")[-1])
                if window_number != 3:
                    return [], {}, {}
                indices = (
                    (0, 1, 2)
                    if direction == "forward"
                    else tuple(right_indices)
                )
                detections = tuple(
                    {
                        **dict(track_map[7][index]),
                        "sample_index": index,
                    }
                    for index in indices
                )
                candidate = self.module.CandidateProfile(
                    candidate_id="7",
                    descriptor=None,
                    overlap_score=1.0,
                    geometry_score=1.0,
                    detection_count=len(detections),
                    metadata={
                        "local_track_id": 7,
                        "tracklet_scope": (
                            "MOTION_CONTINUOUS_STRONG_OVERLAP"
                        ),
                        "tracklet_sample_indices": indices,
                        "tracklet_detections": detections,
                        "overlap_link_samples": len(detections),
                        "overlap_previous_samples": len(detections),
                        "strong_overlap_unique": True,
                        "raw_overlap_score": 1.0,
                    },
                )
                return [candidate], {"7": 7}, {"7": None}

            with patch.dict(
                os.environ,
                {
                    "S3_ACCESS_KEY": "key",
                    "S3_SECRET_KEY": "secret",
                    "S3_BUCKET": "bucket",
                },
                clear=False,
            ), patch.object(
                self.module.legacy,
                "iter_windows",
                return_value=windows,
            ), patch.object(
                self.module,
                "_extract_descriptors_for_tracks",
                side_effect=descriptors,
            ), patch.object(
                self.module,
                "_build_candidate_profiles",
                side_effect=profiles,
            ):
                return self.module.track_player_windowed_reid(
                    "job-midpoint-component",
                    "/tmp/input.mp4",
                    {"t": 13.0, **_bbox()},
                    [
                        {"frame_time_sec": 13.0, "frame_key": "left", **_bbox()},
                        {"frame_time_sec": 29.0, "frame_key": "right", **_bbox()},
                    ],
                    video_duration_sec=42.0,
                    window_sec=10.0,
                    overlap_sec=2.0,
                    fps=2,
                )

        disconnected = run((3, 4, 5))
        midpoint = disconnected["segments"][2]
        self.assertEqual(midpoint["identity_status"], "ABSTAINED")
        self.assertIsNone(midpoint["selected_track_id"])
        self.assertIn(
            "CONFLICTING_CONTINUITY_ROOTS",
            midpoint["reid"]["reason_codes"],
        )

        continuous = run((0, 1, 2))
        midpoint = continuous["segments"][2]
        self.assertEqual(midpoint["identity_status"], "ACCEPTED")
        self.assertEqual(midpoint["selected_track_id"], 7)
        self.assertIn(
            "CONVERGED_CONTINUITY_ROOTS",
            midpoint["reid"]["reason_codes"],
        )

    def test_odd_root_boundary_reconciles_adjacent_overlap_claims(self):
        windows = [
            (0.0, 10.0),
            (8.0, 18.0),
            (16.0, 26.0),
            (24.0, 34.0),
            (32.0, 42.0),
            (40.0, 50.0),
        ]
        left_bbox = _bbox()
        right_bbox = {"x": 0.72, "y": 0.2, "w": 0.1, "h": 0.2}
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

        def track(track_id, bbox, times):
            return {
                track_id: [
                    {
                        "t": float(time_sec),
                        "bbox": dict(bbox),
                        "conf": 0.95,
                        "sample_index": sample_index,
                    }
                    for sample_index, time_sec in enumerate(times)
                ]
            }

        def run(boundary_bbox):
            root_times = [5.0 + 0.5 * index for index in range(9)]
            middle_times = [0.5 * index for index in range(21)]
            right_root_times = [1.0 + 0.5 * index for index in range(9)]
            type(self).track_maps = {
                1: {},
                2: track(10, left_bbox, root_times),
                3: track(20, left_bbox, middle_times),
                4: track(30, boundary_bbox, middle_times),
                5: track(40, boundary_bbox, right_root_times),
                6: {},
            }
            type(self).sample_times = {
                2: root_times,
                3: middle_times,
                4: middle_times,
                5: right_root_times,
            }

            def descriptors(path, _track_map, track_ids):
                window_number = int(Path(path).stem.split("_")[-1])
                return {
                    track_id: (
                        descriptor if window_number in {2, 5} else None
                    )
                    for track_id in track_ids
                }

            with patch.dict(
                os.environ,
                {
                    "S3_ACCESS_KEY": "key",
                    "S3_SECRET_KEY": "secret",
                    "S3_BUCKET": "bucket",
                },
                clear=False,
            ), patch.object(
                self.module.legacy,
                "iter_windows",
                return_value=windows,
            ), patch.object(
                self.module,
                "_extract_descriptors_for_tracks",
                side_effect=descriptors,
            ):
                return self.module.track_player_windowed_reid(
                    "job-odd-boundary",
                    "/tmp/input.mp4",
                    {"t": 15.0, **left_bbox},
                    [
                        {
                            "frame_time_sec": 15.0,
                            "frame_key": "left-root",
                            **left_bbox,
                        },
                        {
                            "frame_time_sec": 35.0,
                            "frame_key": "right-root",
                            **boundary_bbox,
                        },
                    ],
                    video_duration_sec=50.0,
                    window_sec=10.0,
                    overlap_sec=2.0,
                    fps=2,
                )

        converged = run(left_bbox)
        self.assertEqual(
            [
                converged["segments"][index]["identity_status"]
                for index in (2, 3)
            ],
            ["ACCEPTED", "ACCEPTED"],
        )

        conflicted = run(right_bbox)
        for index in (2, 3):
            segment = conflicted["segments"][index]
            self.assertEqual(segment["identity_status"], "ABSTAINED")
            self.assertIn(
                "CONFLICTING_CONTINUITY_ROOTS",
                segment["reid"]["reason_codes"],
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

    def test_anchor_window_uses_dedicated_high_fidelity_profile(self):
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

        with patch.dict(
            os.environ,
            {
                "S3_ACCESS_KEY": "key",
                "S3_SECRET_KEY": "secret",
                "S3_BUCKET": "bucket",
                "PLAYER_REID_ANCHOR_FPS": "5",
                "PLAYER_REID_ANCHOR_DETECTOR_MODEL": "yolo11s.pt",
            },
            clear=False,
        ), patch.object(
            self.module,
            "_extract_descriptors_for_tracks",
            side_effect=lambda _path, _track_map, track_ids: {
                track_id: descriptor for track_id in track_ids
            },
        ):
            output = self.module.track_player_windowed_reid(
                "job-anchor-profile",
                "/tmp/input.mp4",
                {"t": 50.0, **_bbox()},
                [],
                video_duration_sec=115.0,
                fps=1,
                detector_model="yolo11n.pt",
            )

        self.assertFalse(output["tracking_success"])
        self.assertEqual(output["tracking_status"], "ANCHOR_ONLY")
        self.assertEqual(output["tracking_scope_status"], "ANCHOR_ONLY")
        self.assertEqual(output["autonomous_segments_with_player"], 0)
        profiles = {
            item["window_number"]: (item["fps"], item["model"])
            for item in type(self).collection_profiles
        }
        self.assertEqual(profiles[2], (5, "yolo11s.pt"))
        self.assertEqual(profiles[1], (1, "yolo11n.pt"))
        self.assertEqual(profiles[3], (1, "yolo11n.pt"))
        self.assertEqual(
            output["anchor_acquisition"]["detector_model"],
            "yolo11s.pt",
        )
        extraction_profiles = {
            item["start"]: item["accurate"] for item in type(self).segment_extractions
        }
        self.assertTrue(extraction_profiles[35.0])
        self.assertFalse(extraction_profiles[0.0])
        self.assertFalse(extraction_profiles[70.0])

    def test_unmatched_primary_uses_later_anchor_as_seed(self):
        primary = {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2}
        late = {"x": 0.7, "y": 0.2, "w": 0.1, "h": 0.2}
        distant = {"x": 0.75, "y": 0.65, "w": 0.1, "h": 0.2}
        type(self).track_maps = {
            1: self._track_boxes({10: distant}),
            2: self._track_boxes({20: late}),
            3: self._track_boxes({30: late}),
        }
        type(self).sample_times = {3: [8.0, 9.0, 10.0]}
        descriptor = AppearanceDescriptor(
            vector=(1.0, 0.0),
            sample_count=3,
            quality=0.9,
        )

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
            side_effect=lambda _path, _track_map, track_ids: {
                track_id: descriptor for track_id in track_ids
            },
        ):
            output = self.module.track_player_windowed_reid(
                "job-secondary-seed",
                "/tmp/input.mp4",
                {"t": 1.0, **primary},
                [
                    {
                        "frame_time_sec": 1.0,
                        "frame_key": "primary",
                        **primary,
                    },
                    {
                        "frame_time_sec": 78.0,
                        "frame_key": "late",
                        **late,
                    },
                ],
                video_duration_sec=115.0,
            )

        self.assertFalse(output["tracking_success"])
        self.assertEqual(output["tracking_status"], "ANCHOR_ONLY")
        self.assertEqual(output["anchor_acquisition"]["seed_anchor_id"], 2)
        self.assertEqual(output["anchor_acquisition"]["seed_window_index"], 2)
        self.assertEqual(
            [item["status"] for item in output["anchor_matches"]],
            ["TRACK_NOT_FOUND", "MATCHED"],
        )
        matched_anchor_windows = {
            int(item["window_index"])
            for item in output["anchor_matches"]
            if item["status"] == "MATCHED"
        }
        emitted_anchor_windows = {
            int(segment["window_index"])
            for segment in output["segments"]
            if segment["direction"] == "anchor"
        }
        self.assertEqual(emitted_anchor_windows, matched_anchor_windows)
        self.assertEqual(output["segments"][0]["direction"], "backward")
        self.assertEqual(output["segments"][0]["parent_window_index"], 1)

    def test_all_unmatched_anchors_stop_without_legacy_fallback(self):
        primary = {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2}
        late = {"x": 0.7, "y": 0.2, "w": 0.1, "h": 0.2}
        distant = {"x": 0.75, "y": 0.65, "w": 0.1, "h": 0.2}
        type(self).track_maps = {
            1: self._track_boxes({10: distant}),
            2: self._track_boxes({20: distant}),
            3: self._track_boxes({30: distant}),
        }
        type(self).sample_times = {3: [8.0, 9.0, 10.0]}
        fallback_calls = []

        with patch.dict(
            os.environ,
            {
                "S3_ACCESS_KEY": "key",
                "S3_SECRET_KEY": "secret",
                "S3_BUCKET": "bucket",
            },
            clear=False,
        ):
            output = self.module.track_player_windowed_reid(
                "job-no-anchor",
                "/tmp/input.mp4",
                {"t": 1.0, **primary},
                [
                    {
                        "frame_time_sec": 1.0,
                        "frame_key": "primary",
                        **primary,
                    },
                    {
                        "frame_time_sec": 78.0,
                        "frame_key": "late",
                        **late,
                    },
                ],
                video_duration_sec=115.0,
                fallback=lambda *_args, **_kwargs: fallback_calls.append(True),
            )

        self.assertFalse(output["tracking_success"])
        self.assertEqual(output["tracking_status"], "ANCHOR_NOT_FOUND")
        self.assertEqual(output["action_required"], "RESELECT_PLAYER")
        self.assertEqual(output["windows_processed"], 2)
        self.assertIsNone(output["largest_gap_sec"])
        self.assertEqual(fallback_calls, [])
        self.assertNotIn(
            2,
            {item["window_number"] for item in type(self).collection_profiles},
        )

    def test_anchor_descriptor_error_returns_retry_without_legacy_fallback(self):
        type(self).track_maps = {
            1: self._track([30]),
            2: self._track([10]),
            3: self._track([20]),
        }
        fallback_calls = []

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
            side_effect=RuntimeError("descriptor failure"),
        ):
            output = self.module.track_player_windowed_reid(
                "job-anchor-descriptor-error",
                "/tmp/input.mp4",
                {"t": 50.0, **_bbox()},
                [],
                video_duration_sec=115.0,
                fallback=lambda *_args, **_kwargs: fallback_calls.append(True),
            )

        self.assertFalse(output["tracking_success"])
        self.assertEqual(
            output["tracking_status"],
            "ANCHOR_ACQUISITION_ERROR",
        )
        self.assertEqual(output["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(fallback_calls, [])

    def test_mixed_anchor_miss_and_acquisition_error_requires_retry(self):
        primary = {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2}
        late = {"x": 0.7, "y": 0.2, "w": 0.1, "h": 0.2}
        distant = {"x": 0.75, "y": 0.65, "w": 0.1, "h": 0.2}
        type(self).track_maps = {
            1: self._track_boxes({10: distant}),
            2: self._track_boxes({20: distant}),
            3: self._track_boxes({30: late}),
        }
        type(self).sample_times = {3: [8.0, 9.0, 10.0]}

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
            side_effect=RuntimeError("descriptor failure"),
        ):
            output = self.module.track_player_windowed_reid(
                "job-mixed-anchor-failure",
                "/tmp/input.mp4",
                {"t": 1.0, **primary},
                [
                    {
                        "frame_time_sec": 1.0,
                        "frame_key": "primary",
                        **primary,
                    },
                    {
                        "frame_time_sec": 78.0,
                        "frame_key": "late",
                        **late,
                    },
                ],
                video_duration_sec=115.0,
            )

        self.assertFalse(output["tracking_success"])
        self.assertEqual(output["tracking_status"], "ANCHOR_ACQUISITION_ERROR")
        self.assertEqual(output["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(
            [item["status"] for item in output["anchor_matches"]],
            ["TRACK_NOT_FOUND", "DESCRIPTOR_PROCESSING_FAILED"],
        )

    def test_collection_failure_always_resets_tracker_state(self):
        with patch.dict(
            os.environ,
            {
                "S3_ACCESS_KEY": "key",
                "S3_SECRET_KEY": "secret",
                "S3_BUCKET": "bucket",
            },
            clear=False,
        ), patch.object(
            self.module.legacy,
            "_collect_window_samples",
            side_effect=RuntimeError("collector failed"),
        ), patch.object(
            self.module,
            "_reset_tracker",
        ) as reset_tracker:
            output = self.module.track_player_windowed_reid(
                "job-collector-reset",
                "/tmp/input.mp4",
                {"t": 50.0, **_bbox()},
                [],
                video_duration_sec=115.0,
            )

        self.assertFalse(output["tracking_success"])
        self.assertEqual(output["action_required"], "RETRY_ANALYSIS")
        reset_tracker.assert_called_once()

    def test_post_collection_error_fails_closed_without_legacy_fallback(self):
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
        fallback_calls = []

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
            side_effect=lambda _path, _track_map, track_ids: {
                track_id: descriptor for track_id in track_ids
            },
        ), patch.object(
            self.module,
            "_build_candidate_profiles",
            side_effect=RuntimeError("descriptor pipeline failed"),
        ):
            output = self.module.track_player_windowed_reid(
                "job-window-processing-error",
                "/tmp/input.mp4",
                {"t": 50.0, **_bbox()},
                [],
                video_duration_sec=115.0,
                fallback=lambda *_args, **_kwargs: fallback_calls.append(True),
            )

        self.assertFalse(output["tracking_success"])
        self.assertEqual(output["tracking_status"], "WINDOW_PROCESSING_ERROR")
        self.assertEqual(output["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(output["anchors_matched"], 1)
        self.assertEqual(fallback_calls, [])


if __name__ == "__main__":
    unittest.main()
