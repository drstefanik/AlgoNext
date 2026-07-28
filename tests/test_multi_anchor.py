import ast
import unittest
from pathlib import Path

from app.workers.multi_anchor import (
    anchors_for_window,
    assign_anchors_to_windows,
    compute_tracking_window,
    normalize_anchors,
    select_track_id_at_time,
    select_window_track,
    select_window_tracks,
)


class MultiAnchorTests(unittest.TestCase):
    @staticmethod
    def _load_tracking_bbox_builder():
        """Load the production bbox builder without its heavyweight imports."""

        tracking_path = Path(__file__).resolve().parents[1] / "app/workers/tracking.py"
        tree = ast.parse(tracking_path.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_smooth_bbox", "_build_window_bboxes"}
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *functions,
            ],
            type_ignores=[],
        )
        namespace = {
            "select_track_id_at_time": select_track_id_at_time,
        }
        exec(
            compile(
                ast.fix_missing_locations(module),
                str(tracking_path),
                "exec",
            ),
            namespace,
        )
        return namespace["_build_window_bboxes"]

    def test_01_normalizes_flat_selection(self):
        anchors = normalize_anchors(
            [
                {
                    "frame_time_sec": 12.5,
                    "frame_key": "f1.jpg",
                    "x": 0.1,
                    "y": 0.2,
                    "w": 0.1,
                    "h": 0.2,
                }
            ]
        )
        self.assertEqual(anchors[0]["t"], 12.5)
        self.assertEqual(anchors[0]["frame_key"], "f1.jpg")

    def test_02_normalizes_nested_bbox(self):
        anchors = normalize_anchors(
            [{"time_sec": 8, "bbox": {"x": 0.2, "y": 0.3, "w": 0.2, "h": 0.3}}]
        )
        self.assertEqual(anchors[0]["x"], 0.2)

    def test_03_discards_invalid_items(self):
        anchors = normalize_anchors(
            [
                None,
                {"time_sec": -1, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                {"time_sec": 2, "x": 0.95, "y": 0.1, "w": 0.2, "h": 0.1},
            ]
        )
        self.assertEqual(anchors, [])

    def test_04_deduplicates(self):
        item = {"time_sec": 2, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}
        self.assertEqual(len(normalize_anchors([item, dict(item)])), 1)

    def test_05_sorts_and_caps(self):
        items = [
            {"time_sec": t, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}
            for t in [9, 2, 7, 3, 8, 1]
        ]
        self.assertEqual(
            [item["t"] for item in normalize_anchors(items)], [1, 2, 3, 7, 8]
        )

    def test_06_filters_anchors_for_window(self):
        anchors = normalize_anchors(
            [
                {"time_sec": 10, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1},
                {"time_sec": 50, "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1},
            ]
        )
        self.assertEqual([a["t"] for a in anchors_for_window(anchors, 0, 45)], [10])

    def test_07_anchor_selects_correct_track(self):
        anchors = normalize_anchors(
            [
                {
                    "time_sec": 40,
                    "frame_key": "f40",
                    "x": 0.45,
                    "y": 0.2,
                    "w": 0.10,
                    "h": 0.25,
                }
            ]
        )
        track_map = {
            1: [
                {
                    "t": 5,
                    "bbox": {"x": 0.10, "y": 0.2, "w": 0.10, "h": 0.25},
                    "conf": 0.9,
                }
            ],
            7: [
                {
                    "t": 5,
                    "bbox": {"x": 0.45, "y": 0.2, "w": 0.10, "h": 0.25},
                    "conf": 0.9,
                }
            ],
        }
        match = select_window_track(track_map, anchors, window_start=35, window_end=80)
        self.assertEqual(match["track_id"], 7)
        self.assertEqual(match["source"], "anchor")

    def test_08_anchor_overrides_continuity(self):
        anchors = normalize_anchors(
            [{"time_sec": 40, "x": 0.70, "y": 0.2, "w": 0.10, "h": 0.25}]
        )
        track_map = {
            1: [
                {"t": 0, "bbox": {"x": 0.10, "y": 0.2, "w": 0.10, "h": 0.25}},
                {"t": 5, "bbox": {"x": 0.12, "y": 0.2, "w": 0.10, "h": 0.25}},
            ],
            2: [{"t": 5, "bbox": {"x": 0.70, "y": 0.2, "w": 0.10, "h": 0.25}}],
        }
        match = select_window_track(
            track_map,
            anchors,
            window_start=35,
            window_end=80,
            previous_bbox={"x": 0.10, "y": 0.2, "w": 0.10, "h": 0.25},
        )
        self.assertEqual(match["track_id"], 2)
        self.assertEqual(match["source"], "anchor")

    def test_09_continuity_fallback(self):
        match = select_window_track(
            {4: [{"t": 0, "bbox": {"x": 0.21, "y": 0.20, "w": 0.10, "h": 0.25}}]},
            [],
            window_start=45,
            window_end=90,
            previous_bbox={"x": 0.20, "y": 0.20, "w": 0.10, "h": 0.25},
        )
        self.assertEqual(match["track_id"], 4)
        self.assertEqual(match["source"], "continuity")

    def test_10_rejects_bad_continuity(self):
        match = select_window_track(
            {4: [{"t": 0, "bbox": {"x": 0.80, "y": 0.70, "w": 0.03, "h": 0.04}}]},
            [],
            window_start=45,
            window_end=90,
            previous_bbox={"x": 0.20, "y": 0.20, "w": 0.10, "h": 0.25},
        )
        self.assertIsNone(match["track_id"])

    def test_11_reacquires_after_zero_coverage_window(self):
        anchors = normalize_anchors(
            [
                {
                    "time_sec": 82,
                    "frame_key": "late",
                    "x": 0.60,
                    "y": 0.20,
                    "w": 0.10,
                    "h": 0.25,
                }
            ]
        )
        first = select_window_track({}, anchors, window_start=0, window_end=45)
        later = select_window_track(
            {9: [{"t": 2, "bbox": {"x": 0.60, "y": 0.20, "w": 0.10, "h": 0.25}}]},
            anchors,
            window_start=80,
            window_end=125,
            previous_bbox=None,
        )
        self.assertIsNone(first["track_id"])
        self.assertEqual(later["track_id"], 9)
        self.assertEqual(later["source"], "anchor")

    def test_12_window_spans_all_anchors(self):
        self.assertEqual(
            compute_tracking_window([100, 300], 500, 60, 60), (40.0, 320.0)
        )

    def test_13_window_clamps_to_video_duration(self):
        self.assertEqual(compute_tracking_window([490], 500, 60, 60), (430.0, 70.0))

    def test_14_preserves_timestamp_zero(self):
        anchors = normalize_anchors(
            [{"frame_time_sec": 0, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2}]
        )
        self.assertEqual(anchors[0]["t"], 0.0)

    def test_15_pipeline_forwards_multi_anchor_reid_diagnostics(self):
        pipeline_path = Path(__file__).resolve().parents[1] / "app/workers/pipeline.py"
        source = pipeline_path.read_text(encoding="utf-8")
        start = source.index(
            '        if tracking_output.get("mode") == "full_match_windowed":'
        )
        end = source.index("        else:", start)
        full_match_payload = source[start:end]
        compact_payload = "".join(full_match_payload.split())
        for field_name in (
            "identity_mode",
            "anchor_reacquisitions",
            "anchors_total",
            "anchors_matched",
            "anchor_matches",
            "anchors_used",
            "reid_summary",
            "runtime_profile",
            "autonomous_segments_with_player",
            "autonomous_bboxes_count",
            "tracking_scope_status",
        ):
            with self.subTest(field_name=field_name):
                self.assertIn(
                    f'"{field_name}":tracking_output.get("{field_name}")',
                    compact_payload,
                )

    def test_16_assigns_each_anchor_to_one_overlapping_window(self):
        anchors = normalize_anchors(
            [
                {"frame_time_sec": 0, "x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2},
                {"frame_time_sec": 42, "x": 0.2, "y": 0.1, "w": 0.1, "h": 0.2},
                {"frame_time_sec": 78, "x": 0.3, "y": 0.1, "w": 0.1, "h": 0.2},
            ]
        )
        assigned = assign_anchors_to_windows(
            anchors,
            [(0.0, 45.0), (35.0, 80.0), (70.0, 115.0)],
        )

        self.assertEqual(
            [[anchor["t"] for anchor in bucket] for bucket in assigned],
            [[0.0], [42.0], [78.0]],
        )
        self.assertEqual(sum(len(bucket) for bucket in assigned), len(anchors))
        tracking_source = (
            Path(__file__).resolve().parents[1] / "app/workers/tracking.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "anchors_by_window = assign_anchors_to_windows(anchors, windows)",
            tracking_source,
        )
        self.assertIn("anchors_by_window[idx - 1]", tracking_source)

    def test_17_stitches_distinct_local_ids_within_one_window(self):
        anchors = normalize_anchors(
            [
                {
                    "frame_time_sec": 40,
                    "frame_key": "before-cut",
                    "x": 0.20,
                    "y": 0.20,
                    "w": 0.10,
                    "h": 0.25,
                },
                {
                    "frame_time_sec": 60,
                    "frame_key": "after-cut",
                    "x": 0.70,
                    "y": 0.20,
                    "w": 0.10,
                    "h": 0.25,
                },
            ]
        )
        track_map = {
            10: [
                {
                    "t": 5,
                    "bbox": {"x": 0.20, "y": 0.20, "w": 0.10, "h": 0.25},
                    "conf": 0.9,
                }
            ],
            11: [
                {
                    "t": 25,
                    "bbox": {"x": 0.70, "y": 0.20, "w": 0.10, "h": 0.25},
                    "conf": 0.9,
                }
            ],
        }

        match = select_window_tracks(
            track_map,
            anchors,
            window_start=35,
            window_end=80,
        )

        self.assertEqual(match["selected_track_ids"], [10, 11])
        self.assertEqual(
            [item["anchor_frame_key"] for item in match["anchor_matches"]],
            ["before-cut", "after-cut"],
        )
        self.assertEqual(
            select_track_id_at_time(match["anchor_matches"], 49.999),
            10,
        )
        self.assertEqual(
            select_track_id_at_time(match["anchor_matches"], 50.0),
            11,
        )
        tracking_source = (
            Path(__file__).resolve().parents[1] / "app/workers/tracking.py"
        ).read_text(encoding="utf-8")
        self.assertIn("match = select_window_tracks(", tracking_source)
        self.assertIn(
            'anchor_matches=match.get("anchor_matches")',
            tracking_source,
        )
        self.assertIn(
            "if active_track_id != previous_active_track_id:\n"
            "            smoothed = None\n"
            "            previous_active_track_id = active_track_id",
            tracking_source,
        )
        build_window_bboxes = self._load_tracking_bbox_builder()
        samples = [
            {
                "t": local_time,
                "detections": [
                    {
                        "track_id": 10,
                        "bbox": {"x": 0.20, "y": 0.20, "w": 0.10, "h": 0.25},
                        "conf": 0.9,
                    },
                    {
                        "track_id": 11,
                        "bbox": {"x": 0.70, "y": 0.20, "w": 0.10, "h": 0.25},
                        "conf": 0.9,
                    },
                ],
            }
            for local_time in (14.999, 15.0)
        ]
        bboxes, _lost, _last = build_window_bboxes(
            samples,
            match["track_id"],
            fps=5,
            time_offset=35,
            anchor_matches=match["anchor_matches"],
        )
        self.assertEqual([bbox["x"] for bbox in bboxes], [0.20, 0.70])


if __name__ == "__main__":
    unittest.main()
