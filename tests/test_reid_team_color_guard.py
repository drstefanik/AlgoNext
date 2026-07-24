import os
import unittest
from unittest.mock import patch

import numpy as np

from app.reid.team_color_guard import (
    apply_team_color_guard,
    extract_kit_color_signature,
    guard_windowed_reid,
    signature_similarity,
    signatures_compatible,
)


BBOX = {"x": 0.40, "y": 0.20, "w": 0.10, "h": 0.30}
RED = (20, 20, 190)
WHITE = (230, 230, 230)
GREEN = (40, 140, 40)


def frame_with_player(color, bbox=None):
    bbox = bbox or BBOX
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    frame[:] = GREEN
    x1 = int(bbox["x"] * frame.shape[1])
    y1 = int(bbox["y"] * frame.shape[0])
    x2 = int((bbox["x"] + bbox["w"]) * frame.shape[1])
    y2 = int((bbox["y"] + bbox["h"]) * frame.shape[0])
    frame[y1:y2, x1:x2] = color
    return frame


def output_with_segments(anchor_bboxes, other_bboxes, *, legacy=False):
    def segment(direction, track_id, bboxes):
        payload = {
            "direction": direction,
            "window_start": 0.0 if direction == "anchor" else 2.5,
            "window_end": 2.5 if direction == "anchor" else 5.0,
            "selected_track_id": track_id,
            "bboxes": [{**BBOX, "t": time_sec} for time_sec in bboxes],
            "coverage_pct": 20.0,
        }
        if not legacy:
            payload.update(
                {
                    "identity_status": "ACCEPTED",
                    "identity_id": "selected-player",
                    "reid": {
                        "status": "ACCEPTED",
                        "selected_candidate_id": str(track_id),
                        "reason_codes": ["ASSOCIATION_ACCEPTED"],
                    },
                }
            )
        return payload

    return {
        "mode": "full_match_windowed",
        "fps": 1,
        "segments": [
            segment("anchor", 1, anchor_bboxes),
            segment("forward", 2, other_bboxes),
        ],
        "segments_total": 2,
        "segments_with_player": 2,
        "coverage_pct": 80.0,
        "coverage_pct_total": 80.0,
        "reid_summary": {"status": "EXPERIMENTAL", "validated": False},
    }


class TeamColorSignatureTests(unittest.TestCase):
    def test_red_and_white_kits_are_not_treated_as_similar(self):
        red = extract_kit_color_signature(frame_with_player(RED)[60:150, 200:250])
        white = extract_kit_color_signature(frame_with_player(WHITE)[60:150, 200:250])

        self.assertIsNotNone(red)
        self.assertIsNotNone(white)
        self.assertEqual(red.dominant_family, "RED_WARM")
        self.assertEqual(white.dominant_family, "WHITE")
        self.assertLess(signature_similarity(red, white), 0.25)
        self.assertFalse(signatures_compatible(red, white))

    def test_same_red_kit_is_compatible(self):
        first = extract_kit_color_signature(frame_with_player(RED)[60:150, 200:250])
        second = extract_kit_color_signature(
            frame_with_player((25, 25, 175))[60:150, 200:250]
        )
        self.assertTrue(signatures_compatible(first, second))


class TeamColorGuardTests(unittest.TestCase):
    def run_guard(self, output, colors, *, player_ref=None):
        frames = {
            float(time_sec): frame_with_player(color)
            for time_sec, color in colors.items()
        }

        def read(time_sec):
            return frames.get(float(time_sec))

        with patch.dict(
            os.environ,
            {
                "PLAYER_REID_TEAM_COLOR_MIN_SAMPLES": "2",
                "PLAYER_REID_TEAM_COLOR_MIN_CONFIDENCE": "0.35",
                "PLAYER_REID_TEAM_COLOR_MAX_INCOMPATIBLE_FRACTION": "0.20",
                "PLAYER_REID_ANCHOR_MAX_TIME_DELTA_SEC": "1.25",
                "PLAYER_REID_ANCHOR_MIN_IOU": "0.08",
            },
            clear=False,
        ):
            return apply_team_color_guard(
                output,
                input_video_path="unused.mp4",
                player_ref=player_ref or {"t": 1.0, **BBOX},
                frame_reader=read,
            )

    def test_wrong_team_segment_is_downgraded_to_abstention(self):
        result = self.run_guard(
            output_with_segments([1.0, 2.0], [3.0, 4.0]),
            {0.0: RED, 1.0: RED, 2.0: RED, 3.0: WHITE, 4.0: WHITE},
        )

        self.assertEqual(result["segments_with_player"], 1)
        self.assertEqual(result["segments"][0]["identity_status"], "ACCEPTED")
        rejected = result["segments"][1]
        self.assertEqual(rejected["identity_status"], "ABSTAINED")
        self.assertEqual(rejected["bboxes"], [])
        self.assertIsNone(rejected["selected_track_id"])
        self.assertIn(
            "KIT_COLOR_GUARD_REJECTED", rejected["reid"]["reason_codes"]
        )
        self.assertEqual(
            result["reid_summary"]["status"], "EXPERIMENTAL_GUARDED"
        )

    def test_anchor_bbox_mismatch_invalidates_every_identity_link(self):
        shifted = {"x": 0.10, "y": 0.20, "w": 0.10, "h": 0.30}
        output = output_with_segments([1.0, 2.0], [3.0, 4.0])
        output["segments"][0]["bboxes"] = [
            {**shifted, "t": 1.0},
            {**shifted, "t": 2.0},
        ]
        result = self.run_guard(
            output,
            {0.0: RED, 1.0: RED, 2.0: RED, 3.0: RED, 4.0: RED},
        )

        self.assertEqual(result["segments_with_player"], 0)
        self.assertEqual(result["coverage_pct"], 0.0)
        self.assertEqual(result["reid_summary"]["status"], "ANCHOR_REJECTED")
        geometry = result["reid_summary"]["team_color_guard"]["anchor_geometry"]
        self.assertFalse(geometry["passed"])
        self.assertIn("ANCHOR_BBOX_MISMATCH", geometry["reason_codes"])
        self.assertTrue(
            all(segment["bboxes"] == [] for segment in result["segments"])
        )

    def test_mixed_anchor_track_invalidates_every_identity_link(self):
        result = self.run_guard(
            output_with_segments([1.0, 2.0], [3.0, 4.0]),
            {0.0: RED, 1.0: RED, 2.0: WHITE, 3.0: RED, 4.0: RED},
        )

        self.assertEqual(result["segments_with_player"], 0)
        self.assertEqual(result["reid_summary"]["status"], "ANCHOR_REJECTED")
        anchor_decision = result["reid_summary"]["team_color_guard"]["decisions"][0]
        self.assertFalse(anchor_decision["passed"])
        self.assertIn(
            "KIT_COLOR_INCONSISTENT_WITH_ANCHOR",
            anchor_decision["reason_codes"],
        )

    def test_legacy_window_output_is_guarded_too(self):
        result = self.run_guard(
            output_with_segments([1.0, 2.0], [3.0, 4.0], legacy=True),
            {0.0: RED, 1.0: RED, 2.0: RED, 3.0: WHITE, 4.0: WHITE},
        )
        self.assertEqual(result["segments_with_player"], 1)
        self.assertEqual(result["segments"][1]["identity_status"], "ABSTAINED")

    def test_disabled_wrapper_returns_original_output_without_persistence(self):
        original = {
            "segments": [
                {"selected_track_id": 1, "bboxes": [{"t": 1.0, **BBOX}]}
            ]
        }

        def implementation(*_args, **_kwargs):
            return original

        guarded = guard_windowed_reid(implementation)
        with patch.dict(
            os.environ,
            {"PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "0"},
            clear=False,
        ):
            self.assertIs(
                guarded("job", "video.mp4", {"t": 1.0, **BBOX}),
                original,
            )


if __name__ == "__main__":
    unittest.main()
