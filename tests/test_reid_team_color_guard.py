import os
import subprocess
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.reid.team_color_guard import (
    COLOR_FAMILIES,
    KitColorSignature,
    _VideoReader,
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


def bgr_from_hsv(hue, saturation, value):
    pixel = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    return tuple(int(item) for item in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


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


def kit_signature(family, confidence=0.90, quality=0.90):
    distribution = tuple(1.0 if item == family else 0.0 for item in COLOR_FAMILIES)
    return KitColorSignature(
        distribution=distribution,
        dominant_family=family,
        confidence=confidence,
        quality=quality,
    )


def output_with_manual_anchors(
    first_bboxes=(1.0, 2.0),
    second_bboxes=(3.0, 4.0),
    *,
    second_status="MATCHED",
):
    output = output_with_segments(list(first_bboxes), list(second_bboxes))
    selections = [
        {"t": 1.0, **BBOX, "frame_key": "anchor-1.jpg"},
        {"t": 3.0, **BBOX, "frame_key": "anchor-2.jpg"},
    ]
    output["anchor_acquisition"] = {
        "seed_anchor_id": 1,
        "seed_anchor": {"anchor_id": 1, **selections[0]},
    }
    output["anchors_used"] = {
        "player_ref": {"t": 1.0, **BBOX},
        "selections": selections,
    }
    output["anchor_matches"] = [
        {
            "anchor_id": 1,
            "frame_key": "anchor-1.jpg",
            "time_sec": 1.0,
            "window_index": 0,
            "window_start": 0.0,
            "window_end": 2.5,
            "status": "MATCHED",
            "local_track_id": 1,
        },
        {
            "anchor_id": 2,
            "frame_key": "anchor-2.jpg",
            "time_sec": 3.0,
            "window_index": 1,
            "window_start": 2.5,
            "window_end": 5.0,
            "status": second_status,
            "local_track_id": 2 if second_status == "MATCHED" else None,
        },
    ]
    return output


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

    def test_tinted_compressed_white_stays_distinct_from_grass_and_red(self):
        tinted_white = extract_kit_color_signature(
            frame_with_player(bgr_from_hsv(92, 82, 175))[60:150, 200:250]
        )
        shadowed_grass = extract_kit_color_signature(
            frame_with_player(bgr_from_hsv(60, 105, 135))[60:150, 200:250]
        )
        bright_grass = extract_kit_color_signature(
            frame_with_player(bgr_from_hsv(60, 110, 180))[60:150, 200:250]
        )
        red = extract_kit_color_signature(
            frame_with_player(bgr_from_hsv(0, 190, 180))[60:150, 200:250]
        )

        self.assertEqual(tinted_white.dominant_family, "WHITE")
        self.assertEqual(shadowed_grass.dominant_family, "GREEN")
        self.assertEqual(bright_grass.dominant_family, "GREEN")
        self.assertEqual(red.dominant_family, "RED_WARM")
        self.assertFalse(signatures_compatible(tinted_white, shadowed_grass))
        self.assertFalse(signatures_compatible(tinted_white, bright_grass))
        self.assertFalse(signatures_compatible(tinted_white, red))

    def test_low_confidence_observation_remains_unknown_at_default_gate(self):
        with patch.dict(
            os.environ,
            {"PLAYER_REID_TEAM_COLOR_MIN_CONFIDENCE": "0.42"},
            clear=False,
        ):
            self.assertIsNone(
                signatures_compatible(
                    kit_signature("WHITE", confidence=0.90),
                    kit_signature("WHITE", confidence=0.41),
                )
            )


class ExactVideoReaderTests(unittest.TestCase):
    def test_ffmpeg_exact_seek_is_cached(self):
        expected = frame_with_player(WHITE)
        ok, encoded = cv2.imencode(".png", expected)
        self.assertTrue(ok)
        completed = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=0,
            stdout=encoded.tobytes(),
            stderr=b"",
        )

        with patch(
            "app.reid.team_color_guard.subprocess.run",
            return_value=completed,
        ) as run:
            reader = _VideoReader("match.mp4")
            first = reader.read(1322.002)
            second = reader.read(1322.0020001)

        self.assertEqual(run.call_count, 1)
        self.assertTrue(np.array_equal(first, expected))
        self.assertIs(first, second)
        command = run.call_args.args[0]
        self.assertIn("-accurate_seek", command)
        self.assertEqual(command[command.index("-ss") + 1], "1322.002000")
        self.assertLess(command.index("-ss"), command.index("-i"))

    def test_failed_exact_decode_is_cached_without_opencv_seek_fallback(self):
        completed = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout=b"",
            stderr=b"decode failed",
        )
        with patch(
            "app.reid.team_color_guard.subprocess.run",
            return_value=completed,
        ) as run, patch("app.reid.team_color_guard.cv2.VideoCapture") as capture:
            reader = _VideoReader("match.mp4")
            self.assertIsNone(reader.read(5.0))
            self.assertIsNone(reader.read(5.0))

        self.assertEqual(run.call_count, 1)
        capture.assert_not_called()

    def test_undecodable_ffmpeg_payload_fails_closed(self):
        completed = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=0,
            stdout=b"not-an-image",
            stderr=b"",
        )
        with patch(
            "app.reid.team_color_guard.subprocess.run",
            return_value=completed,
        ) as run:
            reader = _VideoReader("match.mp4")
            self.assertIsNone(reader.read(8.0))
            self.assertIsNone(reader.read(8.0))

        self.assertEqual(run.call_count, 1)


class TeamColorGuardTests(unittest.TestCase):
    def run_guard(
        self,
        output,
        colors,
        *,
        player_ref=None,
        minimum_confidence="0.35",
    ):
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
                "PLAYER_REID_TEAM_COLOR_MIN_CONFIDENCE": minimum_confidence,
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
        self.assertIn("KIT_COLOR_GUARD_REJECTED", rejected["reid"]["reason_codes"])
        self.assertEqual(result["reid_summary"]["status"], "EXPERIMENTAL_GUARDED")

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
        self.assertTrue(all(segment["bboxes"] == [] for segment in result["segments"]))

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

    def test_guard_uses_the_resolved_secondary_seed_anchor(self):
        output = output_with_segments([78.0, 79.0], [90.0, 91.0])
        output["segments"][0]["window_start"] = 70.0
        output["segments"][0]["window_end"] = 115.0
        output["segments"][1]["window_start"] = 80.0
        output["segments"][1]["window_end"] = 125.0
        output["anchor_acquisition"] = {
            "seed_anchor_id": 2,
            "seed_anchor": {"t": 78.0, **BBOX},
        }

        result = self.run_guard(
            output,
            {78.0: RED, 79.0: RED, 90.0: RED, 91.0: RED},
            player_ref={"t": 1.0, **BBOX},
        )

        self.assertEqual(result["segments_with_player"], 2)
        self.assertTrue(result["tracking_success"])
        self.assertEqual(
            result["reid_summary"]["team_color_guard"]["seed_anchor_id"],
            2,
        )
        self.assertTrue(
            result["reid_summary"]["team_color_guard"]["anchor_geometry"]["passed"]
        )

    def test_high_confidence_matched_secondary_rescues_low_confidence_seed(self):
        output = output_with_manual_anchors()
        signatures = {
            1.0: kit_signature("WHITE", confidence=0.386446),
            3.0: kit_signature("RED_WARM", confidence=0.92),
        }

        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=lambda _read, ref: signatures[float(ref["t"])],
        ):
            result = self.run_guard(
                output,
                {1.0: RED, 2.0: RED, 3.0: RED, 4.0: RED},
                minimum_confidence="0.42",
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertTrue(result["tracking_success"])
        self.assertEqual(guard["guard_anchor_id"], 2)
        self.assertEqual(guard["prototype_status"], "SELECTED")
        self.assertEqual(
            [item["state"] for item in guard["anchor_candidates"]],
            ["REJECTED", "SELECTED"],
        )
        self.assertIn(
            "ANCHOR_KIT_COLOR_LOW_CONFIDENCE",
            guard["anchor_candidates"][0]["reason_codes"],
        )

    def test_all_matched_manual_anchors_below_gate_fail_once_explicitly(self):
        output = output_with_manual_anchors()
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=[
                kit_signature("WHITE", confidence=0.386446),
                kit_signature("WHITE", confidence=0.40),
            ],
        ):
            result = self.run_guard(
                output,
                {1.0: WHITE, 2.0: WHITE, 3.0: WHITE, 4.0: WHITE},
                minimum_confidence="0.42",
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["segments_with_player"], 0)
        self.assertEqual(guard["prototype_status"], "LOW_CONFIDENCE")
        self.assertIn("ANCHOR_KIT_COLOR_LOW_CONFIDENCE", guard["reason_codes"])
        self.assertEqual(guard["decisions"], [])
        self.assertNotIn("INSUFFICIENT_KIT_COLOR_EVIDENCE", guard["reason_codes"])

    def test_conflicting_usable_manual_anchor_signatures_fail_closed(self):
        output = output_with_manual_anchors()
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=[
                kit_signature("RED_WARM"),
                kit_signature("WHITE"),
            ],
        ):
            result = self.run_guard(
                output,
                {1.0: RED, 2.0: RED, 3.0: WHITE, 4.0: WHITE},
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["segments_with_player"], 0)
        self.assertEqual(guard["prototype_status"], "CONFLICT")
        self.assertIn("ANCHOR_KIT_COLOR_CONFLICT", guard["reason_codes"])
        self.assertEqual(
            guard["anchor_conflicts"],
            [
                {
                    "left_anchor_id": 1,
                    "right_anchor_id": 2,
                    "similarity": 0.0,
                }
            ],
        )
        self.assertEqual(guard["decisions"], [])

    def test_degraded_secondary_does_not_invalidate_verified_seed(self):
        output = output_with_manual_anchors()
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=[
                kit_signature("RED_WARM", confidence=0.90),
                kit_signature("WHITE", confidence=0.30),
            ],
        ):
            result = self.run_guard(
                output,
                {1.0: RED, 2.0: RED, 3.0: RED, 4.0: RED},
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertTrue(result["tracking_success"])
        self.assertEqual(guard["guard_anchor_id"], 1)
        self.assertEqual(guard["prototype_status"], "SELECTED")
        self.assertEqual(guard["anchor_conflicts"], [])
        self.assertIn(
            "ANCHOR_KIT_COLOR_LOW_CONFIDENCE",
            guard["anchor_candidates"][1]["reason_codes"],
        )

    def test_degraded_secondary_does_not_change_segment_sampling(self):
        output = output_with_manual_anchors(second_bboxes=(3.0, 20.0, 21.0))
        output["segments"][1]["window_end"] = 60.0
        output["anchor_matches"][1]["window_end"] = 60.0
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=[
                kit_signature("RED_WARM", confidence=0.90),
                kit_signature("WHITE", confidence=0.30),
            ],
        ):
            result = self.run_guard(
                output,
                {
                    1.0: RED,
                    2.0: RED,
                    # The unusable secondary's exact sample cannot be decoded,
                    # but the broader association still has sufficient evidence.
                    20.0: RED,
                    21.0: RED,
                },
            )

        self.assertTrue(result["tracking_success"])
        secondary_decision = result["reid_summary"]["team_color_guard"]["decisions"][1]
        self.assertTrue(secondary_decision["passed"])
        self.assertEqual(secondary_decision["sampling_mode"], "SEGMENT_EVEN")
        self.assertEqual(secondary_decision["compatible_samples"], 2)
        self.assertEqual(secondary_decision["unknown_samples"], 1)

    def test_unmatched_secondary_cannot_rescue_or_conflict_with_seed(self):
        output = output_with_manual_anchors(second_status="NOT_FOUND")
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            return_value=kit_signature("RED_WARM"),
        ) as anchor_signature:
            result = self.run_guard(
                output,
                {1.0: RED, 2.0: RED, 3.0: RED, 4.0: RED},
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertTrue(result["tracking_success"])
        self.assertEqual(guard["guard_anchor_id"], 1)
        self.assertEqual(anchor_signature.call_count, 1)
        self.assertEqual(guard["anchor_candidates"][1]["match_status"], "NOT_FOUND")
        self.assertIn(
            "ANCHOR_NOT_MATCHED",
            guard["anchor_candidates"][1]["reason_codes"],
        )

    def test_secondary_geometry_is_checked_against_its_own_window(self):
        output = output_with_manual_anchors()
        shifted = {"x": 0.10, "y": 0.20, "w": 0.10, "h": 0.30}
        output["segments"][1]["bboxes"] = [
            {**shifted, "t": 3.0},
            {**shifted, "t": 4.0},
        ]
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=[
                kit_signature("RED_WARM", confidence=0.90),
                kit_signature("RED_WARM", confidence=0.99),
            ],
        ):
            result = self.run_guard(
                output,
                {1.0: RED, 2.0: RED, 3.0: RED, 4.0: RED},
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertTrue(result["tracking_success"])
        self.assertEqual(guard["guard_anchor_id"], 1)
        secondary = guard["anchor_candidates"][1]
        self.assertEqual(secondary["window_indices"], [1])
        self.assertFalse(secondary["geometry"]["passed"])
        self.assertIn("ANCHOR_BBOX_MISMATCH", secondary["geometry"]["reason_codes"])

    def test_anchor_track_evidence_ignores_late_drift_in_same_window(self):
        output = output_with_segments([1.0, 1.5, 20.0], [3.0, 4.0])
        output["segments"][0]["window_end"] = 60.0
        result = self.run_guard(
            output,
            {
                1.0: RED,
                1.5: RED,
                20.0: WHITE,
                3.0: RED,
                4.0: RED,
            },
        )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertTrue(result["tracking_success"])
        anchor_decision = guard["decisions"][0]
        self.assertEqual(anchor_decision["sampling_mode"], "ANCHOR_NEIGHBORHOOD")
        self.assertEqual(
            [item["time_sec"] for item in anchor_decision["evidence"]],
            [1.0, 1.5],
        )

    def test_each_manual_anchor_window_uses_its_local_neighborhood(self):
        output = output_with_manual_anchors(
            first_bboxes=(1.0, 1.5),
            second_bboxes=(3.0, 3.5, 20.0),
        )
        output["segments"][1]["window_end"] = 60.0
        output["anchor_matches"][1]["window_end"] = 60.0
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=[
                kit_signature("RED_WARM"),
                kit_signature("RED_WARM"),
            ],
        ):
            result = self.run_guard(
                output,
                {
                    1.0: RED,
                    1.5: RED,
                    3.0: RED,
                    3.5: RED,
                    20.0: WHITE,
                },
            )

        self.assertTrue(result["tracking_success"])
        secondary_decision = result["reid_summary"]["team_color_guard"]["decisions"][1]
        self.assertEqual(secondary_decision["sampling_mode"], "ANCHOR_NEIGHBORHOOD")
        self.assertEqual(
            [item["time_sec"] for item in secondary_decision["evidence"]],
            [3.0, 3.5],
        )

    def test_same_window_usable_anchors_keep_both_local_neighborhoods(self):
        output = output_with_manual_anchors(
            first_bboxes=(1.0, 1.5, 20.0, 20.5),
            second_bboxes=(70.0, 71.0),
        )
        output["segments"][0]["window_end"] = 60.0
        output["segments"][1]["window_start"] = 60.0
        output["segments"][1]["window_end"] = 120.0
        output["anchors_used"]["selections"][1]["t"] = 20.0
        output["anchor_matches"][0]["window_end"] = 60.0
        output["anchor_matches"][1].update(
            {
                "time_sec": 20.0,
                "window_index": 0,
                "window_start": 0.0,
                "window_end": 60.0,
            }
        )
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=[
                kit_signature("RED_WARM"),
                kit_signature("RED_WARM"),
            ],
        ):
            result = self.run_guard(
                output,
                {
                    1.0: RED,
                    1.5: RED,
                    20.0: WHITE,
                    20.5: WHITE,
                    70.0: RED,
                    71.0: RED,
                },
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertFalse(result["tracking_success"])
        anchor_decision = guard["decisions"][0]
        self.assertEqual(anchor_decision["anchor_times"], [1.0, 20.0])
        self.assertEqual(anchor_decision["compatible_samples"], 2)
        self.assertEqual(anchor_decision["incompatible_samples"], 2)
        self.assertIn(
            "KIT_COLOR_INCONSISTENT_WITH_ANCHOR",
            anchor_decision["reason_codes"],
        )

    def test_exact_frame_extraction_failure_clears_all_identity_links(self):
        completed = subprocess.CompletedProcess(
            args=["ffmpeg"],
            returncode=1,
            stdout=b"",
            stderr=b"decode failed",
        )
        with patch.dict(
            os.environ,
            {"PLAYER_REID_TEAM_COLOR_MIN_CONFIDENCE": "0.42"},
            clear=False,
        ), patch(
            "app.reid.team_color_guard.subprocess.run",
            return_value=completed,
        ):
            result = apply_team_color_guard(
                output_with_segments([1.0, 2.0], [3.0, 4.0]),
                input_video_path="unreadable.mp4",
                player_ref={"t": 1.0, **BBOX},
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "ANCHOR_REJECTED")
        self.assertEqual(result["segments_with_player"], 0)
        self.assertEqual(guard["prototype_status"], "UNAVAILABLE")
        self.assertIn("ANCHOR_KIT_COLOR_UNAVAILABLE", guard["reason_codes"])

    def test_disabled_wrapper_returns_original_output_without_persistence(self):
        original = {
            "segments": [{"selected_track_id": 1, "bboxes": [{"t": 1.0, **BBOX}]}]
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

    def test_wrapper_preserves_structured_anchor_failure(self):
        original = {
            "tracking_success": False,
            "tracking_status": "ANCHOR_NOT_FOUND",
            "action_required": "RESELECT_PLAYER",
            "segments": [],
            "largest_gap_sec": None,
        }

        def implementation(*_args, **_kwargs):
            return original

        guarded = guard_windowed_reid(implementation)
        with patch.dict(
            os.environ,
            {"PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "1"},
            clear=False,
        ):
            self.assertIs(
                guarded("job", "video.mp4", {"t": 1.0, **BBOX}),
                original,
            )

    def test_guard_exception_fails_closed_with_retry_not_reselection(self):
        original = output_with_segments([1.0, 2.0], [3.0, 4.0])

        def implementation(*_args, **_kwargs):
            return original

        guarded = guard_windowed_reid(implementation)
        with patch.dict(
            os.environ,
            {"PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "1"},
            clear=False,
        ), patch(
            "app.reid.team_color_guard.apply_team_color_guard",
            side_effect=RuntimeError("frame read failed"),
        ), patch(
            "app.reid.team_color_guard._repersist_guarded_output",
            side_effect=lambda output, _job_id: output,
        ):
            result = guarded("job", "video.mp4", {"t": 1.0, **BBOX})

        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "TEAM_COLOR_GUARD_ERROR")
        self.assertEqual(result["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(result["segments_with_player"], 0)
        self.assertTrue(
            all(segment.get("bboxes") == [] for segment in result["segments"])
        )


if __name__ == "__main__":
    unittest.main()
