import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.reid.team_color_guard import (
    COLOR_FAMILIES,
    KitColorSignature,
    _VideoReader,
    _anchor_geometry_evidence,
    _repersist_guarded_output,
    _segment_color_evidence,
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


@contextmanager
def lightweight_tracking_module():
    import app.workers

    missing = object()
    previous_module = sys.modules.get("app.workers.tracking", missing)
    previous_attribute = getattr(app.workers, "tracking", missing)
    tracking = types.ModuleType("app.workers.tracking")
    tracking.S3_ENDPOINT_URL = "http://s3.internal"
    tracking._get_s3_client = lambda _endpoint: object()
    tracking._ensure_bucket_exists = lambda *_args, **_kwargs: None
    tracking._upload_file = lambda *_args, **_kwargs: None
    tracking._presign_get_object = (
        lambda bucket, key, _expires: f"https://safe.example/{bucket}/{key}"
    )
    sys.modules["app.workers.tracking"] = tracking
    app.workers.tracking = tracking
    try:
        yield tracking
    finally:
        if previous_module is missing:
            sys.modules.pop("app.workers.tracking", None)
        else:
            sys.modules["app.workers.tracking"] = previous_module
        if previous_attribute is missing:
            app.workers.__dict__.pop("tracking", None)
        else:
            app.workers.tracking = previous_attribute


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
        "fps": 4,
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

    def test_high_similarity_two_tone_argmax_flip_is_unknown(self):
        anchor = KitColorSignature(
            distribution=(
                0.133791,
                0.010083,
                0.189854,
                0.005194,
                0.0,
                0.622915,
                0.02781,
                0.010353,
            ),
            dominant_family="WHITE",
            confidence=0.507582,
            quality=0.778412,
        )
        mixed = KitColorSignature(
            distribution=(
                0.037115,
                0.022012,
                0.547957,
                0.0,
                0.0,
                0.380226,
                0.00125,
                0.01144,
            ),
            dominant_family="GREEN",
            confidence=0.459136,
            quality=0.756611,
        )

        self.assertGreater(signature_similarity(anchor, mixed), 0.90)
        self.assertIsNone(signatures_compatible(anchor, mixed))

    def test_mixed_argmax_sample_cannot_fail_a_compatible_majority(self):
        anchor = KitColorSignature(
            distribution=(
                0.133791,
                0.010083,
                0.189854,
                0.005194,
                0.0,
                0.622915,
                0.02781,
                0.010353,
            ),
            dominant_family="WHITE",
            confidence=0.507582,
            quality=0.778412,
        )
        compatible = kit_signature("WHITE")
        mixed = KitColorSignature(
            distribution=(
                0.037115,
                0.022012,
                0.547957,
                0.0,
                0.0,
                0.380226,
                0.00125,
                0.01144,
            ),
            dominant_family="GREEN",
            confidence=0.459136,
            quality=0.756611,
        )
        segment = {"bboxes": [{"t": float(index), **BBOX} for index in range(5)]}

        with patch(
            "app.reid.team_color_guard._signature_at",
            side_effect=[
                compatible,
                compatible,
                mixed,
                compatible,
                None,
            ],
        ), patch.dict(
            os.environ,
            {
                "PLAYER_REID_TEAM_COLOR_MIN_CONFIDENCE": "0.42",
                "PLAYER_REID_TEAM_COLOR_MIN_SAMPLES": "2",
                "PLAYER_REID_TEAM_COLOR_MAX_INCOMPATIBLE_FRACTION": "0.20",
            },
            clear=False,
        ):
            evidence = _segment_color_evidence(
                lambda _time: None,
                segment,
                anchor,
            )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["compatible_samples"], 3)
        self.assertEqual(evidence["incompatible_samples"], 0)
        self.assertEqual(evidence["unknown_samples"], 2)


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
        self.assertEqual(rejected["reid"]["autonomous_bboxes_count"], 0)
        self.assertEqual(result["reid_summary"]["status"], "ANCHOR_ONLY")
        self.assertEqual(result["reid_summary"]["autonomous_bboxes_count"], 0)
        self.assertEqual(result["reid_summary"]["tracking_scope_status"], "ANCHOR_ONLY")
        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "ANCHOR_ONLY")

    def test_post_guard_overlap_only_evidence_cannot_restore_success(self):
        output = output_with_segments([1.0, 2.0], [2.6, 2.7])
        outside = {
            **output["segments"][1],
            "window_start": 2.5,
            "window_end": 5.0,
            "selected_track_id": 3,
            "bboxes": [{**BBOX, "t": 4.0}, {**BBOX, "t": 5.0}],
            "reid": {
                **output["segments"][1]["reid"],
                "selected_candidate_id": "3",
            },
        }
        output["segments"].append(outside)
        output["segments_total"] = 3
        output["segments_with_player"] = 3

        result = self.run_guard(
            output,
            {
                0.0: RED,
                1.0: RED,
                2.0: RED,
                2.6: RED,
                2.7: RED,
                4.0: WHITE,
                5.0: WHITE,
            },
        )

        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "ANCHOR_ONLY")
        self.assertEqual(result["autonomous_bboxes_count"], 0)
        self.assertEqual(result["tracking_scope_status"], "ANCHOR_ONLY")

    def test_rejected_parent_breaks_post_guard_autonomous_chain(self):
        output = output_with_segments([1.0, 2.0], [3.0, 4.0])
        child = {
            **output["segments"][1],
            "window_start": 5.0,
            "window_end": 7.5,
            "selected_track_id": 3,
            "bboxes": [{**BBOX, "t": 6.0}, {**BBOX, "t": 7.0}],
            "reid": {
                **output["segments"][1]["reid"],
                "selected_candidate_id": "3",
            },
        }
        output["segments"].append(child)
        output["segments_total"] = 3
        output["segments_with_player"] = 3

        result = self.run_guard(
            output,
            {
                1.0: RED,
                2.0: RED,
                3.0: WHITE,
                4.0: WHITE,
                6.0: RED,
                7.0: RED,
            },
        )

        self.assertEqual(result["segments"][1]["identity_status"], "ABSTAINED")
        self.assertEqual(result["segments"][2]["identity_status"], "ABSTAINED")
        self.assertIn(
            "REID_PARENT_CHAIN_BROKEN",
            result["segments"][2]["reid"]["reason_codes"],
        )
        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "ANCHOR_ONLY")
        self.assertEqual(result["segments_with_player"], 1)
        self.assertEqual(result["autonomous_segments_with_player"], 0)
        self.assertEqual(result["autonomous_bboxes_count"], 0)
        self.assertEqual(result["tracking_scope_status"], "ANCHOR_ONLY")
        self.assertEqual(
            result["reid_summary"]["autonomous_segments_with_player"],
            0,
        )
        self.assertEqual(result["reid_summary"]["autonomous_bboxes_count"], 0)
        self.assertEqual(
            result["reid_summary"]["tracking_scope_status"],
            "ANCHOR_ONLY",
        )
        self.assertEqual(result["segments"][2]["reid"]["autonomous_bboxes_count"], 0)

    def test_disconnected_color_pass_is_cleared_after_retained_valid_link(self):
        output = output_with_segments([1.0, 2.0], [3.0, 4.0])
        anchor, valid_link = output["segments"]
        rejected_link = {
            **valid_link,
            "window_start": 5.0,
            "window_end": 7.5,
            "selected_track_id": 3,
            "bboxes": [{**BBOX, "t": 6.0}, {**BBOX, "t": 7.0}],
            "reid": {
                **valid_link["reid"],
                "selected_candidate_id": "3",
            },
        }
        disconnected_link = {
            **valid_link,
            "window_start": 7.5,
            "window_end": 10.0,
            "selected_track_id": 4,
            "bboxes": [{**BBOX, "t": 8.5}, {**BBOX, "t": 9.0}],
            "reid": {
                **valid_link["reid"],
                "selected_candidate_id": "4",
            },
        }
        anchor.update(
            {
                "window_index": 0,
                "parent_window_index": None,
                "processing_direction": "anchor",
            }
        )
        for index, segment in enumerate(
            [valid_link, rejected_link, disconnected_link],
            start=1,
        ):
            segment.update(
                {
                    "window_index": index,
                    "parent_window_index": index - 1,
                    "processing_direction": "forward",
                }
            )
        output["segments"] = [
            anchor,
            valid_link,
            rejected_link,
            disconnected_link,
        ]
        output["segments_total"] = 4
        output["segments_with_player"] = 4

        result = self.run_guard(
            output,
            {
                1.0: RED,
                2.0: RED,
                3.0: RED,
                4.0: RED,
                6.0: WHITE,
                7.0: WHITE,
                8.5: RED,
                9.0: RED,
            },
        )

        self.assertEqual(result["segments"][1]["identity_status"], "ACCEPTED")
        self.assertEqual(result["segments"][2]["identity_status"], "ABSTAINED")
        disconnected = result["segments"][3]
        self.assertEqual(disconnected["identity_status"], "ABSTAINED")
        self.assertEqual(disconnected["bboxes"], [])
        self.assertIsNone(disconnected["selected_track_id"])
        self.assertIn(
            "REID_PARENT_CHAIN_BROKEN",
            disconnected["reid"]["reason_codes"],
        )
        self.assertTrue(disconnected["reid"]["kit_color_guard"]["passed"])
        self.assertEqual(result["segments_with_player"], 2)
        self.assertEqual(result["coverage_pct"], 10.0)
        self.assertEqual(result["autonomous_segments_with_player"], 1)
        self.assertEqual(result["autonomous_bboxes_count"], 2)
        self.assertEqual(
            result["reid_summary"]["autonomous_segments_with_player"],
            1,
        )
        self.assertEqual(result["reid_summary"]["autonomous_bboxes_count"], 2)
        self.assertEqual(disconnected["reid"]["autonomous_bboxes_count"], 0)

    def test_later_manual_anchor_reseeds_post_guard_autonomous_chain(self):
        output = output_with_manual_anchors(
            first_bboxes=(1.0, 2.0),
            second_bboxes=(6.0, 7.0),
        )
        first_anchor, second_anchor = output["segments"]
        rejected_parent = {
            **second_anchor,
            "direction": "forward",
            "window_start": 2.5,
            "window_end": 5.0,
            "selected_track_id": 2,
            "bboxes": [{**BBOX, "t": 3.0}, {**BBOX, "t": 4.0}],
            "reid": {
                **second_anchor["reid"],
                "selected_candidate_id": "2",
            },
        }
        second_anchor.update(
            {
                "direction": "anchor",
                "processing_direction": "forward",
                "window_start": 5.0,
                "window_end": 7.5,
                "selected_track_id": 3,
                "bboxes": [{**BBOX, "t": 6.0}, {**BBOX, "t": 7.0}],
                "reid": {
                    **second_anchor["reid"],
                    "selected_candidate_id": "3",
                },
            }
        )
        autonomous_child = {
            **second_anchor,
            "direction": "forward",
            "processing_direction": "forward",
            "window_start": 7.5,
            "window_end": 10.0,
            "selected_track_id": 4,
            "bboxes": [{**BBOX, "t": 8.5}, {**BBOX, "t": 9.0}],
            "reid": {
                **second_anchor["reid"],
                "selected_candidate_id": "4",
            },
        }
        output["segments"] = [
            first_anchor,
            rejected_parent,
            second_anchor,
            autonomous_child,
        ]
        output["segments_total"] = 4
        output["segments_with_player"] = 4
        for index, segment in enumerate(output["segments"]):
            segment["window_index"] = index
            segment["parent_window_index"] = index - 1 if index else None
        first_anchor["processing_direction"] = "anchor"
        rejected_parent["processing_direction"] = "forward"
        output["anchors_used"]["selections"][1]["t"] = 6.0
        output["anchor_matches"][1].update(
            {
                "time_sec": 6.0,
                "window_index": 2,
                "window_start": 5.0,
                "window_end": 7.5,
                "local_track_id": 3,
            }
        )

        result = self.run_guard(
            output,
            {
                1.0: RED,
                2.0: RED,
                3.0: WHITE,
                4.0: WHITE,
                6.0: RED,
                7.0: RED,
                8.5: RED,
                9.0: RED,
            },
        )

        self.assertEqual(result["segments"][1]["identity_status"], "ABSTAINED")
        self.assertTrue(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "SUCCEEDED")
        self.assertEqual(result["autonomous_segments_with_player"], 1)
        self.assertEqual(result["autonomous_bboxes_count"], 2)
        self.assertEqual(result["tracking_scope_status"], "CROSS_WINDOW_EVIDENCE")
        self.assertEqual(
            result["reid_summary"]["autonomous_segments_with_player"],
            1,
        )
        self.assertEqual(result["reid_summary"]["autonomous_bboxes_count"], 2)
        self.assertEqual(
            result["reid_summary"]["tracking_scope_status"],
            "CROSS_WINDOW_EVIDENCE",
        )
        self.assertEqual(result["segments"][3]["reid"]["autonomous_bboxes_count"], 2)

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
        self.assertEqual(result["tracking_scope_status"], "EMPTY")
        self.assertEqual(result["reid_summary"]["status"], "ANCHOR_REJECTED")
        self.assertEqual(result["reid_summary"]["autonomous_bboxes_count"], 0)
        self.assertEqual(result["reid_summary"]["tracking_scope_status"], "EMPTY")
        self.assertTrue(
            all(
                segment["reid"]["autonomous_bboxes_count"] == 0
                for segment in result["segments"]
            )
        )
        geometry = result["reid_summary"]["team_color_guard"]["anchor_geometry"]
        self.assertFalse(geometry["passed"])
        self.assertIn("ANCHOR_BBOX_MISMATCH", geometry["reason_codes"])
        self.assertTrue(all(segment["bboxes"] == [] for segment in result["segments"]))

    def test_small_late_anchor_raw_sample_passes_geometry(self):
        anchor = {
            "t": 2157.009,
            "x": 0.278125,
            "y": 0.2875,
            "w": 0.03125,
            "h": 0.151388889,
        }
        evidence = _anchor_geometry_evidence(
            [{"bboxes": [{**anchor, "t": 2157.0}]}],
            [0],
            anchor,
        )

        self.assertTrue(evidence["passed"])
        self.assertGreater(evidence["iou"], 0.99)
        self.assertLess(evidence["time_delta_sec"], 0.02)

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
        output = output_with_segments([78.0, 79.0], [117.0, 118.0])
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
            {78.0: RED, 79.0: RED, 117.0: RED, 118.0: RED},
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
        self.assertTrue(
            all(
                segment["reid"]["kit_color_guard"]["passed"]
                for segment in result["segments"]
                if segment.get("bboxes")
            )
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

    def test_near_tied_different_manual_anchor_families_still_conflict(self):
        output = output_with_manual_anchors()
        white = KitColorSignature(
            distribution=(
                0.133791,
                0.010083,
                0.189854,
                0.005194,
                0.0,
                0.622915,
                0.02781,
                0.010353,
            ),
            dominant_family="WHITE",
            confidence=0.507582,
            quality=0.778412,
        )
        green = KitColorSignature(
            distribution=(
                0.037115,
                0.022012,
                0.547957,
                0.0,
                0.0,
                0.380226,
                0.00125,
                0.01144,
            ),
            dominant_family="GREEN",
            confidence=0.459136,
            quality=0.756611,
        )
        with patch(
            "app.reid.team_color_guard._anchor_signature",
            side_effect=[white, green],
        ):
            result = self.run_guard(
                output,
                {1.0: WHITE, 2.0: WHITE, 3.0: WHITE, 4.0: WHITE},
                minimum_confidence="0.42",
            )

        guard = result["reid_summary"]["team_color_guard"]
        self.assertFalse(result["tracking_success"])
        self.assertEqual(guard["prototype_status"], "CONFLICT")
        self.assertIn("ANCHOR_KIT_COLOR_CONFLICT", guard["reason_codes"])

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
        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "ANCHOR_ONLY")
        self.assertEqual(guard["guard_anchor_id"], 1)
        secondary = guard["anchor_candidates"][1]
        self.assertEqual(secondary["window_indices"], [1])
        self.assertFalse(secondary["geometry"]["passed"])
        self.assertIn("ANCHOR_BBOX_MISMATCH", secondary["geometry"]["reason_codes"])

    def test_anchor_track_evidence_ignores_late_drift_in_same_window(self):
        output = output_with_segments([1.0, 1.5, 20.0], [62.0, 63.0])
        output["segments"][0]["window_end"] = 60.0
        result = self.run_guard(
            output,
            {
                1.0: RED,
                1.5: RED,
                20.0: WHITE,
                62.0: RED,
                63.0: RED,
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

    def test_disabled_wrapper_fails_closed_without_running_guard(self):
        original = {
            "tracking_key": "jobs/another-job/tracking/tracking.json",
            "tracking_url": "https://unsafe.example/pre-guard.json",
            "tracking_success": True,
            "segments": [
                {
                    "selected_track_id": 1,
                    "identity_id": "selected-player",
                    "identity_status": "ACCEPTED",
                    "bboxes": [{"t": 1.0, **BBOX}],
                    "reid": {
                        "status": "ACCEPTED",
                        "selected_candidate_id": "1",
                    },
                }
            ],
        }

        def implementation(*_args, **_kwargs):
            return original

        guarded = guard_windowed_reid(implementation)
        with patch.dict(
            os.environ,
            {"PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "0"},
            clear=False,
        ), patch(
            "app.reid.team_color_guard.apply_team_color_guard"
        ) as apply_guard, patch(
            "app.reid.team_color_guard._repersist_guarded_output",
            side_effect=lambda output, _job_id, **_kwargs: output,
        ):
            result = guarded("job", "video.mp4", {"t": 1.0, **BBOX})

        apply_guard.assert_not_called()
        self.assertFalse(result["tracking_success"])
        self.assertEqual(
            result["tracking_status"],
            "TEAM_COLOR_GUARD_DISABLED",
        )
        self.assertEqual(result["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(result["segments_with_player"], 0)
        self.assertEqual(result["autonomous_segments_with_player"], 0)
        self.assertEqual(result["autonomous_bboxes_count"], 0)
        self.assertEqual(result["tracking_scope_status"], "EMPTY")
        self.assertEqual(result["coverage_pct"], 0.0)
        self.assertIsNone(result["largest_gap_sec"])
        self.assertNotIn("tracking_key", result)
        self.assertNotIn("tracking_url", result)
        self.assertNotIn("ASSOCIATION_ACCEPTED", json.dumps(result))
        self.assertTrue(
            all(
                segment["selected_track_id"] is None
                and segment["identity_id"] is None
                and segment["identity_status"] == "ABSTAINED"
                and segment["bboxes"] == []
                for segment in result["segments"]
            )
        )

    def test_invalid_guard_inputs_fail_closed_without_running_guard(self):
        valid_segment = {
            "selected_track_id": 1,
            "identity_id": "selected-player",
            "identity_status": "ACCEPTED",
            "bboxes": [{"t": 1.0, **BBOX}],
            "reid": {
                "status": "ACCEPTED",
                "selected_candidate_id": "1",
            },
        }
        cases = (
            (
                "segments_missing",
                {
                    "tracking_key": "jobs/another-job/tracking/tracking.json",
                    "tracking_url": "https://unsafe.example/pre-guard.json",
                    "tracking_success": True,
                },
                ("job", "video.mp4", {"t": 1.0, **BBOX}),
            ),
            (
                "segments_not_list",
                {
                    "tracking_key": "jobs/another-job/tracking/tracking.json",
                    "tracking_url": "https://unsafe.example/pre-guard.json",
                    "tracking_success": True,
                    "segments": {"unexpected": True},
                },
                ("job", "video.mp4", {"t": 1.0, **BBOX}),
            ),
            (
                "player_ref_missing",
                {
                    "tracking_key": "jobs/another-job/tracking/tracking.json",
                    "tracking_url": "https://unsafe.example/pre-guard.json",
                    "tracking_success": True,
                    "segments": [valid_segment],
                },
                ("job", "video.mp4"),
            ),
            (
                "player_ref_invalid",
                {
                    "tracking_key": "jobs/another-job/tracking/tracking.json",
                    "tracking_url": "https://unsafe.example/pre-guard.json",
                    "tracking_success": True,
                    "segments": [valid_segment],
                },
                ("job", "video.mp4", None),
            ),
            (
                "player_ref_empty",
                {
                    "tracking_key": "jobs/another-job/tracking/tracking.json",
                    "tracking_url": "https://unsafe.example/pre-guard.json",
                    "tracking_success": True,
                    "segments": [valid_segment],
                },
                ("job", "video.mp4", {}),
            ),
            (
                "player_ref_missing_bbox",
                {
                    "tracking_key": "jobs/another-job/tracking/tracking.json",
                    "tracking_url": "https://unsafe.example/pre-guard.json",
                    "tracking_success": True,
                    "segments": [valid_segment],
                },
                ("job", "video.mp4", {"t": 1.0}),
            ),
            (
                "player_ref_missing_time",
                {
                    "tracking_key": "jobs/another-job/tracking/tracking.json",
                    "tracking_url": "https://unsafe.example/pre-guard.json",
                    "tracking_success": True,
                    "segments": [valid_segment],
                },
                ("job", "video.mp4", BBOX),
            ),
            (
                "player_ref_non_finite_time",
                {
                    "tracking_key": "jobs/another-job/tracking/tracking.json",
                    "tracking_url": "https://unsafe.example/pre-guard.json",
                    "tracking_success": True,
                    "segments": [valid_segment],
                },
                ("job", "video.mp4", {"t": float("nan"), **BBOX}),
            ),
        )

        with patch.dict(
            os.environ,
            {"PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "1"},
            clear=False,
        ), patch(
            "app.reid.team_color_guard.apply_team_color_guard"
        ) as apply_guard, patch(
            "app.reid.team_color_guard._repersist_guarded_output",
            side_effect=lambda output, _job_id, **_kwargs: output,
        ):
            for name, original, args in cases:
                with self.subTest(name=name):
                    guarded = guard_windowed_reid(
                        lambda *_args, _original=original, **_kwargs: _original
                    )
                    result = guarded(*args)

                    self.assertFalse(result["tracking_success"])
                    self.assertEqual(
                        result["tracking_status"],
                        "TEAM_COLOR_GUARD_INPUT_INVALID",
                    )
                    self.assertEqual(result["action_required"], "RETRY_ANALYSIS")
                    self.assertEqual(result["segments_with_player"], 0)
                    self.assertEqual(result["autonomous_segments_with_player"], 0)
                    self.assertEqual(result["autonomous_bboxes_count"], 0)
                    self.assertEqual(result["tracking_scope_status"], "EMPTY")
                    self.assertEqual(result["coverage_pct"], 0.0)
                    self.assertIsNone(result["largest_gap_sec"])
                    self.assertNotIn("tracking_key", result)
                    self.assertNotIn("tracking_url", result)
                    self.assertNotIn("ASSOCIATION_ACCEPTED", json.dumps(result))
                    self.assertTrue(
                        all(
                            segment["selected_track_id"] is None
                            and segment["identity_id"] is None
                            and segment["identity_status"] == "ABSTAINED"
                            and segment["bboxes"] == []
                            for segment in result["segments"]
                        )
                    )

        apply_guard.assert_not_called()

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

    def test_false_anchor_only_with_evidence_is_sanitized_and_republished(self):
        original = {
            "tracking_key": "jobs/another-job/tracking/tracking.json",
            "tracking_url": "https://unsafe.example/pre-guard.json",
            "tracking_success": False,
            "tracking_status": "ANCHOR_ONLY",
            "action_required": "RESELECT_PLAYER",
            "segments_with_player": 1,
            "autonomous_segments_with_player": 0,
            "autonomous_bboxes_count": 0,
            "tracking_scope_status": "ANCHOR_ONLY",
            "coverage_pct": 3.5,
            "anchors_total": 2,
            "anchors_matched": 999,
            "anchor_matches": [
                {
                    "anchor_id": 1,
                    "frame_key": "frame_0004.jpg",
                    "time_sec": 719.003,
                    "window_index": 13,
                    "status": "MATCHED",
                    "local_track_id": 7,
                    "source": "primary_player_ref",
                    "nested_secret": {"identity_id": "do-not-copy"},
                },
                {
                    "anchor_id": 2,
                    "frame_key": "frame_0012.jpg",
                    "time_sec": 2157.009,
                    "window_index": 39,
                    "status": "MATCHED",
                    "local_track_id": 11,
                    "source": "selection",
                },
            ],
            "anchors_used": {
                "player_ref": {
                    "t": 719.003,
                    **BBOX,
                    "identity_id": "do-not-copy",
                    "best_time_sec": float("inf"),
                },
                "selections": [
                    {"t": 719.003, **BBOX},
                    {
                        "t": 2157.009,
                        **BBOX,
                        "x": -1.0,
                        "local_track_id": 11,
                    },
                ],
            },
            "anchor_acquisition": {
                "fps": 5,
                "detector_model": "yolo11s.pt",
                "seed_anchor_id": 1,
                "seed_window_index": 13,
            },
            "segments": [
                {
                    "window_index": 0,
                    "direction": "anchor",
                    "selected_track_id": 7,
                    "selected_track_ids": [7],
                    "identity_id": "selected-player",
                    "identity_status": "ACCEPTED",
                    "bboxes": [{"t": 1.0, **BBOX}],
                    "reid": {
                        "status": "ACCEPTED",
                        "selected_candidate_id": "7",
                        "reason_codes": ["ASSOCIATION_ACCEPTED"],
                    },
                }
            ],
            "reid_summary": {
                "status": "EXPERIMENTAL",
                "validated": False,
                "reason_codes": [
                    "AUTONOMOUS_REID_NOT_PROVEN",
                    "ASSOCIATION_ACCEPTED",
                ],
            },
        }
        uploaded = {}

        def upload_file(_client, _bucket, path, key, _content_type):
            uploaded.update(
                {
                    "key": key,
                    "payload": json.loads(Path(path).read_text()),
                }
            )

        guarded = guard_windowed_reid(lambda *_args, **_kwargs: original)
        with lightweight_tracking_module() as tracking, tempfile.TemporaryDirectory() as temporary_root, patch.dict(
            os.environ,
            {
                "PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "1",
                "S3_BUCKET": "tracking",
            },
            clear=False,
        ), patch(
            "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
            Path(temporary_root),
        ), patch(
            "app.reid.team_color_guard.apply_team_color_guard"
        ) as apply_guard:
            tracking._upload_file = upload_file
            result = guarded(
                "job-anchor-only",
                "video.mp4",
                {"t": 1.0, **BBOX},
                analysis_attempt_id="attempt-a",
            )

        apply_guard.assert_not_called()
        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "ANCHOR_ONLY")
        self.assertEqual(result["action_required"], "RESELECT_PLAYER")
        self.assertEqual(result["segments_with_player"], 0)
        self.assertEqual(result["coverage_pct"], 0.0)
        self.assertEqual(result["anchors_matched"], 0)
        diagnostics = result["pre_guard_anchor_diagnostics"]
        self.assertTrue(diagnostics["diagnostic_only"])
        self.assertFalse(diagnostics["validated"])
        self.assertEqual(diagnostics["anchors_total"], 2)
        self.assertEqual(diagnostics["anchors_matched_before_guard"], 2)
        self.assertEqual(
            [item["matched_before_guard"] for item in diagnostics["anchor_matches"]],
            [True, True],
        )
        self.assertTrue(
            all(
                "local_track_id" not in item
                for item in diagnostics["anchor_matches"]
            )
        )
        self.assertNotIn(
            "nested_secret",
            diagnostics["anchor_matches"][0],
        )
        self.assertNotIn(
            "identity_id",
            diagnostics["anchors_used"]["player_ref"],
        )
        self.assertNotIn(
            "best_time_sec",
            diagnostics["anchors_used"]["player_ref"],
        )
        self.assertNotIn(
            "x",
            diagnostics["anchors_used"]["selections"][1],
        )
        self.assertEqual(
            result["reid_summary"]["pre_guard_anchor_diagnostics"],
            diagnostics,
        )
        self.assertEqual(
            result["tracking_key"],
            "jobs/job-anchor-only/attempts/attempt-a/tracking/tracking.json",
        )
        self.assertNotEqual(
            result["tracking_url"],
            "https://unsafe.example/pre-guard.json",
        )
        self.assertEqual(
            result["reid_summary"]["reason_codes"],
            [
                "AUTONOMOUS_REID_NOT_PROVEN",
                "TEAM_COLOR_GUARD_UNVERIFIED_FAILURE_OUTPUT",
            ],
        )
        self.assertEqual(
            result["reid_summary"]["team_color_guard"]["status"],
            "UNVERIFIED_FAILURE_SANITIZED",
        )
        self.assertTrue(
            all(
                segment["selected_track_id"] is None
                and segment["selected_track_ids"] == []
                and segment["identity_id"] is None
                and segment["identity_status"] == "ABSTAINED"
                and segment["bboxes"] == []
                for segment in result["segments"]
            )
        )
        self.assertEqual(
            uploaded["key"],
            "jobs/job-anchor-only/attempts/attempt-a/tracking/tracking.json",
        )
        self.assertEqual(result["analysis_attempt_id"], "attempt-a")
        self.assertEqual(uploaded["payload"]["analysis_attempt_id"], "attempt-a")
        persisted_json = json.dumps(uploaded["payload"])
        self.assertNotIn("selected-player", persisted_json)
        self.assertNotIn("ASSOCIATION_ACCEPTED", persisted_json)
        self.assertNotIn("unsafe.example", persisted_json)
        self.assertNotIn("tracking_key", uploaded["payload"])
        self.assertNotIn("tracking_url", uploaded["payload"])

    def test_guard_exception_fails_closed_with_retry_not_reselection(self):
        original = output_with_segments([1.0, 2.0], [3.0, 4.0])
        original["autonomous_segments_with_player"] = 1
        original["autonomous_bboxes_count"] = 2
        original["tracking_scope_status"] = "CROSS_WINDOW_EVIDENCE"
        original["reid_summary"].update(
            {
                "autonomous_segments_with_player": 1,
                "autonomous_bboxes_count": 2,
                "tracking_scope_status": "CROSS_WINDOW_EVIDENCE",
            }
        )
        for segment in original["segments"]:
            segment["reid"]["autonomous_bboxes_count"] = 1

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
            side_effect=lambda output, _job_id, **_kwargs: output,
        ):
            result = guarded("job", "video.mp4", {"t": 1.0, **BBOX})

        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "TEAM_COLOR_GUARD_ERROR")
        self.assertEqual(result["action_required"], "RETRY_ANALYSIS")
        self.assertEqual(result["segments_with_player"], 0)
        self.assertEqual(result["autonomous_segments_with_player"], 0)
        self.assertEqual(result["autonomous_bboxes_count"], 0)
        self.assertEqual(result["tracking_scope_status"], "EMPTY")
        self.assertEqual(
            result["reid_summary"]["autonomous_segments_with_player"],
            0,
        )
        self.assertEqual(result["reid_summary"]["autonomous_bboxes_count"], 0)
        self.assertEqual(result["reid_summary"]["tracking_scope_status"], "EMPTY")
        self.assertFalse(result["partial"])
        self.assertTrue(
            all(segment.get("bboxes") == [] for segment in result["segments"])
        )
        self.assertTrue(
            all(
                segment["reid"]["autonomous_bboxes_count"] == 0
                for segment in result["segments"]
            )
        )

    def test_repersist_config_missing_drops_stale_asset_references(self):
        guarded_output = {
            "tracking_key": "jobs/another-job/tracking/tracking.json",
            "tracking_url": "https://unsafe.example/pre-guard.json",
            "tracking_success": False,
            "tracking_status": "ANCHOR_REJECTED",
            "segments": [
                {
                    "selected_track_id": None,
                    "identity_id": None,
                    "identity_status": "ABSTAINED",
                    "bboxes": [],
                    "reid": {"status": "ABSTAINED"},
                }
            ],
        }

        with lightweight_tracking_module(), tempfile.TemporaryDirectory() as temporary_root, patch.dict(
            os.environ,
            {"S3_BUCKET": ""},
            clear=False,
        ), patch(
            "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
            Path(temporary_root),
        ):
            result = _repersist_guarded_output(
                guarded_output,
                "job-guarded",
                analysis_attempt_id="attempt-a",
            )
            root_entries = list(Path(temporary_root).iterdir())

        self.assertNotIn("tracking_key", result)
        self.assertNotIn("tracking_url", result)
        self.assertEqual(result["segments"][0]["bboxes"], [])
        self.assertIsNone(result["segments"][0]["selected_track_id"])
        self.assertEqual(root_entries, [])

    def test_guard_wrapper_invalid_job_id_never_reuses_or_persists_asset_refs(self):
        original = output_with_segments([1.0, 2.0], [3.0, 4.0])
        original.update(
            {
                "tracking_key": "jobs/another-job/tracking/tracking.json",
                "tracking_url": "https://unsafe.example/pre-guard.json",
                "tracking_success": True,
            }
        )
        uploads = []
        guarded = guard_windowed_reid(lambda *_args, **_kwargs: original)

        for job_id in (None, "../victim", "/absolute/job"):
            with lightweight_tracking_module(), self.subTest(
                job_id=job_id
            ), tempfile.TemporaryDirectory() as temporary_root, patch.dict(
                os.environ,
                {
                    "PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "1",
                    "S3_BUCKET": "tracking",
                },
                clear=False,
            ), patch(
                "app.reid.team_color_guard.apply_team_color_guard",
                side_effect=lambda output, **_kwargs: dict(output),
            ), patch(
                "app.workers.tracking.S3_ENDPOINT_URL",
                "http://s3.internal",
            ), patch(
                "app.workers.tracking._upload_file",
                side_effect=lambda *_args, **_kwargs: uploads.append(True),
            ), patch(
                "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
                Path(temporary_root),
            ):
                uploads.clear()
                result = guarded(
                    job_id,
                    "video.mp4",
                    {"t": 1.0, **BBOX},
                )

                self.assertNotIn("tracking_key", result)
                self.assertNotIn("tracking_url", result)
                self.assertEqual(uploads, [])
                self.assertEqual(list(Path(temporary_root).iterdir()), [])

    def test_guard_error_upload_failure_exposes_no_pre_guard_asset(self):
        original = output_with_segments([1.0, 2.0], [3.0, 4.0])
        original.update(
            {
                "tracking_key": "jobs/another-job/tracking/tracking.json",
                "tracking_url": "https://unsafe.example/pre-guard.json",
                "tracking_success": True,
            }
        )
        attempted_upload = {}

        def implementation(*_args, **_kwargs):
            return original

        def upload_file(_client, bucket, path, key, content_type):
            attempted_upload.update(
                {
                    "bucket": bucket,
                    "key": key,
                    "content_type": content_type,
                    "payload": json.loads(Path(path).read_text()),
                }
            )
            raise RuntimeError("upload failed")

        guarded = guard_windowed_reid(implementation)
        with lightweight_tracking_module(), tempfile.TemporaryDirectory() as temporary_root, patch.dict(
            os.environ,
            {
                "PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "1",
                "S3_BUCKET": "tracking",
            },
            clear=False,
        ), patch(
            "app.reid.team_color_guard.apply_team_color_guard",
            side_effect=RuntimeError("frame read failed"),
        ), patch(
            "app.workers.tracking.S3_ENDPOINT_URL",
            "http://s3.internal",
        ), patch(
            "app.workers.tracking._get_s3_client",
            return_value=object(),
        ), patch(
            "app.workers.tracking._ensure_bucket_exists",
        ), patch(
            "app.workers.tracking._upload_file",
            side_effect=upload_file,
        ), patch(
            "app.workers.tracking._presign_get_object",
            return_value="https://safe.example/guarded.json",
        ), patch(
            "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
            Path(temporary_root),
        ):
            result = guarded(
                "job-guard-error",
                "video.mp4",
                {"t": 1.0, **BBOX},
                analysis_attempt_id="attempt-a",
            )

        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "TEAM_COLOR_GUARD_ERROR")
        self.assertNotIn("tracking_key", result)
        self.assertNotIn("tracking_url", result)
        self.assertEqual(
            attempted_upload["key"],
            "jobs/job-guard-error/attempts/attempt-a/tracking/tracking.json",
        )
        persisted = attempted_upload["payload"]
        self.assertEqual(persisted["tracking_status"], "TEAM_COLOR_GUARD_ERROR")
        self.assertTrue(
            all(segment.get("bboxes") == [] for segment in persisted["segments"])
        )
        self.assertTrue(
            all(
                segment.get("selected_track_id") is None
                and segment.get("identity_status") == "ABSTAINED"
                for segment in persisted["segments"]
            )
        )
        self.assertNotIn("tracking_key", persisted)
        self.assertNotIn("tracking_url", persisted)

    def test_guard_error_presign_failure_exposes_no_pre_guard_asset(self):
        original = output_with_segments([1.0, 2.0], [3.0, 4.0])
        original.update(
            {
                "tracking_key": "jobs/another-job/tracking/tracking.json",
                "tracking_url": "https://unsafe.example/pre-guard.json",
                "tracking_success": True,
            }
        )
        attempted_upload = {}

        def implementation(*_args, **_kwargs):
            return original

        def upload_file(_client, _bucket, path, key, _content_type):
            attempted_upload.update(
                {
                    "key": key,
                    "payload": json.loads(Path(path).read_text()),
                }
            )

        guarded = guard_windowed_reid(implementation)
        with lightweight_tracking_module(), tempfile.TemporaryDirectory() as temporary_root, patch.dict(
            os.environ,
            {
                "PLAYER_REID_TEAM_COLOR_GUARD_ENABLED": "1",
                "S3_BUCKET": "tracking",
            },
            clear=False,
        ), patch(
            "app.reid.team_color_guard.apply_team_color_guard",
            side_effect=RuntimeError("frame read failed"),
        ), patch(
            "app.workers.tracking.S3_ENDPOINT_URL",
            "http://s3.internal",
        ), patch(
            "app.workers.tracking._get_s3_client",
            return_value=object(),
        ), patch(
            "app.workers.tracking._ensure_bucket_exists",
        ), patch(
            "app.workers.tracking._upload_file",
            side_effect=upload_file,
        ), patch(
            "app.workers.tracking._presign_get_object",
            side_effect=RuntimeError("presign failed"),
        ), patch(
            "app.reid.full_match_runtime.TRACKING_ARTIFACT_ROOT",
            Path(temporary_root),
        ):
            result = guarded(
                "job-guard-presign-error",
                "video.mp4",
                {"t": 1.0, **BBOX},
                analysis_attempt_id="attempt-a",
            )

        self.assertFalse(result["tracking_success"])
        self.assertEqual(result["tracking_status"], "TEAM_COLOR_GUARD_ERROR")
        self.assertNotIn("tracking_key", result)
        self.assertNotIn("tracking_url", result)
        self.assertEqual(
            attempted_upload["key"],
            "jobs/job-guard-presign-error/attempts/attempt-a/tracking/tracking.json",
        )
        self.assertNotIn("tracking_key", attempted_upload["payload"])
        self.assertNotIn("tracking_url", attempted_upload["payload"])


if __name__ == "__main__":
    unittest.main()
