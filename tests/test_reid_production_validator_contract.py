import unittest
from pathlib import Path

import reid_production_validator as validator

ATTEMPT_ID = "f0243750-3488-49a4-ada3-579859961671"
OLD_ATTEMPT_ID = "1fdaf4b6-3c5c-4923-b80d-c542df602e96"
FRAME_KEYS = [
    "jobs/fixture/frames/frame_0004.jpg",
    "jobs/fixture/frames/frame_0012.jpg",
]


def selection_payload():
    return {
        "selections": [
            {
                "frameTimeSec": 719.003,
                "frameKey": FRAME_KEYS[0],
                "bbox": {
                    "x": 0.58125,
                    "y": 0.463888889,
                    "w": 0.11875,
                    "h": 0.536111111,
                },
            },
            {
                "frameTimeSec": 2157.009,
                "frameKey": FRAME_KEYS[1],
                "bbox": {
                    "x": 0.278125,
                    "y": 0.2875,
                    "w": 0.03125,
                    "h": 0.151388889,
                },
            },
        ]
    }


def _bbox(timestamp, index):
    return {
        "t": timestamp,
        "x": 0.1 + (index % 5) * 0.01,
        "y": 0.2 + (index % 3) * 0.01,
        "w": 0.05,
        "h": 0.1,
    }


def _color_signature(family="RED_WARM", *, confidence=0.9, quality=0.9):
    return {
        "version": validator.GUARD_VERSION,
        "dominant_family": family,
        "confidence": confidence,
        "quality": quality,
        "distribution": {
            candidate: 1.0 if candidate == family else 0.0
            for candidate in validator.COLOR_FAMILIES
        },
    }


def _geometry(timestamp):
    return {
        "passed": True,
        "reason_codes": [],
        "nearest_time_sec": timestamp,
        "time_delta_sec": 0.0,
        "iou": 1.0,
        "minimum_iou": 0.08,
        "maximum_time_delta_sec": 1.25,
    }


def _segment_guard(segment, *, anchor_time=None):
    bboxes = segment.get("bboxes") or []
    evidence = [
        {
            "time_sec": round(float(bbox["t"]), 6),
            "status": "COMPATIBLE",
            "similarity": 1.0,
            "signature": _color_signature(),
        }
        for bbox in bboxes[:3]
    ]
    anchor_times = [anchor_time] if anchor_time is not None else []
    return {
        "version": validator.GUARD_VERSION,
        "passed": True,
        "sampling_mode": ("ANCHOR_NEIGHBORHOOD" if anchor_times else "SEGMENT_EVEN"),
        "anchor_times": anchor_times,
        "maximum_anchor_delta_sec": 1.25 if anchor_times else None,
        "compatible_samples": len(evidence),
        "incompatible_samples": 0,
        "unknown_samples": 0,
        "incompatible_fraction": 0.0,
        "reason_codes": [],
        "evidence": evidence,
    }


def genuine_payload(*, sparse=False):
    anchors = {13: 719.003, 39: 2157.009}
    expected_selections = selection_payload()["selections"]
    anchor_selections = {
        13: expected_selections[0],
        39: expected_selections[1],
    }
    segments = []
    for index in range(validator.EXPECTED_WINDOWS):
        start = index * 55.0
        end = min(validator.EXPECTED_DURATION_SEC, start + 60.0)
        if index == 13:
            direction = "anchor"
            processing_direction = "anchor"
            parent = None
        elif index < 13:
            direction = processing_direction = "backward"
            parent = index + 1
        elif index == 39:
            direction = "anchor"
            processing_direction = "forward"
            parent = index - 1
        else:
            direction = processing_direction = "forward"
            parent = index - 1

        retained = not sparse or index in {13, 14, 15, 39}
        if not retained:
            bboxes = []
        elif index in anchors:
            selection = anchor_selections[index]
            bboxes = [
                {
                    "t": anchors[index] + offset,
                    **selection["bbox"],
                }
                for offset in (-0.5, 0.0, 0.5)
            ]
        elif sparse:
            bboxes = [_bbox(start + 10.1, index)]
        else:
            bboxes = [
                _bbox(start + offset, index)
                for offset in (10.1, 20.1, 30.1)
                if start + offset <= end
            ]
        accepted = bool(bboxes)
        reid = {
            "status": "ACCEPTED" if accepted else "ABSTAINED",
            "validated": False,
            "autonomous_bboxes_count": 0,
        }
        if accepted:
            reid["kit_color_guard"] = {"passed": True}
            reid["identity_id"] = "fixture-player"
            if direction != "anchor":
                reid["tracklet_scope"] = "MOTION_CONTINUOUS_STRONG_OVERLAP"
        segments.append(
            {
                "window_index": index,
                "parent_window_index": parent,
                "window_start": start,
                "window_end": end,
                "direction": direction,
                "processing_direction": processing_direction,
                "selected_track_id": f"track-{index}" if accepted else None,
                "selected_track_ids": (
                    [f"track-{index}"] if accepted and direction == "anchor" else []
                ),
                "identity_id": "fixture-player" if accepted else None,
                "identity_status": "ACCEPTED" if accepted else "ABSTAINED",
                "lost_segments": [],
                "bboxes": bboxes,
                "reid": reid,
            }
        )

    anchor_windows = [
        (segments[index]["window_start"], segments[index]["window_end"])
        for index in anchors
    ]
    autonomous_times = set()
    autonomous_segments = 0
    for segment in segments:
        if segment["direction"] == "anchor" or not segment["bboxes"]:
            continue
        outside = {
            round(float(bbox["t"]), 6)
            for bbox in segment["bboxes"]
            if all(
                bbox["t"] < start - 1.0 or bbox["t"] > end + 1.0
                for start, end in anchor_windows
            )
        }
        segment["reid"]["autonomous_bboxes_count"] = len(outside)
        if outside:
            autonomous_segments += 1
            autonomous_times.update(outside)

    guard_decisions = []
    for segment in segments:
        if not segment["bboxes"]:
            continue
        anchor_time = anchors.get(segment["window_index"])
        segment_guard = _segment_guard(
            segment,
            anchor_time=anchor_time,
        )
        segment["reid"]["kit_color_guard"] = segment_guard
        guard_decisions.append(
            {
                "window_index": segment["window_index"],
                **segment_guard,
            }
        )

    all_bboxes = [bbox for segment in segments for bbox in segment.get("bboxes") or []]
    unique_frames = {int(round(bbox["t"])) for bbox in all_bboxes}
    coverage = len(unique_frames) / round(validator.EXPECTED_DURATION_SEC) * 100.0
    rounded_coverage = round(coverage, 2)
    times = sorted({float(bbox["t"]) for bbox in all_bboxes})
    gaps = [times[0]]
    gaps.extend(current - previous for previous, current in zip(times, times[1:]))
    gaps.append(validator.EXPECTED_DURATION_SEC - times[-1])
    largest_gap = round(max(gaps), 2)
    sparse = coverage < 5.0
    status = "SPARSE_CROSS_WINDOW_EVIDENCE" if sparse else "SUCCEEDED"
    evaluation_status = "TRACKING_INCOMPLETE" if sparse else "TRACKING_ONLY"
    score_kind = "tracking_incomplete" if sparse else "tracking_quality"
    motion = validator._image_motion(all_bboxes)
    sample_sufficiency = min(
        100.0,
        len(all_bboxes) / validator.SAMPLE_TARGET * 100.0,
    )
    quality_index = (
        None
        if sparse
        else round(
            coverage * 0.5 + 100.0 * 0.3 + sample_sufficiency * 0.2,
            1,
        )
    )
    signals = {
        "coverage_ratio": round(coverage / 100.0, 6),
        "coverage_pct": rounded_coverage,
        "tracklet_continuity_pct": 0.0 if sparse else 100.0,
        "tracklet_continuity_source": (
            "not_applicable" if sparse else "lost_segments_proxy"
        ),
        "sample_sufficiency_pct": (0.0 if sparse else round(sample_sufficiency, 2)),
        "samples_used": 0 if sparse else len(all_bboxes),
        "segments_total": validator.EXPECTED_WINDOWS,
        "segments_with_player": sum(1 for segment in segments if segment["bboxes"]),
        "largest_gap_sec": None if sparse else largest_gap,
        "image_motion": motion,
    }
    provenance = {
        "kind": score_kind,
        "validated_player_score": False,
        "metrics_scope": "selected_player",
    }
    matched = [
        {
            "anchor_id": 1,
            "frame_key": FRAME_KEYS[0],
            "time_sec": 719.003,
            "window_index": 13,
            "status": "MATCHED",
            "local_track_id": "track-13",
            "source": "primary_player_ref",
        },
        {
            "anchor_id": 2,
            "frame_key": FRAME_KEYS[1],
            "time_sec": 2157.009,
            "window_index": 39,
            "status": "MATCHED",
            "local_track_id": "track-39",
            "source": "selection",
        },
    ]
    expected = expected_selections
    anchors_used = [
        {
            "t": item["frameTimeSec"],
            "frame_key": item["frameKey"],
            **item["bbox"],
        }
        for item in expected
    ]
    retained_count = sum(1 for segment in segments if segment["bboxes"])
    tracking = {
        "analysis_attempt_id": ATTEMPT_ID,
        "mode": "full_match_windowed",
        "tracking_success": True,
        "tracking_status": status,
        "action_required": None,
        "metrics_scope": "selected_player",
        "fps": 1,
        "segments": segments,
        "segments_total": validator.EXPECTED_WINDOWS,
        "segments_with_player": retained_count,
        "autonomous_segments_with_player": autonomous_segments,
        "autonomous_bboxes_count": len(autonomous_times),
        "tracking_scope_status": "CROSS_WINDOW_EVIDENCE",
        "windows_processed": validator.EXPECTED_WINDOWS,
        "anchors_total": 2,
        "anchors_matched": 2,
        "anchor_matches": matched,
        "anchors_used": {
            "player_ref": {
                "t": expected[0]["frameTimeSec"],
                **expected[0]["bbox"],
            },
            "selections": anchors_used,
        },
        "anchor_acquisition": {
            "fps": 5,
            "detector_model": "yolo11s.pt",
            "windows_processed": 2,
            "seed_anchor_id": 1,
            "seed_window_index": 13,
            "seed_anchor": {
                "anchor_id": 1,
                "window_index": 13,
                **anchors_used[0],
            },
        },
        "runtime_profile": {
            "duration_sec": validator.EXPECTED_DURATION_SEC,
            "window_sec": validator.EXPECTED_WINDOW_SEC,
            "overlap_sec": validator.EXPECTED_OVERLAP_SEC,
            "fps": 1,
        },
        "partial": sparse,
        "partial_reason": status if sparse else None,
        "coverage_pct": rounded_coverage,
        "coverage_pct_total": rounded_coverage,
        "largest_gap_sec": largest_gap,
        "bboxes_count": len(all_bboxes),
        "reid_summary": {
            "validated": False,
            "identity_id": "fixture-player",
            "anchor_window_index": 13,
            "anchor_local_track_id": "track-13",
            "anchor_matches": matched,
            "processing_failures": 0,
            "autonomous_segments_with_player": autonomous_segments,
            "autonomous_bboxes_count": len(autonomous_times),
            "tracking_scope_status": "CROSS_WINDOW_EVIDENCE",
            "team_color_guard": {
                "version": validator.GUARD_VERSION,
                "validated": False,
                "status": "APPLIED",
                "anchor_signature": _color_signature(),
                "seed_anchor_id": 1,
                "guard_anchor_id": 1,
                "prototype_status": "SELECTED",
                "prototype_confidence_gate": 0.42,
                "anchor_candidates": [
                    {
                        "anchor_id": 1,
                        "match_status": "MATCHED",
                        "state": "SELECTED",
                        "is_seed": True,
                        "window_indices": [13],
                        "geometry": _geometry(719.003),
                        "signature": _color_signature(),
                        "reason_codes": [],
                    },
                    {
                        "anchor_id": 2,
                        "match_status": "MATCHED",
                        "state": "USABLE",
                        "is_seed": False,
                        "window_indices": [39],
                        "geometry": _geometry(2157.009),
                        "signature": _color_signature(),
                        "reason_codes": [],
                    },
                ],
                "anchor_conflicts": [],
                "anchor_geometry": _geometry(719.003),
                "segments_checked": retained_count,
                "segments_rejected": 0,
                "post_guard_segments_with_player": retained_count,
                "reason_codes": ["TEAM_COLOR_GUARD_EXPERIMENTAL"],
                "decisions": guard_decisions,
            },
        },
    }
    outcome = {
        "analysis_attempt_id": ATTEMPT_ID,
        "pipeline_state": "DONE",
        "tracking_state": "INCOMPLETE" if sparse else "SUCCEEDED",
        "metrics_scope": "selected_player",
        "observed_samples": len(all_bboxes),
        "segments_with_player": retained_count,
        "autonomous_segments_with_player": autonomous_segments,
        "autonomous_bboxes_count": len(autonomous_times),
        "tracking_scope_status": "CROSS_WINDOW_EVIDENCE",
        "windows_processed": validator.EXPECTED_WINDOWS,
        "windows_total": validator.EXPECTED_WINDOWS,
        "anchors_total": 2,
        "anchors_matched": 2,
        "action_required": None,
    }
    result = {
        "analysis_attempt_id": ATTEMPT_ID,
        "tracking": tracking,
        "analysis_outcome": outcome,
        "evaluation_status": evaluation_status,
        "score_kind": score_kind,
        "player_evaluation_available": False,
        "legacy_scores_suppressed": True,
        "tracking_quality_index": quality_index,
        "tracking_signals": signals,
        "tracking_quality": {
            "status": evaluation_status,
            "score_kind": score_kind,
            "player_evaluation_available": False,
            "tracking_quality_index": quality_index,
            "tracking_confidence": "none" if sparse else "low",
            "signals": signals,
            "provenance": provenance,
        },
        "score_provenance": provenance,
        "evidence_metrics": {"image_motion": motion},
        "summary": {
            "evaluation_status": evaluation_status,
            "player_evaluation_available": False,
            "tracking_quality_index": quality_index,
        },
        "radar": {},
        "breakdown": {},
        "skills_computed": {},
        "skills_missing": [],
    }
    return {
        "job": {
            "data": {
                "id": "fixture",
                "status": "PARTIAL",
                "target": {
                    "confirmed": True,
                    "full_match_mode": True,
                    "analysis_attempt_id": ATTEMPT_ID,
                    "tracking": {
                        "status": "PENDING",
                        "analysis_attempt_id": ATTEMPT_ID,
                    },
                },
                "progress": {
                    "step": "DONE",
                    "pct": 100,
                    "analysis_attempt_id": ATTEMPT_ID,
                },
                "result": result,
                "warnings": ["PLAYER_EVALUATION_WITHHELD"],
                "error": None,
                "failure_reason": None,
            }
        },
        "selection": selection_payload(),
        "enqueue": {
            "data": {
                "job_id": "fixture",
                "id": "fixture",
                "status": "QUEUED",
                "analysis_attempt_id": ATTEMPT_ID,
            }
        },
        "before": {
            "data": {
                "id": "fixture",
                "target": {"analysis_attempt_id": OLD_ATTEMPT_ID},
                "result": {"analysis_attempt_id": OLD_ATTEMPT_ID},
            }
        },
    }


def validate(payload):
    return validator.validate_regression_result(
        job_envelope=payload["job"],
        selection_payload=payload["selection"],
        enqueue_envelope=payload["enqueue"],
        fixture_before_envelope=payload["before"],
    )


def runtime_payload(*, revision="expected"):
    return {
        "ok": True,
        "data": {
            "ready": True,
            "required": True,
            "revision": revision,
            "dependencies": {
                "redis": "ready",
                "worker": "ready",
            },
            "worker": {
                "service": "algonext-worker",
                "revision": revision,
                "state": "ready",
            },
            "worker_revision_matches_api": True,
            "worker_age_seconds": 1.5,
            "max_worker_age_seconds": 60.0,
        },
    }


class ReIDProductionValidatorContractTests(unittest.TestCase):
    def test_genuine_payload_passes(self):
        report = validate(genuine_payload())
        self.assertEqual(report["analysis_attempt_id"], ATTEMPT_ID)
        self.assertGreater(report["coverage_pct"], 5.0)

    def test_fabricated_coverage_and_quality_fail(self):
        payload = genuine_payload()
        result = payload["job"]["data"]["result"]
        result["tracking"]["coverage_pct"] = 80.0
        result["tracking"]["coverage_pct_total"] = 80.0
        result["tracking_signals"]["coverage_pct"] = 80.0
        result["tracking_signals"]["coverage_ratio"] = 0.8
        result["tracking_quality_index"] = 72.7
        result["tracking_quality"]["tracking_quality_index"] = 72.7
        result["summary"]["tracking_quality_index"] = 72.7

        with self.assertRaisesRegex(validator.ValidationError, "coverage_pct"):
            validate(payload)

    def test_missing_anchor_acquisition_fails(self):
        payload = genuine_payload()
        payload["job"]["data"]["result"]["tracking"].pop("anchor_acquisition")
        with self.assertRaisesRegex(
            validator.ValidationError,
            "anchor_acquisition",
        ):
            validate(payload)

    def test_stale_attempt_fails(self):
        payload = genuine_payload()
        payload["job"]["data"]["result"]["analysis_attempt_id"] = OLD_ATTEMPT_ID
        with self.assertRaisesRegex(
            validator.ValidationError,
            "analysis_attempt_id",
        ):
            validate(payload)

    def test_nested_preview_candidate_metrics_fail(self):
        payload = genuine_payload()
        payload["job"]["data"]["result"]["evidence_metrics"][
            "preview_candidate_metrics"
        ] = {
            "sampleFramesCount": 4,
            "coveragePct": 0.0,
            "stabilityScore": 33.3,
            "trackingQualityIndex": 11.3,
        }
        with self.assertRaisesRegex(
            validator.ValidationError,
            "Legacy or unvalidated",
        ):
            validate(payload)

    def test_truthful_sparse_payload_fails_release(self):
        payload = genuine_payload(sparse=True)
        with self.assertRaisesRegex(
            validator.ValidationError,
            "too few|sparse",
        ):
            validate(payload)

    def test_runtime_revision_mismatch_fails(self):
        payload = runtime_payload()
        payload["data"]["worker"]["revision"] = "different"
        with self.assertRaisesRegex(
            validator.ValidationError,
            "Worker revision mismatch",
        ):
            validator.validate_runtime_attestation(
                payload,
                expected_revision="expected",
            )

    def test_runtime_failed_envelope_fails(self):
        payload = runtime_payload()
        payload["ok"] = False
        with self.assertRaisesRegex(
            validator.ValidationError,
            "envelope",
        ):
            validator.validate_runtime_attestation(
                payload,
                expected_revision="expected",
            )

    def test_runtime_stopped_or_optional_worker_fails(self):
        payload = runtime_payload()
        payload["data"]["required"] = False
        payload["data"]["dependencies"]["worker"] = "stopped"
        payload["data"]["worker"]["state"] = "stopped"
        with self.assertRaisesRegex(
            validator.ValidationError,
            "readiness|dependencies|state",
        ):
            validator.validate_runtime_attestation(
                payload,
                expected_revision="expected",
            )

    def test_runtime_stale_heartbeat_fails(self):
        payload = runtime_payload()
        payload["data"]["worker_age_seconds"] = 61.0
        with self.assertRaisesRegex(
            validator.ValidationError,
            "heartbeat",
        ):
            validator.validate_runtime_attestation(
                payload,
                expected_revision="expected",
            )

    def test_runtime_cannot_self_attest_an_inflated_heartbeat_limit(self):
        payload = runtime_payload()
        payload["data"]["worker_age_seconds"] = 86400.0
        payload["data"]["max_worker_age_seconds"] = 31536000.0
        with self.assertRaisesRegex(
            validator.ValidationError,
            "max_worker_age_seconds",
        ):
            validator.validate_runtime_attestation(
                payload,
                expected_revision="expected",
            )

    def test_final_target_attempt_mismatch_fails(self):
        payload = genuine_payload()
        payload["job"]["data"]["target"]["analysis_attempt_id"] = OLD_ATTEMPT_ID
        with self.assertRaisesRegex(
            validator.ValidationError,
            "target analysis_attempt_id",
        ):
            validate(payload)

    def test_final_progress_attempt_mismatch_fails(self):
        payload = genuine_payload()
        payload["job"]["data"]["progress"]["analysis_attempt_id"] = OLD_ATTEMPT_ID
        with self.assertRaisesRegex(
            validator.ValidationError,
            "progress analysis_attempt_id",
        ):
            validate(payload)

    def test_nested_previous_attempt_reuse_fails(self):
        payload = genuine_payload()
        payload["before"]["data"]["result"]["tracking"] = {
            "analysis_attempt_id": ATTEMPT_ID,
        }
        with self.assertRaisesRegex(
            validator.ValidationError,
            "reused a previous nested",
        ):
            validate(payload)

    def test_final_job_id_mismatch_fails(self):
        payload = genuine_payload()
        payload["job"]["data"]["id"] = "different-job"
        with self.assertRaisesRegex(
            validator.ValidationError,
            "job ids differ",
        ):
            validate(payload)

    def test_final_failure_reason_fails(self):
        payload = genuine_payload()
        payload["job"]["data"]["failure_reason"] = "TRACKING_FAILED"
        with self.assertRaisesRegex(
            validator.ValidationError,
            "failure/error",
        ):
            validate(payload)

    def test_distinct_identity_per_window_fails(self):
        payload = genuine_payload()
        segments = payload["job"]["data"]["result"]["tracking"]["segments"]
        for segment in segments:
            if not segment["bboxes"]:
                continue
            identity_id = f"identity-{segment['window_index']}"
            segment["identity_id"] = identity_id
            segment["reid"]["identity_id"] = identity_id
        with self.assertRaisesRegex(
            validator.ValidationError,
            "guarded identity proof",
        ):
            validate(payload)

    def test_anchor_local_track_must_exist_in_anchor_segment(self):
        payload = genuine_payload()
        matches = payload["job"]["data"]["result"]["tracking"]["anchor_matches"]
        matches[1]["local_track_id"] = "invented-track"
        with self.assertRaisesRegex(
            validator.ValidationError,
            "local track is not",
        ):
            validate(payload)

    def test_static_guard_flags_without_proof_fail(self):
        payload = genuine_payload()
        payload["job"]["data"]["result"]["tracking"]["reid_summary"][
            "team_color_guard"
        ] = {
            "version": validator.GUARD_VERSION,
            "validated": False,
            "status": "APPLIED",
        }
        with self.assertRaisesRegex(
            validator.ValidationError,
            "prototype|guard",
        ):
            validate(payload)

    def test_contradictory_secondary_anchor_signature_fails(self):
        payload = genuine_payload()
        guard = payload["job"]["data"]["result"]["tracking"]["reid_summary"][
            "team_color_guard"
        ]
        guard["anchor_candidates"][1]["signature"] = _color_signature("CYAN_BLUE")
        with self.assertRaisesRegex(
            validator.ValidationError,
            "anchor conflicts",
        ):
            validate(payload)

    def test_usable_secondary_anchor_without_signature_fails(self):
        payload = genuine_payload()
        guard = payload["job"]["data"]["result"]["tracking"]["reid_summary"][
            "team_color_guard"
        ]
        guard["anchor_candidates"][1]["signature"] = None
        with self.assertRaisesRegex(
            validator.ValidationError,
            "reason codes|state",
        ):
            validate(payload)

    def test_declared_compatible_evidence_with_opposite_signature_fails(self):
        payload = genuine_payload()
        guard = payload["job"]["data"]["result"]["tracking"]["reid_summary"][
            "team_color_guard"
        ]
        evidence = guard["decisions"][0]["evidence"][0]
        evidence["signature"] = _color_signature("CYAN_BLUE")
        with self.assertRaisesRegex(
            validator.ValidationError,
            "similarity|status",
        ):
            validate(payload)

    def test_artifact_sanitizer_handles_key_and_string_variants(self):
        payload = {
            "inputVideoUrl": "https://s3/x?X-Amz-Signature=SECRET",
            "assets": [
                {"signedUrl": "https://s3/x?x-amz-signature=SECRET"},
            ],
            "href": (
                "https://s3/x?X-Amz-Credential=AKIA" "&X-Amz-Security-Token=TOKEN"
            ),
            "downloadURL": "https://cdn/x?token=SECRET",
            "error": "fetch https://s3/x?x-amz-signature=SECRET failed",
            "headers": {
                "Authorization": "Bearer SECRET",
                "authorization": "Bearer SECRET",
            },
            "diagnostics": [
                "AWSAccessKeyId=AKIA",
                "access_token=SECRET",
                "id_token%3DSECRET",
                "api_key: SECRET",
                "client_secret=SECRET",
                "password=SECRET",
            ],
            "anchor_signature": _color_signature(),
            "signature": _color_signature(),
            "note": "signature provenance and token counters",
            "object_key": "jobs/fixture/input.mp4",
            "etag": "fixture-etag",
            "sha256": "fixture-hash",
        }
        cleaned = validator.sanitize_artifact_value(payload)
        validator.assert_sanitized_artifact(cleaned)
        self.assertNotIn("inputVideoUrl", cleaned)
        self.assertEqual(cleaned["assets"], [{}])
        self.assertEqual(cleaned["href"], "[REDACTED_URL]")
        self.assertNotIn("downloadURL", cleaned)
        self.assertEqual(cleaned["error"], "[REDACTED_URL]")
        self.assertEqual(cleaned["headers"], {})
        self.assertEqual(
            cleaned["diagnostics"],
            ["[REDACTED_SENSITIVE_VALUE]"] * 6,
        )
        self.assertEqual(cleaned["anchor_signature"], _color_signature())
        self.assertEqual(cleaned["signature"], _color_signature())
        self.assertEqual(
            cleaned["note"],
            "signature provenance and token counters",
        )
        self.assertEqual(cleaned["object_key"], "jobs/fixture/input.mp4")
        self.assertEqual(cleaned["etag"], "fixture-etag")
        self.assertEqual(cleaned["sha256"], "fixture-hash")

    def test_artifact_post_sanitize_scan_rejects_residual_secrets(self):
        residuals = [
            {"href": "https://example.invalid/runtime"},
            {"note": "x-amz-signature=SECRET"},
            {"note": "credential=AKIA"},
            {"note": "security-token=TOKEN"},
            {"note": "Authorization: Bearer SECRET"},
            {"apiToken": "SECRET"},
            {"note": "AWSAccessKeyId=AKIA"},
            {"note": "access_token=SECRET"},
            {"note": "id_token%3DSECRET"},
            {"note": "api_key: SECRET"},
            {"note": "client_secret=SECRET"},
            {"note": "password=SECRET"},
        ]
        for residual in residuals:
            with self.subTest(residual=residual), self.assertRaises(
                validator.ValidationError
            ):
                validator.assert_sanitized_artifact(residual)

    def test_workflow_derives_runtime_from_exercised_api(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/reid-production-regression.yml"
        ).read_text()
        self.assertNotIn("HEALTH_URL", workflow)
        self.assertNotIn("health_url:", workflow)
        self.assertNotIn("inputs.api_base", workflow)
        self.assertNotIn("api_base:", workflow)
        self.assertIn(
            "API_BASE: https://algonext-frontend.vercel.app/api/backend",
            workflow,
        )
        self.assertIn("API_TARGET_HOST: algonext-frontend.vercel.app", workflow)
        self.assertIn("algonext-production", workflow)
        self.assertIn('runtime_url="${API_BASE%/}/runtime"', workflow)

    def test_workflow_pins_video_and_reserves_final_evidence_time(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/reid-production-regression.yml"
        ).read_text()
        self.assertIn("timeout-minutes: 180", workflow)
        self.assertIn("bytes 0-65535/2179028557", workflow)
        self.assertIn("2e5c54ec42f3f79e989983ac133197f2-260", workflow)
        self.assertIn(
            "0b9ad260c2a404a60496e14da6cb3fce3003004d36d6a218d3dbbe0a29e1f8ea",
            workflow,
        )
        self.assertIn('expected_input_key="jobs/${JOB_ID}/input.mp4"', workflow)
        self.assertIn("path: /tmp/reid-regression-artifacts/", workflow)
        self.assertIn("sanitize_artifact_value", workflow)
        self.assertIn("assert_sanitized_artifact", workflow)
        self.assertIn("staging.rename(destination)", workflow)

    def test_workflow_fences_each_fixture_mutation_with_the_rotated_attempt(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github/workflows/reid-production-regression.yml"
        ).read_text()

        expected_headers = {
            "player-ref": '"${player_ref_attempt_header[@]}"',
            "selection": ('-H "X-Analysis-Attempt-Id: $player_ref_attempt_id"'),
            "enqueue": '-H "X-Analysis-Attempt-Id: $selection_attempt_id"',
        }
        for endpoint, expected_header in expected_headers.items():
            marker = f'"$API_BASE/jobs/$JOB_ID/{endpoint}"'
            endpoint_index = workflow.index(marker)
            curl_index = workflow.rfind("curl ", 0, endpoint_index)
            curl_block = workflow[curl_index:endpoint_index]
            with self.subTest(endpoint=endpoint):
                self.assertIn(expected_header, curl_block)
                self.assertNotIn("--retry", curl_block)

        self.assertIn(
            "jq -r '.data.analysis_attempt_id // empty' "
            "\\\n              /tmp/player-ref-response.json",
            workflow,
        )
        self.assertIn(
            "jq -r '.data.analysis_attempt_id // empty' "
            "\\\n              /tmp/selection-response.json",
            workflow,
        )
        retry_marker = '"$API_BASE/jobs/$JOB_ID/retry"'
        retry_index = workflow.index(retry_marker)
        retry_curl_index = workflow.rfind("curl ", 0, retry_index)
        retry_block = workflow[retry_curl_index:retry_index]
        self.assertIn(
            '-H "X-Analysis-Attempt-Id: $fixture_attempt_id"',
            retry_block,
        )
        self.assertIn("--data @/tmp/supersede-active.json", retry_block)
        self.assertNotIn("--retry", retry_block)
        self.assertIn("force: true", workflow)
        self.assertIn("supersede_active: true", workflow)
        self.assertIn(
            "Active regression fixture target is not confirmed",
            workflow,
        )
        self.assertIn(
            "Active regression fixture player_ref changed",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
