import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core.workspace import (
    InsufficientWorkspaceError,
    cleanup_tracking_workspace,
    require_free_space,
)
from app.reid.association import (
    AppearanceDescriptor,
    CandidateProfile,
    IdentityProfile,
    AssociationThresholds,
    associate_identity,
)
from app.reid.jersey_vision import (
    JerseyCrop,
    JerseyReader,
    JerseyVerifier,
    evaluate_readings,
    parse_reading,
    nearby_confirmation_detections,
)


class JerseyVisionTests(unittest.TestCase):
    def test_context_preserves_order_and_rejects_uncertainty_or_score_digits(self):
        from app.reid.match_context import parse_context, same_match_context

        anchor = parse_context(
            {"legible": True, "left_team": " BOL ", "right_team": "FIO"}
        )
        self.assertTrue(same_match_context(anchor, dict(anchor)))
        for other in [
            {**anchor, "left_team": "FIO", "right_team": "BOL"},
            {**anchor, "legible": False},
            {**anchor, "right_team": "ROM"},
            None,
        ]:
            self.assertFalse(same_match_context(anchor, other))
        for left, right in [("0", "0"), ("11", "20"), ("BOL", None), ("BOL", "BOL")]:
            self.assertFalse(
                parse_context(
                    {"legible": True, "left_team": left, "right_team": right}
                )["legible"]
            )
        with self.assertRaises(ValueError):
            parse_context({"legible": True, "left_team": 8, "right_team": "FIO"})

    def test_context_cannot_reuse_jersey_cache_or_exceed_shared_budget(self):
        from app.reid.match_context import PROMPT, SCHEMA, VERSION, parse_context

        response = Mock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "legible": True,
                                    "left_team": "BOL",
                                    "right_team": "FIO",
                                }
                            )
                        },
                    }
                ],
                "usage": {"total_tokens": 30},
            },
        )
        transport = Mock(side_effect=[response, self.response()])
        with patch.dict(
            "os.environ",
            {
                "JERSEY_OCR_ENABLED": "1",
                "OPENAI_API_KEY": "test",
                "JERSEY_OCR_MAX_CALLS": "2",
            },
        ):
            reader = JerseyReader(transport=transport)
            crop = JerseyCrop(b"same-image", 1, 1)
            kwargs = dict(
                prompt=PROMPT,
                schema=SCHEMA,
                parser=parse_context,
                cache_version=VERSION,
                context=True,
            )
            context = reader.read(crop, **kwargs)
            jersey = reader.read(crop)
            self.assertEqual(context["left_team"], "BOL")
            self.assertEqual(jersey["number"], 8)
            self.assertEqual(reader.context_reads, 1)
            self.assertEqual(reader.legible, 1)
            self.assertEqual(
                reader.read(JerseyCrop(b"another", 2, 1), **kwargs)["status"],
                "BUDGET_EXHAUSTED",
            )
            self.assertEqual(transport.call_count, 2)

    def test_context_proof_must_match_the_jersey_observation_times(self):
        from app.reid.association import _verified_jersey_reacquisition

        anchor = {"legible": True, "left_team": "BOL", "right_team": "FIO"}
        context = {
            "required": True,
            "matched": True,
            "anchor": anchor,
            "readings": [{**anchor, "time_sec": t} for t in (1, 2)],
        }
        candidate = self.jersey_candidate(evidence_changes={"match_context": context})
        self.assertTrue(_verified_jersey_reacquisition(candidate))
        for changes in [
            {"matched": False},
            {"readings": [{**anchor, "time_sec": 50}, {**anchor, "time_sec": 51}]},
            {"readings": [{**anchor, "time_sec": 1}]},
            {
                "readings": [
                    {**anchor, "time_sec": 1},
                    {**anchor, "time_sec": 2, "left_team": "FIO", "right_team": "BOL"},
                ]
            },
        ]:
            invalid = self.jersey_candidate(
                evidence_changes={"match_context": {**context, **changes}}
            )
            self.assertFalse(_verified_jersey_reacquisition(invalid))

    def test_context_conflict_vetoes_even_strong_overlap(self):
        from dataclasses import replace

        candidate = self.jersey_candidate(
            evidence_changes={"status": "CONTEXT_UNVERIFIED"}
        )
        candidate = replace(
            candidate,
            overlap_score=0.99,
            metadata={
                **candidate.metadata,
                "strong_overlap_unique": True,
                "tracklet_scope": "MOTION_CONTINUOUS_STRONG_OVERLAP",
                "overlap_link_samples": 3,
                "overlap_previous_samples": 3,
                "tracklet_sample_indices": [1, 2, 3],
            },
        )
        result = associate_identity(
            IdentityProfile("target", candidate.descriptor),
            [candidate],
            thresholds=AssociationThresholds(require_strong_overlap=True),
        )
        self.assertFalse(result.accepted)
        self.assertIn("MATCH_CONTEXT_UNVERIFIED", result.reason_codes)

    def test_composite_jersey_link_requires_individual_proof_and_exact_box_ownership(
        self,
    ):
        from app.reid.window_logic import (
            _verified_jersey_anchor_link,
            retained_autonomous_chain_indices,
        )

        evidence = copy.deepcopy(self.jersey_candidate().metadata["jersey_evidence"])
        anchor = {
            "direction": "anchor",
            "identity_id": "player",
            "reid": {"jersey_anchor_reading": {"number": 8, "legible": True}},
        }
        components, candidates = [], []
        for index, start in enumerate((10.0, 20.0)):
            current = copy.deepcopy(evidence)
            for n, reading in enumerate(current["readings"]):
                reading["time_sec"] = start + n
            boxes = [
                {"t": start + n, "x": 0.2, "y": 0.3, "w": 0.04, "h": 0.15}
                for n in range(3)
            ]
            components.append(
                {
                    "candidate_id": str(index),
                    "tracklet_scope": "MOTION_CONTINUOUS_JERSEY",
                    "bboxes": boxes,
                }
            )
            candidates.append({"candidate_id": str(index), "jersey_evidence": current})
        segment = {
            "identity_id": "player",
            "bboxes": [b for c in components for b in c["bboxes"]],
            "reid": {
                "tracklet_scope": "INDEPENDENT_JERSEY_TRACKLETS",
                "identity_link": "JERSEY_REACQUISITION",
                "reason_codes": ["JERSEY_AIDED_REACQUISITION_EXPERIMENTAL"],
                "jersey_components": components,
                "candidates": candidates,
            },
        }
        self.assertTrue(_verified_jersey_anchor_link(segment, anchor))
        context_anchor = {"legible": True, "left_team": "BOL", "right_team": "FIO"}
        bound_anchor = {
            **anchor,
            "reid": {**anchor["reid"], "jersey_anchor_match_context": context_anchor},
        }
        self.assertFalse(_verified_jersey_anchor_link(segment, bound_anchor))
        contextual = copy.deepcopy(segment)
        for c in contextual["reid"]["candidates"]:
            c["jersey_evidence"]["match_context"] = {
                "required": True,
                "matched": True,
                "anchor": context_anchor,
                "readings": [
                    {**context_anchor, "time_sec": r["time_sec"]}
                    for r in c["jersey_evidence"]["readings"]
                ],
            }
        self.assertTrue(_verified_jersey_anchor_link(contextual, bound_anchor))
        self.assertFalse(_verified_jersey_anchor_link(contextual, anchor))
        extra = copy.deepcopy(segment)
        extra["bboxes"].append({"t": 25.0, "x": 0.9})
        self.assertFalse(_verified_jersey_anchor_link(extra, anchor))
        graph_anchor = {
            **anchor,
            "window_index": 0,
            "identity_status": "ACCEPTED",
            "bboxes": [{"t": 1.0}],
        }
        graph_extra = {
            **extra,
            "window_index": 1,
            "parent_window_index": 0,
            "direction": "forward",
            "identity_status": "ACCEPTED",
        }
        self.assertEqual(
            retained_autonomous_chain_indices([graph_anchor, graph_extra]), set()
        )
        wrong = copy.deepcopy(segment)
        wrong["reid"]["candidates"][1]["jersey_evidence"]["readings"][0]["number"] = 9
        self.assertFalse(_verified_jersey_anchor_link(wrong, anchor))
        duplicate = copy.deepcopy(segment)
        duplicate["reid"]["jersey_components"][1]["candidate_id"] = "0"
        self.assertFalse(_verified_jersey_anchor_link(duplicate, anchor))
        malformed = copy.deepcopy(segment)
        malformed["reid"]["jersey_components"][1]["bboxes"].append(None)
        self.assertFalse(_verified_jersey_anchor_link(malformed, anchor))
        self.assertFalse(
            _verified_jersey_anchor_link(
                segment, {**anchor, "identity_id": "someone_else"}
            )
        )

    def test_camera_relative_continuity_rejects_spatial_switch_and_crowding(self):
        from app.reid.tracklet_motion import camera_relative_continuity

        first = {
            "t": 1.0,
            "bbox": {"x": 0.1, "y": 0.3, "w": 0.03, "h": 0.1},
            "_motion_context": {"group": 0, "x": 0.0, "y": 0.0, "crowded": False},
        }
        panning = {
            "t": 1.2,
            "bbox": {**first["bbox"], "x": 0.2},
            "_motion_context": {"group": 0, "x": 0.1, "y": 0.0, "crowded": False},
        }
        self.assertTrue(camera_relative_continuity(first, panning))
        switched = {**panning, "bbox": {**panning["bbox"], "x": 0.28}}
        self.assertFalse(camera_relative_continuity(first, switched))
        crowded = {
            **panning,
            "_motion_context": {**panning["_motion_context"], "crowded": True},
        }
        self.assertFalse(camera_relative_continuity(first, crowded))
        unsupported = {
            **panning,
            "_motion_context": {**panning["_motion_context"], "group": 1},
        }
        self.assertFalse(camera_relative_continuity(first, unsupported))

    def test_kit_component_stops_at_unknown_or_opponent_without_rejoining(self):
        from app.reid.jersey_search import trim_kit_component
        from unittest.mock import MagicMock

        detections = [{"t": i, "bbox": {}} for i in range(6)]
        for boundary in (False, None):
            cap = MagicMock()
            cap.set.side_effect = lambda _, time: setattr(cap, "time", int(time / 1000))
            cap.read.side_effect = lambda: (True, cap.time)
            with patch(
                "app.reid.jersey_search.cv2.VideoCapture", return_value=cap
            ), patch(
                "app.reid.jersey_search.crop_from_normalized_bbox",
                side_effect=lambda frame, _: frame,
            ), patch(
                "app.reid.jersey_search.extract_kit_color_signature",
                side_effect=lambda frame: {"time": frame},
            ), patch(
                "app.reid.jersey_search.signatures_compatible",
                side_effect=lambda _, sig: boundary if sig["time"] == 3 else True,
            ):
                result = trim_kit_component("clip", detections, [1, 2], "anchor")
                self.assertEqual([r["t"] for r in result], [0, 1, 2])
                self.assertEqual(
                    trim_kit_component("clip", detections, [2, 4], "anchor"), []
                )
            cap.release.assert_called()

    def batch_response(self, rows):
        return Mock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"readings": rows})},
                    }
                ],
                "usage": {"total_tokens": 50},
            },
        )

    def test_batch_ids_cache_and_independent_confirmation(self):
        rows = [
            dict(
                crop_id=str(i), number=8, legible=True, single_player=True, view="back"
            )
            for i in (1, 0)
        ]
        transport = Mock(side_effect=[self.batch_response(rows), self.response()])
        with patch.dict(
            "os.environ", {"JERSEY_OCR_ENABLED": "1", "OPENAI_API_KEY": "test"}
        ):
            reader = JerseyReader(transport=transport)
        crops = [JerseyCrop(b"one", 1, 0.9, True), JerseyCrop(b"two", 2, 0.9, True)]
        results = [{**r, "kit_compatible": True} for r in reader.read_many(crops)]
        self.assertEqual([r["time_sec"] for r in results], [1, 2])
        self.assertEqual(reader.calls, 1)
        self.assertEqual(evaluate_readings(results, 8)["status"], "UNVERIFIED")
        cached = reader.read(crops[0])
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(reader.calls, 1)
        results.append(
            {**reader.read(JerseyCrop(b"three", 3, 0.9, True)), "kit_compatible": True}
        )
        self.assertEqual(evaluate_readings(results, 8)["status"], "MATCH")
        self.assertEqual(reader.calls, 2)
        body = transport.call_args_list[0].kwargs["json"]
        self.assertNotIn("target", json.dumps(body))
        self.assertFalse(body["store"])

    def test_duplicate_or_unknown_batch_ids_fail_closed_atomically(self):
        row = dict(crop_id="0", number=8, legible=True, single_player=True, view="back")
        for rows in [[row, row], [row, {**row, "crop_id": "99"}], [row]]:
            with self.subTest(rows=rows), patch.dict(
                "os.environ", {"JERSEY_OCR_ENABLED": "1", "OPENAI_API_KEY": "test"}
            ):
                reader = JerseyReader(
                    transport=Mock(return_value=self.batch_response(rows))
                )
                result = reader.read_many(
                    [JerseyCrop(b"one", 1, 0.9), JerseyCrop(b"two", 2, 0.9)]
                )
                self.assertTrue(
                    all(
                        r["status"] == "API_ERROR" and r["number"] is None
                        for r in result
                    )
                )
                self.assertEqual(reader.errors, 1)

    def test_same_batch_cannot_bypass_identity_gate(self):
        candidate = self.jersey_candidate()
        evidence = copy.deepcopy(candidate.metadata["jersey_evidence"])
        for reading in evidence["readings"]:
            reading["request_id"] = "one-batch"
        decision = associate_identity(
            IdentityProfile("player", candidate.descriptor),
            [self.jersey_candidate(evidence_changes=evidence)],
            thresholds=AssociationThresholds(require_strong_overlap=True),
        )
        self.assertFalse(decision.accepted)

    def test_batch_honors_shared_budget_and_deduplicates_images(self):
        rows = [
            dict(crop_id="0", number=8, legible=True, single_player=True, view="back")
        ]
        transport = Mock(return_value=self.batch_response(rows))
        with patch.dict(
            "os.environ",
            {
                "JERSEY_OCR_ENABLED": "1",
                "OPENAI_API_KEY": "test",
                "JERSEY_OCR_MAX_CALLS": "1",
            },
        ):
            reader = JerseyReader(transport=transport)
        result = reader.read_many(
            [JerseyCrop(b"one", 1, 0.9), JerseyCrop(b"one", 2, 0.9)]
        )
        self.assertEqual(result[0]["image_sha256"], result[1]["image_sha256"])
        self.assertEqual(reader.images_read, 1)
        self.assertEqual(
            reader.read_many([JerseyCrop(b"two", 3, 0.9)])[0]["status"],
            "BUDGET_EXHAUSTED",
        )
        self.assertEqual(transport.call_count, 1)

    def test_dense_sampling_retains_readable_location_without_transferring_identity(
        self,
    ):
        from dataclasses import replace

        box = {"x": 0.2, "y": 0.3, "w": 0.1, "h": 0.2}
        coarse = replace(
            self.jersey_candidate(),
            metadata={
                "jersey_evidence": {
                    "target_number": 8,
                    "readings": [
                        {
                            "number": 8,
                            "legible": True,
                            "kit_compatible": True,
                            "time_sec": 150.025,
                        },
                        {
                            "number": 6,
                            "legible": True,
                            "kit_compatible": True,
                            "time_sec": 155,
                        },
                    ],
                },
                "tracklet_detections": [
                    {"t": 50.025, "bbox": box},
                    {"t": 55, "bbox": box},
                ],
            },
        )
        hints = JerseyVerifier.dense_hints([coarse], 100)
        self.assertEqual(len(hints), 1)
        self.assertAlmostEqual(hints[0]["t"], 50.025)
        self.assertEqual(hints[0]["bbox"], box)
        nearby = replace(
            coarse,
            candidate_id="new_id",
            metadata={
                "tracklet_detections": [{"t": 50.025, "bbox": box}],
            },
        )
        other = replace(
            nearby,
            candidate_id="teammate",
            metadata={
                "tracklet_detections": [{"t": 50.025, "bbox": {**box, "x": 0.7}}],
            },
        )
        reordered = JerseyVerifier.prioritize_dense_candidates([other, nearby], hints)
        self.assertEqual(reordered[0].candidate_id, "new_id")
        self.assertEqual(reordered[0].metadata["jersey_preferred_times"], [50.025])
        self.assertNotIn("jersey_evidence", reordered[0].metadata)
        self.assertNotIn("jersey_preferred_times", reordered[1].metadata)

        # A track can be unconfirmed on the original frame and become
        # observable one 3-fps sample later during a camera pan.
        shifted = replace(
            nearby,
            metadata={
                "tracklet_detections": [
                    {"t": 50.358, "bbox": {**box, "x": 0.24}},
                    {"t": 51.358, "bbox": box},
                ],
            },
        )
        result = JerseyVerifier.prioritize_dense_candidates([other, shifted], hints)
        self.assertEqual(result[0].metadata["jersey_preferred_times"], [50.358])
        self.assertNotIn("jersey_evidence", result[0].metadata)

    def test_global_search_requires_read_anchor_kit_and_available_budget(self):
        verifier = JerseyVerifier.__new__(JerseyVerifier)
        verifier.target = 8
        verifier.anchor_signature = {"dominant_family": "WHITE"}
        verifier.anchor_reading = {"number": 8, "legible": True}
        verifier.reader = SimpleNamespace(
            enabled=True, errors=0, max_calls=64, calls=40, max_seconds=240, elapsed=61
        )
        self.assertTrue(verifier.can_reacquire())
        for field, value in [
            ("calls", 63),
            ("elapsed", 238),
            ("errors", 3),
            ("enabled", False),
        ]:
            with patch.object(verifier.reader, field, value):
                self.assertFalse(verifier.can_reacquire())
        for field, value in [
            ("anchor_reading", {"number": 6, "legible": True}),
            ("anchor_signature", None),
        ]:
            with patch.object(verifier, field, value):
                self.assertFalse(verifier.can_reacquire())

    def test_global_window_budget_is_enforced_across_candidates_and_confirmation(self):
        import numpy as np
        from unittest.mock import MagicMock
        from dataclasses import replace

        verifier = JerseyVerifier.__new__(JerseyVerifier)
        verifier.target = 8
        verifier.anchor_signature = {"dominant_family": "WHITE"}
        verifier.anchor_reading = {"number": 8, "legible": True}
        transport = Mock(return_value=self.response())
        with patch.dict(
            "os.environ", {"JERSEY_OCR_ENABLED": "1", "OPENAI_API_KEY": "test"}
        ):
            verifier.reader = JerseyReader(transport=transport)
        cap = MagicMock()
        frames = iter(np.full((100, 40, 3), i, dtype=np.uint8) for i in range(30))
        cap.read.side_effect = lambda: (True, next(frames))
        candidate = replace(
            self.jersey_candidate(),
            metadata={"tracklet_detections": [{"t": i, "bbox": {}} for i in range(8)]},
        )
        with patch("cv2.VideoCapture", return_value=cap), patch(
            "app.reid.appearance.crop_from_normalized_bbox",
            side_effect=lambda frame, _: frame,
        ), patch(
            "app.reid.appearance.evaluate_crop_quality",
            return_value=SimpleNamespace(
                width=40, height=100, sharpness=100, score=0.9
            ),
        ), patch(
            "app.reid.team_color_guard.extract_kit_color_signature",
            return_value={"kit": "white"},
        ), patch(
            "app.reid.team_color_guard.signatures_compatible", return_value=True
        ):
            result = verifier.enrich(
                "video", [candidate, candidate, candidate], 100, max_calls=2
            )
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(len(result), 3)
        self.assertEqual(verifier.reader.calls, 2)

    def test_dense_retry_requires_verified_anchor_matching_read_and_remaining_budget(
        self,
    ):
        verifier = JerseyVerifier.__new__(JerseyVerifier)
        verifier.target = 8
        verifier.anchor_reading = {"number": 8, "legible": True}
        verifier.reader = SimpleNamespace(
            enabled=True, errors=0, max_calls=64, calls=21, max_seconds=240, elapsed=30
        )
        candidate = self.jersey_candidate()
        self.assertTrue(verifier.should_retry_densely([candidate]))
        self.assertFalse(verifier.should_retry_densely([]))
        for attr, value in [
            ("calls", 62),
            ("elapsed", 238),
            ("errors", 3),
            ("enabled", False),
        ]:
            with patch.object(verifier.reader, attr, value):
                self.assertFalse(verifier.should_retry_densely([candidate]))
        verifier.anchor_reading = {"number": None, "legible": False}
        self.assertFalse(verifier.should_retry_densely([candidate]))

    def test_confirmation_samples_are_nearby_and_temporally_independent(self):
        detections = [
            {"t": t} for t in [0, 10, 10.2, 11.001, 12.002, 13.003, 14.004, 15.005, 30]
        ]
        selected = nearby_confirmation_detections(
            detections, [112.002], [112.002, 100, 130], 100
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({d["t"] for d in selected}), 4)
        for detection in selected:
            gap = abs(100 + detection["t"] - 112.002)
            self.assertGreaterEqual(gap, 0.6)
            self.assertLessEqual(gap, 3.0)

    def test_short_fragment_can_confirm_between_two_same_batch_readings(self):
        from app.reid.association import independent_jersey_reads

        detections = [{"t": t} for t in (0.0, 1 / 3, 2 / 3, 1.0)]
        selected = nearby_confirmation_detections(detections, [0.0, 1.0], [0.0, 1.0], 0)
        self.assertEqual({d["t"] for d in selected}, {1 / 3, 2 / 3})
        cached = [
            {"time_sec": t, "image_sha256": str(i) * 64, "request_id": "batch"}
            for i, t in enumerate((0.0, 1.0))
        ]
        self.assertFalse(independent_jersey_reads(cached))
        for detection in selected:
            self.assertTrue(
                independent_jersey_reads(
                    [
                        *cached,
                        {
                            "time_sec": detection["t"],
                            "image_sha256": "f" * 64,
                            "request_id": "fresh",
                        },
                    ]
                )
            )
        self.assertEqual(
            nearby_confirmation_detections([{"t": 0.1}, {"t": 0.2}], [0.0], [0.0], 0),
            [],
        )

    def jersey_candidate(self, *, vector=(1, 0), evidence_changes=None):
        readings = [
            dict(
                number=8,
                legible=True,
                kit_compatible=True,
                image_sha256=digit * 64,
                time_sec=t,
            )
            for digit, t in [("a", 1), ("b", 2)]
        ]
        evidence = dict(
            status="MATCH",
            target_number=8,
            anchor_number=8,
            anchor_legible=True,
            component_match_samples=2,
            readings=readings,
        )
        evidence.update(evidence_changes or {})
        return CandidateProfile(
            "number_8",
            AppearanceDescriptor(vector, 3, 0.9),
            None,
            0.9,
            4,
            {"tracklet_scope": "MOTION_CONTINUOUS_JERSEY", "jersey_evidence": evidence},
        )

    def test_two_shirt_reads_can_bridge_cut_with_strong_appearance(self):
        descriptor = AppearanceDescriptor((1, 0), 3, 0.9)
        identity = IdentityProfile("player", descriptor)
        candidate = self.jersey_candidate()
        similar_teammate = CandidateProfile("teammate", descriptor, None, 0.9, 4)
        decision = associate_identity(
            identity,
            [candidate, similar_teammate],
            thresholds=AssociationThresholds(require_strong_overlap=True),
        )
        self.assertTrue(decision.accepted)
        self.assertFalse(decision.validated)
        self.assertIn("JERSEY_AIDED_REACQUISITION_EXPERIMENTAL", decision.reason_codes)

    def test_reacquisition_rejects_duplicate_reads_bad_kit_unverified_anchor_and_wrong_appearance(
        self,
    ):
        descriptor = AppearanceDescriptor((1, 0), 3, 0.9)
        identity = IdentityProfile("player", descriptor)
        thresholds = AssociationThresholds(require_strong_overlap=True)
        baseline = self.jersey_candidate().metadata["jersey_evidence"]
        changes = [
            {"anchor_legible": False},
            {"anchor_number": 6},
            {"anchor_number": True},
            {"component_match_samples": 1},
            {"component_match_samples": "invalid"},
            {"readings": {"not": "an array"}},
        ]
        for field, value in [
            ("image_sha256", "a" * 64),
            ("kit_compatible", False),
            ("number", 6),
            ("time_sec", 1.1),
        ]:
            readings = copy.deepcopy(baseline["readings"])
            readings[1][field] = value
            changes.append({"readings": readings})
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(
                    associate_identity(
                        identity,
                        [self.jersey_candidate(evidence_changes=change)],
                        thresholds=thresholds,
                    ).accepted
                )
        self.assertFalse(
            associate_identity(
                identity, [self.jersey_candidate(vector=(0, 1))], thresholds=thresholds
            ).accepted
        )

    def test_two_candidates_read_as_same_number_remain_ambiguous(self):
        from dataclasses import replace

        first = self.jersey_candidate()
        second = replace(first, candidate_id="another_8")
        decision = associate_identity(
            IdentityProfile("player", first.descriptor),
            [first, second],
            thresholds=AssociationThresholds(require_strong_overlap=True),
        )
        self.assertFalse(decision.accepted)

    def test_independently_confirmed_disjoint_tracklets_are_not_concurrent_rivals(self):
        from dataclasses import replace

        first = self.jersey_candidate()
        first = replace(
            first,
            metadata={
                **first.metadata,
                "tracklet_detections": [{"t": t} for t in (1.0, 2.0, 3.0)],
            },
        )
        second = replace(
            first,
            candidate_id="later_fragment",
            metadata={
                **first.metadata,
                "tracklet_detections": [{"t": t} for t in (5.0, 6.0, 7.0)],
            },
        )
        identity = IdentityProfile("player", first.descriptor)
        thresholds = AssociationThresholds(require_strong_overlap=True)
        self.assertTrue(
            associate_identity(
                identity, [first, second], thresholds=thresholds
            ).accepted
        )
        for times in [(2.0, 3.0, 4.0), (3.01, 4.0, 5.0), (float("nan"), 4.0, 5.0)]:
            overlapping = replace(
                second,
                metadata={
                    **second.metadata,
                    "tracklet_detections": [{"t": t} for t in times],
                },
            )
            self.assertFalse(
                associate_identity(
                    identity, [first, overlapping], thresholds=thresholds
                ).accepted
            )

    def response(self, number=8, **changes):
        reading = dict(number=number, legible=True, single_player=True, view="back")
        reading.update(changes)
        return Mock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(reading)},
                    }
                ],
                "usage": {"total_tokens": 20},
            },
        )

    def test_request_is_unbiased_and_cached_without_credentials_in_output(self):
        transport = Mock(return_value=self.response())
        with patch.dict(
            "os.environ", {"JERSEY_OCR_ENABLED": "1", "OPENAI_API_KEY": "test-secret"}
        ):
            reader = JerseyReader(transport=transport)
            crop = JerseyCrop(b"image", 1.0, 0.9, True)
            first = reader.read(crop)
            second = reader.read(JerseyCrop(b"image", 4.0, 0.9, True))
        self.assertEqual(first["number"], 8)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["time_sec"], 4)
        self.assertEqual(transport.call_count, 1)
        body = transport.call_args.kwargs["json"]
        self.assertFalse(body["store"])
        self.assertNotIn("target", json.dumps(body))
        self.assertNotIn("test-secret", json.dumps(reader.summary()))
        self.assertEqual(body["response_format"]["json_schema"]["strict"], True)
        self.assertFalse(transport.call_args.kwargs["allow_redirects"])

    def test_budget_and_disabled_key_do_not_call_api(self):
        transport = Mock(return_value=self.response())
        with patch.dict(
            "os.environ",
            {
                "JERSEY_OCR_ENABLED": "1",
                "OPENAI_API_KEY": "test",
                "JERSEY_OCR_MAX_CALLS": "1",
            },
        ):
            reader = JerseyReader(transport=transport)
            reader.read(JerseyCrop(b"one", 0, 1))
            result = reader.read(JerseyCrop(b"two", 1, 1))
        self.assertEqual(result["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(transport.call_count, 1)
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            result = JerseyReader(transport=transport).read(JerseyCrop(b"three", 2, 1))
        self.assertFalse(result["legible"])
        self.assertEqual(transport.call_count, 1)

    def test_errors_circuit_break_and_never_reveal_exception_content(self):
        transport = Mock(side_effect=RuntimeError("secret or image contents"))
        with patch.dict(
            "os.environ", {"JERSEY_OCR_ENABLED": "1", "OPENAI_API_KEY": "test"}
        ):
            reader = JerseyReader(transport=transport)
            with self.assertLogs("app.reid.jersey_vision", level="WARNING") as logs:
                results = [reader.read(JerseyCrop(bytes([i]), i, 1)) for i in range(4)]
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(results[-1]["status"], "CIRCUIT_OPEN")
        self.assertNotIn("secret or image", str(logs.output))

    def test_partial_refused_malformed_or_uncertain_never_becomes_number(self):
        for reading in [
            {"number": True, "legible": True, "single_player": True, "view": "back"},
            {"number": 108, "legible": True, "single_player": True, "view": "back"},
            {"number": "8", "legible": True, "single_player": True, "view": "back"},
        ]:
            with self.assertRaises(ValueError):
                parse_reading(reading)
        for field, value in [
            ("legible", False),
            ("single_player", False),
            ("view", "unknown"),
        ]:
            reading = dict(number=8, legible=True, single_player=True, view="back")
            reading[field] = value
            self.assertIsNone(parse_reading(reading)["number"])
        transport = Mock(
            return_value=Mock(
                status_code=200,
                json=lambda: {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"number":8}'},
                        }
                    ]
                },
            )
        )
        with patch.dict(
            "os.environ", {"JERSEY_OCR_ENABLED": "1", "OPENAI_API_KEY": "test"}
        ):
            self.assertIsNone(
                JerseyReader(transport=transport).read(JerseyCrop(b"a", 0, 1))["number"]
            )

    def test_consensus_needs_distinct_images_times_and_same_kit(self):
        first = dict(
            number=8, legible=True, image_sha256="a", time_sec=0, kit_compatible=True
        )
        second = {**first, "image_sha256": "b", "time_sec": 1}
        self.assertEqual(evaluate_readings([first, second], 8)["status"], "MATCH")
        for invalid in [
            {**second, "image_sha256": "a"},
            {**second, "time_sec": 0.1},
            {**second, "kit_compatible": False},
        ]:
            self.assertEqual(
                evaluate_readings([first, invalid], 8)["status"], "UNVERIFIED"
            )
        self.assertEqual(
            evaluate_readings([first, second, {**second, "number": 6}], 8)["status"],
            "CONFLICT",
        )

    def test_number_cannot_bypass_cv_and_conflict_vetoes_strong_overlap(self):
        descriptor = AppearanceDescriptor((1, 0), 3, 0.9)
        identity = IdentityProfile("player", descriptor)
        metadata = dict(
            strong_overlap_unique=True,
            tracklet_scope="MOTION_CONTINUOUS_STRONG_OVERLAP",
            overlap_link_samples=3,
            overlap_previous_samples=3,
            tracklet_sample_indices=[1, 2, 3],
        )
        thresholds = AssociationThresholds(require_strong_overlap=True)
        conflict = CandidateProfile(
            "candidate",
            descriptor,
            0.95,
            0.95,
            3,
            {**metadata, "jersey_evidence": {"status": "CONFLICT"}},
        )
        decision = associate_identity(identity, [conflict], thresholds=thresholds)
        self.assertFalse(decision.accepted)
        self.assertIn("JERSEY_NUMBER_CONFLICT", decision.reason_codes)
        number_only = CandidateProfile(
            "candidate",
            descriptor,
            None,
            None,
            3,
            {"jersey_evidence": {"status": "MATCH"}},
        )
        self.assertFalse(
            associate_identity(identity, [number_only], thresholds=thresholds).accepted
        )


class WorkspaceTests(unittest.TestCase):
    def test_guard_keeps_only_safe_rejection_diagnostics(self):
        from app.reid.team_color_guard import _team_color_guard_failure_output

        output = _team_color_guard_failure_output(
            {
                "segments": [
                    {
                        "window_index": 1,
                        "bboxes": [{"x": 0.2}],
                        "reid": {
                            "reason_codes": [
                                "LOW_COMBINED_SCORE",
                                "ASSOCIATION_ACCEPTED",
                                "https://secret",
                            ],
                            "best_score": 0.3,
                            "margin": float("nan"),
                        },
                    }
                ],
                "reid_summary": {
                    "jersey_vision": {
                        "calls": 3,
                        "anchor_reading": {"number": 8},
                        "url": "https://secret",
                    }
                },
            },
            status="ANCHOR_ONLY",
            reason_code="TEAM_COLOR_GUARD_UNVERIFIED_FAILURE_OUTPUT",
        )
        diagnostics = output["pre_guard_reid_diagnostics"]
        self.assertEqual(diagnostics["reason_counts"], {"LOW_COMBINED_SCORE": 1})
        self.assertEqual(diagnostics["jersey_vision"]["anchor_number"], 8)
        self.assertTrue(diagnostics["diagnostic_only"])
        self.assertFalse(output["tracking_success"])
        self.assertNotIn("secret", json.dumps(output))
        self.assertEqual(output["segments"][0]["bboxes"], [])

    def test_cleanup_removes_only_attempt_and_refuses_traversal_or_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "job" / "attempts" / "first"
            sibling = target.parent / "second"
            target.mkdir(parents=True)
            sibling.mkdir()
            (target / "window.mp4").write_bytes(b"temporary")
            source = root / "job" / "input.mp4"
            source.write_bytes(b"source")
            self.assertFalse(cleanup_tracking_workspace("../job", "first", root=root))
            self.assertTrue(cleanup_tracking_workspace("job", "first", root=root))
            self.assertTrue(source.exists())
            self.assertTrue(sibling.exists())
            target.symlink_to(sibling, target_is_directory=True)
            self.assertFalse(cleanup_tracking_workspace("job", "first", root=root))
            self.assertTrue(sibling.exists())

    def test_low_disk_fails_before_reserve_is_exhausted(self):
        with patch(
            "app.core.workspace.shutil.disk_usage",
            return_value=SimpleNamespace(free=5 * 1024**3),
        ):
            require_free_space(Path("/tmp"))
            with self.assertRaises(InsufficientWorkspaceError):
                require_free_space(Path("/tmp"), incoming_bytes=2 * 1024**3)


if __name__ == "__main__":
    unittest.main()
