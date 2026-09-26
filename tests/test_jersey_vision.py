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
    evaluate_readings,
    parse_reading,
    nearby_confirmation_detections,
)


class JerseyVisionTests(unittest.TestCase):
    def test_confirmation_samples_are_nearby_and_temporally_independent(self):
        detections = [
            {"t": t} for t in [0, 10, 10.2, 11.001, 12.002, 13.003, 14.004, 15.005, 30]
        ]
        selected = nearby_confirmation_detections(
            detections, [112.002], [112.002, 100, 130], 100
        )
        self.assertEqual({d["t"] for d in selected}, {10.2, 11.001, 13.003, 14.004})

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
