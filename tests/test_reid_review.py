import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.benchmark.reid_review import (
    build_reid_annotation_template,
    materialize_reid_review_pack,
    render_reid_review_html,
)


class ReIDReviewTemplateTests(unittest.TestCase):
    def tracking(self):
        return {
            "fps": 1,
            "segments": [
                {
                    "window_start": 0.0,
                    "window_end": 10.0,
                    "direction": "anchor",
                    "coverage_pct": 20.0,
                    "bboxes": [
                        {"t": 2.0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.2},
                        {"t": 5.0, "x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2},
                        {"t": 8.0, "x": 0.3, "y": 0.2, "w": 0.1, "h": 0.2},
                    ],
                    "reid": {
                        "status": "ACCEPTED",
                        "selected_candidate_id": "7",
                        "best_score": 0.9,
                        "margin": 0.1,
                        "reason_codes": ["MANUAL_ANCHOR"],
                        "candidates": [
                            {
                                "candidate_id": "7",
                                "combined_score": 0.9,
                                "appearance_similarity": 0.95,
                                "evidence": [
                                    {
                                        "time_sec": 5.0,
                                        "frame_index": 5,
                                        "bbox": {
                                            "x": 0.2,
                                            "y": 0.2,
                                            "w": 0.1,
                                            "h": 0.2,
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                },
                {
                    "window_start": 10.0,
                    "window_end": 20.0,
                    "direction": "forward",
                    "bboxes": [],
                    "reid": {
                        "status": "ABSTAINED",
                        "best_score": 0.7,
                        "reason_codes": ["LOW_COMBINED_SCORE"],
                        "candidates": [],
                    },
                },
            ],
        }

    def test_builds_one_review_item_per_window_with_prediction_context(self):
        template = build_reid_annotation_template(
            self.tracking(),
            video_id="job-1",
            identity="selected-player",
            samples_per_window=3,
        )

        self.assertEqual(len(template["windows"]), 2)
        first = template["windows"][0]
        second = template["windows"][1]
        self.assertEqual(first["target_visibility"], "UNCERTAIN")
        self.assertIsNone(first["candidate_state"])
        self.assertEqual(first["review_context"]["decision"], "ACCEPTED")
        self.assertEqual(len(first["evidence_frames"]), 3)
        self.assertIn("bbox", first["evidence_frames"][0])
        candidate = first["review_context"]["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "7")
        self.assertEqual(candidate["evidence"][0]["frame_index"], 5)
        self.assertEqual(second["review_context"]["decision"], "ABSTAINED")
        self.assertEqual(
            [frame["time_sec"] for frame in second["evidence_frames"]],
            [12.5, 15.0, 17.5],
        )

    def test_html_contains_window_and_frame_export_tools(self):
        template = build_reid_annotation_template(
            self.tracking(),
            video_id="job-1",
            identity="selected-player",
        )
        html = render_reid_review_html(template)
        self.assertIn("Esporta annotazioni finestre", html)
        self.assertIn("Esporta annotazioni frame / IDF1", html)
        self.assertIn("selected_track_is_target", html)
        self.assertIn("tracking-annotation-v1", html)
        self.assertIn("localStorage.setItem", html)
        self.assertIn("Azzera salvataggio locale", html)

    def test_html_escapes_video_id_in_document_title(self):
        template = build_reid_annotation_template(
            self.tracking(),
            video_id='<job & "unsafe">',
            identity="selected-player",
        )
        html = render_reid_review_html(template)
        self.assertIn("&lt;job &amp; &quot;unsafe&quot;&gt;", html)
        self.assertNotIn('<title>AlgoNext ReID review — <job', html)

    def test_materialize_pack_deduplicates_frame_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "video.mp4"
            video.write_bytes(b"placeholder")

            def fake_extract(_video, output, **_kwargs):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"jpeg")

            with patch(
                "app.benchmark.reid_review._extract_frame", side_effect=fake_extract
            ) as extract:
                result = materialize_reid_review_pack(
                    self.tracking(),
                    video_path=video,
                    output_dir=root / "pack",
                    video_id="job-1",
                    identity="selected-player",
                )

            self.assertTrue(Path(result["html_path"]).is_file())
            self.assertTrue(Path(result["template_path"]).is_file())
            # Candidate evidence at t=5 is shared with the selected-track sample.
            self.assertEqual(extract.call_count, result["frames_extracted"])
            self.assertEqual(result["frames_extracted"], 6)

    def test_rejects_zero_samples(self):
        with self.assertRaises(ValueError):
            build_reid_annotation_template(
                {"segments": []},
                video_id="job-1",
                identity="selected-player",
                samples_per_window=0,
            )


if __name__ == "__main__":
    unittest.main()
