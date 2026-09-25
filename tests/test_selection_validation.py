import unittest

from pydantic import ValidationError
from app.schemas import (
    JobCreate,
    PickPlayerPayload,
    PlayerRefPayload,
    SelectionBox,
    TrackSelectionBox,
    TargetSelectionPayload,
)

BOX = {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.3}


class SelectionValidationTests(unittest.TestCase):
    def test_zero_time_is_valid_for_every_selection_alias(self):
        for model, names in (
            (TrackSelectionBox, ("frame_time_sec", "time_sec", "frameTimeSec")),
            (
                TargetSelectionPayload,
                ("time_sec", "timeSec", "frame_time_sec", "frameTimeSec"),
            ),
        ):
            for name in names:
                with self.subTest(model=model, name=name):
                    item = model.model_validate({name: 0, "bbox": BOX})
                    self.assertEqual(
                        getattr(
                            item, "frame_time_sec", getattr(item, "time_sec", None)
                        ),
                        0,
                    )

    def test_zero_track_id_is_preserved(self):
        self.assertEqual(
            PickPlayerPayload.model_validate(
                {"track_id": 0, "frame_key": "frame.jpg"}
            ).track_id,
            0,
        )

    def test_rejects_nonfinite_and_outside_boxes(self):
        for model in (
            PlayerRefPayload,
            SelectionBox,
            TrackSelectionBox,
            TargetSelectionPayload,
        ):
            for field, value in (
                ("x", float("nan")),
                ("w", float("inf")),
                ("x", None),
                ("h", True),
                ("x", -0.1),
                ("w", 1.5),
            ):
                with self.subTest(
                    model=model, field=field, value=value
                ), self.assertRaises(ValidationError):
                    model.model_validate(
                        {
                            "frame_time_sec": 0,
                            "time_sec": 0,
                            "bbox": {**BOX, field: value},
                        }
                    )

    def test_rejects_negative_and_nonfinite_time(self):
        for model in (
            PlayerRefPayload,
            SelectionBox,
            TrackSelectionBox,
            TargetSelectionPayload,
        ):
            for value in (-1, float("inf"), float("nan")):
                with self.subTest(model=model, value=value), self.assertRaises(
                    ValidationError
                ):
                    model.model_validate(
                        {"frame_time_sec": value, "time_sec": value, "bbox": BOX}
                    )

    def test_empty_source_or_context_is_rejected_before_queueing(self):
        for values in ({"video_url": "   "}, {"role": " "}, {"category": " "}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                JobCreate.model_validate(
                    {
                        "video_url": "https://example.org/match.mp4",
                        "role": "MF",
                        "category": "U18",
                        **values,
                    }
                )


if __name__ == "__main__":
    unittest.main()
