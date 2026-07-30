import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.reid.appearance import (
    aggregate_appearance_descriptors,
    crop_from_normalized_bbox,
    extract_appearance_descriptor,
)
from app.reid.association import cosine_similarity
from app.reid.association import DESCRIPTOR_VERSION
from app.reid.osnet_embedding import OSNET_DESCRIPTOR_VERSION


def player_crop(upper_bgr, lower_bgr):
    image = np.zeros((128, 64, 3), dtype=np.uint8)
    image[:70, :] = np.array(upper_bgr, dtype=np.uint8)
    image[70:, :] = np.array(lower_bgr, dtype=np.uint8)
    cv2.line(image, (2, 2), (61, 125), (255, 255, 255), 2)
    return image


class ReIdAppearanceTests(unittest.TestCase):
    def test_same_uniform_is_more_similar_than_different_uniform(self):
        anchor = extract_appearance_descriptor(
            player_crop((20, 20, 220), (20, 20, 80))
        )
        same = extract_appearance_descriptor(
            player_crop((25, 25, 210), (25, 25, 85))
        )
        different = extract_appearance_descriptor(
            player_crop((220, 40, 20), (80, 30, 20))
        )

        self.assertIsNotNone(anchor)
        self.assertIsNotNone(same)
        self.assertIsNotNone(different)
        self.assertGreater(
            cosine_similarity(anchor, same),
            cosine_similarity(anchor, different),
        )

    def test_tiny_crop_is_rejected(self):
        tiny = np.zeros((8, 4, 3), dtype=np.uint8)
        self.assertIsNone(extract_appearance_descriptor(tiny))

    def test_aggregation_increases_sample_count(self):
        first = extract_appearance_descriptor(
            player_crop((20, 20, 220), (20, 20, 80))
        )
        second = extract_appearance_descriptor(
            player_crop((25, 25, 210), (25, 25, 85))
        )
        aggregate = aggregate_appearance_descriptors([first, second])

        self.assertIsNotNone(aggregate)
        self.assertEqual(aggregate.sample_count, 2)
        self.assertGreater(cosine_similarity(aggregate, first), 0.95)

    def test_normalized_bbox_crop(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[20:80, 50:100] = 255
        crop = crop_from_normalized_bbox(
            frame,
            {"x": 0.25, "y": 0.20, "w": 0.25, "h": 0.60},
            padding=0.0,
        )

        self.assertIsNotNone(crop)
        self.assertEqual(crop.shape[:2], (60, 50))
        self.assertEqual(int(crop.mean()), 255)

    def test_osnet_backend_combines_learned_and_colour_features(self):
        with patch.dict(
            "os.environ",
            {"PLAYER_REID_DESCRIPTOR_BACKEND": "osnet_hybrid"},
        ), patch(
            "app.reid.appearance.extract_osnet_embedding",
            return_value=(1.0, 0.0, 0.0, 0.0),
        ):
            descriptor = extract_appearance_descriptor(
                player_crop((20, 20, 220), (20, 20, 80))
            )

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.version, OSNET_DESCRIPTOR_VERSION)
        self.assertGreater(len(descriptor.vector), 4)

    def test_osnet_backend_falls_back_without_model(self):
        with patch.dict(
            "os.environ",
            {
                "PLAYER_REID_DESCRIPTOR_BACKEND": "osnet_hybrid",
                "PLAYER_REID_LEARNED_FAIL_OPEN": "1",
            },
        ), patch(
            "app.reid.appearance.extract_osnet_embedding",
            return_value=None,
        ):
            descriptor = extract_appearance_descriptor(
                player_crop((20, 20, 220), (20, 20, 80))
            )

        self.assertIsNotNone(descriptor)
        self.assertNotEqual(descriptor.version, OSNET_DESCRIPTOR_VERSION)

    def test_aggregation_prefers_surviving_learned_descriptors(self):
        crop = player_crop((20, 20, 220), (20, 20, 80))
        with patch.dict(
            "os.environ",
            {"PLAYER_REID_DESCRIPTOR_BACKEND": "osnet_hybrid"},
        ), patch(
            "app.reid.appearance.extract_osnet_embedding",
            side_effect=[(1.0, 0.0, 0.0, 0.0), None],
        ):
            learned = extract_appearance_descriptor(crop)
            fallback = extract_appearance_descriptor(crop)

        self.assertEqual(learned.version, OSNET_DESCRIPTOR_VERSION)
        self.assertEqual(fallback.version, DESCRIPTOR_VERSION)
        aggregate = aggregate_appearance_descriptors([fallback, learned])
        self.assertIsNotNone(aggregate)
        self.assertEqual(aggregate.version, OSNET_DESCRIPTOR_VERSION)
        self.assertEqual(aggregate.sample_count, 1)


if __name__ == "__main__":
    unittest.main()
