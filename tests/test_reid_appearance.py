import unittest

import cv2
import numpy as np

from app.reid.appearance import (
    aggregate_appearance_descriptors,
    crop_from_normalized_bbox,
    extract_appearance_descriptor,
)
from app.reid.association import cosine_similarity


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


if __name__ == "__main__":
    unittest.main()
