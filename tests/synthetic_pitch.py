from __future__ import annotations

import cv2
import numpy as np


def make_pitch_frame(*, brightness: float = 1.0, width: int = 640, height: int = 360) -> np.ndarray:
    brightness = max(0.2, min(1.0, brightness))
    frame = np.full(
        (height, width, 3),
        (
            int(35 * brightness),
            int(130 * brightness),
            int(35 * brightness),
        ),
        dtype=np.uint8,
    )
    points = np.array(
        [
            [int(width * 0.11), int(height * 0.14)],
            [int(width * 0.89), int(height * 0.14)],
            [int(width * 0.985), int(height * 0.92)],
            [int(width * 0.015), int(height * 0.92)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(
        frame,
        points,
        (
            int(40 * brightness),
            int(150 * brightness),
            int(40 * brightness),
        ),
    )
    white = (245, 245, 245)
    thickness = max(2, int(round(width / 160)))
    cv2.polylines(frame, [points], True, white, thickness)
    cv2.line(
        frame,
        (width // 2, int(height * 0.14)),
        (width // 2, int(height * 0.92)),
        white,
        thickness,
    )
    cv2.circle(
        frame,
        (width // 2, int(height * 0.53)),
        int(min(width, height) * 0.15),
        white,
        thickness,
    )
    cv2.rectangle(
        frame,
        (int(width * 0.015), int(height * 0.34)),
        (int(width * 0.19), int(height * 0.72)),
        white,
        thickness,
    )
    cv2.rectangle(
        frame,
        (int(width * 0.81), int(height * 0.34)),
        (int(width * 0.985), int(height * 0.72)),
        white,
        thickness,
    )
    return frame


def make_non_pitch_frame(*, width: int = 640, height: int = 360) -> np.ndarray:
    frame = np.full((height, width, 3), (40, 40, 210), dtype=np.uint8)
    cv2.rectangle(
        frame,
        (int(width * 0.30), int(height * 0.10)),
        (int(width * 0.70), int(height * 0.96)),
        (80, 150, 210),
        -1,
    )
    cv2.circle(
        frame,
        (width // 2, int(height * 0.28)),
        int(min(width, height) * 0.12),
        (100, 180, 220),
        -1,
    )
    return frame
