"""Canonical alignment position test (the test Phase 4's align.py prompt
asks for, added now since align.py itself was built in Phase 1 -- enrollment
can't embed a face without it).

Draws small white markers at "detected" landmark positions on a synthetic
image (an exact scale+translate of the canonical reference points -- a
similarity transform, no rotation), runs align_face, then locates the
brightest pixel near each expected canonical position in the OUTPUT crop and
asserts it lands within 2px, per the roadmap's own acceptance criterion.
"""

import cv2
import numpy as np

from pipeline.align import ARCFACE_REFERENCE_LANDMARKS_112, align_face


def _make_synthetic_face(scale: float, offset: tuple[float, float], canvas_size: int = 300):
    """Places a small white marker at each canonical landmark position,
    scaled and offset -- simulating what a detector would report for a face
    that isn't already perfectly aligned.
    """
    image = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    src_landmarks = []
    for x, y in ARCFACE_REFERENCE_LANDMARKS_112:
        sx, sy = x * scale + offset[0], y * scale + offset[1]
        cv2.circle(image, (int(round(sx)), int(round(sy))), radius=2, color=(255, 255, 255), thickness=-1)
        src_landmarks.append((sx, sy))
    return image, tuple(src_landmarks)


def test_align_face_recovers_canonical_positions():
    image, src_landmarks = _make_synthetic_face(scale=1.5, offset=(20.0, 15.0))

    aligned = align_face(image, src_landmarks, output_size=112)
    assert aligned.shape == (112, 112, 3)

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    for expected_x, expected_y in ARCFACE_REFERENCE_LANDMARKS_112:
        # Search a small window around the expected position for the marker.
        x0, x1 = max(0, int(expected_x) - 5), min(112, int(expected_x) + 6)
        y0, y1 = max(0, int(expected_y) - 5), min(112, int(expected_y) + 6)
        window = gray[y0:y1, x0:x1]
        assert window.size > 0
        local_y, local_x = np.unravel_index(np.argmax(window), window.shape)
        found_x, found_y = x0 + local_x, y0 + local_y

        assert abs(found_x - expected_x) <= 2, f"x off by {abs(found_x - expected_x)}"
        assert abs(found_y - expected_y) <= 2, f"y off by {abs(found_y - expected_y)}"
