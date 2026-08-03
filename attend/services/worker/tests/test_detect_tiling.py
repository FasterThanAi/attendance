"""Phase 3 deliverable 2: tile grid coverage, NMS merge across a tile
boundary, and tile-local-to-frame coordinate mapping.

Uses a FakeRawDetector (simple bright-rectangle finder via cv2 connected
components) instead of a real insightface SCRFD session, so these run
without onnxruntime/insightface installed -- only numpy/opencv, which this
sandbox actually has. Real-model behaviour (does SCRFD itself find a face)
is NOT what's under test here; the tiling/merge/coordinate-mapping LOGIC is,
and that logic doesn't care what produced the per-tile boxes.
"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.detect import (
    Detection,
    DetectorModel,
    compute_tile_grid,
    detect_faces_tiled,
    non_max_suppression,
)
from pipeline.params import PipelineParams


# --------------------------------------------------------------------------
# compute_tile_grid: coverage + overlap
# --------------------------------------------------------------------------


def test_compute_tile_grid_covers_full_frame_with_no_gaps():
    width, height, tile_size, overlap = 3000, 1800, 1280, 256
    tiles = compute_tile_grid(width, height, tile_size, overlap)

    # Every tile is within bounds and exactly tile-sized (except possibly
    # clipped by the frame edge, which for these dimensions doesn't happen
    # since tile_size <= both dimensions).
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
        assert (x1 - x0) == tile_size
        assert (y1 - y0) == tile_size

    # Coverage: every pixel column/row of the frame is covered by at least
    # one tile on each axis -- check via the union of x-ranges and y-ranges.
    x_ranges = sorted({(x0, x1) for x0, _, x1, _ in tiles})
    covered_x = np.zeros(width, dtype=bool)
    for x0, x1 in x_ranges:
        covered_x[x0:x1] = True
    assert covered_x.all(), "gap found along x axis -- some columns never covered by any tile"

    y_ranges = sorted({(y0, y1) for _, y0, _, y1 in tiles})
    covered_y = np.zeros(height, dtype=bool)
    for y0, y1 in y_ranges:
        covered_y[y0:y1] = True
    assert covered_y.all(), "gap found along y axis -- some rows never covered by any tile"

    # The last tile's far edge must exactly reach the frame boundary (the
    # whole point of _axis_tile_origins pinning the last origin).
    assert max(x1 for _, _, x1, _ in tiles) == width
    assert max(y1 for _, _, _, y1 in tiles) == height


def test_compute_tile_grid_neighbouring_tiles_overlap_by_requested_amount():
    width, height, tile_size, overlap = 3000, 1800, 1280, 256
    tiles = compute_tile_grid(width, height, tile_size, overlap)

    x_origins = sorted({x0 for x0, _, _, _ in tiles})
    # Interior consecutive tiles (not the last, clamped one) should be
    # exactly `stride` apart, i.e. overlap by exactly `overlap` px.
    stride = tile_size - overlap
    for a, b in zip(x_origins, x_origins[1:-1]):
        assert b - a == stride


def test_compute_tile_grid_small_frame_is_a_single_tile():
    # Enrollment-selfie-sized frame: no tiling needed at all.
    tiles = compute_tile_grid(640, 480, 1280, 256)
    assert len(tiles) == 1
    assert tiles[0] == (0, 0, 640, 480)


# --------------------------------------------------------------------------
# non_max_suppression: merges duplicate detections across a tile boundary
# --------------------------------------------------------------------------


def _det(x1, y1, x2, y2, score):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, landmarks=tuple((0.0, 0.0) for _ in range(5)))


def test_nms_merges_overlapping_duplicate_detections_keeping_highest_score():
    # Two near-identical boxes for the "same" face (as tiling + the
    # whole-frame pass would both produce), plus one clearly distinct face.
    same_face_a = _det(100, 100, 180, 180, score=0.70)
    same_face_b = _det(104, 102, 183, 181, score=0.91)  # slightly shifted, higher score
    distinct_face = _det(900, 900, 980, 980, score=0.85)

    kept = non_max_suppression([same_face_a, same_face_b, distinct_face], iou_threshold=0.4)

    assert len(kept) == 2
    scores = sorted(d.score for d in kept)
    assert scores == [0.85, 0.91]


def test_nms_keeps_detections_below_iou_threshold_separate():
    # IoU well under the 0.4 threshold -- both must survive.
    a = _det(0, 0, 50, 50, score=0.8)
    b = _det(45, 45, 95, 95, score=0.6)
    kept = non_max_suppression([a, b], iou_threshold=0.4)
    assert len(kept) == 2


# --------------------------------------------------------------------------
# detect_faces_tiled: tile-local coordinates map correctly back to frame
# coordinates, and the tile + whole-frame-downscaled passes merge into one
# --------------------------------------------------------------------------


class _FakeRawDetector:
    """Stands in for insightface's raw SCRFD model: finds bright rectangles
    via connected components and returns them in the same (bboxes, kpss)
    shape `detect_faces` expects, with a constant score. Deterministic and
    independent of tile boundaries -- exactly what's needed to test the
    coordinate-mapping/merge logic without a real ONNX model.
    """

    def detect(self, image_bgr: np.ndarray, metric: str = "default"):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        kpss = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 5 or h < 5:
                continue
            bboxes.append([float(x), float(y), float(x + w), float(y + h), 0.95])
            cx, cy = x + w / 2, y + h / 2
            kpss.append([[cx, cy]] * 5)

        if not bboxes:
            return np.zeros((0, 5)), np.zeros((0, 5, 2))
        return np.array(bboxes), np.array(kpss)


def test_detect_faces_tiled_maps_coordinates_back_to_frame_space():
    width, height = 3000, 1800
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # A "face" near the far edge of the frame -- with tile_size=1280,
    # overlap=256 the tile grid's last x-origin is pinned to 3000-1280=1720,
    # so this box (x in [2900, 2980)) falls inside the tile spanning
    # [1720, 3000) x [0, 1280) -- i.e. it genuinely exercises tile-local ->
    # frame coordinate translation, not just the (0,0)-origin tile.
    face_x1, face_y1, face_x2, face_y2 = 2900, 100, 2980, 180
    cv2.rectangle(image, (face_x1, face_y1), (face_x2, face_y2), (255, 255, 255), thickness=-1)

    model = DetectorModel(_FakeRawDetector())
    params = PipelineParams(
        tile_trigger_long_side_px=2000,
        tile_size_px=1280,
        tile_overlap_px=256,
        nms_iou_threshold=0.4,
        detector_score_min=0.5,
    )

    detections = detect_faces_tiled(image, model, params)

    # The tiled pass and the whole-frame-downscaled pass both find this same
    # face; NMS must merge them into exactly one final detection.
    assert len(detections) == 1
    d = detections[0]

    # Recovered box should land close to the true box -- some slack for the
    # whole-frame-downscaled pass's rescale rounding, tolerance chosen to be
    # tight enough to catch a real off-by-tile-origin bug (which would be
    # off by hundreds of pixels, not a handful).
    assert abs(d.x1 - face_x1) <= 6
    assert abs(d.y1 - face_y1) <= 6
    assert abs(d.x2 - face_x2) <= 6
    assert abs(d.y2 - face_y2) <= 6


def test_detect_faces_tiled_skips_tiling_for_small_frames():
    # Below tile_trigger_long_side_px: must behave exactly like plain
    # detect_faces (single call, no tiling, no whole-frame-downscale pass).
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (150, 150), (255, 255, 255), thickness=-1)

    model = DetectorModel(_FakeRawDetector())
    params = PipelineParams(tile_trigger_long_side_px=2000, detector_score_min=0.5)

    detections = detect_faces_tiled(image, model, params)
    assert len(detections) == 1
    d = detections[0]
    assert abs(d.x1 - 100) <= 1
    assert abs(d.y1 - 100) <= 1
