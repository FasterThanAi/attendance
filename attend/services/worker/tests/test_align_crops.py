"""Phase 4 deliverable 2 tests: align_crops's memmap output has the right
shape/dtype and each row's content actually matches align_face's own
(already Phase-1-proven) output; aligned_index.parquet correctly maps row
index back to det_id; a fully-empty accepted_df doesn't crash.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from pipeline.align import align_crops, align_face
from pipeline.params import PipelineParams

FRONTAL_LANDMARKS = ((130.0, 150.0), (170.0, 150.0), (150.0, 170.0), (135.0, 190.0), (165.0, 190.0))


def _write_frame(frame_dir, frame_index: int) -> np.ndarray:
    frame_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(frame_index)
    image = rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_dir / f"frame_{frame_index + 1:05d}.jpg"), image)
    return image


def _accepted_row(frame_index: int, det_id: str, quality_score: float, landmarks=FRONTAL_LANDMARKS) -> dict:
    row = {"frame_index": frame_index, "det_id": det_id, "quality_score": quality_score, "accepted": True, "reject_reason": None}
    for i, (lx, ly) in enumerate(landmarks):
        row[f"lmk_x{i + 1}"] = lx
        row[f"lmk_y{i + 1}"] = ly
    return row


def test_align_crops_writes_correctly_shaped_memmap_and_index(tmp_path):
    frame_dir = tmp_path / "extract"
    image0 = _write_frame(frame_dir, 0)
    image1 = _write_frame(frame_dir, 1)

    accepted_df = pd.DataFrame([
        _accepted_row(0, "0_0", quality_score=0.9),
        _accepted_row(1, "1_0", quality_score=0.7),
    ])

    out_dir = tmp_path / "align"
    result = align_crops(accepted_df, frame_dir, out_dir, PipelineParams())

    assert result.count == 2
    assert result.aligned_npy_path == out_dir / "aligned.npy"
    assert result.aligned_index_parquet_path == out_dir / "aligned_index.parquet"

    aligned_array = np.load(result.aligned_npy_path, mmap_mode="r")
    assert aligned_array.shape == (2, 112, 112, 3)
    assert aligned_array.dtype == np.uint8

    # Row 0 must match align_face's own output for the same frame/landmarks
    # exactly (align_crops is a thin loop around align_face, not a
    # reimplementation) -- ties this test back to Phase 1's already-proven
    # alignment math instead of re-asserting it from scratch. Compare
    # against align_face run on the RE-READ jpg (not the original in-memory
    # `image0`) -- align_crops reads frames back off disk same as it does,
    # and JPEG is lossy, so the on-disk bytes are what actually matters here.
    reloaded_image0 = cv2.imread(str(frame_dir / "frame_00001.jpg"))
    expected_row0 = align_face(reloaded_image0, FRONTAL_LANDMARKS, output_size=112)
    np.testing.assert_array_equal(np.array(aligned_array[0]), expected_row0)

    index_df = pd.read_parquet(result.aligned_index_parquet_path)
    assert list(index_df["det_id"]) == ["0_0", "1_0"]
    assert list(index_df["row_index"]) == [0, 1]
    assert list(index_df["frame_index"]) == [0, 1]


def test_align_crops_empty_accepted_df_does_not_crash(tmp_path):
    frame_dir = tmp_path / "extract"
    frame_dir.mkdir()
    accepted_df = pd.DataFrame(columns=["frame_index", "det_id", "quality_score", "lmk_x1", "lmk_y1", "lmk_x2", "lmk_y2", "lmk_x3", "lmk_y3", "lmk_x4", "lmk_y4", "lmk_x5", "lmk_y5"])

    out_dir = tmp_path / "align"
    result = align_crops(accepted_df, frame_dir, out_dir, PipelineParams())

    assert result.count == 0
    aligned_array = np.load(result.aligned_npy_path)
    assert aligned_array.shape == (0, 112, 112, 3)

    index_df = pd.read_parquet(result.aligned_index_parquet_path)
    assert len(index_df) == 0
    assert "det_id" in index_df.columns
