"""Face alignment: 5-point similarity transform to a canonical 112x112 crop.

Built in full now (Phase 1), not stubbed, because enrollment cannot embed a
face without an aligned crop, and the alignment math itself never changes
across phases -- Phase 4's `align_crops` (the batch, dataframe-driven,
memmap-writing version for classroom video) is a thin wrapper around the
same `align_face` function defined here.

The reference landmark positions below are the standard ArcFace/InsightFace
112x112 canonical positions (the same constants used by insightface's
`face_align.norm_crop`). Getting these wrong silently degrades every
embedding downstream, which is why there's a test asserting eyes land within
2px of the expected position for a synthetic, known input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.params import PipelineParams

logger = logging.getLogger("attend.worker.align")

# Standard ArcFace/InsightFace canonical reference points for a 112x112
# aligned crop, in (x, y) order: left eye, right eye, nose tip, left mouth
# corner, right mouth corner.
ARCFACE_REFERENCE_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(
    image_bgr: np.ndarray,
    landmarks: tuple[tuple[float, float], ...],
    output_size: int = 112,
) -> np.ndarray:
    """Warp `image_bgr` so the given 5 landmarks map onto the canonical
    ArcFace reference positions, producing an `output_size` x `output_size`
    BGR uint8 crop.

    `landmarks` must be in the same order as ARCFACE_REFERENCE_LANDMARKS_112
    (left eye, right eye, nose, left mouth corner, right mouth corner) -- this
    is the order insightface's SCRFD detector returns them in (see detect.py).
    """
    src = np.array(landmarks, dtype=np.float32)

    if output_size != 112:
        # Scale the reference points if a caller ever wants a different
        # output size; 112 is the only size ArcFace r100 actually accepts,
        # so this is defensive rather than expected to be exercised.
        scale = output_size / 112.0
        dst = ARCFACE_REFERENCE_LANDMARKS_112 * scale
    else:
        dst = ARCFACE_REFERENCE_LANDMARKS_112

    transform_matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform_matrix is None:
        raise ValueError("cv2.estimateAffinePartial2D failed to find a transform for the given landmarks")

    aligned = cv2.warpAffine(image_bgr, transform_matrix, (output_size, output_size), borderValue=0.0)
    return aligned


# --------------------------------------------------------------------------
# Phase 4: batch alignment for classroom video -- a memmap-writing wrapper
# around the same align_face used by Phase 1's enrollment
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedSet:
    aligned_npy_path: Path
    aligned_index_parquet_path: Path
    count: int


def align_crops(accepted_df: pd.DataFrame, frame_dir: Path, out_dir: Path, params: PipelineParams) -> AlignedSet:
    """Phase 4 deliverable 2. `accepted_df` is expected to already be
    filtered to quality.parquet's accepted rows -- this function aligns
    every row it's given and does not re-check an `accepted` column itself;
    run_align_stage (below) does that filtering, so this stays a plain
    "given these rows, align them" transform.

    Writes aligned.npy as a SINGLE (N, 112, 112, 3) uint8 BGR memmap, not
    individual JPEGs -- later phases (embedding, clustering) load this array
    repeatedly, and per-crop JPEG decode overhead would dominate at that
    point (Phase 4 prompt, verbatim). Uses np.lib.format.open_memmap rather
    than a bare np.memmap so the written file is a self-describing .npy
    (shape/dtype in its header) -- later phases can just
    `np.load(path, mmap_mode="r")` without needing N/dtype/shape passed out
    of band. aligned_index.parquet maps each memmap row index back to
    det_id/frame_index/quality_score so later phases can join back to
    quality.parquet without re-deriving anything.

    Source frames are cached per frame_index (not re-read per crop) for the
    same reason as quality.score_detections.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output_size = params.embed_input_size

    aligned_npy_path = out_dir / "aligned.npy"
    aligned_index_parquet_path = out_dir / "aligned_index.parquet"

    n = len(accepted_df)
    if n == 0:
        # np.memmap/open_memmap don't like a zero-length leading dimension
        # on some platforms -- write a plain (still self-describing, still
        # loadable via np.load) empty array instead of crashing on a video
        # where every single detection was rejected.
        np.save(aligned_npy_path, np.zeros((0, output_size, output_size, 3), dtype=np.uint8))
        pd.DataFrame(columns=["row_index", "det_id", "frame_index", "quality_score"]).to_parquet(
            aligned_index_parquet_path
        )
        logger.warning("align stage: 0 accepted crops -- nothing to align (every detection was rejected upstream)")
        return AlignedSet(aligned_npy_path=aligned_npy_path, aligned_index_parquet_path=aligned_index_parquet_path, count=0)

    memmap = np.lib.format.open_memmap(
        aligned_npy_path, mode="w+", dtype=np.uint8, shape=(n, output_size, output_size, 3)
    )

    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    frames_cache: dict[int, np.ndarray | None] = {}
    index_rows: list[dict] = []
    unreadable_frame_count = 0

    for position, (_, row) in enumerate(accepted_df.iterrows()):
        frame_index = int(row["frame_index"])
        if frame_index not in frames_cache:
            image = None
            if 0 <= frame_index < len(frame_paths):
                image = cv2.imread(str(frame_paths[frame_index]))
            frames_cache[frame_index] = image

        image = frames_cache[frame_index]
        landmarks = tuple((float(row[f"lmk_x{i + 1}"]), float(row[f"lmk_y{i + 1}"])) for i in range(5))

        if image is None:
            # Should not happen for a row that passed the quality gate (that
            # gate already read this same frame successfully) -- but fail
            # into a black crop rather than crashing the whole stage over
            # one unreadable frame; the accepted_count vs. written-crop
            # count staying honest matters more than any single crop.
            aligned = np.zeros((output_size, output_size, 3), dtype=np.uint8)
            unreadable_frame_count += 1
        else:
            aligned = align_face(image, landmarks, output_size=output_size)

        memmap[position] = aligned
        index_rows.append({
            "row_index": position,
            "det_id": row["det_id"],
            "frame_index": frame_index,
            "quality_score": float(row["quality_score"]) if "quality_score" in row else float("nan"),
        })

    memmap.flush()
    del memmap  # release the file handle before returning -- callers may want to np.load() it immediately

    if unreadable_frame_count:
        logger.warning("align stage: %d crop(s) had an unreadable source frame, written as black", unreadable_frame_count)

    pd.DataFrame(index_rows).to_parquet(aligned_index_parquet_path)

    return AlignedSet(aligned_npy_path=aligned_npy_path, aligned_index_parquet_path=aligned_index_parquet_path, count=n)


def run_align_stage(quality_parquet_path: Path, frame_dir: Path, out_dir: Path, params: PipelineParams) -> AlignedSet:
    """The I/O wrapper run.py's orchestrator calls: reads quality.parquet,
    filters to accepted==True, and hands that off to align_crops. Kept
    separate from align_crops itself for the same reason
    quality.run_quality_stage is separate from score_detections -- the core
    transform stays a plain DataFrame-in transform, testable without a real
    quality.parquet file on disk.
    """
    quality_df = pd.read_parquet(quality_parquet_path)
    # .astype(bool) matters more than it looks: on a genuinely empty (0-row)
    # quality_df, "accepted" round-trips as an object-dtype column, and
    # boolean-masking a DataFrame with an object-dtype (not bool-dtype)
    # column -- even one with zero rows -- silently returns a DataFrame with
    # NO COLUMNS AT ALL in pandas, not just zero rows. Without this cast, a
    # video with zero detections would crash align_crops with a confusing
    # KeyError on "det_id" instead of correctly producing zero aligned crops.
    accepted_df = quality_df[quality_df["accepted"].astype(bool)].reset_index(drop=True)

    result = align_crops(accepted_df, frame_dir, out_dir, params)
    logger.info("align stage: %d accepted crop(s) aligned to %sx%s", result.count, params.embed_input_size, params.embed_input_size)
    return result
