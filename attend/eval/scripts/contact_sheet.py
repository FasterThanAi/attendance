#!/usr/bin/env python3
"""Phase 4 deliverable 4: is the quality gate actually making sensible calls?

Reads a job's quality.parquet (written by pipeline.quality.run_quality_stage)
and the frame JPEGs it was computed from, samples N random accepted and N
random rejected detections, crops each out of its source frame, and writes
one grid image with every crop labelled (quality score for accepted crops,
reject_reason for rejected ones). Phase 4's own definition of done, verbatim:
"Manually inspecting 30 accepted and 30 rejected crops confirms the gate is
making sensible calls" -- this script is exactly that inspection, made fast.

What to look for: accepted crops should all look like genuinely usable,
roughly-frontal, in-focus faces; rejected crops should look like exactly
what their reject_reason says (a "too_blurred" crop that looks perfectly
sharp to your eye means blur_laplacian_min needs recalibrating in Phase 7,
not that this script is wrong).

Standalone on purpose, same as gallery_sanity.py and draw_detections.py:
only pandas/numpy/opencv, no worker package dependency.

Usage:
    python eval/scripts/contact_sheet.py --job-dir /data/jobs/<job_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

THUMB_SIZE = 160
LABEL_HEIGHT = 24
PAD = 6
GRID_COLUMNS = 8

ACCEPTED_LABEL_COLOR = (0, 200, 0)  # BGR: green
REJECTED_LABEL_COLOR = (0, 0, 220)  # BGR: red


def _crop_from_frame(frame_paths: list[Path], row: pd.Series) -> np.ndarray | None:
    frame_index = int(row["frame_index"])
    if not (0 <= frame_index < len(frame_paths)):
        return None
    image = cv2.imread(str(frame_paths[frame_index]))
    if image is None:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _make_tile(crop: np.ndarray | None, label: str, color: tuple[int, int, int]) -> np.ndarray:
    if crop is not None and crop.size > 0:
        thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
    else:
        thumb = np.full((THUMB_SIZE, THUMB_SIZE, 3), 40, dtype=np.uint8)  # dark grey -- "couldn't even crop this one"

    tile = np.full((THUMB_SIZE + LABEL_HEIGHT, THUMB_SIZE, 3), 255, dtype=np.uint8)
    tile[:THUMB_SIZE] = thumb
    cv2.putText(tile, label[:24], (2, THUMB_SIZE + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return tile


def build_contact_sheet(
    quality_df: pd.DataFrame, frame_dir: Path, n_accepted: int, n_rejected: int, seed: int = 0
) -> np.ndarray | None:
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))

    # .astype(bool): on a genuinely empty quality.parquet (0 detections),
    # "accepted" round-trips as object-dtype, and boolean-masking a
    # DataFrame with an object-dtype column -- even an empty one -- silently
    # drops every column in pandas, not just every row.
    accepted_mask = quality_df["accepted"].astype(bool)
    accepted_rows = quality_df[accepted_mask]
    rejected_rows = quality_df[~accepted_mask]

    sampled_accepted = (
        accepted_rows.sample(n=min(n_accepted, len(accepted_rows)), random_state=seed) if len(accepted_rows) else accepted_rows
    )
    sampled_rejected = (
        rejected_rows.sample(n=min(n_rejected, len(rejected_rows)), random_state=seed) if len(rejected_rows) else rejected_rows
    )

    tiles = []
    for _, row in sampled_accepted.iterrows():
        crop = _crop_from_frame(frame_paths, row)
        label = f"OK q={row['quality_score']:.2f}"
        tiles.append(_make_tile(crop, label, ACCEPTED_LABEL_COLOR))
    for _, row in sampled_rejected.iterrows():
        crop = _crop_from_frame(frame_paths, row)
        label = str(row["reject_reason"])
        tiles.append(_make_tile(crop, label, REJECTED_LABEL_COLOR))

    if not tiles:
        return None

    n_cols = GRID_COLUMNS
    n_rows = -(-len(tiles) // n_cols)  # ceil division
    tile_h, tile_w = tiles[0].shape[:2]

    sheet = np.full((n_rows * (tile_h + PAD) + PAD, n_cols * (tile_w + PAD) + PAD, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, n_cols)
        y0 = PAD + r * (tile_h + PAD)
        x0 = PAD + c * (tile_w + PAD)
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    return sheet


def run(job_dir: Path, out_path: Path, n_accepted: int, n_rejected: int, seed: int) -> int:
    quality_parquet = job_dir / "quality" / "quality.parquet"
    frame_dir = job_dir / "extract"

    if not quality_parquet.exists():
        print(f"No quality.parquet at {quality_parquet} -- has the quality stage run for this job?")
        return 1
    if not frame_dir.exists():
        print(f"No frame directory at {frame_dir} -- has the extract stage run for this job?")
        return 1

    quality_df = pd.read_parquet(quality_parquet)
    accepted_n = int(quality_df["accepted"].sum())
    rejected_n = len(quality_df) - accepted_n
    print(f"quality.parquet: {len(quality_df)} total, {accepted_n} accepted, {rejected_n} rejected.")

    if len(quality_df):
        print("\nRejection reasons:")
        print(quality_df.loc[~quality_df["accepted"], "reject_reason"].value_counts().to_string())

    sheet = build_contact_sheet(quality_df, frame_dir, n_accepted, n_rejected, seed)
    if sheet is None:
        print("\nNo accepted or rejected rows found -- nothing to draw.")
        return 1

    cv2.imwrite(str(out_path), sheet)
    print(f"\nWrote contact sheet ({min(n_accepted, accepted_n)} accepted + "
          f"{min(n_rejected, rejected_n)} rejected tiles) to {out_path}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path, help="Job directory, e.g. /data/jobs/42")
    parser.add_argument("--out-path", type=Path, default=None, help="Defaults to <job-dir>/contact_sheet.jpg")
    parser.add_argument("--n-accepted", type=int, default=30, help="Phase 4's own definition of done: inspect 30")
    parser.add_argument("--n-rejected", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0, help="Random sample seed, for reproducible sheets")
    args = parser.parse_args()

    out_path = args.out_path or (args.job_dir / "contact_sheet.jpg")
    return run(args.job_dir, out_path, args.n_accepted, args.n_rejected, args.seed)


if __name__ == "__main__":
    sys.exit(main())
