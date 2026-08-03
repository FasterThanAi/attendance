#!/usr/bin/env python3
"""Phase 5 deliverable 4: is clustering actually finding one person per
cluster?

Reads a job's clusters.parquet, cluster_summary.parquet, aligned.npy, and
aligned_index.parquet (all written by pipeline.cluster.run_cluster_stage /
pipeline.align.align_crops), and produces:

  - a summary: number of clusters, number of noise points, cluster size
    distribution, cluster tightness distribution
  - one contact-sheet image PER CLUSTER: a grid of up to 12 member crops

The per-cluster contact sheets are the main learning artifact of this phase
(Phase 5 prompt, verbatim: "Looking at them is how you will understand what
your system is actually doing"). What to look for: every crop in one
cluster's sheet should be the SAME person. A sheet with two different faces
in it means DBSCAN merged two people (a "merging" failure mode); the same
person appearing in the sheets of two DIFFERENT clusters means DBSCAN split
one person into two identities (an "over-splitting" failure mode). Both are
named, expected failure modes per the roadmap, not signs something is
broken -- this script exists to help you SEE which one (if either) you have
and how often.

Standalone on purpose, same as the other eval/scripts/*.py: only
pandas/numpy/opencv, no worker package dependency.

Usage:
    python eval/scripts/cluster_report.py --job-dir /data/jobs/<job_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

THUMB_SIZE = 112  # aligned.npy crops are already exactly this size
LABEL_HEIGHT = 20
PAD = 4
GRID_COLUMNS = 4
MAX_MEMBERS_PER_SHEET = 12


def _print_histogram(label: str, values: pd.Series, bins: int = 10) -> None:
    print(f"\n{label} (n={len(values)})")
    if len(values) == 0:
        print("  (no data)")
        return
    arr = values.to_numpy(dtype=float)
    print(f"  mean={arr.mean():.3f}  min={arr.min():.3f}  p50={np.percentile(arr, 50):.3f}  max={arr.max():.3f}")
    counts, edges = np.histogram(arr, bins=bins)
    max_count = max(counts.max(), 1)
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(40 * count / max_count)
        print(f"  [{lo:6.2f}, {hi:6.2f}) {bar} ({count})")


def _make_tile(crop: np.ndarray, label: str) -> np.ndarray:
    tile = np.full((THUMB_SIZE + LABEL_HEIGHT, THUMB_SIZE, 3), 255, dtype=np.uint8)
    tile[:THUMB_SIZE] = crop
    cv2.putText(tile, label[:16], (2, THUMB_SIZE + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def build_cluster_sheet(member_crops: list[np.ndarray], member_labels: list[str]) -> np.ndarray:
    tiles = [_make_tile(crop, label) for crop, label in zip(member_crops, member_labels)]
    n_cols = min(GRID_COLUMNS, len(tiles))
    n_rows = -(-len(tiles) // n_cols)
    tile_h, tile_w = tiles[0].shape[:2]

    sheet = np.full((n_rows * (tile_h + PAD) + PAD, n_cols * (tile_w + PAD) + PAD, 3), 230, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, n_cols)
        y0 = PAD + r * (tile_h + PAD)
        x0 = PAD + c * (tile_w + PAD)
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return sheet


def run(job_dir: Path, out_dir: Path, max_members_per_sheet: int, seed: int) -> int:
    clusters_parquet = job_dir / "cluster" / "clusters.parquet"
    cluster_summary_parquet = job_dir / "cluster" / "cluster_summary.parquet"
    aligned_npy = job_dir / "align" / "aligned.npy"
    aligned_index_parquet = job_dir / "align" / "aligned_index.parquet"

    for path in (clusters_parquet, cluster_summary_parquet, aligned_npy, aligned_index_parquet):
        if not path.exists():
            print(f"Missing {path} -- has the cluster stage (and align, for aligned.npy) run for this job?")
            return 1

    clusters_df = pd.read_parquet(clusters_parquet)
    summary_df = pd.read_parquet(cluster_summary_parquet)
    aligned_index_df = pd.read_parquet(aligned_index_parquet)
    aligned = np.load(aligned_npy, mmap_mode="r")

    det_id_to_row = dict(zip(aligned_index_df["det_id"], aligned_index_df["row_index"]))

    noise_count = int((clusters_df["cluster_id"] == -1).sum())
    cluster_count = len(summary_df)
    print(f"clusters.parquet: {len(clusters_df)} total detections, {cluster_count} clusters, {noise_count} noise points.")

    if cluster_count == 0:
        print("\nNo clusters found -- nothing to report on.")
        return 1

    _print_histogram("Cluster size distribution (member_count)", summary_df["member_count"])
    _print_histogram("Cluster tightness distribution (intra-cluster mean cosine similarity)", summary_df["tightness"])

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    written = 0
    for _, cluster_row in summary_df.sort_values("cluster_id").iterrows():
        cluster_id = int(cluster_row["cluster_id"])
        member_det_ids = clusters_df.loc[clusters_df["cluster_id"] == cluster_id, "det_id"].tolist()
        if not member_det_ids:
            continue

        if len(member_det_ids) > max_members_per_sheet:
            member_det_ids = list(rng.choice(member_det_ids, size=max_members_per_sheet, replace=False))

        crops, labels = [], []
        for det_id in member_det_ids:
            row_index = det_id_to_row.get(det_id)
            if row_index is None:
                continue
            crops.append(np.asarray(aligned[int(row_index)]))
            labels.append(str(det_id))

        if not crops:
            continue

        sheet = build_cluster_sheet(crops, labels)
        out_path = out_dir / f"cluster_{cluster_id:03d}.jpg"
        cv2.imwrite(str(out_path), sheet)
        written += 1

    print(f"\nWrote {written} per-cluster contact sheet(s) to {out_dir}.")
    print("Look for: every crop in one sheet is the same person (merging failure = "
          "two faces in one sheet; over-splitting failure = the same face appearing "
          "in two different cluster sheets -- cross-check visually).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path, help="Job directory, e.g. /data/jobs/42")
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to <job-dir>/cluster_reports")
    parser.add_argument("--max-members-per-sheet", type=int, default=MAX_MEMBERS_PER_SHEET)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.job_dir / "cluster_reports")
    return run(args.job_dir, out_dir, args.max_members_per_sheet, args.seed)


if __name__ == "__main__":
    sys.exit(main())
