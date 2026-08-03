#!/usr/bin/env python3
"""Phase 5 deliverable 5: parameter sweep across cluster_eps x
cluster_min_samples, reusing the SAME cached embeddings.npy for every grid
point -- never re-running detection or embedding. That's the entire point
of Phase 2's stage caching: without it, sweeping 5 eps values x 4
min_samples values (20 combinations) would mean re-running SCRFD detection
and ArcFace embedding 20 times over, for no reason -- clustering is the
only thing changing.

Unlike gallery_sanity.py/contact_sheet.py/cluster_report.py, this script
DOES import the worker's pipeline package (pipeline.cluster.cluster_embeddings,
the pure function) rather than staying fully standalone -- it needs the
exact same clustering logic run.py's real cluster stage uses, not a
reimplementation that could quietly drift from it. Run it in the worker's
own environment (`docker-compose exec worker python eval/scripts/sweep_cluster.py ...`).

Usage:
    python eval/scripts/sweep_cluster.py --job-dir /data/jobs/<job_id>
    python eval/scripts/sweep_cluster.py --job-dir /data/jobs/<job_id> \\
        --ground-truth labels.csv   # optional: det_id,true_label columns, enables purity
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

# Add services/worker to the path so `pipeline.*` imports resolve when this
# script is run directly (not installed as a package) -- same convention as
# the worker's own pytest config (pythonpath = ["."]).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "worker"))

from pipeline.cluster import cluster_embeddings  # noqa: E402
from pipeline.params import PipelineParams  # noqa: E402

# Phase 5 prompt, verbatim grid.
EPS_GRID = [round(v, 2) for v in np.arange(0.30, 0.55 + 1e-9, 0.02)]
MIN_SAMPLES_GRID = [2, 3, 4, 5]


def load_ground_truth(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["det_id"], df["true_label"]))


def compute_purity(assignments_df: pd.DataFrame, ground_truth: dict[str, str]) -> float | None:
    """Standard clustering purity: for each predicted (non-noise) cluster,
    take its majority TRUE label among members with known ground truth, sum
    those majority counts, divide by the total labelled+clustered points.
    None if no ground-truth det_id overlaps this job's detections at all.
    """
    labelled = assignments_df[assignments_df["det_id"].isin(ground_truth)]
    clustered = labelled[labelled["cluster_id"] != -1]
    if len(clustered) == 0:
        return None

    correct = 0
    for _, group in clustered.groupby("cluster_id"):
        true_labels = group["det_id"].map(ground_truth)
        correct += true_labels.value_counts().iloc[0]
    return correct / len(clustered)


def run(job_dir: Path, ground_truth_path: Path | None) -> int:
    embeddings_npy = job_dir / "embed" / "embeddings.npy"
    quality_parquet = job_dir / "quality" / "quality.parquet"
    aligned_index_parquet = job_dir / "align" / "aligned_index.parquet"

    for path in (embeddings_npy, quality_parquet, aligned_index_parquet):
        if not path.exists():
            print(f"Missing {path} -- run extract/detect/quality/align/embed for this job first.")
            return 1

    embeddings = np.asarray(np.load(embeddings_npy, mmap_mode="r"))
    quality_df_full = pd.read_parquet(quality_parquet)
    # .astype(bool): see pipeline/align.py's run_align_stage for why this
    # cast matters -- an empty quality_df's "accepted" column is
    # object-dtype, and boolean-masking with it (even 0 rows) silently
    # drops every column, not just every row.
    accepted_df = quality_df_full[quality_df_full["accepted"].astype(bool)].reset_index(drop=True)
    aligned_index_df = pd.read_parquet(aligned_index_parquet)

    if list(accepted_df["det_id"]) != list(aligned_index_df["det_id"]):
        print(
            "WARNING: quality.parquet's accepted rows don't match aligned_index.parquet's "
            "det_id order -- results below may be misaligned. Re-run 'align' after any "
            "'quality' param change before trusting this sweep."
        )
    if len(accepted_df) != embeddings.shape[0]:
        print(f"ERROR: {len(accepted_df)} accepted crops but {embeddings.shape[0]} embeddings -- stages out of sync.")
        return 1

    ground_truth = load_ground_truth(ground_truth_path) if ground_truth_path else None

    base_params = PipelineParams()
    print(
        f"Sweeping cluster_eps in {EPS_GRID} x cluster_min_samples in {MIN_SAMPLES_GRID} "
        f"over {len(accepted_df)} CACHED embeddings (detect/embed are not re-run).\n"
    )

    header = f"{'eps':>6} {'min_samples':>12} {'clusters':>9} {'noise':>7}"
    if ground_truth:
        header += f" {'purity':>8}"
    print(header)

    results = []
    for eps in EPS_GRID:
        for min_samples in MIN_SAMPLES_GRID:
            params = replace(base_params, cluster_eps=float(eps), cluster_min_samples=int(min_samples))
            result = cluster_embeddings(embeddings, accepted_df, params)

            cluster_count = len(result.diagnostics)
            noise_count = int((result.assignments_df["cluster_id"] == -1).sum())
            purity = compute_purity(result.assignments_df, ground_truth) if ground_truth else None

            row = f"{eps:6.2f} {min_samples:12d} {cluster_count:9d} {noise_count:7d}"
            row += f" {purity:8.3f}" if purity is not None else (" " + f"{'n/a':>8}" if ground_truth else "")
            print(row)

            results.append({
                "cluster_eps": eps, "cluster_min_samples": min_samples,
                "cluster_count": cluster_count, "noise_count": noise_count, "purity": purity,
            })

    out_path = job_dir / "cluster_sweep.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nWrote full sweep results to {out_path}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path, help="Job directory, e.g. /data/jobs/42")
    parser.add_argument(
        "--ground-truth", type=Path, default=None,
        help="Optional CSV with det_id,true_label columns, needed to report purity",
    )
    args = parser.parse_args()
    return run(args.job_dir, args.ground_truth)


if __name__ == "__main__":
    sys.exit(main())
