#!/usr/bin/env python3
"""Phase 7 deliverable 3: match_threshold calibration sweep.

Re-runs ONLY match_clusters (never detect/embed/cluster) across
match_threshold from 0.25 to 0.60 in steps of 0.01, against each labelled
session's ALREADY-CACHED cluster representatives and gallery vectors --
cheap enough to do 36 times per session. For each threshold value, reports
precision/recall/F1 aggregated (by summing confusion counts, not averaging
per-session ratios -- a 90-student session and an 8-student session
shouldn't count equally) across every session passed in.

Produces:
  - a printed table (one row per threshold)
  - {out-dir}/precision_recall_curve.png
  - {out-dir}/fp_fn_vs_threshold.png

Does NOT choose an operating point for you. Phase 7's prompt is explicit
that this is a judgment call the roadmap wants a WRITTEN PARAGRAPH
justifying ("false positives break the anti-proxy claim, false negatives
break teacher trust -- decide which you are optimising for and say so"),
made by a person looking at real curves from real labelled sessions. This
script's job ends at giving you the curves and the table; once you've
picked a value, write it into pipeline/params.py's match_threshold field
yourself, with a comment recording the date, the dataset, and your
reasoning -- see the ASSUMPTION comment already there for the format every
other calibrated constant in this file follows.

Usage:
    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \\
        python eval/scripts/sweep_threshold.py --job-data-dir /data/jobs \\
        --datasets-dir eval/datasets --out-dir eval/reports
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display in a container -- write PNGs directly
import matplotlib.pyplot as plt
import pandas as pd

import eval_lib as el
from pipeline.params import PipelineParams

DEFAULT_THRESHOLDS = [round(0.25 + 0.01 * i, 2) for i in range(36)]  # 0.25 .. 0.60 inclusive


def sweep_all_sessions(
    database_url: str, datasets_dir: Path, job_data_dir: Path, base_params: PipelineParams,
    thresholds: list[float], session_ids: list[str] | None,
) -> pd.DataFrame:
    all_session_dirs = sorted(p.name for p in datasets_dir.iterdir() if p.is_dir()) if datasets_dir.exists() else []
    target_sessions = session_ids if session_ids else all_session_dirs

    per_session_dfs = []
    for session_id in target_sessions:
        truth_path = datasets_dir / session_id / "truth.csv"
        if not truth_path.exists():
            print(f"Skipping session {session_id}: no truth.csv.")
            continue
        truth_df = el.load_truth_csv(truth_path)

        try:
            ctx = el.fetch_session_context(database_url, int(session_id), job_data_dir)
            cluster_reps = el.load_cluster_representatives(ctx.job_dir)
        except Exception as e:
            print(f"Skipping session {session_id}: {e}")
            continue

        sweep_df = el.sweep_match_threshold(
            cluster_reps, ctx.gallery, ctx.roll_number_by_student_id, truth_df, base_params, thresholds,
        )
        sweep_df["session_id"] = session_id
        per_session_dfs.append(sweep_df)
        print(f"session {session_id}: swept {len(thresholds)} threshold value(s).")

    if not per_session_dfs:
        return pd.DataFrame(columns=["match_threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1"])

    combined = pd.concat(per_session_dfs, ignore_index=True)
    agg = combined.groupby("match_threshold")[["tp", "fp", "fn", "tn"]].sum().reset_index()

    precisions, recalls, f1s = [], [], []
    for _, row in agg.iterrows():
        counts = el.ConfusionCounts(tp=int(row["tp"]), fp=int(row["fp"]), fn=int(row["fn"]), tn=int(row["tn"]))
        p, r, f1 = el.precision_recall_f1(counts)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    agg["precision"] = precisions
    agg["recall"] = recalls
    agg["f1"] = f1s
    return agg.sort_values("match_threshold").reset_index(drop=True)


def plot_precision_recall_curve(agg: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(agg["recall"], agg["precision"], marker="o", markersize=3, linewidth=1)
    for _, row in agg.iloc[::5].iterrows():  # label every 5th point so it stays readable
        ax.annotate(f"{row['match_threshold']:.2f}", (row["recall"], row["precision"]), fontsize=7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve across match_threshold sweep")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fp_fn_vs_threshold(agg: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agg["match_threshold"], agg["fp"], marker="o", markersize=3, label="False positives (proxy hole)", color="tab:red")
    ax.plot(agg["match_threshold"], agg["fn"], marker="o", markersize=3, label="False negatives (trust killer)", color="tab:blue")
    ax.set_xlabel("match_threshold")
    ax.set_ylabel("Count")
    ax.set_title("False positive / false negative count vs match_threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_table(agg: pd.DataFrame) -> None:
    print(f"\n{'threshold':>9}  {'tp':>4}  {'fp':>4}  {'fn':>4}  {'tn':>4}  {'precision':>9}  {'recall':>7}  {'f1':>6}")
    for _, row in agg.iterrows():
        # groupby(...).sum() can upcast tp/fp/fn/tn to float64 even though
        # they started as Python ints -- cast back explicitly rather than
        # assume the dtype survived the aggregation.
        print(f"{row['match_threshold']:9.2f}  {int(row['tp']):4d}  {int(row['fp']):4d}  "
              f"{int(row['fn']):4d}  {int(row['tn']):4d}  "
              f"{row['precision']:9.3f}  {row['recall']:7.3f}  {row['f1']:6.3f}")


def run(
    database_url: str, datasets_dir: Path, job_data_dir: Path, out_dir: Path, base_params: PipelineParams,
    thresholds: list[float], session_ids: list[str] | None,
) -> int:
    agg = sweep_all_sessions(database_url, datasets_dir, job_data_dir, base_params, thresholds, session_ids)
    if len(agg) == 0:
        print("No sessions could be swept -- nothing to report.")
        return 1

    print_table(agg)

    out_dir.mkdir(parents=True, exist_ok=True)
    pr_path = out_dir / "precision_recall_curve.png"
    fpfn_path = out_dir / "fp_fn_vs_threshold.png"
    plot_precision_recall_curve(agg, pr_path)
    plot_fp_fn_vs_threshold(agg, fpfn_path)
    print(f"\nWrote {pr_path} and {fpfn_path}.")

    best_f1_row = agg.loc[agg["f1"].idxmax()]
    print(f"\nFor reference only (NOT a recommendation -- see this script's docstring): the threshold "
          f"with the highest F1 in this sweep is {best_f1_row['match_threshold']:.2f} "
          f"(precision={best_f1_row['precision']:.3f}, recall={best_f1_row['recall']:.3f}). Maximising F1 "
          f"treats a false positive and a false negative as equally costly, which the roadmap explicitly "
          f"says they are not (\"false positives break the anti-proxy claim, false negatives break teacher "
          f"trust\") -- look at the FP/FN-vs-threshold plot and the table above, decide which failure mode "
          f"this deployment can tolerate less, and choose accordingly. Then update pipeline/params.py's "
          f"match_threshold with a comment recording today's date, which sessions you calibrated against, "
          f"and why.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", type=Path, default=Path("eval/datasets"))
    parser.add_argument("--job-data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/reports"))
    parser.add_argument("--session-ids", default=None, help="Comma-separated class_session_ids; default: every labelled session found")
    parser.add_argument("--min-threshold", type=float, default=0.25)
    parser.add_argument("--max-threshold", type=float, default=0.60)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("Set DATABASE_URL, e.g.:\n"
              "  DATABASE_URL=postgresql://attend:attend@localhost:5432/attend "
              "python eval/scripts/sweep_threshold.py --job-data-dir /data/jobs")
        return 1
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    n_steps = round((args.max_threshold - args.min_threshold) / args.step) + 1
    thresholds = [round(args.min_threshold + args.step * i, 2) for i in range(n_steps)]

    session_ids = args.session_ids.split(",") if args.session_ids else None
    return run(database_url, args.datasets_dir, args.job_data_dir, args.out_dir, PipelineParams(), thresholds, session_ids)


if __name__ == "__main__":
    sys.exit(main())
