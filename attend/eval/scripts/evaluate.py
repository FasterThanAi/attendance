#!/usr/bin/env python3
"""Phase 7 deliverable 2: the core evaluation harness.

For a set of labelled sessions (eval/datasets/{class_session_id}/truth.csv)
and a given PipelineParams, computes and prints:

  - STUDENT-LEVEL PRESENCE METRICS: precision, recall, F1, accuracy, with
    false positives (proxy hole) and false negatives (trust killer) reported
    separately and prominently, never folded into one "error rate."
  - CLUSTERING QUALITY: cluster count vs actual present count, and (if you've
    spot-labelled any clusters -- see eval_lib.clustering_quality) purity/
    over-split rate/merge rate.
  - PIPELINE YIELD: mean detections per present student, and the fraction of
    present students who produced zero accepted crops at all -- the
    roadmap's own framing: "these are unrecoverable failures -- the most
    important single diagnostic."
  - STRATIFIED BREAKDOWN by row_number, seat_position, and wears_glasses --
    mandatory, not optional (Phase 7 prompt, verbatim), because an aggregate
    number can hide a system that only works in the front row.

Every metric is recomputed from CACHED artifacts (embeddings.npy,
quality.parquet, cluster_summary.parquet) already sitting under each
session's job_dir, never by re-detecting/re-embedding a video -- exactly
the same "expensive stages stay cached" principle pipeline/run.py's own
manifest system exists for (Phase 2, non-negotiable rule #2).

ABLATIONS (Phase 7 deliverable 6) -- --ablation {none,temporal-coherence-off,
quality-weighting-off,clustering-off} recomputes clustering/matching
in-memory from cached embeddings with that one thing disabled. Two of the
roadmap's five ablations ("tiled detection off", "quality gating off")
change what gets DETECTED/ACCEPTED in the first place, not just how
detected crops get clustered/matched -- those need a REAL pipeline re-run
(a fresh processing_job at modified PipelineParams, through the normal
POST /sessions/{id}/process) before evaluate.py has anything cached to read
for them. Point --job-data-dir/--processing-job-id-override at that job's
output once it's finished to evaluate it the same way.

Usage:
    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \\
        python eval/scripts/evaluate.py --job-data-dir /data/jobs \\
        --datasets-dir eval/datasets

Standalone-ish: needs services/worker on PYTHONPATH for pipeline.cluster/
pipeline.match/pipeline.params (pure functions only -- see those modules'
own "DB imports deferred" design), plus psycopg2 for the one DB lookup this
script does per session (fetch_session_context).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import eval_lib as el
from pipeline.cluster import cluster_embeddings
from pipeline.match import ClusterRepresentative, match_clusters
from pipeline.params import PipelineParams

ABLATIONS = ["none", "temporal-coherence-off", "quality-weighting-off", "clustering-off"]
STRATIFY_COLUMNS = ["row_number", "seat_position", "wears_glasses"]


def _load_cached_stage_outputs(job_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    embeddings = np.asarray(np.load(job_dir / "embed" / "embeddings.npy", mmap_mode="r"))
    aligned_index_df = pd.read_parquet(job_dir / "align" / "aligned_index.parquet")
    quality_df_full = pd.read_parquet(job_dir / "quality" / "quality.parquet")
    # .astype(bool): see align.py's run_align_stage docstring -- an empty
    # "accepted" column round-trips as object dtype, not bool.
    accepted_df = quality_df_full[quality_df_full["accepted"].astype(bool)].reset_index(drop=True)

    if list(accepted_df["det_id"]) != list(aligned_index_df["det_id"]):
        raise ValueError(
            f"{job_dir}: quality.parquet's accepted rows don't match aligned_index.parquet's "
            "det_id order -- the align/quality stages may be out of sync for this job."
        )
    return embeddings, accepted_df, aligned_index_df


def build_cluster_representatives(job_dir: Path, params: PipelineParams, ablation: str) -> list[ClusterRepresentative]:
    """`ablation="none"` reads the already-cached cluster_summary.parquet
    (the fast path, identical to what run_match_stage would see in
    production). Every other ablation re-runs pipeline.cluster.
    cluster_embeddings in-memory against cached embeddings/quality output --
    cheap (no detection, no ONNX inference), and never writes anything back.
    """
    if ablation == "none":
        return el.load_cluster_representatives(job_dir)
    if ablation == "clustering-off":
        raise ValueError("build_cluster_representatives: 'clustering-off' has no cluster representatives at "
                          "all -- use eval_lib.majority_vote_predicted_present directly instead.")

    embeddings, accepted_df, _ = _load_cached_stage_outputs(job_dir)
    run_params = params
    if ablation == "temporal-coherence-off":
        run_params = replace(params, temporal_coherence_enabled=False)

    result = cluster_embeddings(embeddings, accepted_df, run_params)

    reps = []
    for diag in result.diagnostics:
        vector = diag.representative
        if ablation == "quality-weighting-off":
            vector = el.plain_mean_representative(embeddings, diag.member_indices)
        reps.append(ClusterRepresentative(
            cluster_id=diag.cluster_id, vector=vector, best_crop_uri="",
            member_count=len(diag.member_indices), mean_quality=diag.mean_quality,
        ))
    return reps


def evaluate_session(
    database_url: str, class_session_id: str, job_data_dir: Path, params: PipelineParams, ablation: str,
    truth_df: pd.DataFrame, cluster_labels_df: pd.DataFrame | None,
    processing_job_id_override: int | None = None,
) -> dict:
    """Returns a dict with this session's scored rows_df, matches (or None
    for clustering-off), cluster_reps (or None), and pipeline-yield inputs
    -- everything the caller needs to fold into the cross-session totals
    and stratified breakdown.
    """
    ctx = el.fetch_session_context(database_url, int(class_session_id), job_data_dir)
    if processing_job_id_override is not None:
        ctx = replace(ctx, processing_job_id=processing_job_id_override, job_dir=job_data_dir / str(processing_job_id_override))

    present_roll_numbers = set(truth_df.loc[truth_df["actually_present"], "roll_number"])
    present_student_ids = {
        sid for sid, roll in ctx.roll_number_by_student_id.items() if roll in present_roll_numbers
    }

    if ablation == "clustering-off":
        embeddings, _, _ = _load_cached_stage_outputs(ctx.job_dir)
        predicted = el.majority_vote_predicted_present(embeddings, ctx.gallery, params)
        matches, cluster_reps = None, None
    else:
        cluster_reps = build_cluster_representatives(ctx.job_dir, params, ablation)
        match_result = match_clusters(cluster_reps, ctx.gallery, params)
        matches = match_result.matches
        predicted = el.predicted_present_from_matches(matches, set(ctx.gallery.keys()))

    scored = el.rows_for_session(predicted, ctx.roll_number_by_student_id, truth_df)
    unmapped = scored[scored["predicted_present"].isna()]
    if len(unmapped) > 0:
        print(f"  WARNING: session {class_session_id}: {len(unmapped)} truth.csv roll_number(s) not found "
              f"in this course's enrollment/gallery: {list(unmapped['roll_number'])[:5]}"
              f"{' ...' if len(unmapped) > 5 else ''} -- excluded from scoring, not counted as either present or absent.")
    scored = scored.dropna(subset=["predicted_present"])
    scored["session_id"] = class_session_id

    yield_stats = None
    if matches is not None and cluster_reps is not None:
        yield_stats = el.pipeline_yield_for_session(matches, cluster_reps, set(ctx.gallery.keys()), present_student_ids)

    clustering_stats = None
    if cluster_reps is not None:
        clustering_stats = el.clustering_quality(cluster_reps, len(present_student_ids), cluster_labels_df)

    return {"scored": scored, "yield_stats": yield_stats, "clustering_stats": clustering_stats}


def _print_metrics_block(label: str, summary: el.MetricsSummary) -> None:
    c = summary.counts
    print(f"\n{label}")
    print(f"  precision={summary.precision:.3f}  recall={summary.recall:.3f}  "
          f"f1={summary.f1:.3f}  accuracy={summary.accuracy:.3f}")
    print(f"  true positive={c.tp}  false positive (proxy hole)={c.fp}  "
          f"false negative (trust killer)={c.fn}  true negative={c.tn}")


def run(
    database_url: str, datasets_dir: Path, job_data_dir: Path, params: PipelineParams, ablation: str,
    session_ids: list[str] | None,
) -> int:
    all_session_dirs = sorted(p.name for p in datasets_dir.iterdir() if p.is_dir()) if datasets_dir.exists() else []
    target_sessions = session_ids if session_ids else all_session_dirs
    if not target_sessions:
        print(f"No session directories found under {datasets_dir}.")
        return 1

    all_scored = []
    all_yield_stats = []
    all_clustering_stats = []

    for session_id in target_sessions:
        truth_path = datasets_dir / session_id / "truth.csv"
        if not truth_path.exists():
            print(f"Skipping session {session_id}: no truth.csv at {truth_path}.")
            continue
        truth_df = el.load_truth_csv(truth_path)

        labels_path = datasets_dir / session_id / "cluster_labels.csv"
        cluster_labels_df = pd.read_csv(labels_path, dtype=str) if labels_path.exists() else None

        print(f"\n=== session {session_id} ({len(truth_df)} labelled student(s)) ===")
        try:
            result = evaluate_session(
                database_url, session_id, job_data_dir, params, ablation, truth_df, cluster_labels_df,
            )
        except Exception as e:
            print(f"  ERROR evaluating session {session_id}: {e}")
            continue

        all_scored.append(result["scored"])
        if result["yield_stats"] is not None:
            all_yield_stats.append(result["yield_stats"])
        if result["clustering_stats"] is not None:
            all_clustering_stats.append(result["clustering_stats"])

    if not all_scored:
        print("\nNo sessions could be scored -- nothing to report.")
        return 1

    combined = pd.concat(all_scored, ignore_index=True)

    print(f"\n{'=' * 70}\nAGGREGATE ACROSS {combined['session_id'].nunique()} SESSION(S), "
          f"{len(combined)} SCORED (SESSION, STUDENT) PAIR(S), ablation={ablation}\n{'=' * 70}")
    _print_metrics_block("STUDENT-LEVEL PRESENCE (headline)", el.summarize(combined))

    if all_yield_stats:
        mean_detections = float(np.mean([s["mean_detections_per_present_student"] for s in all_yield_stats]))
        mean_zero_crop_fraction = float(np.mean([s["zero_accepted_crop_fraction"] for s in all_yield_stats]))
        print("\nPIPELINE YIELD (mean across sessions)")
        print(f"  mean detections per present student: {mean_detections:.2f}")
        print(f"  fraction of present students with ZERO accepted crops (unrecoverable): "
              f"{mean_zero_crop_fraction:.3f}")

    if all_clustering_stats:
        total_clusters = sum(s["cluster_count"] for s in all_clustering_stats)
        total_actual = sum(s["actual_present_count"] for s in all_clustering_stats)
        print("\nCLUSTERING QUALITY (summed across sessions)")
        print(f"  total clusters: {total_clusters}  total actual present: {total_actual}  "
              f"difference: {total_clusters - total_actual:+d}")
        labelled = [s for s in all_clustering_stats if s["purity"] is not None]
        if labelled:
            mean_purity = float(np.mean([s["purity"] for s in labelled]))
            mean_over_split = float(np.mean([s["over_split_rate"] for s in labelled]))
            mean_merge = float(np.mean([s["merge_rate"] for s in labelled]))
            print(f"  (from spot-labelled clusters, {sum(s['labelled_cluster_count'] for s in labelled)} labelled) "
                  f"purity={mean_purity:.3f}  over_split_rate={mean_over_split:.3f}  merge_rate={mean_merge:.3f}")
        else:
            print("  (no cluster_labels.csv found for any session -- purity/over-split/merge rate unavailable; "
                  "see eval_lib.clustering_quality's docstring for the spot-labelling format)")

    print(f"\n{'=' * 70}\nSTRATIFIED BREAKDOWN (mandatory, not optional)\n{'=' * 70}")
    breakdown = el.stratified_breakdown(combined, STRATIFY_COLUMNS)
    for _, row in breakdown.iterrows():
        print(f"  {row['group']}={row['value']!r:20} n={row['n']:4d}  precision={row['precision']:.3f}  "
              f"recall={row['recall']:.3f}  f1={row['f1']:.3f}  (fp={row['fp']}, fn={row['fn']})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", type=Path, default=Path("eval/datasets"))
    parser.add_argument("--job-data-dir", type=Path, required=True, help="Same directory pipeline.run uses as job_dir's parent")
    parser.add_argument("--ablation", choices=ABLATIONS, default="none")
    parser.add_argument("--session-ids", default=None, help="Comma-separated class_session_ids; default: every labelled session found")
    parser.add_argument("--match-threshold", type=float, default=None, help="Override PipelineParams.match_threshold")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("Set DATABASE_URL, e.g.:\n"
              "  DATABASE_URL=postgresql://attend:attend@localhost:5432/attend "
              "python eval/scripts/evaluate.py --job-data-dir /data/jobs")
        return 1
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    params = PipelineParams()
    if args.match_threshold is not None:
        params = replace(params, match_threshold=args.match_threshold)

    session_ids = args.session_ids.split(",") if args.session_ids else None
    return run(database_url, args.datasets_dir, args.job_data_dir, params, args.ablation, session_ids)


if __name__ == "__main__":
    sys.exit(main())
