"""Identity clustering (Phase 5): DBSCAN over ArcFace embeddings, plus a
temporal-coherence post-pass that exploits the fact that the camera pans --
a given real person is only visible during one contiguous window of frames,
which DBSCAN alone has no way to know.

Non-negotiable rule #1: cluster_embeddings itself is a pure function --
numpy arrays and a DataFrame in, a ClusterResult out, no filesystem/DB
access. The I/O wrapper (run_cluster_stage, at the bottom of this file)
reads embeddings.npy/aligned_index.parquet/quality.parquet/aligned.npy and
writes clusters.parquet/cluster_summary.parquet/best-crop JPEGs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from pipeline.params import PipelineParams

logger = logging.getLogger("attend.worker.cluster")

NOISE_LABEL = -1


@dataclass(frozen=True)
class ClusterDiagnostics:
    cluster_id: int
    member_indices: tuple[int, ...]  # positions into the embeddings/quality_df arrays (row order), NOT det_ids
    mean_quality: float
    tightness: float  # intra-cluster mean pairwise cosine similarity
    first_frame: int
    last_frame: int
    representative: np.ndarray  # (512,) float32, L2-normalised, quality-weighted mean
    best_crop_row_index: int  # highest-quality member's row index, for saving a JPEG


@dataclass(frozen=True)
class ClusterResult:
    assignments_df: pd.DataFrame  # one row per input embedding: det_id, cluster_id, distance_to_representative
    diagnostics: list[ClusterDiagnostics]  # one entry per cluster (excludes noise)
    merge_log: list[str]
    split_log: list[str]


def _weighted_mean_vector(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The QUALITY-WEIGHTED mean, then re-L2-normalised (Phase 5 prompt,
    step c) -- this is the representative vector for a cluster.

    Why weighting beats a plain mean: a back-row student's forty crops are
    NOT equally trustworthy. A blurry, extreme-yaw crop's embedding sits
    further from that student's "true" identity vector than a sharp,
    frontal crop's does -- it's not that it's a weak signal, Phase 4's own
    framing is that it can be a confidently WRONG one. A plain mean lets
    however many bad crops happen to exist pull the representative away
    from where the good evidence points, in direct proportion to their
    count rather than their reliability. Weighting by the same composite
    quality score Phase 4 already computed -- rather than inventing a
    second, redundant notion of "trustworthy" -- means the representative is
    dominated by the crops most likely to embed correctly. This IS the
    noise-averaging payoff Section 2.2 describes: not a better model, better
    aggregation.
    """
    weights = np.clip(weights, 1e-6, None)  # guard an all-zero-weight cluster from nulling the sum
    weighted_sum = (vectors * weights[:, None]).sum(axis=0)
    norm = np.linalg.norm(weighted_sum)
    if norm < 1e-12:
        return weighted_sum.astype(np.float32)
    return (weighted_sum / norm).astype(np.float32)


def _pairwise_cosine_mean(vectors: np.ndarray) -> float:
    """"Intra-cluster mean cosine similarity" (Phase 5 prompt, step d) --
    the mean of every pairwise similarity among a cluster's members,
    excluding self-similarity. Vectors are already L2-normalised (embed_
    aligned's contract), so a dot product IS cosine similarity. A
    single-member cluster is trivially, perfectly tight (nothing to be
    inconsistent with).
    """
    m = vectors.shape[0]
    if m <= 1:
        return 1.0
    similarity = vectors @ vectors.T
    off_diagonal_sum = float(similarity.sum() - np.trace(similarity))
    count = m * (m - 1)
    return off_diagonal_sum / count


def _diagnostics_for_group(
    cluster_id: int, indices: np.ndarray, embeddings: np.ndarray, quality_scores: np.ndarray, frame_indices: np.ndarray
) -> ClusterDiagnostics:
    member_vectors = embeddings[indices]
    member_quality = quality_scores[indices]

    representative = _weighted_mean_vector(member_vectors, member_quality)
    tightness = _pairwise_cosine_mean(member_vectors)
    first_frame = int(frame_indices[indices].min())
    last_frame = int(frame_indices[indices].max())
    best_local_position = int(np.argmax(member_quality))  # step e: highest-quality member is the best_crop

    return ClusterDiagnostics(
        cluster_id=cluster_id,
        member_indices=tuple(int(i) for i in indices),
        mean_quality=float(member_quality.mean()),
        tightness=tightness,
        first_frame=first_frame,
        last_frame=last_frame,
        representative=representative,
        best_crop_row_index=int(indices[best_local_position]),
    )


def _build_all_diagnostics(
    embeddings: np.ndarray, quality_scores: np.ndarray, frame_indices: np.ndarray, labels: np.ndarray
) -> dict[int, ClusterDiagnostics]:
    diagnostics: dict[int, ClusterDiagnostics] = {}
    for cluster_id in sorted(set(int(l) for l in labels)):
        if cluster_id == NOISE_LABEL:
            continue
        indices = np.where(labels == cluster_id)[0]
        diagnostics[cluster_id] = _diagnostics_for_group(cluster_id, indices, embeddings, quality_scores, frame_indices)
    return diagnostics


def _frame_overlap_fraction(a: ClusterDiagnostics, b: ClusterDiagnostics) -> float:
    overlap = max(0, min(a.last_frame, b.last_frame) - max(a.first_frame, b.first_frame) + 1)
    span_a = a.last_frame - a.first_frame + 1
    span_b = b.last_frame - b.first_frame + 1
    shorter_span = min(span_a, span_b)
    return overlap / shorter_span if shorter_span > 0 else 0.0


class _UnionFind:
    """Minimal union-find over a fixed set of cluster ids -- used to collapse
    a whole chain of pairwise merges (A merges with B, B merges with C) into
    one final group per Phase 5's merge rule, without needing a second
    "did anything change" loop: every pair is checked once, and transitive
    merges fall out of the union-find structure for free.
    """

    def __init__(self, ids: list[int]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _merge_pass(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    frame_indices: np.ndarray,
    labels: np.ndarray,
    diagnostics: dict[int, ClusterDiagnostics],
    params: PipelineParams,
) -> tuple[np.ndarray, dict[int, ClusterDiagnostics], list[str]]:
    """Temporal coherence step 1 (Phase 5 prompt): merge two clusters whose
    representatives are within cluster_eps * cluster_merge_distance_factor
    AND whose frame ranges overlap by at least temporal_overlap_min_fraction
    of the shorter cluster's span -- likely the same person, over-split by
    DBSCAN because an embedding drifts slightly across a long pan.
    """
    merge_log: list[str] = []
    cluster_ids = list(diagnostics.keys())
    if len(cluster_ids) < 2:
        return labels, diagnostics, merge_log

    uf = _UnionFind(cluster_ids)
    threshold = params.cluster_eps * params.cluster_merge_distance_factor

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            a_id, b_id = cluster_ids[i], cluster_ids[j]
            if uf.find(a_id) == uf.find(b_id):
                continue

            a, b = diagnostics[a_id], diagnostics[b_id]
            distance = 1.0 - float(np.dot(a.representative, b.representative))
            overlap_fraction = _frame_overlap_fraction(a, b)

            if distance < threshold and overlap_fraction >= params.temporal_overlap_min_fraction:
                uf.union(a_id, b_id)
                merge_log.append(
                    f"merged cluster {b_id} into {a_id}: representative distance={distance:.4f} "
                    f"(threshold={threshold:.4f}), frame overlap={overlap_fraction:.2f}"
                )

    if not merge_log:
        return labels, diagnostics, merge_log

    root_to_new_id: dict[int, int] = {}
    old_to_new: dict[int, int] = {}
    next_id = 0
    for cid in cluster_ids:
        root = uf.find(cid)
        if root not in root_to_new_id:
            root_to_new_id[root] = next_id
            next_id += 1
        old_to_new[cid] = root_to_new_id[root]

    new_labels = np.array([old_to_new[int(l)] if int(l) != NOISE_LABEL else NOISE_LABEL for l in labels])
    new_diagnostics = _build_all_diagnostics(embeddings, quality_scores, frame_indices, new_labels)
    return new_labels, new_diagnostics, merge_log


def _split_pass(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    frame_indices: np.ndarray,
    labels: np.ndarray,
    diagnostics: dict[int, ClusterDiagnostics],
    params: PipelineParams,
    total_frames: int,
) -> tuple[np.ndarray, dict[int, ClusterDiagnostics], list[str]]:
    """Temporal coherence step 2 (Phase 5 prompt): a cluster whose members
    span more than cluster_split_frame_span_fraction of the whole video AND
    whose tightness is below cluster_split_tightness_max is FLAGGED as
    likely two different people DBSCAN merged into one. A second, tighter-
    eps DBSCAN pass restricted to just that cluster's members is tried.

    If that tighter pass genuinely produces 2+ non-noise sub-clusters, the
    split is applied (new cluster ids appended after the current max). If
    it doesn't (the tighter pass still can't separate them), the cluster is
    left alone -- but the flag is still logged. "I looked and couldn't split
    it" is real information for Phase 7, not a failure to hide.
    """
    split_log: list[str] = []
    labels = labels.copy()
    next_new_id = (max(diagnostics.keys()) + 1) if diagnostics else 0

    for cluster_id, diag in list(diagnostics.items()):
        span_fraction = (diag.last_frame - diag.first_frame + 1) / total_frames if total_frames > 0 else 0.0
        is_split_candidate = (
            span_fraction > params.cluster_split_frame_span_fraction
            and diag.tightness < params.cluster_split_tightness_max
        )
        if not is_split_candidate:
            continue

        indices = np.array(diag.member_indices)
        sub_embeddings = embeddings[indices]
        tighter_eps = params.cluster_eps * params.cluster_split_eps_factor
        sub_labels = DBSCAN(metric="cosine", eps=tighter_eps, min_samples=params.cluster_min_samples).fit_predict(
            sub_embeddings
        )

        non_noise_sub_ids = sorted(set(int(l) for l in sub_labels) - {NOISE_LABEL})
        if len(non_noise_sub_ids) < 2:
            split_log.append(
                f"cluster {cluster_id} flagged (frame span={span_fraction:.2f}, tightness={diag.tightness:.3f}) "
                "but the tighter-eps re-pass did not separate it -- left as one cluster."
            )
            continue

        for sub_id in non_noise_sub_ids:
            sub_member_indices = indices[sub_labels == sub_id]
            labels[sub_member_indices] = next_new_id
            split_log.append(
                f"split cluster {cluster_id}: {len(sub_member_indices)} member(s) -> new cluster {next_new_id}"
            )
            next_new_id += 1

        sub_noise_indices = indices[sub_labels == NOISE_LABEL]
        labels[sub_noise_indices] = NOISE_LABEL  # members the re-pass couldn't place become global noise

    new_diagnostics = _build_all_diagnostics(embeddings, quality_scores, frame_indices, labels)
    return labels, new_diagnostics, split_log


def _build_assignments_df(
    embeddings: np.ndarray, det_ids: pd.Series, labels: np.ndarray, diagnostics: dict[int, ClusterDiagnostics]
) -> pd.DataFrame:
    distances = np.full(len(labels), np.nan, dtype=np.float32)
    for cluster_id, diag in diagnostics.items():
        idx = np.array(diag.member_indices)
        distances[idx] = 1.0 - (embeddings[idx] @ diag.representative)
    # Noise rows keep NaN -- "distance to representative" isn't meaningful
    # for a point that doesn't belong to any cluster.

    return pd.DataFrame({
        "det_id": det_ids.to_numpy(),
        "cluster_id": labels,
        "distance_to_representative": distances,
    })


def cluster_embeddings(embeddings: np.ndarray, quality_df: pd.DataFrame, params: PipelineParams) -> ClusterResult:
    """Phase 5 deliverable 2. `embeddings` is (N, 512) float32 L2-normalised
    (embed_aligned's output); `quality_df` is quality.parquet's rows FILTERED
    to accepted==True, in the SAME row order as `embeddings` (run_cluster_
    stage, below, is responsible for that alignment and asserts it before
    calling this). Returns a ClusterResult; does NOT write parquet or crop
    JPEGs itself -- that's run_cluster_stage's job -- so this stays a plain
    array/DataFrame transform, testable on synthetic data with no filesystem
    access (non-negotiable rule #1).
    """
    n = embeddings.shape[0]
    if n == 0 or len(quality_df) == 0:
        empty_df = pd.DataFrame(columns=["det_id", "cluster_id", "distance_to_representative"])
        return ClusterResult(assignments_df=empty_df, diagnostics=[], merge_log=[], split_log=[])

    quality_scores = quality_df["quality_score"].to_numpy(dtype=np.float64)
    frame_indices = quality_df["frame_index"].to_numpy(dtype=np.int64)

    labels = DBSCAN(metric="cosine", eps=params.cluster_eps, min_samples=params.cluster_min_samples).fit_predict(
        embeddings
    )
    # -1 = noise. NOT discarded (Phase 5 prompt, step b) -- every embedding's
    # row survives into assignments_df regardless of its label.
    diagnostics = _build_all_diagnostics(embeddings, quality_scores, frame_indices, labels)

    merge_log: list[str] = []
    split_log: list[str] = []
    if params.temporal_coherence_enabled and diagnostics:
        labels, diagnostics, merge_log = _merge_pass(embeddings, quality_scores, frame_indices, labels, diagnostics, params)
        total_frames = int(frame_indices.max()) + 1
        labels, diagnostics, split_log = _split_pass(
            embeddings, quality_scores, frame_indices, labels, diagnostics, params, total_frames
        )

    assignments_df = _build_assignments_df(embeddings, quality_df["det_id"], labels, diagnostics)

    return ClusterResult(
        assignments_df=assignments_df,
        diagnostics=sorted(diagnostics.values(), key=lambda d: d.cluster_id),
        merge_log=merge_log,
        split_log=split_log,
    )


# --------------------------------------------------------------------------
# I/O wrapper: what run.py actually calls
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterStageSummary:
    total_embeddings: int
    cluster_count: int
    noise_count: int
    merge_count: int
    split_log: list[str]
    clusters_parquet_path: Path
    cluster_summary_parquet_path: Path


def run_cluster_stage(
    embeddings_npy_path: Path,
    aligned_npy_path: Path,
    aligned_index_parquet_path: Path,
    quality_parquet_path: Path,
    out_dir: Path,
    params: PipelineParams,
) -> ClusterStageSummary:
    """Reads embeddings.npy, quality.parquet (filtered to accepted rows),
    and aligned.npy (for best_crop JPEGs); runs cluster_embeddings; writes
    clusters.parquet + cluster_summary.parquet + one best-crop JPEG per
    cluster; logs the stage summary including every merge/split decision.

    Cross-checks that quality.parquet's accepted-row det_id order matches
    aligned_index.parquet's det_id order (both SHOULD be identical, since
    align_crops derived its row order from the exact same accepted-filter --
    see Phase 4's run_align_stage) before trusting embeddings' row order to
    line up with quality scores/frame indices. If the two stages are out of
    sync (e.g. quality was re-run with different params but align wasn't),
    this fails loudly instead of silently attaching the wrong quality score
    or frame index to a cluster -- which would corrupt every diagnostic and
    every merge/split decision without ever raising an error.
    """
    embeddings = np.asarray(np.load(embeddings_npy_path, mmap_mode="r"))
    aligned_index_df = pd.read_parquet(aligned_index_parquet_path)
    quality_df_full = pd.read_parquet(quality_parquet_path)

    # .astype(bool): see align.py's run_align_stage for why this cast is
    # required, not cosmetic -- an empty quality_df's "accepted" column is
    # object-dtype, and boolean-masking with an object-dtype column (even
    # empty) silently drops every column, not just every row.
    accepted_df = quality_df_full[quality_df_full["accepted"].astype(bool)].reset_index(drop=True)

    if list(accepted_df["det_id"]) != list(aligned_index_df["det_id"]):
        raise ValueError(
            "run_cluster_stage: quality.parquet's accepted rows don't match "
            "aligned_index.parquet's det_id order -- the align/quality stages "
            "may be out of sync (re-run 'align' after any 'quality' param change)."
        )
    if len(accepted_df) != embeddings.shape[0]:
        raise ValueError(
            f"run_cluster_stage: {len(accepted_df)} accepted crops but "
            f"{embeddings.shape[0]} embeddings -- embed/align/quality stages out of sync."
        )

    result = cluster_embeddings(embeddings, accepted_df, params)

    out_dir.mkdir(parents=True, exist_ok=True)
    clusters_parquet_path = out_dir / "clusters.parquet"
    result.assignments_df.to_parquet(clusters_parquet_path)

    aligned_memmap = np.load(aligned_npy_path, mmap_mode="r") if aligned_npy_path.exists() else None
    crops_dir = out_dir / "best_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for diag in result.diagnostics:
        best_crop_uri = ""
        if aligned_memmap is not None and aligned_memmap.shape[0] > 0:
            crop = np.asarray(aligned_memmap[diag.best_crop_row_index])
            best_crop_path = crops_dir / f"cluster_{diag.cluster_id}.jpg"
            cv2.imwrite(str(best_crop_path), crop)
            best_crop_uri = str(best_crop_path)

        summary_rows.append({
            "cluster_id": diag.cluster_id,
            "member_count": len(diag.member_indices),
            "mean_quality": diag.mean_quality,
            "tightness": diag.tightness,
            "first_frame": diag.first_frame,
            "last_frame": diag.last_frame,
            "best_crop_uri": best_crop_uri,
            # Phase 6's integration contract: "Consumes: cluster_summary.parquet
            # WITH REPRESENTATIVE VECTORS." Stored as raw float32 bytes, same
            # convention as gallery_embedding.vector in the DB -- matching's
            # gallery-similarity computation needs this, not just the scalar
            # diagnostics above.
            "representative_vector": diag.representative.astype(np.float32).tobytes(),
        })

    cluster_summary_parquet_path = out_dir / "cluster_summary.parquet"
    summary_columns = [
        "cluster_id", "member_count", "mean_quality", "tightness", "first_frame", "last_frame",
        "best_crop_uri", "representative_vector",
    ]
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(columns=summary_columns)
    summary_df.to_parquet(cluster_summary_parquet_path)

    noise_count = int((result.assignments_df["cluster_id"] == NOISE_LABEL).sum())
    merge_count = len(result.merge_log)

    logger.info(
        "cluster stage: %d embeddings, %d clusters, %d noise points, %d merges, %d split decisions",
        len(accepted_df), len(result.diagnostics), noise_count, merge_count, len(result.split_log),
    )
    for line in result.merge_log:
        logger.info("cluster stage merge: %s", line)
    for line in result.split_log:
        logger.info("cluster stage split: %s", line)

    return ClusterStageSummary(
        total_embeddings=len(accepted_df),
        cluster_count=len(result.diagnostics),
        noise_count=noise_count,
        merge_count=merge_count,
        split_log=result.split_log,
        clusters_parquet_path=clusters_parquet_path,
        cluster_summary_parquet_path=cluster_summary_parquet_path,
    )
