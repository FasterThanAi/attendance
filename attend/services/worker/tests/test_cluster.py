"""Phase 5 deliverable 6 tests:
- cluster_embeddings on synthetic data with three well-separated Gaussian
  blobs returns exactly three clusters.
- quality weighting shifts the representative toward high-quality members.
- the merge rule fires on constructed overlapping clusters.

Plus a few extra tests for behavior the roadmap calls out elsewhere in
Phase 5's prompt: noise (-1) is kept, not discarded; empty input doesn't
crash; tightness is 1.0 for a single-member cluster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.cluster import (
    ClusterDiagnostics,
    _frame_overlap_fraction,
    _merge_pass,
    _pairwise_cosine_mean,
    _weighted_mean_vector,
    cluster_embeddings,
)
from pipeline.params import PipelineParams


def _l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _make_quality_df(n: int, frame_indices=None, quality_scores=None) -> pd.DataFrame:
    return pd.DataFrame({
        "det_id": [f"det_{i}" for i in range(n)],
        "frame_index": frame_indices if frame_indices is not None else list(range(n)),
        "quality_score": quality_scores if quality_scores is not None else [1.0] * n,
    })


def test_cluster_embeddings_three_gaussian_blobs_returns_three_clusters():
    rng = np.random.default_rng(42)
    dim = 64
    per_blob = 15

    # Three random, well-separated unit directions -- checked pairwise so the
    # test itself doesn't flake on an unlucky draw of two nearly-identical
    # directions.
    while True:
        centers = _l2_normalize_rows(rng.normal(size=(3, dim)))
        pairwise_cos = centers @ centers.T
        off_diagonal = pairwise_cos[~np.eye(3, dtype=bool)]
        if off_diagonal.max() < 0.3:  # well-separated: <0.3 cosine similarity between any two centers
            break

    blobs = []
    for center in centers:
        noise = rng.normal(scale=0.05, size=(per_blob, dim))
        samples = _l2_normalize_rows(center + noise)
        blobs.append(samples)
    embeddings = np.concatenate(blobs, axis=0).astype(np.float32)

    quality_df = _make_quality_df(len(embeddings))
    params = PipelineParams(cluster_eps=0.42, cluster_min_samples=3, temporal_coherence_enabled=False)

    result = cluster_embeddings(embeddings, quality_df, params)

    assert len(result.diagnostics) == 3, f"expected 3 clusters, got {len(result.diagnostics)}"
    # Every point should have been clustered into one of the 3 blobs' worth
    # of members -- allow a SMALL amount of noise (DBSCAN edge effects) but
    # not entire blobs going missing.
    cluster_sizes = sorted(d.mean_quality * 0 + len(d.member_indices) for d in result.diagnostics)
    assert min(cluster_sizes) >= per_blob - 3, f"a cluster is suspiciously small: {cluster_sizes}"


def test_cluster_embeddings_keeps_noise_not_discarded():
    rng = np.random.default_rng(0)
    dim = 32
    # One real cluster of 5 close points, plus 3 totally unrelated random
    # points that shouldn't cluster with anything or each other.
    center = _l2_normalize_rows(rng.normal(size=(1, dim)))[0]
    cluster_points = _l2_normalize_rows(center + rng.normal(scale=0.02, size=(5, dim)))
    scattered_points = _l2_normalize_rows(rng.normal(size=(3, dim)))
    embeddings = np.concatenate([cluster_points, scattered_points], axis=0).astype(np.float32)

    quality_df = _make_quality_df(len(embeddings))
    params = PipelineParams(cluster_eps=0.3, cluster_min_samples=3, temporal_coherence_enabled=False)

    result = cluster_embeddings(embeddings, quality_df, params)

    assert len(result.assignments_df) == len(embeddings), "every input row must appear in the output, noise or not"
    noise_rows = result.assignments_df[result.assignments_df["cluster_id"] == -1]
    assert len(noise_rows) >= 1, "expected at least the 3 scattered points to be noise"


def test_weighted_mean_vector_shifts_toward_high_quality_members():
    # A "true identity" direction, plus a deliberately different outlier
    # direction (still close enough that DBSCAN would put them in one
    # cluster, but far enough to visibly pull a plain mean off course).
    true_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    outlier_direction = np.array([0.7, 0.7, 0.0], dtype=np.float32)
    outlier_direction /= np.linalg.norm(outlier_direction)

    vectors = np.stack([true_direction, true_direction, true_direction, outlier_direction])

    plain_mean = vectors.mean(axis=0)
    plain_mean /= np.linalg.norm(plain_mean)

    weights = np.array([1.0, 1.0, 1.0, 0.01])  # outlier's quality score is nearly zero
    weighted_mean = _weighted_mean_vector(vectors, weights)

    sim_plain = float(np.dot(plain_mean, true_direction))
    sim_weighted = float(np.dot(weighted_mean, true_direction))

    assert sim_weighted > sim_plain, (
        f"weighted mean (sim={sim_weighted:.4f}) should be closer to the true direction "
        f"than the plain mean (sim={sim_plain:.4f})"
    )
    assert sim_weighted == pytest.approx(1.0, abs=1e-3)  # near-zero-weight outlier should barely move it at all


def test_quality_weighting_shifts_cluster_representative_end_to_end():
    # Same idea as above, but through the full cluster_embeddings path --
    # one cluster, 3 good members near [1,0,0,...] and 1 low-quality
    # outlier member nearby (within eps) but off-axis.
    dim = 8
    true_direction = np.zeros(dim, dtype=np.float32)
    true_direction[0] = 1.0
    outlier_direction = np.zeros(dim, dtype=np.float32)
    outlier_direction[0] = 0.7
    outlier_direction[1] = 0.7
    outlier_direction /= np.linalg.norm(outlier_direction)

    embeddings = np.stack([true_direction, true_direction, true_direction, outlier_direction]).astype(np.float32)
    quality_df = _make_quality_df(4, quality_scores=[1.0, 1.0, 1.0, 0.01])
    params = PipelineParams(cluster_eps=0.42, cluster_min_samples=2, temporal_coherence_enabled=False)

    result = cluster_embeddings(embeddings, quality_df, params)
    assert len(result.diagnostics) == 1, "expected all 4 points in one cluster (outlier within eps)"

    representative = result.diagnostics[0].representative
    assert float(np.dot(representative, true_direction)) > 0.95


def test_pairwise_cosine_mean_single_member_is_trivially_tight():
    assert _pairwise_cosine_mean(np.array([[1.0, 0.0, 0.0]])) == 1.0


def test_pairwise_cosine_mean_identical_vectors_is_one():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert _pairwise_cosine_mean(vectors) == pytest.approx(1.0)


def test_merge_rule_fires_on_constructed_overlapping_clusters():
    # Two clusters, hand-constructed (not via DBSCAN) per the roadmap's own
    # phrasing: "the merge rule fires on constructed overlapping clusters."
    # Cluster 0: 3 members, frames 0-20. Cluster 1: 3 members, frames 10-30
    # (overlaps cluster 0's frame range) -- and their representative vectors
    # are close (distance 0.05), well under cluster_eps(0.42) *
    # cluster_merge_distance_factor(1.3) = 0.546.
    dim = 8
    rep_a = np.zeros(dim, dtype=np.float32)
    rep_a[0] = 1.0
    # A vector at cosine distance ~0.05 from rep_a (i.e. cosine similarity ~0.95).
    rep_b = np.zeros(dim, dtype=np.float32)
    rep_b[0] = 0.95
    rep_b[1] = np.sqrt(1 - 0.95 ** 2)

    embeddings = np.stack([rep_a, rep_a, rep_a, rep_b, rep_b, rep_b]).astype(np.float32)
    quality_scores = np.ones(6)
    frame_indices = np.array([0, 10, 20, 10, 20, 30])

    diag_a = ClusterDiagnostics(
        cluster_id=0, member_indices=(0, 1, 2), mean_quality=1.0, tightness=1.0,
        first_frame=0, last_frame=20, representative=rep_a, best_crop_row_index=0,
    )
    diag_b = ClusterDiagnostics(
        cluster_id=1, member_indices=(3, 4, 5), mean_quality=1.0, tightness=1.0,
        first_frame=10, last_frame=30, representative=rep_b, best_crop_row_index=3,
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    diagnostics = {0: diag_a, 1: diag_b}

    params = PipelineParams(cluster_eps=0.42, cluster_merge_distance_factor=1.3, temporal_overlap_min_fraction=0.3)

    new_labels, new_diagnostics, merge_log = _merge_pass(
        embeddings, quality_scores, frame_indices, labels, diagnostics, params
    )

    assert len(merge_log) == 1, f"expected exactly one merge decision, got: {merge_log}"
    assert len(new_diagnostics) == 1, "the two clusters should have merged into one"
    assert len(set(new_labels)) == 1, "every point should now share the same cluster id"


def test_merge_rule_does_not_fire_when_frame_ranges_dont_overlap():
    # Same close representative vectors as above, but NON-overlapping frame
    # ranges (cluster 0: frames 0-5, cluster 1: frames 500-505) -- these are
    # two DIFFERENT people who just happen to look similar, seen at
    # different points in the pan; must NOT merge.
    dim = 8
    rep_a = np.zeros(dim, dtype=np.float32)
    rep_a[0] = 1.0
    rep_b = np.zeros(dim, dtype=np.float32)
    rep_b[0] = 0.95
    rep_b[1] = np.sqrt(1 - 0.95 ** 2)

    embeddings = np.stack([rep_a, rep_a, rep_b, rep_b]).astype(np.float32)
    quality_scores = np.ones(4)
    frame_indices = np.array([0, 5, 500, 505])

    diag_a = ClusterDiagnostics(
        cluster_id=0, member_indices=(0, 1), mean_quality=1.0, tightness=1.0,
        first_frame=0, last_frame=5, representative=rep_a, best_crop_row_index=0,
    )
    diag_b = ClusterDiagnostics(
        cluster_id=1, member_indices=(2, 3), mean_quality=1.0, tightness=1.0,
        first_frame=500, last_frame=505, representative=rep_b, best_crop_row_index=2,
    )
    labels = np.array([0, 0, 1, 1])
    diagnostics = {0: diag_a, 1: diag_b}

    params = PipelineParams(cluster_eps=0.42, cluster_merge_distance_factor=1.3, temporal_overlap_min_fraction=0.3)

    new_labels, new_diagnostics, merge_log = _merge_pass(
        embeddings, quality_scores, frame_indices, labels, diagnostics, params
    )

    assert merge_log == []
    assert len(new_diagnostics) == 2


def test_frame_overlap_fraction():
    a = ClusterDiagnostics(0, (), 0, 0, first_frame=0, last_frame=20, representative=np.zeros(1), best_crop_row_index=0)
    b = ClusterDiagnostics(1, (), 0, 0, first_frame=10, last_frame=30, representative=np.zeros(1), best_crop_row_index=0)
    # overlap = [10,20] -> 11 frames; shorter span = 21 (a: 0-20)
    assert _frame_overlap_fraction(a, b) == pytest.approx(11 / 21)


def test_cluster_embeddings_empty_input_does_not_crash():
    embeddings = np.zeros((0, 512), dtype=np.float32)
    quality_df = pd.DataFrame(columns=["det_id", "frame_index", "quality_score"])
    result = cluster_embeddings(embeddings, quality_df, PipelineParams())
    assert result.diagnostics == []
    assert len(result.assignments_df) == 0
