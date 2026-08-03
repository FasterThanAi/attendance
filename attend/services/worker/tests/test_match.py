"""Phase 6 deliverable 7 tests:
- Hungarian assignment beats greedy on a constructed matrix where they differ.
- Band boundaries are exact at the threshold values.
- A student not enrolled in the course is never matched (pure-function
  layer: build_gallery_matrix/match_clusters never introduce a student id
  beyond what the gallery dict contains -- the DB-level "only query enrolled
  students" scoping is exercised separately against run_match_stage).
- The band totals invariant holds, and raises loudly when violated.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.match import (
    CONFIDENT,
    UNCERTAIN,
    UNMATCHED,
    ClusterMatchRow,
    ClusterRepresentative,
    _decide_band,
    build_gallery_matrix,
    build_session_summary,
    match_clusters,
)
from pipeline.params import PipelineParams


def _rep(cluster_id: int, vector: np.ndarray) -> ClusterRepresentative:
    return ClusterRepresentative(
        cluster_id=cluster_id, vector=vector.astype(np.float32), best_crop_uri=f"crop_{cluster_id}.jpg",
        member_count=10, mean_quality=0.8,
    )


def _greedy_assignment(similarity: np.ndarray) -> dict[int, int]:
    """Naive greedy top-1: process clusters (rows) in order, each claims its
    own highest-similarity gallery column that hasn't already been claimed.
    Standing in for "the wrong approach" the Phase 6 prompt warns against,
    so the test can show Hungarian genuinely does better, not just differently.
    """
    assigned_cols: set[int] = set()
    result: dict[int, int] = {}
    for row in range(similarity.shape[0]):
        candidates = [c for c in range(similarity.shape[1]) if c not in assigned_cols]
        if not candidates:
            continue
        best_col = max(candidates, key=lambda c: similarity[row, c])
        result[row] = best_col
        assigned_cols.add(best_col)
    return result


def test_hungarian_beats_greedy_on_constructed_matrix():
    # Two orthonormal gallery directions -- similarity to student0 is the
    # x-component, similarity to student1 is the y-component, exactly.
    student0_id, student1_id = 100, 200
    gallery = {
        student0_id: np.array([1.0, 0.0], dtype=np.float32),
        student1_id: np.array([0.0, 1.0], dtype=np.float32),
    }

    # cluster0 at cos=0.9, sin=+0.4359 -- similarity 0.9 to student0, 0.4359 to student1.
    # cluster1 at cos=0.9, sin=-0.4359 -- similarity 0.9 to student0, -0.4359 to student1.
    # Both clusters' individually-best match is student0 (0.9 > their other option) --
    # exactly the "two clusters, same top pick" scenario the roadmap warns about.
    sin_component = float(np.sqrt(1 - 0.9 ** 2))
    cluster0 = _rep(0, np.array([0.9, sin_component]))
    cluster1 = _rep(1, np.array([0.9, -sin_component]))
    cluster_reps = [cluster0, cluster1]

    similarity = np.stack([c.vector for c in cluster_reps]) @ np.stack([gallery[student0_id], gallery[student1_id]]).T

    greedy = _greedy_assignment(similarity)
    greedy_total = sum(similarity[row, col] for row, col in greedy.items())
    # Greedy: cluster0 (row 0) claims student0 first (its argmax), leaving
    # cluster1 stuck with student1 at a NEGATIVE similarity.
    assert greedy[0] == 0  # cluster0 -> student0 (column 0)
    assert greedy[1] == 1  # cluster1 -> student1 (column 1), forced

    params = PipelineParams(match_threshold=0.0, match_margin_min=0.0, uncertain_band=0.0)
    result = match_clusters(cluster_reps, gallery, params)
    matches_by_cluster = {m.cluster_id: m for m in result.matches}

    hungarian_total = sum(
        m.similarity for m in result.matches if m.similarity is not None
    )

    # Hungarian must find the GLOBALLY better arrangement: cluster0 -> student1,
    # cluster1 -> student0 -- the opposite of greedy's choice, and a strictly
    # higher total similarity.
    assert matches_by_cluster[0].student_id == student1_id
    assert matches_by_cluster[1].student_id == student0_id
    assert hungarian_total > greedy_total


def test_match_clusters_never_assigns_a_student_outside_the_given_gallery():
    # The gallery passed in represents "students enrolled in this course."
    # match_clusters must never produce a student_id that wasn't a key in
    # that dict -- this is the pure-function half of "a student not enrolled
    # in the course is never matched" (the DB half is that the gallery query
    # itself is scoped to enrollment, exercised against run_match_stage).
    enrolled_gallery = {1: np.array([1.0, 0.0], dtype=np.float32), 2: np.array([0.0, 1.0], dtype=np.float32)}
    cluster_reps = [_rep(0, np.array([1.0, 0.0])), _rep(1, np.array([0.0, 1.0]))]

    result = match_clusters(cluster_reps, enrolled_gallery, PipelineParams(match_threshold=0.0, match_margin_min=0.0, uncertain_band=0.0))

    matched_student_ids = {m.student_id for m in result.matches if m.student_id is not None}
    assert matched_student_ids <= set(enrolled_gallery.keys())


def test_match_clusters_empty_gallery_marks_every_cluster_unmatched():
    cluster_reps = [_rep(0, np.array([1.0, 0.0])), _rep(1, np.array([0.0, 1.0]))]
    result = match_clusters(cluster_reps, {}, PipelineParams())
    assert all(m.decision == UNMATCHED for m in result.matches)
    assert all(m.student_id is None for m in result.matches)


def test_build_gallery_matrix_preserves_row_order():
    gallery = {5: np.array([1.0, 0.0]), 3: np.array([0.0, 1.0])}
    matrix, student_ids = build_gallery_matrix(gallery)
    assert student_ids == [5, 3]
    assert matrix.shape == (2, 2)
    np.testing.assert_array_equal(matrix[0], gallery[5])
    np.testing.assert_array_equal(matrix[1], gallery[3])


def test_build_gallery_matrix_empty():
    matrix, student_ids = build_gallery_matrix({})
    assert matrix.shape == (0, 0)
    assert student_ids == []


# --------------------------------------------------------------------------
# Band boundary tests
# --------------------------------------------------------------------------



# NOTE: these boundary tests deliberately use binary-exact fractions (0.5,
# 0.25, 0.125, 0.0625, 0.375, 0.4375, 0.1875 -- all n/2^k) rather than the
# real defaults (0.38, 0.05, 0.08). Decimal values like 0.38 - 0.05 don't
# subtract exactly in IEEE 754 double (0.38 - 0.05 == 0.32999999999999996),
# which would make an "exact boundary" test flaky depending on which side
# of the float-rounding coin it lands on. Binary-exact fractions remove
# that noise so these tests check _decide_band's own >=/> logic, not
# floating-point subtraction.


def test_decide_band_confident_at_exact_boundary():
    params = PipelineParams(match_threshold=0.5, match_margin_min=0.125, uncertain_band=0.25)
    # similarity exactly at threshold, margin exactly at match_margin_min -> CONFIDENT
    assert _decide_band(similarity=0.5, runner_up_similarity=0.375, params=params) == CONFIDENT


def test_decide_band_just_below_margin_is_not_confident():
    params = PipelineParams(match_threshold=0.5, match_margin_min=0.125, uncertain_band=0.25)
    # similarity at threshold but margin (0.0625) just under match_margin_min (0.125)
    # -> falls to UNCERTAIN (similarity 0.5 >= threshold-uncertain_band == 0.25)
    assert _decide_band(similarity=0.5, runner_up_similarity=0.4375, params=params) == UNCERTAIN


def test_decide_band_uncertain_at_exact_lower_boundary():
    params = PipelineParams(match_threshold=0.5, match_margin_min=0.125, uncertain_band=0.25)
    # similarity exactly at (match_threshold - uncertain_band) = 0.25 -> UNCERTAIN
    assert _decide_band(similarity=0.25, runner_up_similarity=0.0, params=params) == UNCERTAIN


def test_decide_band_just_below_uncertain_boundary_is_unmatched():
    params = PipelineParams(match_threshold=0.5, match_margin_min=0.125, uncertain_band=0.25)
    assert _decide_band(similarity=0.1875, runner_up_similarity=0.0, params=params) == UNMATCHED


def test_decide_band_no_runner_up_uses_similarity_as_margin():
    # A gallery of exactly one student -- no "other" entry to be confused
    # with, so margin collapses to similarity itself.
    params = PipelineParams(match_threshold=0.5, match_margin_min=0.125, uncertain_band=0.25)
    assert _decide_band(similarity=0.9, runner_up_similarity=None, params=params) == CONFIDENT


# --------------------------------------------------------------------------
# Session summary invariant tests
# --------------------------------------------------------------------------


def test_build_session_summary_happy_path_partitions_enrolled_students():
    matches = [
        ClusterMatchRow(cluster_id=0, student_id=1, similarity=0.9, runner_up_similarity=0.1, decision=CONFIDENT),
        ClusterMatchRow(cluster_id=1, student_id=2, similarity=0.5, runner_up_similarity=0.45, decision=UNCERTAIN),
        ClusterMatchRow(cluster_id=2, student_id=None, similarity=None, runner_up_similarity=None, decision=UNMATCHED),
    ]
    # Student 3 is enrolled but no cluster was assigned to them at all.
    summary = build_session_summary(matches, enrolled_student_ids=[1, 2, 3], preflight_had_warnings=False, params=PipelineParams())

    assert summary.total_enrolled == 3
    assert summary.proposed_present == 1
    assert summary.needs_review == 1
    assert summary.proposed_absent == 1
    assert summary.unrecognised_clusters == 1
    assert summary.proposed_present + summary.needs_review + summary.proposed_absent == summary.total_enrolled
    assert summary.mean_confident_similarity == pytest.approx(0.9)


def test_build_session_summary_raises_when_invariant_violated():
    # The same student appears as BOTH a confident match (cluster 1) and an
    # uncertain match (cluster 2) -- real Hungarian output could never
    # produce this (one-to-one assignment), but build_session_summary must
    # still detect and refuse it rather than silently double-counting a student.
    matches = [
        ClusterMatchRow(cluster_id=1, student_id=500, similarity=0.9, runner_up_similarity=0.1, decision=CONFIDENT),
        ClusterMatchRow(cluster_id=2, student_id=500, similarity=0.5, runner_up_similarity=0.3, decision=UNCERTAIN),
    ]
    with pytest.raises(ValueError, match="invariant violated"):
        build_session_summary(matches, enrolled_student_ids=[500], preflight_had_warnings=False, params=PipelineParams())


def test_build_session_summary_no_enrolled_students_is_zero_coverage_not_a_crash():
    summary = build_session_summary([], enrolled_student_ids=[], preflight_had_warnings=False, params=PipelineParams())
    assert summary.total_enrolled == 0
    assert summary.coverage_percent == 0.0
    assert summary.mean_confident_similarity is None


def test_session_health_poor_on_preflight_warnings_even_with_full_coverage():
    matches = [ClusterMatchRow(cluster_id=0, student_id=1, similarity=0.9, runner_up_similarity=0.1, decision=CONFIDENT)]
    summary = build_session_summary(matches, enrolled_student_ids=[1], preflight_had_warnings=True, params=PipelineParams())
    assert summary.coverage_percent == 100.0
    assert summary.session_health == "poor"


def test_session_health_good_when_coverage_and_similarity_are_high():
    matches = [ClusterMatchRow(cluster_id=0, student_id=1, similarity=0.9, runner_up_similarity=0.1, decision=CONFIDENT)]
    summary = build_session_summary(matches, enrolled_student_ids=[1], preflight_had_warnings=False, params=PipelineParams())
    assert summary.session_health == "good"
