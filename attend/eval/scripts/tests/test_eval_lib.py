"""Phase 7 deliverable 2/3 tests: confusion-matrix counting, precision/
recall/F1, the stratified breakdown, and the match-threshold sweep against
a small constructed (not real) set of clusters/gallery/truth.

Run from eval/scripts/ with PYTHONPATH including this directory and
services/worker (eval_lib imports pipeline.match/pipeline.params for the
re-match sweep helper):

    PYTHONPATH=services/worker:eval/scripts python -m pytest eval/scripts/tests/test_eval_lib.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eval_lib as el
from pipeline.match import ClusterMatchRow, ClusterRepresentative
from pipeline.params import PipelineParams


def _truth_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["actually_present"] = df["actually_present"].astype(bool)
    df["wears_glasses"] = df["wears_glasses"].astype(bool)
    return df


# --------------------------------------------------------------------------
# Confusion matrix / precision / recall / F1
# --------------------------------------------------------------------------


def test_compute_confusion_basic_counts():
    rows = pd.DataFrame({
        "actually_present": [True, True, False, False],
        "predicted_present": [True, False, True, False],
    })
    counts = el.compute_confusion(rows)
    assert counts == el.ConfusionCounts(tp=1, fp=1, fn=1, tn=1)


def test_precision_recall_f1_all_correct():
    counts = el.ConfusionCounts(tp=5, fp=0, fn=0, tn=3)
    precision, recall, f1 = el.precision_recall_f1(counts)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_precision_recall_f1_no_positives_at_all_is_zero_not_a_crash():
    counts = el.ConfusionCounts(tp=0, fp=0, fn=0, tn=10)
    precision, recall, f1 = el.precision_recall_f1(counts)
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_precision_recall_f1_all_false_positives():
    # Every "present" claim was wrong -- precision should be 0, recall
    # undefined-but-0-by-convention since there were no true positives.
    counts = el.ConfusionCounts(tp=0, fp=4, fn=2, tn=1)
    precision, recall, f1 = el.precision_recall_f1(counts)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test_accuracy():
    counts = el.ConfusionCounts(tp=3, fp=1, fn=1, tn=5)
    assert el.accuracy(counts) == pytest.approx(8 / 10)


# --------------------------------------------------------------------------
# Stratified breakdown -- the "row 1 vs row 6" scenario from the roadmap
# --------------------------------------------------------------------------


def test_stratified_breakdown_reveals_a_row_number_gap_aggregate_hides():
    # Front row (row_number=1): perfect. Back row (row_number=6): every
    # present student is a false negative. Aggregate accuracy looks fine;
    # the breakdown must not.
    rows = pd.DataFrame({
        "actually_present": [True, True, True, True],
        "predicted_present": [True, True, False, False],
        "row_number": [1, 1, 6, 6],
    })
    breakdown = el.stratified_breakdown(rows, ["row_number"])
    by_row = {row["value"]: row for _, row in breakdown.iterrows()}

    assert by_row[1]["recall"] == pytest.approx(1.0)
    assert by_row[6]["recall"] == pytest.approx(0.0)
    assert by_row[1]["n"] == 2
    assert by_row[6]["n"] == 2


def test_stratified_breakdown_multiple_group_columns():
    rows = pd.DataFrame({
        "actually_present": [True, True, True],
        "predicted_present": [True, False, True],
        "row_number": [1, 2, 3],
        "wears_glasses": [True, True, False],
    })
    breakdown = el.stratified_breakdown(rows, ["row_number", "wears_glasses"])
    assert set(breakdown["group"]) == {"row_number", "wears_glasses"}
    # 3 distinct row_number values + 2 distinct wears_glasses values
    assert len(breakdown) == 3 + 2


# --------------------------------------------------------------------------
# predicted_present_from_matches -- the CONFIDENT-only scoring rule
# --------------------------------------------------------------------------


def test_predicted_present_only_counts_confident():
    matches = [
        ClusterMatchRow(cluster_id=0, student_id=1, similarity=0.9, runner_up_similarity=0.1, decision="confident"),
        ClusterMatchRow(cluster_id=1, student_id=2, similarity=0.5, runner_up_similarity=0.45, decision="uncertain"),
        ClusterMatchRow(cluster_id=2, student_id=None, similarity=None, runner_up_similarity=None, decision="unmatched"),
    ]
    predicted = el.predicted_present_from_matches(matches, gallery_student_ids={1, 2, 3})
    assert predicted == {1: True, 2: False, 3: False}


# --------------------------------------------------------------------------
# sweep_match_threshold -- re-running match at many thresholds against the
# SAME cached cluster representatives/gallery
# --------------------------------------------------------------------------


def _rep(cluster_id: int, vector: np.ndarray) -> ClusterRepresentative:
    return ClusterRepresentative(
        cluster_id=cluster_id, vector=vector.astype(np.float32), best_crop_uri=f"crop_{cluster_id}.jpg",
        member_count=5, mean_quality=0.8,
    )


def test_sweep_match_threshold_recall_drops_as_threshold_rises():
    # One cluster, similarity 0.5 to its true student. At a low threshold
    # it's CONFIDENT (counted present); at a high threshold it isn't.
    gallery = {100: np.array([1.0, 0.0], dtype=np.float32)}
    cluster_reps = [_rep(0, np.array([0.5, np.sqrt(1 - 0.25)]))]  # cosine similarity to gallery[100] == 0.5
    roll_number_by_student_id = {100: "R100"}
    truth_df = _truth_df([
        {"roll_number": "R100", "actually_present": True, "row_number": 1, "seat_position": "left", "wears_glasses": False, "notes": ""},
    ])
    base_params = PipelineParams(match_margin_min=0.0, uncertain_band=0.0)

    sweep_df = el.sweep_match_threshold(
        cluster_reps, gallery, roll_number_by_student_id, truth_df, base_params,
        thresholds=[0.3, 0.5, 0.7],
    )

    by_threshold = {row["match_threshold"]: row for _, row in sweep_df.iterrows()}
    assert by_threshold[0.3]["recall"] == pytest.approx(1.0)  # 0.5 >= 0.3 -> confident -> present -> recall 1
    assert by_threshold[0.7]["recall"] == pytest.approx(0.0)  # 0.5 < 0.7 -> not confident -> absent -> recall 0


def test_rows_for_session_drops_students_not_in_gallery_mapping():
    predicted_present = {1: True}
    roll_number_by_student_id = {1: "R1"}  # student 2 deliberately missing
    truth_df = _truth_df([
        {"roll_number": "R1", "actually_present": True, "row_number": 1, "seat_position": "left", "wears_glasses": False, "notes": ""},
        {"roll_number": "R2", "actually_present": True, "row_number": 1, "seat_position": "left", "wears_glasses": False, "notes": ""},
    ])
    result = el.rows_for_session(predicted_present, roll_number_by_student_id, truth_df)
    # R2 has no prediction at all (NaN) -- caller is responsible for dropping
    # or flagging it, rows_for_session itself must not silently invent False.
    r2_row = result.loc[result["roll_number"] == "R2"].iloc[0]
    assert pd.isna(r2_row["predicted_present"])


# --------------------------------------------------------------------------
# pipeline_yield_for_session
# --------------------------------------------------------------------------


def test_pipeline_yield_flags_zero_crop_present_student_as_unrecoverable():
    cluster_reps = [_rep(0, np.array([1.0, 0.0]))]
    cluster_reps[0] = ClusterRepresentative(
        cluster_id=0, vector=np.array([1.0, 0.0], dtype=np.float32), best_crop_uri="c.jpg",
        member_count=12, mean_quality=0.9,
    )
    matches = [
        ClusterMatchRow(cluster_id=0, student_id=1, similarity=0.9, runner_up_similarity=0.1, decision="confident"),
    ]
    # student 2 is present but NO cluster was ever assigned to them.
    result = el.pipeline_yield_for_session(
        matches, cluster_reps, gallery_student_ids={1, 2}, present_student_ids={1, 2},
    )
    assert result["zero_accepted_crop_count"] == 1
    assert result["zero_accepted_crop_fraction"] == pytest.approx(0.5)
    assert result["mean_detections_per_present_student"] == pytest.approx((12 + 0) / 2)


# --------------------------------------------------------------------------
# clustering_quality
# --------------------------------------------------------------------------


def test_clustering_quality_no_labels_returns_none_metrics():
    cluster_reps = [_rep(0, np.array([1.0, 0.0])), _rep(1, np.array([0.0, 1.0]))]
    result = el.clustering_quality(cluster_reps, actual_present_count=3, cluster_labels_df=None)
    assert result["cluster_count"] == 2
    assert result["cluster_count_minus_actual"] == -1
    assert result["purity"] is None


def test_majority_vote_predicted_present_requires_enough_votes():
    gallery = {100: np.array([1.0, 0.0], dtype=np.float32), 200: np.array([0.0, 1.0], dtype=np.float32)}
    # 3 crops strongly matching student 100, 1 crop strongly matching student 200.
    embeddings = np.array([
        [1.0, 0.0],
        [0.95, np.sqrt(1 - 0.95 ** 2)],
        [0.98, np.sqrt(1 - 0.98 ** 2)],
        [0.0, 1.0],
    ], dtype=np.float32)
    params = PipelineParams(match_threshold=0.5, cluster_min_samples=3)

    predicted = el.majority_vote_predicted_present(embeddings, gallery, params)
    assert predicted[100] is True  # 3 votes >= min_votes(3)
    assert predicted[200] is False  # only 1 vote < min_votes(3)


def test_majority_vote_predicted_present_abstains_below_threshold():
    gallery = {100: np.array([1.0, 0.0], dtype=np.float32)}
    # A crop that's only weakly similar to the one gallery student.
    embeddings = np.array([[0.1, np.sqrt(1 - 0.01)]], dtype=np.float32)
    params = PipelineParams(match_threshold=0.5, cluster_min_samples=1)
    predicted = el.majority_vote_predicted_present(embeddings, gallery, params)
    assert predicted[100] is False


def test_majority_vote_predicted_present_empty_gallery_or_embeddings():
    assert el.majority_vote_predicted_present(np.zeros((0, 2)), {}, PipelineParams()) == {}
    gallery = {100: np.array([1.0, 0.0], dtype=np.float32)}
    result = el.majority_vote_predicted_present(np.zeros((0, 2)), gallery, PipelineParams())
    assert result == {100: False}


def test_clustering_quality_detects_over_split_and_merge():
    cluster_reps = [_rep(i, np.array([1.0, 0.0])) for i in range(3)]
    # Person "R1" was over-split into clusters 0 and 1; clusters 1 and 2 were
    # merged (cluster 1 also incorrectly labelled as containing "R2").
    labels_df = pd.DataFrame({
        "cluster_id": [0, 1, 1, 2],
        "roll_number": ["R1", "R1", "R2", "R3"],
    })
    result = el.clustering_quality(cluster_reps, actual_present_count=3, cluster_labels_df=labels_df)
    assert result["labelled_cluster_count"] == 3  # clusters 0, 1, 2
    # 3 distinct roll numbers (R1, R2, R3); only R1 is labelled against more
    # than one cluster_id (0 and 1) -- over-split.
    assert result["over_split_rate"] == pytest.approx(1 / 3)
    assert result["merge_rate"] == pytest.approx(1 / 3)  # cluster 1 has two conflicting roll numbers
