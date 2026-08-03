"""Phase 7 shared library: everything evaluate.py, sweep_threshold.py, and
failure_gallery.py have in common.

Two halves, same pure/impure split every other phase in this project uses:

  - PURE (no DB, no filesystem beyond a parquet path you hand it): truth.csv
    loading, confusion-matrix counting, precision/recall/F1, the stratified
    breakdown, and the "re-run match at arbitrary params against cached
    clusters" helper. All directly unit-testable (see tests/test_eval_lib.py
    -- run from the repo root with PYTHONPATH including this directory and
    services/worker, since this module imports pipeline.match/pipeline.params
    for the re-match helper).
  - IMPURE: `fetch_session_context`, a plain-psycopg2 DB lookup (same style
    as gallery_sanity.py/match_report.py -- standalone, no api/worker package
    dependency) that turns a class_session_id into everything the pure
    functions need: the course's enrolled students, their cached gallery
    vectors, and the job_dir of the most recent successfully-processed video
    for that session.

CONVENTION this phase introduces: an eval dataset directory's name
(eval/datasets/{session_id}/) IS the class_session_id, as a string. This is
what lets these scripts go from "a folder of truth.csv files" straight to
"the right course's enrollment, the right cached embeddings" without a
separate mapping file.

Evaluation scoring rule (documented once, here, since every script depends
on it): a CONFIDENT decision counts as "system says present"; UNCERTAIN
(awaiting teacher review) and UNMATCHED both count as "system says absent"
for metrics purposes. This scores the pipeline's AUTOMATIC decision honestly
-- crediting a correct guess that a human still has to confirm would
overstate what the system itself achieved, and Phase 8's review workflow is
a downstream mitigation, not part of what Phase 7 is measuring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.match import CONFIDENT, ClusterRepresentative, match_clusters
from pipeline.params import PipelineParams

TRUTH_COLUMNS = ["roll_number", "actually_present", "row_number", "seat_position", "wears_glasses", "notes"]


# --------------------------------------------------------------------------
# Ground truth loading
# --------------------------------------------------------------------------


def load_truth_csv(path: Path) -> pd.DataFrame:
    """Reads one session's truth.csv (Phase 7 deliverable 1's format).
    roll_number stays a string throughout (some institutions' roll numbers
    have leading zeros or letters) -- never coerced to int.
    """
    df = pd.read_csv(path, dtype={"roll_number": str})
    missing = [c for c in TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s): {missing}")
    df["actually_present"] = _coerce_bool_column(df["actually_present"])
    df["wears_glasses"] = _coerce_bool_column(df["wears_glasses"])
    df["row_number"] = df["row_number"].astype(int)
    return df


def _coerce_bool_column(series: pd.Series) -> pd.Series:
    """pandas' read_csv auto-infers a clean "True"/"False" column as native
    bool dtype, in which case a plain .astype(bool) is a safe no-op -- but
    it silently does NOT do this for other truthy spellings a hand-edited
    or differently-written truth.csv might contain ("TRUE"/"FALSE", "1"/"0",
    "yes"/"no"), and naively calling .astype(bool) on a resulting
    object/string column is a real trap: the non-empty STRING "False"
    astype(bool)s to True, since Python truthiness of a non-empty string is
    always True. Handled explicitly here rather than trusted to pandas'
    inference -- the same category of dtype gotcha as the empty-"accepted"-
    column bug found in Phase 5's quality_df masking, guarded against
    directly instead of assuming a repeat can't happen.
    """
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_all_truth(datasets_dir: Path) -> pd.DataFrame:
    """Every eval/datasets/{session_id}/truth.csv, concatenated, with a
    `session_id` column added (the directory name -- see module docstring's
    convention). Sessions with no truth.csv yet (mid-labelling) are skipped,
    not errored on.
    """
    frames = []
    if not datasets_dir.exists():
        return pd.DataFrame(columns=TRUTH_COLUMNS + ["session_id"])

    for session_dir in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
        truth_path = session_dir / "truth.csv"
        if not truth_path.exists():
            continue
        df = load_truth_csv(truth_path)
        df["session_id"] = session_dir.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=TRUTH_COLUMNS + ["session_id"])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Confusion matrix + headline metrics (pure)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int  # present, system says present
    fp: int  # ABSENT, system says present -- the proxy hole
    fn: int  # present, system says ABSENT -- the trust killer
    tn: int  # absent, system says absent


def compute_confusion(rows_df: pd.DataFrame) -> ConfusionCounts:
    """`rows_df` needs exactly two boolean columns: `actually_present` and
    `predicted_present`. One row per (session, student) pair being scored.
    """
    actual = rows_df["actually_present"].astype(bool)
    predicted = rows_df["predicted_present"].astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def precision_recall_f1(counts: ConfusionCounts) -> tuple[float, float, float]:
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) > 0 else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def accuracy(counts: ConfusionCounts) -> float:
    total = counts.tp + counts.fp + counts.fn + counts.tn
    return (counts.tp + counts.tn) / total if total > 0 else 0.0


@dataclass(frozen=True)
class MetricsSummary:
    counts: ConfusionCounts
    precision: float
    recall: float
    f1: float
    accuracy: float

    def to_dict(self) -> dict:
        return {
            "tp": self.counts.tp, "fp": self.counts.fp, "fn": self.counts.fn, "tn": self.counts.tn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1, "accuracy": self.accuracy,
        }


def summarize(rows_df: pd.DataFrame) -> MetricsSummary:
    counts = compute_confusion(rows_df)
    precision, recall, f1 = precision_recall_f1(counts)
    return MetricsSummary(counts=counts, precision=precision, recall=recall, f1=f1, accuracy=accuracy(counts))


# --------------------------------------------------------------------------
# Stratified breakdown (Phase 7 deliverable 2, "mandatory, not optional")
# --------------------------------------------------------------------------


def stratified_breakdown(rows_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """One row per (group_col, value) -- e.g. (row_number, 1), (row_number, 2),
    (seat_position, "left"), (wears_glasses, True) -- each with its own full
    confusion-matrix-derived metrics. Deliberately a separate row per
    dimension rather than a full cross-product: the roadmap's own examples
    ("row 1 vs row 6", "students wearing glasses") are single-dimension
    breakdowns, and a full cross-product fragments an 8-session dataset into
    cells too small to mean anything.
    """
    records = []
    for col in group_cols:
        for value, group_df in rows_df.groupby(col):
            summary = summarize(group_df)
            records.append({
                "group": col, "value": value, "n": len(group_df),
                **summary.to_dict(),
            })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Cached-artifact loading (job_dir files, no DB)
# --------------------------------------------------------------------------


def load_cluster_representatives(job_dir: Path) -> list[ClusterRepresentative]:
    """Reads job_dir/cluster/cluster_summary.parquet into the same
    ClusterRepresentative shape match_clusters expects -- using the
    parquet's own LOCAL cluster_id (not a detected_cluster DB id), since
    evaluate.py/sweep_threshold.py re-run matching entirely in-memory and
    never write cluster_match rows (that would corrupt real session data
    with speculative re-matches at experimental threshold values).
    """
    summary_path = job_dir / "cluster" / "cluster_summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found -- has this job's cluster stage run?")

    summary_df = pd.read_parquet(summary_path)
    reps = []
    for _, row in summary_df.iterrows():
        vector = np.frombuffer(row["representative_vector"], dtype=np.float32)
        reps.append(
            ClusterRepresentative(
                cluster_id=int(row["cluster_id"]),
                vector=vector,
                best_crop_uri=row["best_crop_uri"],
                member_count=int(row["member_count"]),
                mean_quality=float(row["mean_quality"]),
            )
        )
    return reps


def majority_vote_predicted_present(
    embeddings: np.ndarray,
    gallery: dict[int, np.ndarray],
    params: PipelineParams,
    min_votes: int | None = None,
) -> dict[int, bool]:
    """Ablation-only (Phase 7 deliverable 6, "clustering off (match every
    crop individually and majority vote)"): bypasses DBSCAN and the
    Hungarian algorithm entirely. Every ACCEPTED crop's embedding
    independently "votes" for its single nearest gallery student (plain
    cosine-similarity argmax, no one-to-one constraint -- with clustering
    off there's no cluster-level aggregation left to make one-to-one
    assignment meaningful), abstaining if even its best similarity falls
    below `match_threshold`. A student is then marked "system says present"
    if enough INDEPENDENT crops voted for them.

    ASSUMPTION: "majority vote" doesn't specify a vote count -- reusing
    `cluster_min_samples` (this project's existing notion of "how many
    corroborating crops does it take to trust a detection," Phase 5) as the
    default vote threshold, rather than inventing a fourth unrelated
    tunable just for this one ablation.
    """
    if min_votes is None:
        min_votes = params.cluster_min_samples

    gallery_matrix, student_ids = np.zeros((0, 0)), []
    if gallery:
        student_ids = list(gallery.keys())
        gallery_matrix = np.stack([gallery[sid] for sid in student_ids]).astype(np.float32)

    votes: dict[int, int] = {sid: 0 for sid in student_ids}
    if embeddings.shape[0] > 0 and gallery_matrix.shape[0] > 0:
        similarity = embeddings.astype(np.float32) @ gallery_matrix.T  # (N crops, S students)
        best_col = similarity.argmax(axis=1)
        best_sim = similarity[np.arange(similarity.shape[0]), best_col]
        for crop_idx in range(similarity.shape[0]):
            if best_sim[crop_idx] >= params.match_threshold:
                votes[student_ids[best_col[crop_idx]]] += 1

    return {sid: (count >= min_votes) for sid, count in votes.items()}


def plain_mean_representative(embeddings: np.ndarray, member_indices: tuple[int, ...]) -> np.ndarray:
    """Ablation-only (Phase 7 deliverable 6, "quality weighting off"):
    recomputes a cluster's representative as an UNWEIGHTED mean of its
    member embeddings, bypassing pipeline.cluster's quality-weighted mean
    entirely. Deliberately NOT a pipeline.cluster code path -- this exists
    only to answer "how much does quality weighting help," and the shipped
    clustering code (already verified on real hardware in Phase 5) has no
    reason to grow an ablation-only toggle.
    """
    vectors = embeddings[np.array(member_indices)]
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        return mean.astype(np.float32)
    return (mean / norm).astype(np.float32)


# --------------------------------------------------------------------------
# Predicted-present derivation + the threshold sweep (pure)
# --------------------------------------------------------------------------


def predicted_present_from_matches(matches, gallery_student_ids: set[int]) -> dict[int, bool]:
    """See module docstring for the CONFIDENT-only scoring rule. Every
    enrolled student in `gallery_student_ids` gets an entry, defaulting to
    False -- a student no cluster was ever assigned to is exactly as
    "system says absent" as one that was assigned but decided UNCERTAIN/
    UNMATCHED.
    """
    confident_ids = {m.student_id for m in matches if m.decision == CONFIDENT and m.student_id is not None}
    return {sid: (sid in confident_ids) for sid in gallery_student_ids}


def rows_for_session(
    predicted_present: dict[int, bool],
    roll_number_by_student_id: dict[int, str],
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Joins a {student_id: bool} prediction map to one session's truth.csv
    rows via roll_number (truth.csv's only student identifier -- see Phase 7
    deliverable 1's format). Students in truth.csv with no matching
    roll_number in the gallery (e.g. a labelling typo) are dropped with a
    printed warning by the caller, not silently included as a guaranteed
    false negative.
    """
    predicted_by_roll = {
        roll_number_by_student_id[sid]: present
        for sid, present in predicted_present.items()
        if sid in roll_number_by_student_id
    }
    merged = truth_df.copy()
    merged["predicted_present"] = merged["roll_number"].map(predicted_by_roll)
    return merged


def sweep_match_threshold(
    cluster_reps: list[ClusterRepresentative],
    gallery: dict[int, np.ndarray],
    roll_number_by_student_id: dict[int, str],
    truth_df: pd.DataFrame,
    base_params: PipelineParams,
    thresholds: list[float],
) -> pd.DataFrame:
    """Phase 7 deliverable 3's core loop: re-runs ONLY match_clusters (never
    detect/embed/cluster) at every threshold value, against the SAME cached
    cluster representatives and gallery -- cheap enough to do 36 times
    (0.25 to 0.60 step 0.01) per session.
    """
    rows = []
    for threshold in thresholds:
        params = replace(base_params, match_threshold=threshold)
        result = match_clusters(cluster_reps, gallery, params)
        predicted = predicted_present_from_matches(result.matches, set(gallery.keys()))
        scored = rows_for_session(predicted, roll_number_by_student_id, truth_df)
        scored = scored.dropna(subset=["predicted_present"])
        summary = summarize(scored)
        rows.append({"match_threshold": round(threshold, 2), **summary.to_dict()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pipeline yield (Phase 7 deliverable 2)
# --------------------------------------------------------------------------


def pipeline_yield_for_session(
    matches,
    cluster_reps: list[ClusterRepresentative],
    gallery_student_ids: set[int],
    present_student_ids: set[int],
) -> dict:
    """detections-per-present-student and the zero-accepted-crop fraction.

    ASSUMPTION (the roadmap doesn't specify how to attribute a raw crop to a
    student who was never matched): "detections attributed to a student" =
    member_count of whichever cluster was ASSIGNED to them by the Hungarian
    algorithm, regardless of band (confident/uncertain/unmatched) -- the
    assignment is the only crop-to-identity link this pipeline produces at
    all. A present student with NO cluster assigned to them gets 0, which is
    exactly the roadmap's "unrecoverable failure": no crop-level evidence
    ever got attached to them, as opposed to evidence existing but being
    mis-scored. A present student who exists in `gallery_student_ids` but
    was assigned a cluster with a WRONG identity would still show a nonzero
    count here (it's someone else's crops) -- a limitation worth stating in
    docs/EVALUATION.md, not something this function can detect without
    per-crop ground truth.
    """
    member_count_by_cluster_id = {c.cluster_id: c.member_count for c in cluster_reps}

    detections_by_student: dict[int, int] = {sid: 0 for sid in gallery_student_ids}
    for m in matches:
        if m.student_id is not None and m.student_id in detections_by_student:
            detections_by_student[m.student_id] += member_count_by_cluster_id.get(m.cluster_id, 0)

    present_counts = [detections_by_student.get(sid, 0) for sid in present_student_ids]
    zero_crop_present = sum(1 for c in present_counts if c == 0)

    return {
        "mean_detections_per_present_student": float(np.mean(present_counts)) if present_counts else 0.0,
        "zero_accepted_crop_count": zero_crop_present,
        "zero_accepted_crop_fraction": (zero_crop_present / len(present_counts)) if present_counts else 0.0,
    }


# --------------------------------------------------------------------------
# Clustering quality (Phase 7 deliverable 2)
# --------------------------------------------------------------------------


def clustering_quality(
    cluster_reps: list[ClusterRepresentative],
    actual_present_count: int,
    cluster_labels_df: pd.DataFrame | None = None,
) -> dict:
    """`cluster_labels_df` is this project's OWN addition (not specified by
    the roadmap beyond "requires spot-labelled clusters; support partial
    labelling"): an optional eval/datasets/{session_id}/cluster_labels.csv
    with columns (cluster_id, roll_number), covering as many or as few
    clusters as you've had time to spot-check by eye against
    cluster_report.py's contact sheets. Purity/over-split/merge rate are
    computed only over the labelled subset; None if nothing's labelled yet.
    """
    result = {
        "cluster_count": len(cluster_reps),
        "actual_present_count": actual_present_count,
        "cluster_count_minus_actual": len(cluster_reps) - actual_present_count,
    }

    if cluster_labels_df is None or len(cluster_labels_df) == 0:
        result.update({"purity": None, "over_split_rate": None, "merge_rate": None, "labelled_cluster_count": 0})
        return result

    labelled = cluster_labels_df.dropna(subset=["roll_number"])
    labelled_cluster_ids = set(labelled["cluster_id"])

    # Purity: a cluster is "pure" if every spot-labelled crop -- here,
    # simplified to one label per cluster since cluster_labels.csv labels
    # whole clusters, not individual crops -- names exactly one person. With
    # one label per cluster this is trivially 100% UNLESS the same person's
    # true identity was manually recorded against two different cluster_ids
    # (that's over-splitting, counted separately below) or two different
    # people were recorded against the SAME cluster_id (that would mean the
    # label file itself has a duplicate cluster_id with conflicting roll
    # numbers -- a labelling error, flagged rather than silently averaged).
    duplicate_cluster_ids = labelled["cluster_id"][labelled["cluster_id"].duplicated(keep=False)]
    conflicting = 0
    for cid in set(duplicate_cluster_ids):
        rolls = set(labelled.loc[labelled["cluster_id"] == cid, "roll_number"])
        if len(rolls) > 1:
            conflicting += 1

    pure_count = len(labelled_cluster_ids) - conflicting
    purity = pure_count / len(labelled_cluster_ids) if labelled_cluster_ids else None

    # Over-split: the same roll_number labelled against more than one cluster_id.
    roll_to_clusters: dict[str, set] = {}
    for _, row in labelled.iterrows():
        roll_to_clusters.setdefault(row["roll_number"], set()).add(row["cluster_id"])
    over_split_people = sum(1 for clusters in roll_to_clusters.values() if len(clusters) > 1)
    over_split_rate = over_split_people / len(roll_to_clusters) if roll_to_clusters else None

    # Merge rate: a single cluster_id labelled with more than one distinct
    # roll_number (the conflicting count above, reused).
    merge_rate = conflicting / len(labelled_cluster_ids) if labelled_cluster_ids else None

    result.update({
        "purity": purity,
        "over_split_rate": over_split_rate,
        "merge_rate": merge_rate,
        "labelled_cluster_count": len(labelled_cluster_ids),
    })
    return result


# --------------------------------------------------------------------------
# DB lookup (impure) -- turns a class_session_id into everything the pure
# functions above need. psycopg2 is imported LOCALLY, not at module level,
# for the same reason pipeline/match.py defers sqlalchemy/config/db: every
# function above this point stays importable/testable without a database
# driver installed at all.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionContext:
    class_session_id: int
    course_id: int
    processing_job_id: int
    job_dir: Path
    roll_number_by_student_id: dict[int, str]
    full_name_by_student_id: dict[int, str]
    gallery: dict[int, np.ndarray]  # only students with a cached gallery_mean_vector (Phase 1)
    enrolled_student_ids: set[int]  # ALL enrolled, including those with no gallery vector yet


def fetch_session_context(database_url: str, class_session_id: int, job_data_dir: Path) -> SessionContext:
    """`job_data_dir` is the same directory pipeline.run.process_session
    uses (settings.job_data_dir on the worker) -- passed in explicitly
    rather than imported from worker.config, so this module has zero
    dependency on either the api or worker package, same convention as
    every other eval/scripts/*.py.
    """
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT course_id FROM class_session WHERE id = %s", (class_session_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"No class_session with id={class_session_id}")
            course_id = row[0]

            cur.execute(
                "SELECT id FROM processing_job WHERE class_session_id = %s AND state = 'succeeded' "
                "ORDER BY finished_at DESC LIMIT 1",
                (class_session_id,),
            )
            job_row = cur.fetchone()
            if job_row is None:
                raise ValueError(f"No succeeded processing_job for class_session_id={class_session_id}")
            processing_job_id = job_row[0]

            cur.execute(
                "SELECT s.id, s.roll_number, s.full_name, s.gallery_mean_vector "
                "FROM student s JOIN enrollment e ON e.student_id = s.id "
                "WHERE e.course_id = %s",
                (course_id,),
            )
            student_rows = cur.fetchall()
    finally:
        conn.close()

    roll_number_by_student_id = {r[0]: r[1] for r in student_rows}
    full_name_by_student_id = {r[0]: r[2] for r in student_rows}
    enrolled_student_ids = {r[0] for r in student_rows}
    gallery = {r[0]: np.frombuffer(r[3], dtype=np.float32) for r in student_rows if r[3] is not None}

    return SessionContext(
        class_session_id=class_session_id,
        course_id=course_id,
        processing_job_id=processing_job_id,
        job_dir=job_data_dir / str(processing_job_id),
        roll_number_by_student_id=roll_number_by_student_id,
        full_name_by_student_id=full_name_by_student_id,
        gallery=gallery,
        enrolled_student_ids=enrolled_student_ids,
    )


def fetch_gallery_photo_uris(database_url: str, student_id: int) -> list[str]:
    """Every gallery_photo.storage_uri for a student, best quality_score
    first -- used by failure_gallery.py to show "the student's enrollment
    photos" (plural; the roadmap's own phrasing) next to whatever the
    pipeline actually detected for them.
    """
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT storage_uri FROM gallery_photo WHERE student_id = %s ORDER BY quality_score DESC",
                (student_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]
