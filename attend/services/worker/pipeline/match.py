"""Gallery matching and three-band decisions (Phase 6).

Assigns names to detected clusters, and -- more importantly -- decides when
NOT to. The three-band decision (confident / uncertain / unmatched) is what
makes this system safe to deploy: a system that is 92% accurate and honest
about the other 8% is usable; a system that is 92% accurate and silent
about it is not.

Non-negotiable rule #1: match_clusters and build_session_summary are pure
functions -- numpy/dataclasses in, dataclasses out, no DB. The I/O wrapper
(run_match_stage, near the bottom of this file) is the only part that
touches the database, mirroring enrollment.py's split between
process_enrollment_video (pure) and enroll_student (DB-writing orchestrator).

sqlalchemy/config/db are imported LOCALLY inside run_match_stage, not at
module level, so that match_clusters/build_session_summary/build_gallery_matrix
stay importable (and unit-testable, see tests/test_match.py) in any
environment that has numpy/pandas/scipy but not the full worker DB stack --
the same reason pipeline/params.py stays dependency-free. Every other pure
stage module here (detect.py, quality.py, align.py, embed.py, cluster.py)
gets this property for free because they simply never touch the DB;
match.py is the first stage module where DB access is unavoidable
somewhere in the file, so the split has to be explicit.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pipeline.params import PipelineParams

logger = logging.getLogger("attend.worker.match")

CONFIDENT = "confident"
UNCERTAIN = "uncertain"
UNMATCHED = "unmatched"


@dataclass(frozen=True)
class ClusterRepresentative:
    cluster_id: int
    vector: np.ndarray  # (512,) float32, L2-normalised
    best_crop_uri: str
    member_count: int
    mean_quality: float


@dataclass(frozen=True)
class ClusterMatchRow:
    cluster_id: int
    student_id: int | None  # None whenever decision == UNMATCHED, even if Hungarian nominally assigned one
    similarity: float | None
    runner_up_similarity: float | None
    decision: str  # CONFIDENT | UNCERTAIN | UNMATCHED


@dataclass(frozen=True)
class MatchResult:
    matches: list[ClusterMatchRow]


def build_gallery_matrix(gallery: dict[int, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """`gallery`: student_id -> L2-normalised (512,) cached mean embedding,
    for students enrolled in THIS course ONLY (Phase 6 prompt, step a:
    "Only students in the course's enrollment table -- never the whole
    institution"). The caller (run_match_stage) is responsible for that
    filtering; this function just stacks whatever it's given into a matrix,
    preserving a stable student_id order for every downstream column index.

    Returns (matrix (S, 512), student_ids in matching row order). Empty
    gallery returns a (0, 0) matrix and an empty id list, not an error --
    a course with zero enrolled/enrolled-but-never-enrolled-biometrically
    students is a valid (if unhelpful) state.
    """
    student_ids = list(gallery.keys())
    if not student_ids:
        return np.zeros((0, 0), dtype=np.float32), student_ids
    matrix = np.stack([gallery[sid] for sid in student_ids]).astype(np.float32)
    return matrix, student_ids


def _decide_band(similarity: float, runner_up_similarity: float | None, params: PipelineParams) -> str:
    """Phase 6 deliverable 2, verbatim:
        margin = similarity - runner_up_similarity
        CONFIDENT if similarity >= match_threshold AND margin >= match_margin_min
        UNCERTAIN if similarity >= (match_threshold - uncertain_band) AND not confident
        UNMATCHED otherwise

    When there's no "other" gallery entry to compute a runner-up against
    (a gallery of exactly one student), margin collapses to `similarity`
    itself -- there's nothing to be confused with, so the margin gate can't
    meaningfully fail.
    """
    margin = similarity - runner_up_similarity if runner_up_similarity is not None else similarity
    if similarity >= params.match_threshold and margin >= params.match_margin_min:
        return CONFIDENT
    if similarity >= (params.match_threshold - params.uncertain_band):
        return UNCERTAIN
    return UNMATCHED


def match_clusters(
    cluster_reps: list[ClusterRepresentative], gallery: dict[int, np.ndarray], params: PipelineParams
) -> MatchResult:
    """Phase 6 deliverables 1-2.

    b. Cosine similarity matrix via ONE matrix multiply: both cluster
       representatives and gallery vectors are already L2-normalised, so a
       dot product IS cosine similarity -- no per-pair loop.
    c. ONE-TO-ONE assignment via the Hungarian algorithm
       (scipy.optimize.linear_sum_assignment) on the NEGATED similarity
       matrix (the library minimises cost; negating similarity turns
       "maximise total similarity" into that native minimisation).

       Why not greedy top-1: two different clusters can each have their
       SINGLE highest similarity pointed at the SAME gallery student -- two
       students who happen to look alike, or two noisy crops that both
       embed slightly toward a third identity. Greedy assignment (each
       cluster independently claims its best match) would assign BOTH
       clusters to that one student, violating the roadmap's non-negotiable
       one-to-one constraint ("one physical person cannot be two students")
       and silently stranding the other cluster with whatever was left,
       even if a globally better arrangement existed. The Hungarian
       algorithm instead finds the single bijection between clusters and
       gallery entries that maximises TOTAL similarity summed across every
       pair -- which by construction never double-books a gallery entry.
    d. For each ASSIGNED pair, record similarity and runner_up_similarity:
       the best similarity to any OTHER gallery entry (this cluster's own
       similarity row with the assigned column excluded) -- NOT the
       second-best row in the overall Hungarian solution, which is a
       different, less meaningful number (some other cluster's runner-up).
    """
    if not cluster_reps:
        return MatchResult(matches=[])

    gallery_matrix, student_ids = build_gallery_matrix(gallery)
    cluster_matrix = np.stack([c.vector for c in cluster_reps]).astype(np.float32)

    if gallery_matrix.shape[0] == 0:
        # No gallery to match against at all -- every cluster is unrecognised.
        return MatchResult(matches=[
            ClusterMatchRow(cluster_id=c.cluster_id, student_id=None, similarity=None,
                             runner_up_similarity=None, decision=UNMATCHED)
            for c in cluster_reps
        ])

    similarity = cluster_matrix @ gallery_matrix.T  # (C, S), cosine similarity

    row_indices, col_indices = linear_sum_assignment(-similarity)
    assigned_student_col_by_row: dict[int, int] = dict(zip(row_indices.tolist(), col_indices.tolist()))

    matches: list[ClusterMatchRow] = []
    for row, cluster_rep in enumerate(cluster_reps):
        if row not in assigned_student_col_by_row:
            # More clusters than gallery entries -- this one has no assignment at all.
            matches.append(ClusterMatchRow(
                cluster_id=cluster_rep.cluster_id, student_id=None, similarity=None,
                runner_up_similarity=None, decision=UNMATCHED,
            ))
            continue

        col = assigned_student_col_by_row[row]
        sim = float(similarity[row, col])

        other_sims = np.delete(similarity[row], col)
        runner_up = float(other_sims.max()) if other_sims.size > 0 else None

        decision = _decide_band(sim, runner_up, params)
        matches.append(ClusterMatchRow(
            cluster_id=cluster_rep.cluster_id,
            student_id=student_ids[col] if decision != UNMATCHED else None,
            similarity=sim,
            runner_up_similarity=runner_up,
            decision=decision,
        ))

    return MatchResult(matches=matches)


# --------------------------------------------------------------------------
# Session summary (Phase 6 deliverable 4)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionSummary:
    total_enrolled: int
    proposed_present: int
    needs_review: int
    proposed_absent: int
    unrecognised_clusters: int
    coverage_percent: float
    mean_confident_similarity: float | None
    session_health: str  # "good" | "fair" | "poor"

    def to_json_dict(self) -> dict:
        return {
            "total_enrolled": self.total_enrolled,
            "proposed_present": self.proposed_present,
            "needs_review": self.needs_review,
            "proposed_absent": self.proposed_absent,
            "unrecognised_clusters": self.unrecognised_clusters,
            "coverage_percent": self.coverage_percent,
            "mean_confident_similarity": self.mean_confident_similarity,
            "session_health": self.session_health,
        }


def _compute_session_health(
    coverage_percent: float, mean_confident_similarity: float | None, preflight_had_warnings: bool, params: PipelineParams
) -> str:
    """ASSUMPTION thresholds (see params.py) -- the roadmap specifies the
    THREE conditions ("poor when coverage_percent is low, mean_confident_
    similarity is low, or the pre-flight check returned warnings") but not
    numeric cutoffs; Phase 7 calibrates these against real sessions.
    """
    poor = (
        coverage_percent < params.session_health_poor_coverage_percent
        or (mean_confident_similarity is not None and mean_confident_similarity < params.session_health_poor_mean_similarity)
        or preflight_had_warnings
    )
    if poor:
        return "poor"
    return "fair" if coverage_percent < params.session_health_fair_coverage_percent else "good"


def build_session_summary(
    matches: list[ClusterMatchRow],
    enrolled_student_ids: list[int],
    preflight_had_warnings: bool,
    params: PipelineParams,
) -> SessionSummary:
    """Phase 6 deliverable 4. Asserts, as a hard invariant (prompt, verbatim:
    "fail the job loudly rather than producing a roster that silently omits
    a student"), that proposed_present + needs_review + proposed_absent ==
    total_enrolled EXACTLY -- every enrolled student falls into exactly one
    of those three buckets, never zero, never two.
    """
    enrolled_set = set(enrolled_student_ids)
    total_enrolled = len(enrolled_set)

    confident_student_ids = {m.student_id for m in matches if m.decision == CONFIDENT}
    uncertain_student_ids = {m.student_id for m in matches if m.decision == UNCERTAIN}

    proposed_present = len(confident_student_ids & enrolled_set)
    needs_review = len(uncertain_student_ids & enrolled_set)
    matched_ids = confident_student_ids | uncertain_student_ids
    proposed_absent = len(enrolled_set - matched_ids)

    unrecognised_clusters = sum(1 for m in matches if m.decision == UNMATCHED)

    if proposed_present + needs_review + proposed_absent != total_enrolled:
        raise ValueError(
            "build_session_summary: invariant violated -- proposed_present + needs_review + "
            f"proposed_absent ({proposed_present} + {needs_review} + {proposed_absent}) != "
            f"total_enrolled ({total_enrolled}). Refusing to produce a roster that silently "
            "omits or double-counts a student; this points at a bug in match_clusters or the "
            "gallery/enrollment query, not a data quality issue to route around."
        )

    coverage_percent = ((proposed_present + needs_review) / total_enrolled * 100.0) if total_enrolled > 0 else 0.0

    confident_similarities = [m.similarity for m in matches if m.decision == CONFIDENT and m.similarity is not None]
    mean_confident_similarity = float(np.mean(confident_similarities)) if confident_similarities else None

    session_health = _compute_session_health(coverage_percent, mean_confident_similarity, preflight_had_warnings, params)

    return SessionSummary(
        total_enrolled=total_enrolled,
        proposed_present=proposed_present,
        needs_review=needs_review,
        proposed_absent=proposed_absent,
        unrecognised_clusters=unrecognised_clusters,
        coverage_percent=coverage_percent,
        mean_confident_similarity=mean_confident_similarity,
        session_health=session_health,
    )


# --------------------------------------------------------------------------
# I/O + DB wrapper: what run.py actually calls (Phase 6 deliverables 3, 5, 6)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchStageSummary:
    cluster_count: int
    confident_count: int
    uncertain_count: int
    unmatched_count: int
    session_summary: SessionSummary


def run_match_stage(
    cluster_summary_parquet_path: Path,
    class_session_id: int,
    params: PipelineParams,
    processing_job_id: int,
) -> MatchStageSummary:
    """The DB-aware orchestrator match_clusters/build_session_summary don't
    need to be (non-negotiable rule #1) but this stage inescapably does:
    Phase 6 deliverable 1a says the gallery is scoped to "students in the
    course's enrollment table -- never the whole institution," which means
    a live query against `enrollment`/`course`/`class_session`, not a file
    under job_dir like every earlier stage.

    Steps, in order:
      1. Read cluster_summary.parquet (already excludes noise -- see
         cluster.py's _build_all_diagnostics, which never emits a
         diagnostics entry for NOISE_LABEL).
      2. Insert one detected_cluster row per cluster FIRST, so the DB-
         generated id can be used directly as ClusterRepresentative.cluster_id
         -- no separate local-id-to-DB-id mapping step needed afterward.
      3. Look up the session's course, every enrolled student, and each
         enrolled student's cached gallery_mean_vector (Phase 1). Students
         enrolled but never gallery-enrolled (gallery_mean_vector IS NULL)
         are excluded from the MATCHING gallery (nothing to compare against)
         but still counted in enrolled_student_ids, so build_session_summary
         correctly places them in proposed_absent rather than silently
         dropping them.
      4. match_clusters (pure) -> cluster_match rows, inserted with decision
         values that line up exactly with ClusterMatchDecision's enum values
         (both are "confident"/"uncertain"/"unmatched" -- see models.py).
      5. build_session_summary (pure), fed preflight_had_warnings from
         video_upload.preflight_status_json (Phase 6 retrofit -- see db.py's
         module docstring and routers/upload.py).
      6. Persist class_session.draft_summary_json and flip its status to
         AWAITING_REVIEW. Does NOT write attendance_record rows -- turning a
         draft into committed attendance is Phase 8's job, not this stage's.
    """
    # Deferred imports -- see module docstring: keeps match_clusters/
    # build_session_summary importable without the full worker DB stack.
    from sqlalchemy import insert, select, update

    from config import settings
    from db import get_engine, table

    engine = get_engine()

    cluster_summary_df = pd.read_parquet(cluster_summary_parquet_path)

    class_session_t = table("class_session")
    course_t = table("course")
    enrollment_t = table("enrollment")
    student_t = table("student")
    processing_job_t = table("processing_job")
    video_upload_t = table("video_upload")
    detected_cluster_t = table("detected_cluster")
    cluster_match_t = table("cluster_match")

    now = datetime.now(timezone.utc)
    retention_expires_at = now + timedelta(days=settings.biometric_retention_days)

    with engine.begin() as conn:
        session_row = conn.execute(
            select(class_session_t).where(class_session_t.c.id == class_session_id)
        ).mappings().first()
        if session_row is None:
            raise ValueError(f"run_match_stage: no class_session with id={class_session_id}")

        course_row = conn.execute(
            select(course_t).where(course_t.c.id == session_row["course_id"])
        ).mappings().first()
        if course_row is None:
            raise ValueError(
                f"run_match_stage: class_session {class_session_id} references "
                f"missing course_id={session_row['course_id']}"
            )

        # Step 1a of the roadmap's matching contract: ONLY this course's
        # enrolled students, never the whole institution.
        enrolled_rows = conn.execute(
            select(student_t)
            .select_from(enrollment_t.join(student_t, student_t.c.id == enrollment_t.c.student_id))
            .where(enrollment_t.c.course_id == course_row["id"])
        ).mappings().all()

        enrolled_student_ids = [int(r["id"]) for r in enrolled_rows]
        gallery: dict[int, np.ndarray] = {}
        for r in enrolled_rows:
            if r["gallery_mean_vector"] is None:
                logger.warning(
                    "run_match_stage: student %s is enrolled in course %s but has no cached "
                    "gallery vector yet (never completed Phase 1 enrollment) -- excluded from "
                    "matching, will land in proposed_absent.",
                    r["id"], course_row["id"],
                )
                continue
            gallery[int(r["id"])] = np.frombuffer(r["gallery_mean_vector"], dtype=np.float32)

        # Step 2: insert detected_cluster rows first, so their DB-generated
        # ids can be used directly as the ClusterRepresentative.cluster_id --
        # ClusterMatch.cluster_id is an FK into detected_cluster, not into
        # cluster_summary.parquet's own (job-local) cluster_id column.
        cluster_reps: list[ClusterRepresentative] = []
        for _, row in cluster_summary_df.iterrows():
            db_cluster_id = conn.execute(
                insert(detected_cluster_t).values(
                    processing_job_id=processing_job_id,
                    representative_vector=row["representative_vector"],
                    crop_count=int(row["member_count"]),
                    mean_quality=float(row["mean_quality"]),
                    best_crop_uri=row["best_crop_uri"],
                    created_at=now,
                    retention_expires_at=retention_expires_at,
                )
            ).inserted_primary_key[0]

            vector = np.frombuffer(row["representative_vector"], dtype=np.float32)
            cluster_reps.append(
                ClusterRepresentative(
                    cluster_id=int(db_cluster_id),
                    vector=vector,
                    best_crop_uri=row["best_crop_uri"],
                    member_count=int(row["member_count"]),
                    mean_quality=float(row["mean_quality"]),
                )
            )

        # Step 3/4: pure matching logic, no DB involved in the decision itself.
        match_result = match_clusters(cluster_reps, gallery, params)

        for m in match_result.matches:
            conn.execute(
                insert(cluster_match_t).values(
                    cluster_id=m.cluster_id,
                    student_id=m.student_id,
                    similarity=m.similarity,
                    runner_up_similarity=m.runner_up_similarity,
                    decision=m.decision,
                )
            )

        # Step 5: pre-flight warnings, persisted back in Phase 6's own
        # retrofit of video_upload.preflight_status_json (see db.py).
        job_row = conn.execute(
            select(processing_job_t).where(processing_job_t.c.id == processing_job_id)
        ).mappings().first()
        preflight_had_warnings = False
        if job_row is not None:
            video_row = conn.execute(
                select(video_upload_t).where(video_upload_t.c.id == job_row["video_upload_id"])
            ).mappings().first()
            if video_row is not None and video_row["preflight_status_json"]:
                preflight_dict = json.loads(video_row["preflight_status_json"])
                preflight_had_warnings = preflight_dict.get("status") != "pass"

        session_summary = build_session_summary(
            match_result.matches, enrolled_student_ids, preflight_had_warnings, params
        )

        # Step 6: persist the draft and move the session out of "processing."
        conn.execute(
            update(class_session_t)
            .where(class_session_t.c.id == class_session_id)
            .values(
                status="awaiting_review",
                draft_summary_json=json.dumps(session_summary.to_json_dict()),
            )
        )

    confident_count = sum(1 for m in match_result.matches if m.decision == CONFIDENT)
    uncertain_count = sum(1 for m in match_result.matches if m.decision == UNCERTAIN)
    unmatched_count = sum(1 for m in match_result.matches if m.decision == UNMATCHED)

    logger.info(
        "match stage: %d clusters -- %d confident, %d uncertain, %d unmatched; "
        "session_health=%s, coverage=%.1f%%",
        len(cluster_reps), confident_count, uncertain_count, unmatched_count,
        session_summary.session_health, session_summary.coverage_percent,
    )

    return MatchStageSummary(
        cluster_count=len(cluster_reps),
        confident_count=confident_count,
        uncertain_count=uncertain_count,
        unmatched_count=unmatched_count,
        session_summary=session_summary,
    )
