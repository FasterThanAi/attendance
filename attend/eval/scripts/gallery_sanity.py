#!/usr/bin/env python3
"""Phase 1 deliverable 6: is your enrollment data usable?

Computes within-student and between-student cosine similarity distributions
across every enrolled student's gallery embeddings, prints them as a text
histogram, and flags any student whose within-student mean similarity is
below 0.55 for re-enrollment.

If the two distributions overlap substantially, matching will not work no
matter how well-tuned the rest of the pipeline is -- this script is meant to
be the first thing you run after enrolling a test class, before building
anything else on top of it (Phase 1's own framing).

Usage (run on the host, or `docker-compose exec worker python
eval/scripts/gallery_sanity.py`):

    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \\
        python eval/scripts/gallery_sanity.py

Standalone on purpose: a plain psycopg2 connection + numpy, no dependency on
either the api or worker package, so it always runs regardless of which
service's virtualenv you happen to have active.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import psycopg2

EMBEDDING_DIM = 512
WITHIN_STUDENT_FLAG_THRESHOLD = 0.55


def fetch_embeddings(database_url: str) -> dict[int, list[np.ndarray]]:
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT student_id, vector FROM gallery_embedding ORDER BY student_id")
            rows = cur.fetchall()
    finally:
        conn.close()

    by_student: dict[int, list[np.ndarray]] = defaultdict(list)
    for student_id, raw_vector in rows:
        vec = np.frombuffer(raw_vector, dtype=np.float32)
        if vec.shape[0] != EMBEDDING_DIM:
            print(f"WARNING: student_id={student_id} has a vector of unexpected "
                  f"dimension {vec.shape[0]} (expected {EMBEDDING_DIM}), skipping it.")
            continue
        by_student[student_id].append(vec)
    return by_student


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already L2-normalised by embed.py, but don't assume that
    # blindly here -- a script meant to catch data problems shouldn't itself
    # silently produce wrong numbers if that assumption is ever violated.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def within_student_similarities(by_student: dict[int, list[np.ndarray]]) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    for student_id, vectors in by_student.items():
        sims = [
            cosine_similarity(vectors[i], vectors[j])
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        ]
        result[student_id] = sims
    return result


def between_student_similarities(by_student: dict[int, list[np.ndarray]], sample_pairs: int = 2000) -> list[float]:
    """Mean vector per student, then all pairwise cross-student similarities.
    Capped at `sample_pairs` random pairs so this stays fast with a large
    gallery -- exact enumeration isn't needed to characterise the distribution.
    """
    student_ids = list(by_student.keys())
    means = {sid: np.mean(by_student[sid], axis=0) for sid in student_ids}

    rng = np.random.default_rng(seed=0)
    sims: list[float] = []
    n = len(student_ids)
    if n < 2:
        return sims

    total_possible_pairs = n * (n - 1) // 2
    num_pairs = min(sample_pairs, total_possible_pairs)
    seen = set()
    while len(sims) < num_pairs:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        sims.append(cosine_similarity(means[student_ids[i]], means[student_ids[j]]))
    return sims


def print_histogram(label: str, values: list[float], bins: int = 20) -> None:
    print(f"\n{label} (n={len(values)})")
    if not values:
        print("  (no data)")
        return
    arr = np.array(values)
    print(f"  mean={arr.mean():.3f}  p5={np.percentile(arr, 5):.3f}  "
          f"p50={np.percentile(arr, 50):.3f}  p95={np.percentile(arr, 95):.3f}")
    counts, edges = np.histogram(arr, bins=bins, range=(-1.0, 1.0))
    max_count = max(counts.max(), 1)
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(40 * count / max_count)
        print(f"  [{lo:5.2f}, {hi:5.2f}) {bar} ({count})")


def main() -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("Set DATABASE_URL, e.g.:\n"
              "  DATABASE_URL=postgresql://attend:attend@localhost:5432/attend python eval/scripts/gallery_sanity.py")
        return 1
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    by_student = fetch_embeddings(database_url)
    if not by_student:
        print("No gallery_embedding rows found. Enroll at least one student first.")
        return 1

    within = within_student_similarities(by_student)
    all_within = [s for sims in within.values() for s in sims]
    between = between_student_similarities(by_student)

    print_histogram("WITHIN-STUDENT similarity", all_within)
    print_histogram("BETWEEN-STUDENT similarity", between)

    if all_within and between:
        overlap_lo = max(min(all_within), min(between))
        overlap_hi = min(max(all_within), max(between))
        if overlap_hi > overlap_lo:
            print(f"\nOVERLAP RANGE: [{overlap_lo:.3f}, {overlap_hi:.3f}] -- "
                  "some pairs in this range could go either way. The narrower this is, the better.")
        else:
            print("\nNo overlap between the two distributions -- good sign.")

    print("\nStudents flagged for re-enrollment (within-student mean similarity "
          f"below {WITHIN_STUDENT_FLAG_THRESHOLD}):")
    flagged = False
    for student_id, sims in within.items():
        if not sims:
            print(f"  student_id={student_id}: only 1 embedding, cannot compute within-student similarity "
                  "-- needs more enrollment photos regardless.")
            flagged = True
            continue
        mean_sim = float(np.mean(sims))
        if mean_sim < WITHIN_STUDENT_FLAG_THRESHOLD:
            print(f"  student_id={student_id}: mean={mean_sim:.3f}")
            flagged = True
    if not flagged:
        print("  (none)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
