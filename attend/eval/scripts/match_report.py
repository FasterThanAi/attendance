#!/usr/bin/env python3
"""Phase 6 deliverable 6: is matching actually deciding correctly?

Reads a single processing_job's cluster_match/detected_cluster rows (plus
the student they resolved to, if any) directly from Postgres and reports:

  - counts per band (confident / uncertain / unmatched)
  - a similarity distribution PER BAND -- the confident band should sit
    clearly above match_threshold, unmatched clearly below; heavy overlap
    between bands near the threshold is a sign match_threshold/uncertain_band
    need recalibrating (Phase 7's job, this script is how you'd notice)
  - the ten lowest-margin CONFIDENT matches -- these are the confident
    decisions closest to being reclassified as uncertain if the margin gate
    were tightened even slightly; worth a manual look before trusting a
    threshold change

Standalone on purpose, same as gallery_sanity.py: a plain psycopg2
connection + numpy, no dependency on either the api or worker package, so
it always runs regardless of which service's virtualenv you happen to have
active. Match data lives in Postgres (not a job_dir file, unlike every
other eval/scripts/*.py report) because the match stage itself is the one
pipeline stage that writes its output to the DB, not the filesystem -- see
pipeline/match.py's module docstring.

Usage:
    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \\
        python eval/scripts/match_report.py --processing-job-id 42
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import psycopg2


@dataclass
class MatchRow:
    cluster_id: int
    best_crop_uri: str
    student_id: int | None
    full_name: str | None
    roll_number: str | None
    similarity: float | None
    runner_up_similarity: float | None
    decision: str


def fetch_matches(database_url: str, processing_job_id: int) -> list[MatchRow]:
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT dc.id, dc.best_crop_uri, cm.student_id, s.full_name, s.roll_number,
                       cm.similarity, cm.runner_up_similarity, cm.decision
                FROM detected_cluster dc
                JOIN cluster_match cm ON cm.cluster_id = dc.id
                LEFT JOIN student s ON s.id = cm.student_id
                WHERE dc.processing_job_id = %s
                ORDER BY dc.id
                """,
                (processing_job_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        MatchRow(
            cluster_id=r[0], best_crop_uri=r[1], student_id=r[2], full_name=r[3], roll_number=r[4],
            similarity=r[5], runner_up_similarity=r[6], decision=r[7],
        )
        for r in rows
    ]


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


def _margin(row: MatchRow) -> float:
    if row.similarity is None:
        return float("-inf")  # sorts to the front -- shouldn't happen for a CONFIDENT row, flag it loudly if it does
    if row.runner_up_similarity is None:
        return row.similarity
    return row.similarity - row.runner_up_similarity


def run(database_url: str, processing_job_id: int) -> int:
    rows = fetch_matches(database_url, processing_job_id)
    if not rows:
        print(f"No cluster_match rows found for processing_job_id={processing_job_id}. "
              "Has the match stage run for this job?")
        return 1

    by_decision: dict[str, list[MatchRow]] = {"confident": [], "uncertain": [], "unmatched": []}
    for row in rows:
        by_decision.setdefault(row.decision, []).append(row)

    print(f"processing_job_id={processing_job_id}: {len(rows)} cluster(s) total.")
    for decision in ("confident", "uncertain", "unmatched"):
        print(f"  {decision}: {len(by_decision.get(decision, []))}")

    for decision in ("confident", "uncertain", "unmatched"):
        sims = [r.similarity for r in by_decision.get(decision, []) if r.similarity is not None]
        print_histogram(f"{decision.upper()} similarity distribution", sims)

    confident_rows = by_decision.get("confident", [])
    print(f"\nTen lowest-margin CONFIDENT matches (margin = similarity - runner_up_similarity, n={len(confident_rows)}):")
    if not confident_rows:
        print("  (none)")
    else:
        lowest_margin = sorted(confident_rows, key=_margin)[:10]
        for row in lowest_margin:
            margin = _margin(row)
            runner_up_str = f"{row.runner_up_similarity:.3f}" if row.runner_up_similarity is not None else "n/a"
            print(
                f"  cluster_id={row.cluster_id:4d}  student={row.full_name!r} ({row.roll_number})  "
                f"similarity={row.similarity:.3f}  runner_up={runner_up_str}  margin={margin:.3f}  "
                f"crop={row.best_crop_uri}"
            )

    unmatched_rows = by_decision.get("unmatched", [])
    if unmatched_rows:
        print(f"\n{len(unmatched_rows)} unrecognised cluster(s) -- best crops for a manual look:")
        for row in unmatched_rows:
            print(f"  cluster_id={row.cluster_id:4d}  crop={row.best_crop_uri}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processing-job-id", required=True, type=int)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("Set DATABASE_URL, e.g.:\n"
              "  DATABASE_URL=postgresql://attend:attend@localhost:5432/attend "
              "python eval/scripts/match_report.py --processing-job-id 42")
        return 1
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    return run(database_url, args.processing_job_id)


if __name__ == "__main__":
    sys.exit(main())
