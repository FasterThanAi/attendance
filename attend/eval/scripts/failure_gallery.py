#!/usr/bin/env python3
"""Phase 7 deliverable 4: for every false negative and false positive,
produce a page showing the student's enrollment photos, any crops that were
detected for them, what cluster they landed in, and what the system
decided -- "this is how you diagnose causes rather than guessing" (the
roadmap's own framing).

For each FN/FP student in a session, writes one JPEG "page" to
{out-dir}/{session_id}/{FN,FP}_{roll_number}.jpg containing:
  - up to 4 enrollment photos (gallery_photo rows, best quality_score first)
  - up to 8 crops from whichever cluster the Hungarian assignment linked
    them to, if any (UNCERTAIN and even UNMATCHED assignments still carry a
    student_id in pipeline.match's output -- only a genuinely unassigned
    cluster count leaves this empty; see pipeline/match.py's ClusterMatchRow)
  - a text header: roll_number, name, actual vs predicted, decision band,
    similarity/runner_up_similarity if a cluster was linked at all

A student with ZERO detected crops at all (the "unrecoverable failure" from
eval_lib.pipeline_yield_for_session) gets a page that says exactly that --
itself a diagnosis, not a script failure.

Standalone-ish, same as evaluate.py/sweep_threshold.py: services/worker on
PYTHONPATH for pipeline.match/pipeline.params, psycopg2 for the gallery
photo/session lookups, opencv/pandas/numpy for the page itself.

Usage:
    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \\
        python eval/scripts/failure_gallery.py --job-data-dir /data/jobs \\
        --datasets-dir eval/datasets --out-dir eval/reports/failures
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import eval_lib as el
from pipeline.match import UNMATCHED, ClusterMatchRow, match_clusters
from pipeline.params import PipelineParams

THUMB_SIZE = 160
LABEL_HEIGHT = 22
PAD = 6
GRID_COLUMNS = 4
MAX_ENROLLMENT_PHOTOS = 4
MAX_DETECTED_CROPS = 8
HEADER_HEIGHT = 130


def _make_tile(image: np.ndarray | None, label: str) -> np.ndarray:
    tile = np.full((THUMB_SIZE + LABEL_HEIGHT, THUMB_SIZE, 3), 255, dtype=np.uint8)
    if image is not None:
        resized = cv2.resize(image, (THUMB_SIZE, THUMB_SIZE))
        tile[:THUMB_SIZE] = resized
    else:
        cv2.putText(tile, "missing", (10, THUMB_SIZE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(tile, label[:20], (2, THUMB_SIZE + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def _build_row(images: list[np.ndarray | None], labels: list[str], row_title: str) -> np.ndarray:
    if not images:
        blank = np.full((THUMB_SIZE + LABEL_HEIGHT + 24, GRID_COLUMNS * (THUMB_SIZE + PAD) + PAD, 3), 255, dtype=np.uint8)
        cv2.putText(blank, f"{row_title}: none found", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1, cv2.LINE_AA)
        return blank

    tiles = [_make_tile(img, lbl) for img, lbl in zip(images, labels)]
    n_cols = min(GRID_COLUMNS, len(tiles))
    n_rows = -(-len(tiles) // n_cols)
    tile_h, tile_w = tiles[0].shape[:2]

    grid = np.full((n_rows * (tile_h + PAD) + PAD, n_cols * (tile_w + PAD) + PAD, 3), 230, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, n_cols)
        y0, x0 = PAD + r * (tile_h + PAD), PAD + c * (tile_w + PAD)
        grid[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    title_bar = np.full((24, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(title_bar, row_title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([title_bar, grid])


def load_crops_for_cluster(job_dir: Path, cluster_id: int, max_crops: int = MAX_DETECTED_CROPS) -> list[np.ndarray]:
    clusters_path = job_dir / "cluster" / "clusters.parquet"
    aligned_index_path = job_dir / "align" / "aligned_index.parquet"
    aligned_path = job_dir / "align" / "aligned.npy"
    if not (clusters_path.exists() and aligned_index_path.exists() and aligned_path.exists()):
        return []

    clusters_df = pd.read_parquet(clusters_path)
    aligned_index_df = pd.read_parquet(aligned_index_path)
    aligned = np.load(aligned_path, mmap_mode="r")
    det_id_to_row = dict(zip(aligned_index_df["det_id"], aligned_index_df["row_index"]))

    member_det_ids = clusters_df.loc[clusters_df["cluster_id"] == cluster_id, "det_id"].tolist()[:max_crops]
    crops = []
    for det_id in member_det_ids:
        row_index = det_id_to_row.get(det_id)
        if row_index is not None:
            crops.append(np.asarray(aligned[int(row_index)]))
    return crops


def load_enrollment_photos(database_url: str, student_id: int, max_photos: int = MAX_ENROLLMENT_PHOTOS) -> list[np.ndarray]:
    uris = el.fetch_gallery_photo_uris(database_url, student_id)[:max_photos]
    photos = []
    for uri in uris:
        image = cv2.imread(uri)
        photos.append(image)  # may be None if the file isn't reachable from here -- _make_tile handles that
    return photos


def build_failure_page(
    failure_type: str,  # "FN" or "FP"
    roll_number: str,
    full_name: str,
    actually_present: bool,
    match_row: ClusterMatchRow | None,
    enrollment_photos: list[np.ndarray],
    detected_crops: list[np.ndarray],
) -> np.ndarray:
    body_rows = [
        _build_row(enrollment_photos, [f"photo {i}" for i in range(len(enrollment_photos))], "Enrollment photos"),
        _build_row(detected_crops, [f"crop {i}" for i in range(len(detected_crops))], "Detected crops (linked cluster)"),
    ]
    max_width = max(r.shape[1] for r in body_rows)
    padded_rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, max_width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for r in body_rows
    ]
    body = np.vstack(padded_rows)

    header = np.full((HEADER_HEIGHT, body.shape[1], 3), 245, dtype=np.uint8)
    kind_color = (0, 0, 220) if failure_type == "FN" else (0, 140, 255)
    cv2.putText(header, f"{failure_type}: {roll_number} -- {full_name}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, kind_color, 2, cv2.LINE_AA)
    cv2.putText(header, f"actually_present={actually_present}", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    if match_row is None:
        cv2.putText(header, "No cluster was ever assigned to this student -- zero linkable evidence.", (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        sim_str = f"{match_row.similarity:.3f}" if match_row.similarity is not None else "n/a"
        ru_str = f"{match_row.runner_up_similarity:.3f}" if match_row.runner_up_similarity is not None else "n/a"
        cv2.putText(
            header,
            f"linked cluster_id={match_row.cluster_id}  decision={match_row.decision}  "
            f"similarity={sim_str}  runner_up={ru_str}",
            (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )
    cv2.putText(
        header, "system said PRESENT only for CONFIDENT decisions (see eval_lib.py's scoring rule)",
        (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA,
    )

    return np.vstack([header, body])


def run(database_url: str, datasets_dir: Path, job_data_dir: Path, out_dir: Path, params: PipelineParams,
        session_ids: list[str] | None) -> int:
    all_session_dirs = sorted(p.name for p in datasets_dir.iterdir() if p.is_dir()) if datasets_dir.exists() else []
    target_sessions = session_ids if session_ids else all_session_dirs

    total_pages = 0
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

        match_result = match_clusters(cluster_reps, ctx.gallery, params)
        match_by_student_id = {m.student_id: m for m in match_result.matches if m.student_id is not None}
        predicted = el.predicted_present_from_matches(match_result.matches, set(ctx.gallery.keys()))

        roll_to_student_id = {roll: sid for sid, roll in ctx.roll_number_by_student_id.items()}

        session_out_dir = out_dir / session_id
        session_pages = 0
        for _, truth_row in truth_df.iterrows():
            roll_number = truth_row["roll_number"]
            student_id = roll_to_student_id.get(roll_number)
            if student_id is None:
                continue  # not in this course's enrollment/gallery -- evaluate.py already warns about this

            actually_present = bool(truth_row["actually_present"])
            predicted_present = predicted.get(student_id, False)

            if actually_present and not predicted_present:
                failure_type = "FN"
            elif not actually_present and predicted_present:
                failure_type = "FP"
            else:
                continue  # correct decision -- not a failure, no page needed

            match_row = match_by_student_id.get(student_id)
            detected_crops = load_crops_for_cluster(ctx.job_dir, match_row.cluster_id) if match_row else []
            enrollment_photos = load_enrollment_photos(database_url, student_id)
            full_name = ctx.full_name_by_student_id.get(student_id, "(unknown name)")

            page = build_failure_page(
                failure_type, roll_number, full_name, actually_present, match_row,
                enrollment_photos, detected_crops,
            )
            session_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = session_out_dir / f"{failure_type}_{roll_number}.jpg"
            cv2.imwrite(str(out_path), page)
            session_pages += 1

        print(f"session {session_id}: wrote {session_pages} failure page(s) to {session_out_dir}.")
        total_pages += session_pages

    print(f"\nWrote {total_pages} failure page(s) total across {len(target_sessions)} session(s).")
    return 0 if total_pages >= 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", type=Path, default=Path("eval/datasets"))
    parser.add_argument("--job-data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/reports/failures"))
    parser.add_argument("--session-ids", default=None)
    parser.add_argument("--match-threshold", type=float, default=None)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("Set DATABASE_URL, e.g.:\n"
              "  DATABASE_URL=postgresql://attend:attend@localhost:5432/attend "
              "python eval/scripts/failure_gallery.py --job-data-dir /data/jobs")
        return 1
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    params = PipelineParams()
    if args.match_threshold is not None:
        from dataclasses import replace
        params = replace(params, match_threshold=args.match_threshold)

    session_ids = args.session_ids.split(",") if args.session_ids else None
    return run(database_url, args.datasets_dir, args.job_data_dir, args.out_dir, params, session_ids)


if __name__ == "__main__":
    sys.exit(main())
