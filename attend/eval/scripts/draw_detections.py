#!/usr/bin/env python3
"""Phase 3 deliverable 4: is tiled detection actually finding the back row?

Reads a job's detections.parquet (written by pipeline.detect.detect_all_frames)
and the frames it was computed from, draws each detection's box plus its
face_width_px and score onto a sample of frames, and writes them to an output
directory so you can eyeball whether tiling is working -- boxes should cover
front AND back rows, with no obvious duplicate boxes on the same face (that
would mean nms_iou_threshold needs tuning) and no obvious gaps at tile
boundaries (that would mean tile_overlap_px is too small).

Standalone on purpose, same as gallery_sanity.py: only pandas/numpy/opencv,
no dependency on the worker package's config/db modules, so it runs against
any job_dir you point it at without needing DATABASE_URL or a live worker.

Usage:
    python eval/scripts/draw_detections.py \\
        --job-dir /data/jobs/<job_id> \\
        --out-dir /data/jobs/<job_id>/debug_detections \\
        --sample-every 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BOX_COLOR = (0, 220, 0)  # BGR: green
TEXT_COLOR = (0, 220, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_frame(frame: np.ndarray, frame_detections: pd.DataFrame) -> np.ndarray:
    annotated = frame.copy()
    for _, row in frame_detections.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"w={row['face_width_px']:.0f}px score={row['score']:.2f}"
        label_y = max(y1 - 8, 12)
        cv2.putText(annotated, label, (x1, label_y), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    return annotated


def run(job_dir: Path, out_dir: Path, sample_every: int) -> int:
    parquet_path = job_dir / "detect" / "detections.parquet"
    frame_dir = job_dir / "extract"

    if not parquet_path.exists():
        print(f"No detections.parquet at {parquet_path} -- has the detect stage run for this job?")
        return 1
    if not frame_dir.exists():
        print(f"No frame directory at {frame_dir} -- has the extract stage run for this job?")
        return 1

    detections = pd.read_parquet(parquet_path)
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        print(f"No frame_*.jpg files found in {frame_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_indices = list(range(0, len(frame_paths), sample_every))
    print(f"{len(frame_paths)} frames total, {len(detections)} detections total, "
          f"drawing {len(sampled_indices)} sampled frame(s) every {sample_every}.")

    per_frame_counts = detections.groupby("frame_index").size()

    for frame_index in sampled_indices:
        frame_path = frame_paths[frame_index]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"  frame_index={frame_index}: could not read {frame_path}, skipping")
            continue

        frame_detections = detections[detections["frame_index"] == frame_index]
        annotated = draw_frame(frame, frame_detections)

        out_path = out_dir / f"annotated_{frame_index:06d}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"  frame_index={frame_index}: {len(frame_detections)} detection(s) -> {out_path}")

    print("\nPer-frame detection count summary (all frames, not just sampled):")
    if len(per_frame_counts):
        print(f"  mean={per_frame_counts.mean():.2f}  min={per_frame_counts.min()}  "
              f"max={per_frame_counts.max()}  frames_with_zero={ (len(frame_paths) - len(per_frame_counts)) }")
    else:
        print("  (no detections at all -- check the model loaded correctly and "
              "detector_score_min isn't filtering everything out)")

    print(f"\nWrote {len(sampled_indices)} annotated frame(s) to {out_dir}. "
          "Look for: back-row faces boxed (tiling working), no duplicate boxes "
          "on one face (NMS working), no gaps at tile seams (overlap sufficient).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path, help="Job directory, e.g. /data/jobs/42")
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to <job-dir>/debug_detections")
    parser.add_argument("--sample-every", type=int, default=30, help="Draw every Nth frame (default 30)")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.job_dir / "debug_detections")
    return run(args.job_dir, out_dir, args.sample_every)


if __name__ == "__main__":
    sys.exit(main())
