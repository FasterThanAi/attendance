"""Face detection (SCRFD via insightface).

Phase 1 built single-frame, non-tiled detection (`detect_faces`) for
enrollment selfie video, where a face never needs more than ~640px of input
to detect reliably. Phase 3 (below) adds the tiled-detection path for
classroom video: split a 4K frame into overlapping tiles, detect on each at
native resolution, map back to frame coordinates, and merge with NMS --
without tiling, feeding SCRFD a downscaled 4K frame makes back-row faces
vanish entirely (Section on tiled detection in the roadmap).

Model: SCRFD det_10g (insightface `buffalo_l` model pack), run via
onnxruntime. Loaded once per process and cached -- constructing an ONNX
session per frame/call is the single most common way to make this stage
10-20x slower than it needs to be (see detect_all_frames's parallelism note).
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.params import PipelineParams


@dataclass(frozen=True)
class Detection:
    """One detected face. Coordinates are in the ORIGINAL image's pixel
    space (not tile-local -- that mapping only matters once Phase 3 adds
    tiling, and is handled there, not here).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    # 5 landmarks in (x, y) order: left eye, right eye, nose, left mouth
    # corner, right mouth corner -- the standard insightface/ArcFace ordering.
    landmarks: tuple[tuple[float, float], ...]

    @property
    def face_width_px(self) -> float:
        return self.x2 - self.x1

    @property
    def face_height_px(self) -> float:
        return self.y2 - self.y1


class DetectorModel:
    """Thin wrapper around an insightface SCRFD model_zoo instance.

    Kept as a real class (rather than passing the raw insightface object
    around) so callers -- and tests -- depend on this narrow interface
    (`.raw.detect(...)`) instead of insightface's exact internals, which
    have changed across versions.
    """

    def __init__(self, raw_model) -> None:
        self.raw = raw_model


_detector_singleton: DetectorModel | None = None


def load_detector(model_dir: Path | None = None, ctx_id: int = -1, det_size: tuple[int, int] = (640, 640)) -> DetectorModel:
    """Load SCRFD det_10g once per process.

    `ctx_id=-1` means CPU (insightface convention; >=0 selects a GPU device
    index). This is deliberately a module-level singleton -- see the
    docstring above -- so importing this module and calling load_detector()
    repeatedly (e.g. once per frame in a naive implementation) is cheap after
    the first call.
    """
    global _detector_singleton
    if _detector_singleton is not None:
        return _detector_singleton

    from insightface.model_zoo import model_zoo

    if model_dir is None or not (model_dir / "det_10g.onnx").exists():
        default_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
        if (default_dir / "det_10g.onnx").exists():
            model_dir = default_dir
        elif model_dir is not None:
            raise FileNotFoundError(f"det_10g.onnx not found in {model_dir} or {default_dir}")
        else:
            raise FileNotFoundError(f"det_10g.onnx not found in {default_dir}")

    raw_model = model_zoo.get_model(str(model_dir / "det_10g.onnx"))
    raw_model.prepare(ctx_id=ctx_id, input_size=det_size)

    _detector_singleton = DetectorModel(raw_model)
    return _detector_singleton


def detect_faces(image_bgr: np.ndarray, model: DetectorModel, score_min: float = 0.0) -> list[Detection]:
    """Run SCRFD on a single, already-loaded BGR image (uint8, HxWx3).

    Non-tiled: fine for enrollment selfie video frames and for any frame
    whose longer side is under ~2000px. Phase 3 adds a tiled variant for full
    4K classroom frames, per the roadmap's Section on tiled detection.

    `score_min` is a cheap pre-filter; the real accept/reject decision using
    params.detector_score_min (plus size/blur/pose) happens in the quality
    gate (quality.py), not here -- this stage's job is only detection.
    """
    bboxes, kpss = model.raw.detect(image_bgr, metric="default")

    detections: list[Detection] = []
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2, score = bboxes[i]
        if score < score_min:
            continue
        landmarks = tuple((float(kpss[i, j, 0]), float(kpss[i, j, 1])) for j in range(5))
        detections.append(
            Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                score=float(score), landmarks=landmarks,
            )
        )
    return detections


def largest_face(detections: list[Detection]) -> Detection | None:
    """Enrollment videos have exactly one subject. If a frame has more than
    one face in it (someone walked behind the student, a poster with a face
    on it, etc.), the ENROLLMENT job discards the whole frame rather than
    guessing which face is the enrolling student -- see Phase 1's prompt:
    "if a frame contains two faces, discard that frame entirely and log it."
    This helper just finds the largest candidate; the caller decides whether
    multiple detections means "discard the frame."
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d.face_width_px * d.face_height_px)


# --------------------------------------------------------------------------
# Phase 3: tiled detection for classroom (4K) video
# --------------------------------------------------------------------------


def _axis_tile_origins(dim: int, tile_size: int, stride: int) -> list[int]:
    """Origins along one axis (x or y) for tiles of `tile_size`, `stride`
    apart, guaranteed to cover [0, dim) with no gap -- the last origin is
    pinned to `dim - tile_size` even if that means slightly more overlap
    with its neighbour than `stride` alone would give, rather than leaving a
    strip of the frame untiled.
    """
    if dim <= tile_size:
        return [0]

    origins = list(range(0, dim - tile_size + 1, stride))
    last_origin = dim - tile_size
    if origins[-1] != last_origin:
        origins.append(last_origin)
    return origins


def compute_tile_grid(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Returns (x0, y0, x1, y1) bounds for each tile, covering the full
    `width` x `height` frame with `overlap` px of overlap between
    horizontally/vertically adjacent tiles (see Phase 3 prompt: "Tile size
    1280 px, overlap 256 px. Compute the grid so tiles cover the full frame").

    If the frame is smaller than one tile in a given dimension, that whole
    dimension is a single tile (no tiling needed on that axis).
    """
    stride = tile_size - overlap
    x_origins = _axis_tile_origins(width, tile_size, stride)
    y_origins = _axis_tile_origins(height, tile_size, stride)

    tiles = []
    for y0 in y_origins:
        for x0 in x_origins:
            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            tiles.append((x0, y0, x1, y1))
    return tiles


def _iou(a: Detection, b: Detection) -> float:
    x0, y0 = max(a.x1, b.x1), max(a.y1, b.y1)
    x1, y1 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def non_max_suppression(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy NMS: highest-score detection wins, anything overlapping it
    above `iou_threshold` is dropped, repeat. This is what merges the SAME
    face detected in two overlapping tiles (or a tile and the whole-frame
    downscaled pass) into a single detection (Phase 3 prompt, step d).
    """
    remaining = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []

    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if _iou(best, d) < iou_threshold]

    return kept


def detect_faces_tiled(image_bgr: np.ndarray, model: DetectorModel, params: PipelineParams) -> list[Detection]:
    """The full Phase 3 detection strategy for one frame:
      a-d. if the frame is large, tile it, detect on each tile at native
           size, map tile-local detections back to frame coordinates.
      e.   ALSO run detection once on the whole frame downscaled to
           tile_size_px, to catch large front-row faces a tile boundary may
           have cut in half.
      Then merge everything with NMS and apply the score threshold.

    Small frames (enrollment selfies, pre-flight samples) skip tiling
    entirely and this is equivalent to plain detect_faces.
    """
    height, width = image_bgr.shape[:2]
    long_side = max(height, width)

    all_detections: list[Detection] = []

    if long_side > params.tile_trigger_long_side_px:
        for x0, y0, x1, y1 in compute_tile_grid(width, height, params.tile_size_px, params.tile_overlap_px):
            tile = image_bgr[y0:y1, x0:x1]
            for d in detect_faces(tile, model, score_min=0.0):
                all_detections.append(Detection(
                    x1=d.x1 + x0, y1=d.y1 + y0, x2=d.x2 + x0, y2=d.y2 + y0,
                    score=d.score,
                    landmarks=tuple((lx + x0, ly + y0) for lx, ly in d.landmarks),
                ))

        scale = params.tile_size_px / long_side
        small = cv2.resize(image_bgr, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        for d in detect_faces(small, model, score_min=0.0):
            all_detections.append(Detection(
                x1=d.x1 / scale, y1=d.y1 / scale, x2=d.x2 / scale, y2=d.y2 / scale,
                score=d.score,
                landmarks=tuple((lx / scale, ly / scale) for lx, ly in d.landmarks),
            ))
    else:
        all_detections = detect_faces(image_bgr, model, score_min=0.0)

    merged = non_max_suppression(all_detections, params.nms_iou_threshold)
    return [d for d in merged if d.score >= params.detector_score_min]


# --------------------------------------------------------------------------
# Batch detection over a whole video's extracted frames (Phase 3 deliverable
# 2-3): multiprocessing, one detector load per worker process, parquet output
# --------------------------------------------------------------------------

_worker_model_dir: Path | None = None


def _init_detection_worker(model_dir_str: str) -> None:
    """multiprocessing.Pool initializer: runs once when each worker PROCESS
    starts, loading the ONNX session exactly once per process -- not once
    per frame, which is "the single most common way to make this stage
    20x slower" per the Phase 3 prompt. Each worker process gets its own
    fresh `_detector_singleton` (see load_detector), so this really is a
    per-process load, not shared/pickled state.
    """
    global _worker_model_dir
    _worker_model_dir = Path(model_dir_str)
    load_detector(_worker_model_dir)


def _detect_one_frame(task: tuple[int, str, float, PipelineParams]) -> list[dict]:
    frame_index, frame_path_str, fps, params = task
    model = load_detector(_worker_model_dir)  # cached from the initializer; cheap

    image = cv2.imread(frame_path_str)
    if image is None:
        return []

    rows = []
    for det_index, d in enumerate(detect_faces_tiled(image, model, params)):
        row = {
            "frame_index": frame_index,
            "frame_timestamp_s": frame_index / fps if fps > 0 else 0.0,
            "det_id": f"{frame_index}_{det_index}",
            "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
            "score": d.score,
            "face_width_px": d.face_width_px,
        }
        for i, (lx, ly) in enumerate(d.landmarks):
            row[f"lmk_x{i + 1}"] = lx
            row[f"lmk_y{i + 1}"] = ly
        rows.append(row)
    return rows


@dataclass
class DetectionSummary:
    total_frames: int
    total_detections: int
    detections_per_frame_mean: float
    detections_per_frame_min: int
    detections_per_frame_max: int
    face_width_histogram: dict[str, int]  # "0-10", "10-20", ... -> count
    parquet_path: Path


def detect_all_frames(
    frame_dir: Path,
    out_dir: Path,
    params: PipelineParams,
    model_dir: Path,
    fps: float,
) -> DetectionSummary:
    """Runs tiled detection over every frame_*.jpg in `frame_dir`, using a
    multiprocessing Pool sized min(cpu_count, 4) (Phase 3 prompt), and writes
    one row per detection to `out_dir`/detections.parquet with columns:
    frame_index, frame_timestamp_s, det_id, x1, y1, x2, y2, score,
    lmk_x1..lmk_x5, lmk_y1..lmk_y5, face_width_px.

    No crops are produced here -- that's Phase 4's job (align.py already
    exists from Phase 1's enrollment use; wiring it into the classroom
    pipeline is Phase 4's, not this function's).
    """
    import pandas as pd

    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    tasks = [(i, str(p), fps, params) for i, p in enumerate(frame_paths)]

    n_workers = min(multiprocessing.cpu_count(), 4)
    with multiprocessing.Pool(
        processes=n_workers, initializer=_init_detection_worker, initargs=(str(model_dir),)
    ) as pool:
        per_frame_rows = pool.map(_detect_one_frame, tasks)

    all_rows = [row for frame_rows in per_frame_rows for row in frame_rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "detections.parquet"
    pd.DataFrame(all_rows).to_parquet(parquet_path)

    counts = [len(rows) for rows in per_frame_rows]
    widths = [row["face_width_px"] for row in all_rows]

    histogram: dict[str, int] = {}
    for w in widths:
        bucket_start = int(w // 10) * 10
        key = f"{bucket_start}-{bucket_start + 10}"
        histogram[key] = histogram.get(key, 0) + 1

    return DetectionSummary(
        total_frames=len(frame_paths),
        total_detections=len(all_rows),
        detections_per_frame_mean=(sum(counts) / len(counts)) if counts else 0.0,
        detections_per_frame_min=min(counts) if counts else 0,
        detections_per_frame_max=max(counts) if counts else 0,
        face_width_histogram=dict(sorted(histogram.items(), key=lambda kv: int(kv[0].split("-")[0]))),
        parquet_path=parquet_path,
    )
