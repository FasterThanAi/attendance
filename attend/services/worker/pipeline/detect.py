"""Face detection (SCRFD via insightface), Phase 1 subset.

Phase 1 only needs single-frame, non-tiled detection: enrollment videos are a
student's own phone selfie, not a 4K classroom pan, so a face never needs
more than ~640px of input to detect reliably. Phase 3 extends this module
with the tiled-detection path (splitting a 4K frame into overlapping tiles,
detecting on each at native resolution, merging with NMS) for classroom
video, where back-row faces are small enough that downscaling loses them.
`detect_faces` below is written so Phase 3 can add a `detect_faces_tiled`
alongside it without touching this function.

Model: SCRFD det_10g (insightface `buffalo_l` model pack), run via
onnxruntime. Loaded once per process and cached -- constructing an ONNX
session per frame/call is the single most common way to make this stage 10-
20x slower than it needs to be (see Phase 3's parallelism note).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def load_detector(model_dir: Path, ctx_id: int = -1, det_size: tuple[int, int] = (640, 640)) -> DetectorModel:
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
