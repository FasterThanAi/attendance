"""Quality metrics on a face crop: blur, brightness, yaw/pitch estimate.

Phase 1 subset: enrollment needs yaw (to bucket frames into left/frontal/
right poses) and a basic accept/reject gate before spending an embedding
call on a bad crop. Phase 4 builds the full composite score_detections
pipeline stage (a documented weighted combination of size/blur/frontality,
run over a whole detections.parquet, producing quality.parquet with
reject reasons) -- it imports and reuses the exact functions below rather
than redefining the geometry, so a pose estimate computed during enrollment
and one computed during classroom-video quality gating always agree.

ASSUMPTIONS I MADE (per the global brief's "when unsure" rule):
  The roadmap describes the yaw/pitch geometry only in words ("yaw from the
  horizontal offset of the nose tip relative to the midpoint of the two
  eyes, normalised by inter-ocular distance; pitch from the vertical offset
  of the nose relative to the eye-mouth midpoint") without giving the
  ratio-to-degrees scale factor. YAW_SCALE_DEG / PITCH_SCALE_DEG below are a
  reasonable first guess (a nose offset equal to the full inter-ocular
  distance is treated as ~65 degrees of rotation), not a calibrated pose
  estimator. Phase 7 (threshold calibration against real labelled sessions)
  is exactly where you'd tighten this if stratified accuracy by yaw looks
  wrong -- flag it if the bucketing in gallery_sanity.py output looks off.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

YAW_SCALE_DEG = 65.0
PITCH_SCALE_DEG = 65.0


@dataclass(frozen=True)
class PoseEstimate:
    yaw_deg: float
    pitch_deg: float


def estimate_pose(landmarks: tuple[tuple[float, float], ...]) -> PoseEstimate:
    """landmarks order: left eye, right eye, nose, left mouth, right mouth
    (the same order detect.py returns and align.py expects).
    """
    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])
    nose = np.array(landmarks[2])
    left_mouth = np.array(landmarks[3])
    right_mouth = np.array(landmarks[4])

    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0
    inter_ocular = float(np.linalg.norm(right_eye - left_eye))

    if inter_ocular < 1e-6:
        # Degenerate landmarks (shouldn't happen with a real detector, but a
        # crafted test input might hit this) -- treat as maximally rotated
        # rather than dividing by zero, so the quality gate rejects it.
        return PoseEstimate(yaw_deg=90.0, pitch_deg=90.0)

    yaw_ratio = (nose[0] - eye_mid[0]) / inter_ocular
    eye_mouth_mid = (eye_mid + mouth_mid) / 2.0
    pitch_ratio = (nose[1] - eye_mouth_mid[1]) / inter_ocular

    yaw_deg = float(np.clip(yaw_ratio * YAW_SCALE_DEG, -90.0, 90.0))
    pitch_deg = float(np.clip(pitch_ratio * PITCH_SCALE_DEG, -90.0, 90.0))
    return PoseEstimate(yaw_deg=yaw_deg, pitch_deg=pitch_deg)


def blur_score(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian, computed on a crop resized to a FIXED
    128x128 first. Resizing first is what makes this comparable across
    crops of different original face sizes -- without it, a large face
    naturally has more high-frequency detail and looks "sharper" than a
    small face even at identical true sharpness (see Phase 4's prompt).
    """
    resized = cv2.resize(crop_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def simple_quality_score(face_width_px: float, blur: float, yaw_deg: float, max_abs_yaw_deg: float) -> float:
    """A placeholder composite score for Phase 1's enrollment "select the
    best N crops per pose bucket" step. This is deliberately simple (an
    unweighted mean of three min-max-normalised sub-scores) and NOT the
    tunable, documented-weights composite score Phase 4's score_detections
    builds for the full classroom-video quality gate (that one gets its
    weights as PipelineParams fields per the Phase 4 prompt). This function
    exists so enrollment has *some* principled way to rank crops now,
    without inventing Phase 4's real design early.

    Normalisation references (heuristic, not calibrated):
      - face_width_px: 150px treated as "fully good", scales linearly below that
      - blur: Laplacian variance of 300 treated as "fully sharp"
      - frontality: 1.0 at yaw=0, 0.0 at yaw=+-max_abs_yaw_deg
    """
    size_score = min(face_width_px / 150.0, 1.0)
    blur_score_norm = min(blur / 300.0, 1.0)
    frontality = max(0.0, 1.0 - abs(yaw_deg) / max_abs_yaw_deg)
    return (size_score + blur_score_norm + frontality) / 3.0
