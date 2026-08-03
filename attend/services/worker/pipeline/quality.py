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

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pipeline.params import PipelineParams

logger = logging.getLogger("attend.worker.quality")

YAW_SCALE_DEG = 65.0
PITCH_SCALE_DEG = 65.0

# Normalisation references for the composite quality score below (and for
# Phase 1's simpler simple_quality_score, which now reuses these same two
# constants rather than its own hardcoded copies, so the two scoring
# functions can never silently drift on what "fully good" size/blur means).
FACE_WIDTH_NORM_PX = 150.0  # face_width_px at or above this is "fully good"
BLUR_NORM = 300.0  # Laplacian variance at or above this is "fully sharp"


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

    Normalisation references (heuristic, not calibrated -- see
    FACE_WIDTH_NORM_PX/BLUR_NORM above, shared with Phase 4's
    composite_quality_score):
      - face_width_px: FACE_WIDTH_NORM_PX treated as "fully good", scales linearly below that
      - blur: Laplacian variance of BLUR_NORM treated as "fully sharp"
      - frontality: 1.0 at yaw=0, 0.0 at yaw=+-max_abs_yaw_deg
    """
    size_score = min(face_width_px / FACE_WIDTH_NORM_PX, 1.0)
    blur_score_norm = min(blur / BLUR_NORM, 1.0)
    frontality = max(0.0, 1.0 - abs(yaw_deg) / max_abs_yaw_deg)
    return (size_score + blur_score_norm + frontality) / 3.0


# --------------------------------------------------------------------------
# Phase 4: composite quality score (tunable weights) + full quality gate
# over a video's detections.parquet
# --------------------------------------------------------------------------


def composite_quality_score(
    face_width_px: float, blur: float, yaw_deg: float, pitch_deg: float, params: PipelineParams
) -> float:
    """The real, tunable composite score the Phase 4 prompt asks for: a
    WEIGHTED combination of normalised face-size, blur, and frontality
    sub-scores, with the three weights as PipelineParams fields
    (quality_weight_size/blur/frontality) so Phase 7's calibration pass can
    tune them without a code change.

    Differs from Phase 1's simple_quality_score in two ways: the weights are
    tunable rather than an unweighted mean, and frontality considers BOTH
    yaw and pitch (enrollment's simpler score only needed yaw for pose
    bucketing; classroom video's accept/reject gate cares about both, so the
    quality score should too). Each angle's frontality is normalised against
    its own accept/reject bound (max_abs_yaw_deg/max_abs_pitch_deg), so a
    face right at the rejection boundary always scores frontality=0,
    regardless of which bound it is.
    """
    size_score = min(face_width_px / FACE_WIDTH_NORM_PX, 1.0)
    blur_score_norm = min(max(blur, 0.0) / BLUR_NORM, 1.0)

    frontality_yaw = max(0.0, 1.0 - abs(yaw_deg) / params.max_abs_yaw_deg) if params.max_abs_yaw_deg > 0 else 0.0
    frontality_pitch = (
        max(0.0, 1.0 - abs(pitch_deg) / params.max_abs_pitch_deg) if params.max_abs_pitch_deg > 0 else 0.0
    )
    frontality = (frontality_yaw + frontality_pitch) / 2.0

    return (
        params.quality_weight_size * size_score
        + params.quality_weight_blur * blur_score_norm
        + params.quality_weight_frontality * frontality
    )


def _reject_reason(
    detector_score: float,
    face_width_px: float,
    blur: float,
    yaw_deg: float,
    pitch_deg: float,
    bright: float,
    crop_is_valid: bool,
    params: PipelineParams,
) -> str | None:
    """The Phase 4 prompt's reject rules, checked in a fixed order (a crop
    can fail more than one; the first failing check in this order is the
    one recorded). None means accepted.
    """
    if not crop_is_valid:
        return "invalid_crop"  # bbox fell (partly) outside the frame -- not one of the roadmap's named rules, but a real failure mode of stored det coordinates that has to go SOMEWHERE, not silently crash score_detections.
    if detector_score < params.detector_score_min:
        return "low_detector_score"
    if face_width_px < params.min_face_px:
        return "too_small"
    if blur < params.blur_laplacian_min:
        return "too_blurred"
    if abs(yaw_deg) > params.max_abs_yaw_deg:
        return "yaw_too_extreme"
    if abs(pitch_deg) > params.max_abs_pitch_deg:
        return "pitch_too_extreme"
    if not (params.brightness_min <= bright <= params.brightness_max):
        return "bad_brightness"
    return None


@dataclass(frozen=True)
class QualityResult:
    quality_df: pd.DataFrame
    accepted_count: int
    rejected_count: int
    reject_reason_counts: dict[str, int]


def score_detections(detections_df: pd.DataFrame, frame_dir: Path, params: PipelineParams) -> QualityResult:
    """Phase 4 deliverable 1: for every row in `detections_df` (one row per
    detected face, from detect_all_frames's detections.parquet), crop it out
    of its source frame, compute blur/brightness/pose/composite quality
    score, and apply the accept/reject gate. EVERY input row is kept in the
    output -- rejected rows too, with a `reject_reason` -- per the Phase 4
    prompt: "Phase 7 needs to analyse what was thrown away."

    Frames are read once per frame_index (grouping first), not once per
    detection row -- a 60s pan at 4fps has ~240 frames but can easily have
    several thousand detections, so re-decoding the same JPEG per detection
    would dominate this stage's runtime for no reason.

    Deliberately does NOT write quality.parquet itself -- see
    run_quality_stage below for that -- so this stays a plain DataFrame-in,
    DataFrame-out transform, testable without touching a real detections
    parquet file on disk (non-negotiable rule #1).
    """
    extra_columns = ["blur", "brightness", "yaw_deg", "pitch_deg", "quality_score", "accepted", "reject_reason"]

    if detections_df.empty:
        # A genuinely faceless video (or a stub/test fixture) -- return an
        # empty result with the FULL expected column set anyway, so callers
        # (align's run_align_stage does quality_df["accepted"]) never
        # KeyError on a 0-row DataFrame that has no columns at all.
        columns = list(detections_df.columns) + [c for c in extra_columns if c not in detections_df.columns]
        return QualityResult(
            quality_df=pd.DataFrame(columns=columns), accepted_count=0, rejected_count=0, reject_reason_counts={}
        )

    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))

    rows: list[dict] = []
    reject_counts: dict[str, int] = {}
    accepted_count = 0

    for frame_index, group in detections_df.groupby("frame_index"):
        frame_index = int(frame_index)
        image = None
        if 0 <= frame_index < len(frame_paths):
            image = cv2.imread(str(frame_paths[frame_index]))

        for _, det_row in group.iterrows():
            x1, y1, x2, y2 = int(det_row["x1"]), int(det_row["y1"]), int(det_row["x2"]), int(det_row["y2"])
            crop = None
            if image is not None:
                h, w = image.shape[:2]
                cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if cx2 > cx1 and cy2 > cy1:
                    crop = image[cy1:cy2, cx1:cx2]

            if crop is None or crop.size == 0:
                blur, bright, yaw_deg, pitch_deg, quality_score = 0.0, 0.0, 90.0, 90.0, 0.0
                reason = "invalid_crop"
            else:
                landmarks = tuple(
                    (float(det_row[f"lmk_x{i + 1}"]), float(det_row[f"lmk_y{i + 1}"])) for i in range(5)
                )
                blur = blur_score(crop)
                bright = brightness(crop)
                pose = estimate_pose(landmarks)
                yaw_deg, pitch_deg = pose.yaw_deg, pose.pitch_deg
                quality_score = composite_quality_score(
                    float(det_row["face_width_px"]), blur, yaw_deg, pitch_deg, params
                )
                reason = _reject_reason(
                    float(det_row["score"]), float(det_row["face_width_px"]),
                    blur, yaw_deg, pitch_deg, bright, True, params,
                )

            accepted = reason is None
            if accepted:
                accepted_count += 1
            else:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1

            rows.append({
                **det_row.to_dict(),
                "blur": blur,
                "brightness": bright,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "quality_score": quality_score,
                "accepted": accepted,
                "reject_reason": reason,
            })

    quality_df = pd.DataFrame(rows)
    return QualityResult(
        quality_df=quality_df,
        accepted_count=accepted_count,
        rejected_count=len(quality_df) - accepted_count,
        reject_reason_counts=reject_counts,
    )


@dataclass(frozen=True)
class QualityStageSummary:
    input_count: int
    accepted_count: int
    rejected_count: int
    reject_reason_counts: dict[str, int]
    quality_parquet_path: Path


def run_quality_stage(
    detections_parquet_path: Path, frame_dir: Path, out_dir: Path, params: PipelineParams
) -> QualityStageSummary:
    """The I/O wrapper run.py's orchestrator actually calls: reads
    detections.parquet, runs score_detections, writes quality.parquet, and
    logs the stage summary (Phase 4 deliverable 3: "input detections,
    accepted count, rejected count broken down by reason, and the
    accepted-crop quality score distribution").
    """
    detections_df = pd.read_parquet(detections_parquet_path)
    result = score_detections(detections_df, frame_dir, params)

    out_dir.mkdir(parents=True, exist_ok=True)
    quality_parquet_path = out_dir / "quality.parquet"
    result.quality_df.to_parquet(quality_parquet_path)

    accepted_scores = result.quality_df.loc[result.quality_df["accepted"], "quality_score"]
    if len(accepted_scores):
        dist = (
            f"mean={accepted_scores.mean():.3f} min={accepted_scores.min():.3f} "
            f"max={accepted_scores.max():.3f}"
        )
    else:
        dist = "(no accepted crops)"

    logger.info(
        "quality stage: %d input, %d accepted, %d rejected -- reasons: %s -- accepted score dist: %s",
        len(detections_df), result.accepted_count, result.rejected_count, result.reject_reason_counts, dist,
    )

    return QualityStageSummary(
        input_count=len(detections_df),
        accepted_count=result.accepted_count,
        rejected_count=result.rejected_count,
        reject_reason_counts=result.reject_reason_counts,
        quality_parquet_path=quality_parquet_path,
    )
