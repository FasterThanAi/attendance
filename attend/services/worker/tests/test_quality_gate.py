"""Phase 4 deliverable 5: "each reject rule fires on a crafted input" and
"blur metric is scale-invariant across two sizes of the same face."

score_detections takes a plain in-memory DataFrame and a directory of real
frame JPEGs (written here with cv2, same as any other test) -- no parquet
file needs to exist on disk for these tests, since parquet I/O only happens
in run_quality_stage, not in score_detections itself (see quality.py's
docstring for why that split exists).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from pipeline.detect import DETECTION_COLUMNS
from pipeline.params import PipelineParams
from pipeline.quality import blur_score, composite_quality_score, score_detections

# left eye, right eye, nose, left mouth corner, right mouth corner -- a
# frontal face, well inside every one of PipelineParams' default bounds.
FRONTAL_LANDMARKS = ((130.0, 150.0), (170.0, 150.0), (150.0, 170.0), (135.0, 190.0), (165.0, 190.0))


def _write_frame(tmp_path, frame_index: int, sharp: bool = True, brightness_value: int = 128) -> None:
    frame_dir = tmp_path / "extract"
    frame_dir.mkdir(exist_ok=True)
    if sharp:
        rng = np.random.default_rng(frame_index)
        image = rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)
    else:
        image = np.full((300, 300, 3), brightness_value, dtype=np.uint8)
    cv2.imwrite(str(frame_dir / f"frame_{frame_index + 1:05d}.jpg"), image)


def _detection_row(frame_index: int, det_id: str, x1=100, y1=100, x2=200, y2=200, score=0.9, landmarks=FRONTAL_LANDMARKS) -> dict:
    row = {
        "frame_index": frame_index, "frame_timestamp_s": frame_index / 4.0, "det_id": det_id,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score,
        "face_width_px": x2 - x1, "tile_origin_x": -1, "tile_origin_y": -1,
    }
    for i, (lx, ly) in enumerate(landmarks):
        row[f"lmk_x{i + 1}"] = lx
        row[f"lmk_y{i + 1}"] = ly
    return row


def test_score_detections_accepts_a_good_frontal_crop(tmp_path):
    _write_frame(tmp_path, 0, sharp=True)
    df = pd.DataFrame([_detection_row(0, "0_0")])

    result = score_detections(df, tmp_path / "extract", PipelineParams())

    assert result.accepted_count == 1
    assert result.rejected_count == 0
    row = result.quality_df.iloc[0]
    assert row["accepted"]
    assert row["reject_reason"] is None
    assert 0.0 <= row["quality_score"] <= 1.0


def test_score_detections_rejects_low_detector_score(tmp_path):
    _write_frame(tmp_path, 0, sharp=True)
    params = PipelineParams()
    df = pd.DataFrame([_detection_row(0, "0_0", score=params.detector_score_min - 0.01)])

    result = score_detections(df, tmp_path / "extract", params)
    assert result.quality_df.iloc[0]["reject_reason"] == "low_detector_score"
    assert not result.quality_df.iloc[0]["accepted"]


def test_score_detections_rejects_too_small_face(tmp_path):
    _write_frame(tmp_path, 0, sharp=True)
    params = PipelineParams()
    # face_width_px = x2-x1 = 10, well under min_face_px=50
    df = pd.DataFrame([_detection_row(0, "0_0", x1=100, y1=100, x2=110, y2=110)])

    result = score_detections(df, tmp_path / "extract", params)
    assert result.quality_df.iloc[0]["reject_reason"] == "too_small"


def test_score_detections_rejects_too_blurred(tmp_path):
    _write_frame(tmp_path, 0, sharp=False, brightness_value=128)  # uniform crop -> zero blur
    df = pd.DataFrame([_detection_row(0, "0_0")])

    result = score_detections(df, tmp_path / "extract", PipelineParams())
    assert result.quality_df.iloc[0]["reject_reason"] == "too_blurred"


def test_score_detections_rejects_yaw_too_extreme(tmp_path):
    _write_frame(tmp_path, 0, sharp=True)
    # eye_mid=(150,150), inter_ocular=40; nose at x=190 -> yaw_ratio=1.0 ->
    # yaw_deg=65 (YAW_SCALE_DEG), well past the default max_abs_yaw_deg=35.
    # nose y stays at the eye_mouth_mid's y so pitch is unaffected (isolates
    # this test to only the yaw rule).
    landmarks = ((130.0, 150.0), (170.0, 150.0), (190.0, 170.0), (135.0, 190.0), (165.0, 190.0))
    df = pd.DataFrame([_detection_row(0, "0_0", landmarks=landmarks)])

    result = score_detections(df, tmp_path / "extract", PipelineParams())
    assert result.quality_df.iloc[0]["reject_reason"] == "yaw_too_extreme"


def test_score_detections_rejects_pitch_too_extreme(tmp_path):
    _write_frame(tmp_path, 0, sharp=True)
    # Push the nose far below the eye-mouth midpoint -> large pitch.
    landmarks = ((130.0, 150.0), (170.0, 150.0), (150.0, 210.0), (135.0, 190.0), (165.0, 190.0))
    df = pd.DataFrame([_detection_row(0, "0_0", landmarks=landmarks)])

    result = score_detections(df, tmp_path / "extract", PipelineParams())
    assert result.quality_df.iloc[0]["reject_reason"] == "pitch_too_extreme"


def test_score_detections_rejects_bad_brightness(tmp_path):
    _write_frame(tmp_path, 0, sharp=False, brightness_value=250)  # over brightness_max=215, also blurred
    # Use a params override with a very low blur floor so brightness is the
    # ONLY failing check (isolating this rule from "too_blurred").
    params = PipelineParams(blur_laplacian_min=-1.0)
    df = pd.DataFrame([_detection_row(0, "0_0")])

    result = score_detections(df, tmp_path / "extract", params)
    assert result.quality_df.iloc[0]["reject_reason"] == "bad_brightness"


def test_score_detections_rejects_invalid_crop_outside_frame(tmp_path):
    _write_frame(tmp_path, 0, sharp=True)  # 300x300 frame
    df = pd.DataFrame([_detection_row(0, "0_0", x1=400, y1=400, x2=500, y2=500)])  # entirely outside

    result = score_detections(df, tmp_path / "extract", PipelineParams())
    assert result.quality_df.iloc[0]["reject_reason"] == "invalid_crop"


def test_score_detections_keeps_rejected_rows_not_drops_them(tmp_path):
    # Phase 4 prompt, verbatim: "Keep the rejected rows -- do not drop them."
    _write_frame(tmp_path, 0, sharp=True)
    params = PipelineParams()
    df = pd.DataFrame([
        _detection_row(0, "0_0", score=params.detector_score_min - 0.1),  # rejected
        _detection_row(0, "0_1"),  # accepted
    ])

    result = score_detections(df, tmp_path / "extract", params)
    assert len(result.quality_df) == 2
    assert result.accepted_count == 1
    assert result.rejected_count == 1


def test_score_detections_empty_input_has_full_schema_not_crash(tmp_path):
    (tmp_path / "extract").mkdir()
    empty_df = pd.DataFrame(columns=DETECTION_COLUMNS)

    result = score_detections(empty_df, tmp_path / "extract", PipelineParams())
    assert result.accepted_count == 0
    assert result.rejected_count == 0
    assert "accepted" in result.quality_df.columns
    assert "reject_reason" in result.quality_df.columns
    assert len(result.quality_df) == 0


def test_blur_score_is_scale_invariant_across_two_crop_sizes():
    # Same underlying, realistically-textured face-sized crop, sampled at
    # two different NATIVE resolutions -- a smooth-ish random pattern
    # (low-res noise upsampled with linear interpolation, not raw per-pixel
    # noise, which aliases unrealistically under resampling) standing in
    # for real face texture. blur_score resizes to a fixed 128x128 FIRST
    # internally (Phase 4 prompt: "otherwise the metric scales with face
    # size and is not comparable across rows"), so a face crop captured at
    # 60px and the same face captured at 200px should land in the same
    # ballpark once both are normalised to 128x128, not off by orders of
    # magnitude the way raw (un-resized) Laplacian variance would be.
    rng = np.random.default_rng(0)
    low_res_texture = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
    base = cv2.resize(low_res_texture, (256, 256), interpolation=cv2.INTER_LINEAR)

    small = cv2.resize(base, (90, 90), interpolation=cv2.INTER_AREA)  # a small/far face crop
    large = cv2.resize(base, (180, 180), interpolation=cv2.INTER_AREA)  # a large/close face crop

    score_small = blur_score(small)
    score_large = blur_score(large)

    # Not asserting exact equality (resizing itself introduces some blur
    # differences -- more so the further a crop's native size is from the
    # canonical 128x128 in either direction), just that they're in the same
    # ballpark, not off by an order of magnitude: the whole point of
    # resizing to a fixed size first is that a small crop and a large crop
    # of equivalently-sharp content shouldn't be wildly apart.
    ratio = max(score_small, score_large) / max(min(score_small, score_large), 1e-6)
    assert ratio < 4.0, f"blur scores not scale-invariant enough: small={score_small} large={score_large}"


def test_composite_quality_score_prefers_larger_sharper_frontal_faces():
    params = PipelineParams()
    good = composite_quality_score(face_width_px=150, blur=300, yaw_deg=0, pitch_deg=0, params=params)
    small = composite_quality_score(face_width_px=30, blur=300, yaw_deg=0, pitch_deg=0, params=params)
    blurry = composite_quality_score(face_width_px=150, blur=20, yaw_deg=0, pitch_deg=0, params=params)
    rotated = composite_quality_score(face_width_px=150, blur=300, yaw_deg=34, pitch_deg=0, params=params)
    pitched = composite_quality_score(face_width_px=150, blur=300, yaw_deg=0, pitch_deg=24, params=params)

    assert good == pytest.approx(1.0)
    assert small < good
    assert blurry < good
    assert rotated < good
    assert pitched < good


def test_composite_quality_score_weights_are_configurable():
    # All weight on size -- blur/frontality shouldn't move the score at all.
    params = PipelineParams(quality_weight_size=1.0, quality_weight_blur=0.0, quality_weight_frontality=0.0)
    sharp = composite_quality_score(face_width_px=150, blur=300, yaw_deg=0, pitch_deg=0, params=params)
    blurry = composite_quality_score(face_width_px=150, blur=0, yaw_deg=90, pitch_deg=90, params=params)
    assert sharp == pytest.approx(blurry)
