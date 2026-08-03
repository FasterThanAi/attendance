import numpy as np

from pipeline.quality import blur_score, brightness, simple_quality_score


def test_blur_score_sharp_vs_flat():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)  # noise = "sharp" (high Laplacian variance)
    flat = np.full((112, 112, 3), 128, dtype=np.uint8)  # uniform = perfectly "blurred"

    assert blur_score(sharp) > blur_score(flat)
    assert blur_score(flat) == 0.0


def test_brightness_matches_uniform_value():
    crop = np.full((50, 50, 3), 100, dtype=np.uint8)
    assert brightness(crop) == 100.0


def test_simple_quality_score_prefers_larger_sharper_frontal_faces():
    good = simple_quality_score(face_width_px=150, blur=300, yaw_deg=0, max_abs_yaw_deg=35.0)
    small = simple_quality_score(face_width_px=30, blur=300, yaw_deg=0, max_abs_yaw_deg=35.0)
    blurry = simple_quality_score(face_width_px=150, blur=20, yaw_deg=0, max_abs_yaw_deg=35.0)
    rotated = simple_quality_score(face_width_px=150, blur=300, yaw_deg=34, max_abs_yaw_deg=35.0)

    assert good == 1.0
    assert small < good
    assert blurry < good
    assert rotated < good
