"""Phase 1 deliverable 7: "pose bucketing on synthetic landmark inputs."

`estimate_pose`'s exact formula is documented (and flagged as a heuristic
assumption) in quality.py's module docstring, so these tests assert precise
expected values computed by hand from that formula -- not just "roughly
plausible" -- to catch any accidental change to the geometry.
"""

import pytest

from enrollment import pose_bucket
from pipeline.quality import PITCH_SCALE_DEG, YAW_SCALE_DEG, estimate_pose

# left eye, right eye, nose, left mouth corner, right mouth corner
FRONTAL_LANDMARKS = ((30.0, 50.0), (70.0, 50.0), (50.0, 70.0), (35.0, 90.0), (65.0, 90.0))


def test_estimate_pose_frontal_is_near_zero():
    pose = estimate_pose(FRONTAL_LANDMARKS)
    assert pose.yaw_deg == pytest.approx(0.0, abs=1e-6)
    assert pose.pitch_deg == pytest.approx(0.0, abs=1e-6)


def test_estimate_pose_yaw_shifts_with_nose_x():
    # inter-ocular distance is 40; nose shifted +10px right of eye midpoint
    # -> yaw_ratio = 10/40 = 0.25
    landmarks = ((30.0, 50.0), (70.0, 50.0), (60.0, 70.0), (35.0, 90.0), (65.0, 90.0))
    pose = estimate_pose(landmarks)
    assert pose.yaw_deg == pytest.approx(0.25 * YAW_SCALE_DEG, abs=1e-6)


def test_estimate_pose_pitch_shifts_with_nose_y():
    # eye_mid=(50,50), mouth_mid=(50,90) -> eye_mouth_mid=(50,70).
    # nose shifted +10px below that -> pitch_ratio = 10/40 = 0.25
    landmarks = ((30.0, 50.0), (70.0, 50.0), (50.0, 80.0), (35.0, 90.0), (65.0, 90.0))
    pose = estimate_pose(landmarks)
    assert pose.pitch_deg == pytest.approx(0.25 * PITCH_SCALE_DEG, abs=1e-6)


def test_estimate_pose_degenerate_landmarks_does_not_crash():
    # Left and right eye at the same point -> inter-ocular distance is zero.
    # Must return a clamped, maximally-rotated estimate, not raise or NaN.
    landmarks = ((50.0, 50.0), (50.0, 50.0), (50.0, 70.0), (35.0, 90.0), (65.0, 90.0))
    pose = estimate_pose(landmarks)
    assert pose.yaw_deg == 90.0
    assert pose.pitch_deg == 90.0


@pytest.mark.parametrize(
    "yaw_deg, split_deg, expected",
    [
        (0.0, 15.0, "frontal"),
        (14.9, 15.0, "frontal"),
        (-14.9, 15.0, "frontal"),
        (15.1, 15.0, "right"),
        (-15.1, 15.0, "left"),
        (40.0, 15.0, "right"),
        (-40.0, 15.0, "left"),
    ],
)
def test_pose_bucket(yaw_deg, split_deg, expected):
    assert pose_bucket(yaw_deg, split_deg) == expected
