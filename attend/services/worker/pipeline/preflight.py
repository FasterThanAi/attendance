"""Pre-flight quality check (Phase 2 deliverable 2b).

Runs on 15-20 sampled frames and must finish in well under 30 seconds -- the
whole point is telling the teacher a video is unusable while they're still
in the classroom and can re-shoot immediately, not twenty minutes later when
the class has ended. See Section 1.4 of the roadmap: "the highest-value
fifty lines of code in the project." (This is closer to two hundred, because
every check needs its own honest failure message, but the spirit holds.)

ASSUMPTION, flagged clearly and repeated from params.py: every threshold
here is a first-pass guess. I have no real 4K classroom pan video to
calibrate against in this environment. Treat the exact numbers as
placeholders to retune once you have real footage -- the STRUCTURE (which
checks exist, what they measure, pass/warn/fail semantics) is the actual
Phase 2 deliverable; the numbers are Phase 7's job to calibrate properly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.detect import DetectorModel, detect_faces
from pipeline.params import PipelineParams


@dataclass(frozen=True)
class CheckResult:
    code: str
    severity: str  # "info" | "warn" | "fail"
    message: str


@dataclass
class PreflightResult:
    status: str  # "pass" | "warn" | "fail"
    checks: list[CheckResult] = field(default_factory=list)


def _sample_frames(video_path: Path, count: int) -> list[np.ndarray]:
    """Grabs `count` evenly spaced frames directly via OpenCV's frame-index
    seeking -- no ffmpeg subprocess, no intermediate JPEGs, since these
    frames are only used for in-memory measurements and never stored.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        indices = np.linspace(0, total_frames - 1, num=min(count, total_frames), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        return frames
    finally:
        cap.release()


def _frame_sharpness(frame_bgr: np.ndarray) -> float:
    # Downscale before measuring so this stays fast on a full 4K frame --
    # unlike quality.py's per-crop blur_score, this is a whole-frame check,
    # so the fixed resize target is larger (960px) to still catch genuine
    # softness without being dominated by fine sensor noise at full res.
    h, w = frame_bgr.shape[:2]
    scale = 960.0 / max(h, w)
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def preflight_check(
    video_path: Path,
    expected_students: int,
    params: PipelineParams,
    detector: DetectorModel,
) -> PreflightResult:
    frames = _sample_frames(video_path, params.preflight_sample_count)
    checks: list[CheckResult] = []

    if not frames:
        return PreflightResult(
            status="fail",
            checks=[CheckResult("unreadable_video", "fail", "This file could not be read as a video.")],
        )

    frame_height, frame_width = frames[0].shape[:2]

    # --- sharpness ---
    sharpness_values = [_frame_sharpness(f) for f in frames]
    median_sharpness = float(np.median(sharpness_values))
    if median_sharpness < params.preflight_sharpness_min:
        checks.append(CheckResult(
            "blurry_video", "fail",
            "The video is blurry. Hold the phone steadier and pan more slowly.",
        ))

    # --- detection on each sampled frame (whole-frame, downscaled to 1280 --
    # same "catch large faces on a full-frame pass" idea Phase 3 uses
    # alongside tiling, used here alone since preflight only needs a rough
    # signal, not per-crop accuracy) ---
    mean_x_per_frame: list[float] = []
    detection_counts: list[int] = []
    face_luminances: list[float] = []
    upper_third_luminances: list[float] = []

    for frame in frames:
        h, w = frame.shape[:2]
        scale = 1280.0 / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        detections = detect_faces(small, detector, score_min=params.detector_score_min)
        detection_counts.append(len(detections))

        if detections:
            xs = [(d.x1 + d.x2) / 2.0 for d in detections]
            mean_x_per_frame.append(float(np.mean(xs)) / small.shape[1])  # normalised 0..1

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            face_lums = []
            for d in detections:
                x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                region = gray[max(0, y1):y2, max(0, x1):x2]
                if region.size:
                    face_lums.append(float(region.mean()))
            if face_lums:
                face_luminances.append(float(np.mean(face_lums)))

            upper_third = gray[: gray.shape[0] // 3, :]
            upper_third_luminances.append(float(upper_third.mean()))

    # --- face yield ---
    mean_detections_per_frame = float(np.mean(detection_counts)) if detection_counts else 0.0
    if expected_students > 0:
        yield_ratio = mean_detections_per_frame / expected_students
        if yield_ratio < params.preflight_min_face_yield_ratio:
            checks.append(CheckResult(
                "low_face_yield", "fail",
                "Very few faces were found. Check that you are facing the students and the room is lit.",
            ))

    # --- pan detection + range/coverage (same underlying metric, two checks) ---
    if len(mean_x_per_frame) >= 2:
        x_range = max(mean_x_per_frame) - min(mean_x_per_frame)
        if x_range < params.preflight_min_pan_range_fraction:
            checks.append(CheckResult(
                "no_pan_detected", "fail",
                "The camera did not move across the room. Pan slowly from one side to the other.",
            ))
        elif x_range < params.preflight_min_coverage_fraction:
            checks.append(CheckResult(
                "partial_coverage", "warn",
                "Part of the room may have been missed. Pan the full width.",
            ))

        # pan speed: total normalised x movement, scaled to pixels, over the
        # sampled duration -- a rough proxy since we don't have per-frame
        # exact timestamps here, just evenly spaced samples across the video.
        cap = cv2.VideoCapture(str(video_path))
        video_duration_s = 0.0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            video_duration_s = frame_count / fps if fps > 0 else 0.0
        finally:
            cap.release()

        if video_duration_s > 0:
            total_px_movement = sum(
                abs(mean_x_per_frame[i] - mean_x_per_frame[i - 1]) * frame_width
                for i in range(1, len(mean_x_per_frame))
            )
            px_per_sec = total_px_movement / video_duration_s
            if px_per_sec > params.preflight_max_pan_speed_px_per_sec:
                checks.append(CheckResult(
                    "pan_too_fast", "fail",
                    "You panned too quickly. Take at least 15 seconds for each sweep.",
                ))

    # --- backlighting ---
    if face_luminances and upper_third_luminances:
        mean_face_lum = float(np.mean(face_luminances))
        mean_upper_lum = float(np.mean(upper_third_luminances))
        if mean_face_lum > 1e-6 and mean_upper_lum / mean_face_lum > params.preflight_backlight_luminance_ratio_max:
            checks.append(CheckResult(
                "backlit", "fail",
                "The windows are behind the students. Move so the windows are behind you.",
            ))

    fail_checks = [c for c in checks if c.severity == "fail"]
    warn_checks = [c for c in checks if c.severity == "warn"]

    if fail_checks:
        status = "fail"
    elif warn_checks:
        status = "warn"
    else:
        status = "pass"

    return PreflightResult(status=status, checks=checks)
