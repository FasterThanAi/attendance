"""Phase 1: enrollment job.

Given a student's 5-second turning-head video, produces a pose-diverse set
of high-quality face embeddings and writes them to the gallery tables.

Split into a pure function (`process_enrollment_video`, no DB, no network --
independently testable, non-negotiable rule #1) and a thin DB-writing
orchestrator (`enroll_student`), the same split the classroom pipeline will
use in run.py (Phase 2).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import insert, select, update

from config import settings
from consent import assert_consent_valid
from db import get_engine, table
from pipeline.align import align_face
from pipeline.detect import DetectorModel, detect_faces, largest_face, load_detector
from pipeline.embed import EmbedModel, embed_batch, l2_normalize, load_model
from pipeline.extract import extract_frames
from pipeline.params import PipelineParams
from pipeline.quality import brightness, blur_score, estimate_pose, simple_quality_score

logger = logging.getLogger("attend.worker.enrollment")

MODEL_VERSION = "arcface-r100-buffalo_l"  # see embed.py's ASSUMPTION docstring


def pose_bucket(yaw_deg: float, split_deg: float) -> str:
    """Pure function, unit-tested directly on synthetic yaw values (Phase 1
    deliverable 7: "pose bucketing on synthetic landmark inputs").
    """
    if yaw_deg < -split_deg:
        return "left"
    if yaw_deg > split_deg:
        return "right"
    return "frontal"


@dataclass(frozen=True)
class EnrollmentCrop:
    bucket: str
    quality_score: float
    yaw_deg: float
    pitch_deg: float
    blur: float
    brightness: float
    aligned_bgr: np.ndarray  # (112, 112, 3) uint8, not yet embedded
    frame_index: int


@dataclass
class EnrollmentSummary:
    total_embeddings: int
    pose_coverage: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def process_enrollment_video(
    video_path: Path,
    work_dir: Path,
    params: PipelineParams,
    detector: DetectorModel,
) -> list[EnrollmentCrop]:
    """Extract, detect, quality-gate, pose-bucket, and select the best crops
    from a single enrollment video. Does not embed and does not touch the
    database -- see enroll_student for that.
    """
    manifest = extract_frames(video_path, work_dir / "frames", params.enrollment_sample_fps)

    candidates: list[EnrollmentCrop] = []
    discarded_multi_face = 0

    for frame_index, frame_path in enumerate(sorted((work_dir / "frames").glob("frame_*.jpg"))):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue

        detections = detect_faces(image, detector, score_min=params.detector_score_min)

        # Enrollment videos have exactly one subject. Two-or-more faces in a
        # frame (someone walked past, a poster in the background) means
        # discard the WHOLE frame rather than guess which face is the
        # student -- Phase 1 prompt, verbatim.
        if len(detections) >= 2:
            discarded_multi_face += 1
            continue
        if len(detections) == 0:
            continue

        det = largest_face(detections)
        if det is None or det.face_width_px < params.min_face_px:
            continue

        aligned = align_face(image, det.landmarks, params.embed_input_size)
        pose = estimate_pose(det.landmarks)

        if abs(pose.yaw_deg) > params.max_abs_yaw_deg or abs(pose.pitch_deg) > params.max_abs_pitch_deg:
            continue

        blur = blur_score(aligned)
        if blur < params.blur_laplacian_min:
            continue

        bright = brightness(aligned)
        if not (params.brightness_min <= bright <= params.brightness_max):
            continue

        quality_score = simple_quality_score(det.face_width_px, blur, pose.yaw_deg, params.max_abs_yaw_deg)
        bucket = pose_bucket(pose.yaw_deg, params.enrollment_pose_split_deg)

        candidates.append(
            EnrollmentCrop(
                bucket=bucket,
                quality_score=quality_score,
                yaw_deg=pose.yaw_deg,
                pitch_deg=pose.pitch_deg,
                blur=blur,
                brightness=bright,
                aligned_bgr=aligned,
                frame_index=frame_index,
            )
        )

    if discarded_multi_face:
        logger.warning("enrollment: discarded %d frame(s) with multiple faces", discarded_multi_face)

    # Select the top-N by quality per bucket, so the final set spans poses
    # rather than being N near-identical frontal frames (Phase 1 prompt).
    selected: list[EnrollmentCrop] = []
    for bucket_name in ("left", "frontal", "right"):
        bucket_crops = sorted(
            (c for c in candidates if c.bucket == bucket_name),
            key=lambda c: c.quality_score,
            reverse=True,
        )
        selected.extend(bucket_crops[: params.enrollment_crops_per_pose])

    # Hard ceiling: if all three buckets delivered a full quota, trim the
    # lowest-quality overall down to enrollment_max_embeddings.
    if len(selected) > params.enrollment_max_embeddings:
        selected = sorted(selected, key=lambda c: c.quality_score, reverse=True)[: params.enrollment_max_embeddings]

    return selected


def enroll_student(
    student_id: int,
    video_path: Path,
    work_dir: Path,
    params: PipelineParams | None = None,
) -> EnrollmentSummary:
    """The RQ job entrypoint. Consent is checked FIRST, before any video
    processing runs at all -- not just before the DB writes at the end.
    """
    params = params or PipelineParams()
    video_path = Path(video_path)
    work_dir = Path(work_dir)
    engine = get_engine()

    with engine.begin() as conn:
        assert_consent_valid(conn, student_id)  # raises ConsentError and aborts if invalid

    detector = load_detector(Path(settings.insightface_home))
    embed_model = load_model(Path(settings.insightface_home))

    crops = process_enrollment_video(video_path, work_dir, params, detector)

    warnings: list[str] = []
    pose_coverage = {"left": 0, "frontal": 0, "right": 0}
    for c in crops:
        pose_coverage[c.bucket] += 1

    if len(crops) < params.gallery_min_embeddings:
        warnings.append(
            f"Only {len(crops)} usable crops found (need at least {params.gallery_min_embeddings}). "
            "Re-record the enrollment video: slower head turn, better lighting, closer to camera."
        )
    for bucket_name, count in pose_coverage.items():
        if count == 0:
            warnings.append(f"No usable '{bucket_name}' pose crops found -- turn further in that direction.")

    if not crops:
        return EnrollmentSummary(total_embeddings=0, pose_coverage=pose_coverage, warnings=warnings)

    aligned_batch = np.stack([c.aligned_bgr for c in crops], axis=0)
    embeddings = embed_batch(embed_model, aligned_batch)

    now = datetime.now(timezone.utc)
    retention_expires_at = now + timedelta(days=settings.biometric_retention_days)

    photo_dir = Path(settings.job_data_dir) / "enrollment" / str(student_id)
    photo_dir.mkdir(parents=True, exist_ok=True)

    gallery_photo = table("gallery_photo")
    gallery_embedding = table("gallery_embedding")
    student = table("student")

    with engine.begin() as conn:
        for crop, vector in zip(crops, embeddings):
            crop_filename = f"{crop.bucket}_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = photo_dir / crop_filename
            cv2.imwrite(str(crop_path), crop.aligned_bgr)

            photo_id = conn.execute(
                insert(gallery_photo).values(
                    student_id=student_id,
                    storage_uri=str(crop_path),
                    captured_at=now,
                    quality_score=crop.quality_score,
                    pose_bucket=crop.bucket,
                )
            ).inserted_primary_key[0]

            conn.execute(
                insert(gallery_embedding).values(
                    student_id=student_id,
                    vector=vector.tobytes(),
                    source_photo_id=photo_id,
                    model_version=MODEL_VERSION,
                    created_at=now,
                    retention_expires_at=retention_expires_at,
                )
            )

        # Recompute the cached mean vector from ALL of this student's
        # embeddings (not just this run's), so re-enrollment or a second
        # session correctly blends with what's already there.
        all_vectors_rows = conn.execute(
            select(gallery_embedding.c.vector).where(gallery_embedding.c.student_id == student_id)
        ).all()
        all_vectors = np.stack(
            [np.frombuffer(row[0], dtype=np.float32) for row in all_vectors_rows], axis=0
        )
        mean_vector = l2_normalize(all_vectors.mean(axis=0, keepdims=True))[0]

        conn.execute(
            update(student)
            .where(student.c.id == student_id)
            .values(gallery_mean_vector=mean_vector.astype(np.float32).tobytes(), gallery_updated_at=now)
        )

    return EnrollmentSummary(total_embeddings=len(crops), pose_coverage=pose_coverage, warnings=warnings)
