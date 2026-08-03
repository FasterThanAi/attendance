"""Every tunable number in the pipeline lives here, and nowhere else
(non-negotiable rule #3). Every job record stores the exact PipelineParams
used, serialised to JSON, so re-running a stage with different numbers is
always reproducible and comparable.

Fields are grouped by which phase introduced them. Phase 1 only needs the
enrollment-related fields plus the shared detection/quality/embedding fields
that enrollment already exercises; match/cluster fields are included now
(with documented defaults) because they're declared once, here, even though
nothing uses them until Phases 5-6 -- adding a field later would mean an
older job's stored params_json is missing a key the code now expects.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineParams:
    # --- frame sampling ---
    # Classroom video (Phase 3): 4 fps is enough coverage of a ~60s pan while
    # cutting a 30fps/1800-frame video down to ~240 frames (7/8 less compute).
    sample_fps: float = 4.0

    # Enrollment video (Phase 1): the student's 5s turning-head clip is much
    # shorter, so a higher fps still yields a manageable frame count (~30
    # frames) while giving enough samples to find 3 distinct head poses.
    enrollment_sample_fps: float = 6.0

    # --- tiled detection (Phase 3) ---
    # A frame is only tiled if its longer side exceeds this -- a 4K frame
    # (3840x2160) is; enrollment's selfie-video frames and the pre-flight
    # check's downscaled samples are not, and skip tiling entirely.
    tile_trigger_long_side_px: int = 2000
    # 1280px tiles with 256px overlap, per the roadmap's tiled-detection
    # section verbatim -- overlap exists so a face straddling a tile boundary
    # is still fully visible in at least one tile.
    tile_size_px: int = 1280
    tile_overlap_px: int = 256
    # IoU above which two detections (from adjacent tiles, or a tile vs. the
    # whole-frame downscaled pass) are considered the same face and merged.
    nms_iou_threshold: float = 0.4

    # --- detection quality gate ---
    # Below this SCRFD confidence, treat it as noise, not a face.
    detector_score_min: float = 0.60
    # ArcFace needs ~112x112 aligned pixels to embed reliably (Section 1.1 of
    # the roadmap); below this the crop is unusable regardless of anything else.
    min_face_px: int = 50
    # Beyond this yaw/pitch, the face is too rotated for a reliable embedding
    # (profile shots confuse ArcFace, which was trained mostly on near-frontal
    # faces). These are the ACCEPT/REJECT bounds -- not to be confused with
    # enrollment_pose_split_deg below, which just buckets already-accepted
    # frames into left/frontal/right for pose diversity.
    max_abs_yaw_deg: float = 35.0
    max_abs_pitch_deg: float = 25.0
    # Variance of the Laplacian on a 128x128-normalised crop (see quality.py's
    # blur_score -- resizing first is what makes this threshold comparable
    # across near/far faces). Below this, the crop is too blurred to trust.
    blur_laplacian_min: float = 90.0
    # Mean luminance (0-255) outside this range means the crop is too dark
    # (underexposed / backlit) or blown out to embed reliably.
    brightness_min: float = 40.0
    brightness_max: float = 215.0

    # --- pose bucketing (enrollment only, Phase 1) ---
    # A frame's yaw below -enrollment_pose_split_deg is "left", above
    # +enrollment_pose_split_deg is "right", otherwise "frontal". This is a
    # DIFFERENT, tighter boundary than max_abs_yaw_deg on purpose: it's about
    # spreading enrollment coverage across poses, not about rejecting anything.
    enrollment_pose_split_deg: float = 15.0
    # Best-quality crops kept per pose bucket. 3 buckets x 3 = up to 9 crops,
    # trimmed down to enrollment_max_embeddings if every bucket over-delivers.
    enrollment_crops_per_pose: int = 3
    # Hard ceiling on stored embeddings per student -- keeps storage and
    # later matching cost bounded even if all three buckets are full quality.
    enrollment_max_embeddings: int = 8
    # An enrollment run producing fewer than this many accepted, bucketed
    # crops is flagged for re-recording rather than silently accepted with a
    # thin gallery (Phase 1's definition of done: "at least five gallery
    # embeddings from at least three distinct head poses").
    gallery_min_embeddings: int = 5

    # --- alignment / embedding ---
    embed_input_size: int = 112
    embed_batch_size: int = 64

    # --- clustering (Phase 5) ---
    cluster_eps: float = 0.42  # DBSCAN cosine distance
    cluster_min_samples: int = 3
    # Merge two DBSCAN clusters if their representative vectors are closer
    # than cluster_eps * this factor AND their frame ranges overlap
    # substantially -- see Phase 5's temporal-coherence post-pass.
    cluster_merge_distance_factor: float = 1.3
    temporal_coherence_enabled: bool = True

    # --- matching (Phase 6) ---
    match_threshold: float = 0.38  # cosine similarity, ArcFace r100
    match_margin_min: float = 0.05  # top1 minus top2 (Hungarian assignment)
    uncertain_band: float = 0.08  # below threshold by this much -> review

    # --- pre-flight quality check (Phase 2) ---
    # ASSUMPTION: these thresholds are a first-pass guess, not calibrated
    # against real classroom footage -- I have no real 4K pan video to test
    # against in this environment. Expect to retune all of these once you've
    # run a few real uploads through it and can see which checks fire
    # correctly vs. which cry wolf (or miss something obvious).
    preflight_sample_count: int = 18  # frames sampled, evenly spaced across the video
    preflight_sharpness_min: float = 60.0  # whole-frame Laplacian variance (looser than per-crop blur_laplacian_min)
    preflight_backlight_luminance_ratio_max: float = 1.8  # upper-third mean / face-region mean luminance
    preflight_min_pan_range_fraction: float = 0.25  # (max-min mean detection x) / frame width, else "didn't pan"
    preflight_max_pan_speed_px_per_sec: float = 400.0  # else "panned too fast" -> motion blur
    preflight_min_face_yield_ratio: float = 0.15  # (mean detections per frame) / expected_students, else "too few faces"
    preflight_min_coverage_fraction: float = 0.6  # same x-range metric as pan range, used for the "missed part of room" check

    def to_json_dict(self) -> dict:
        """What gets stored on processing_job.params_json (see models.py)."""
        return asdict(self)
