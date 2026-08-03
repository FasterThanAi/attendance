"""api-side helper for creating processing_job rows.

DEFAULT_PIPELINE_PARAMS is a duplicate of PipelineParams()'s default field
values from services/worker/pipeline/params.py -- not imported directly for
the same reason as GALLERY_MIN_EMBEDDINGS in services/enrollment.py: that
module lives in the worker's separate Docker image/dependency set. Unlike
that one constant, this is the WHOLE defaults dict, which is more surface
area to drift -- if you add a field to PipelineParams, add it here too, or
job creation will serialise an incomplete params_json that the worker's
`PipelineParams(**json.loads(row["params_json"]))` will reject with a
missing-argument TypeError (which is at least a loud, honest failure, not a
silent wrong-default -- see run.py's fail-loudly design).
"""

DEFAULT_PIPELINE_PARAMS: dict = {
    "sample_fps": 4.0,
    "enrollment_sample_fps": 6.0,
    "tile_trigger_long_side_px": 2000,
    "tile_size_px": 1280,
    "tile_overlap_px": 256,
    "nms_iou_threshold": 0.4,
    "detector_score_min": 0.60,
    "min_face_px": 50,
    "max_abs_yaw_deg": 35.0,
    "max_abs_pitch_deg": 25.0,
    "blur_laplacian_min": 90.0,
    "brightness_min": 40.0,
    "brightness_max": 215.0,
    "enrollment_pose_split_deg": 15.0,
    "enrollment_crops_per_pose": 3,
    "enrollment_max_embeddings": 8,
    "gallery_min_embeddings": 5,
    "embed_input_size": 112,
    "embed_batch_size": 64,
    "cluster_eps": 0.42,
    "cluster_min_samples": 3,
    "cluster_merge_distance_factor": 1.3,
    "temporal_coherence_enabled": True,
    "match_threshold": 0.38,
    "match_margin_min": 0.05,
    "uncertain_band": 0.08,
    "preflight_sample_count": 18,
    "preflight_sharpness_min": 60.0,
    "preflight_backlight_luminance_ratio_max": 1.8,
    "preflight_min_pan_range_fraction": 0.25,
    "preflight_max_pan_speed_px_per_sec": 400.0,
    "preflight_min_face_yield_ratio": 0.15,
    "preflight_min_coverage_fraction": 0.6,
}
