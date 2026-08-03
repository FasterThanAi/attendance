"""Job orchestration (Phase 2 deliverable 3-4; Phase 3 wires in real
extract/detect stages).

`process_session` is the single RQ entrypoint for the classroom-video
pipeline. It runs each stage in order, skipping any stage whose on-disk
manifest already matches the current params (so re-running match calibration
a hundred times during Phase 7 doesn't re-extract frames every time -- see
non-negotiable rule #2), and fails loudly with the stage name and traceback
on any exception (rule: "a job that fails visibly is far better than a job
that quietly marks students absent").

Per the Phase 2 prompt, ALL stages started as stubs (log + write an empty
manifest). Phase 3 replaces exactly two of them -- extract and detect --
with real implementations (pipeline.extract.extract_frames and
pipeline.detect.detect_all_frames); quality/align/embed/cluster/match stay
stubs until Phases 4-6, one at a time, so it's always clear from this file
which phase delivered which stage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update

from config import settings
from db import get_engine, table
from pipeline.detect import detect_all_frames
from pipeline.extract import extract_frames
from pipeline.params import PipelineParams

logger = logging.getLogger("attend.worker.run")

STAGE_ORDER = ["extract", "detect", "quality", "align", "embed", "cluster", "match"]

# Which PipelineParams fields each stage's output depends on. Used to build
# a cumulative hash per stage (see compute_stage_hashes) so that changing a
# parameter invalidates that stage and everything after it, but NOT stages
# before it -- non-negotiable requirement from the Phase 2 prompt, and the
# reason Phase 7's threshold sweeps are affordable at all.
STAGE_PARAM_FIELDS: dict[str, list[str]] = {
    "extract": ["sample_fps"],
    "detect": [
        "detector_score_min", "tile_trigger_long_side_px", "tile_size_px",
        "tile_overlap_px", "nms_iou_threshold",
    ],
    "quality": ["min_face_px", "max_abs_yaw_deg", "max_abs_pitch_deg", "blur_laplacian_min", "brightness_min", "brightness_max"],
    "align": ["embed_input_size"],
    "embed": ["embed_batch_size"],
    "cluster": ["cluster_eps", "cluster_min_samples", "cluster_merge_distance_factor", "temporal_coherence_enabled"],
    "match": ["match_threshold", "match_margin_min", "uncertain_band"],
}


def compute_stage_hashes(params: PipelineParams) -> dict[str, str]:
    """A chained hash: each stage's hash covers its own relevant fields PLUS
    every upstream stage's hash. That chaining is what makes "invalidates
    this stage and everything after, not before" true -- a downstream-only
    field change never touches an upstream stage's hash, because upstream
    hashes are computed first and don't know downstream fields exist.
    """
    params_dict = params.to_json_dict()
    hashes: dict[str, str] = {}
    upstream_hash = ""

    for stage in STAGE_ORDER:
        own_fields = {name: params_dict[name] for name in STAGE_PARAM_FIELDS[stage]}
        payload = json.dumps({"upstream": upstream_hash, "own": own_fields}, sort_keys=True)
        stage_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        hashes[stage] = stage_hash
        upstream_hash = stage_hash

    return hashes


@dataclass(frozen=True)
class StageManifest:
    stage: str
    params_hash: str
    completed_at: str
    item_count: int


def _manifest_path(stage_dir: Path) -> Path:
    return stage_dir / "manifest.json"


def is_stage_complete(stage_dir: Path, expected_hash: str) -> bool:
    manifest_path = _manifest_path(stage_dir)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return manifest.get("params_hash") == expected_hash


def write_stage_manifest(stage_dir: Path, stage: str, params_hash: str, item_count: int) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = StageManifest(
        stage=stage,
        params_hash=params_hash,
        completed_at=datetime.now(timezone.utc).isoformat(),
        item_count=item_count,
    )
    _manifest_path(stage_dir).write_text(json.dumps(manifest.__dict__))


def _run_stub_stage(stage: str, stage_dir: Path, job_id: int) -> int:
    """Phase 2's stub, still used for quality/align/embed/cluster/match until
    Phases 4-6 replace them one at a time -- see _run_real_stage below for
    the pattern each of those will follow.
    """
    logger.info("job %s: stage '%s' running (stub)", job_id, stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return 0  # item_count -- nothing produced yet, this is a stub


def _run_extract_stage(stage_dir: Path, video_path: Path, params: PipelineParams, job_id: int) -> int:
    logger.info("job %s: stage 'extract' running (video=%s, fps=%s)", job_id, video_path, params.sample_fps)
    manifest = extract_frames(video_path, stage_dir, params.sample_fps)
    return manifest.frame_count


def _run_detect_stage(stage_dir: Path, frame_dir: Path, params: PipelineParams, model_dir: Path, job_id: int) -> int:
    logger.info("job %s: stage 'detect' running (frames=%s, model_dir=%s)", job_id, frame_dir, model_dir)
    fps = params.sample_fps
    summary = detect_all_frames(frame_dir, stage_dir, params, model_dir, fps)
    logger.info(
        "job %s: stage 'detect' done -- %s frames, %s detections, %.2f/frame mean",
        job_id, summary.total_frames, summary.total_detections, summary.detections_per_frame_mean,
    )
    return summary.total_detections


def run_all_stages(
    job_dir: Path,
    params: PipelineParams,
    job_id: int,
    on_stage_start: Callable[[str], None] | None = None,
    video_path: Path | None = None,
    model_dir: Path | None = None,
) -> None:
    """The actual stage loop: no database, no RQ, no network -- just the
    filesystem under `job_dir`, `params`, and (for the two real Phase 3
    stages) `video_path`/`model_dir`. Independently testable per
    non-negotiable rule #1; `process_session` below is a thin DB-aware
    wrapper around this.

    `video_path`/`model_dir` are only required if the 'extract'/'detect'
    stages actually need to run (i.e. aren't already cached) -- tests that
    only exercise the stub stages, or that pre-seed extract/detect's
    manifests, can omit them, same as before Phase 3 added real stages here.

    `on_stage_start`, if given, is called with the stage name before each
    stage runs (process_session uses it to update processing_job.stage in
    the DB; tests can pass a list.append to record the call order).
    """
    stage_hashes = compute_stage_hashes(params)

    for stage in STAGE_ORDER:
        if on_stage_start:
            on_stage_start(stage)

        stage_dir = job_dir / stage
        if is_stage_complete(stage_dir, stage_hashes[stage]):
            logger.info("job %s: stage '%s' already complete, skipping", job_id, stage)
            continue

        if stage == "extract":
            if video_path is None:
                raise ValueError("run_all_stages: 'extract' stage needs video_path, none given")
            item_count = _run_extract_stage(stage_dir, video_path, params, job_id)
        elif stage == "detect":
            if model_dir is None:
                raise ValueError("run_all_stages: 'detect' stage needs model_dir, none given")
            item_count = _run_detect_stage(stage_dir, job_dir / "extract", params, model_dir, job_id)
        else:
            item_count = _run_stub_stage(stage, stage_dir, job_id)

        write_stage_manifest(stage_dir, stage, stage_hashes[stage], item_count)


def process_session(job_id: int) -> None:
    engine = get_engine()
    processing_job = table("processing_job")
    video_upload = table("video_upload")

    with engine.begin() as conn:
        row = conn.execute(select(processing_job).where(processing_job.c.id == job_id)).mappings().first()
        if row is None:
            raise ValueError(f"No processing_job with id={job_id}")

        video_row = conn.execute(
            select(video_upload).where(video_upload.c.id == row["video_upload_id"])
        ).mappings().first()
        if video_row is None:
            raise ValueError(f"processing_job {job_id} references missing video_upload_id={row['video_upload_id']}")

        params = PipelineParams(**json.loads(row["params_json"]))
        conn.execute(
            update(processing_job)
            .where(processing_job.c.id == job_id)
            .values(state="running", started_at=datetime.now(timezone.utc))
        )

    job_dir = Path(settings.job_data_dir) / str(job_id)
    video_path = Path(video_row["storage_uri"])
    model_dir = Path(settings.insightface_home)
    current_stage = {"name": None}

    def on_stage_start(stage: str) -> None:
        current_stage["name"] = stage
        with engine.begin() as conn:
            conn.execute(update(processing_job).where(processing_job.c.id == job_id).values(stage=stage))

    try:
        run_all_stages(
            job_dir, params, job_id,
            on_stage_start=on_stage_start,
            video_path=video_path,
            model_dir=model_dir,
        )

    except Exception as exc:
        logger.error(
            "job %s: FAILED at stage '%s': %s\n%s",
            job_id, current_stage["name"], exc, traceback.format_exc(),
        )
        with engine.begin() as conn:
            conn.execute(
                update(processing_job)
                .where(processing_job.c.id == job_id)
                .values(
                    state="failed",
                    error_text=f"stage={current_stage['name']}: {exc}",
                    finished_at=datetime.now(timezone.utc),
                )
            )
        raise  # RQ must see this job as failed -- never swallow it (global brief: fail loudly)

    with engine.begin() as conn:
        conn.execute(
            update(processing_job)
            .where(processing_job.c.id == job_id)
            .values(state="succeeded", finished_at=datetime.now(timezone.utc))
        )
