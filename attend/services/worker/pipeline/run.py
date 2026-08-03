"""Job orchestration (Phase 2 deliverable 3-4; Phase 3 wired in real
extract/detect; Phase 4 wired in real quality/align; Phase 5 wired in real
embed/cluster; Phase 6 wires in the real match stage).

`process_session` is the single RQ entrypoint for the classroom-video
pipeline. It runs each stage in order, skipping any stage whose on-disk
manifest already matches the current params (so re-running match calibration
a hundred times during Phase 7 doesn't re-extract frames every time -- see
non-negotiable rule #2), and fails loudly with the stage name and traceback
on any exception (rule: "a job that fails visibly is far better than a job
that quietly marks students absent").

Per the Phase 2 prompt, ALL stages started as stubs (log + write an empty
manifest). Phase 3 replaced extract/detect, Phase 4 replaced quality/align,
Phase 5 replaced embed/cluster, and Phase 6 replaces `match`
(pipeline.match.run_match_stage) -- every stage in STAGE_ORDER is now a real
implementation.

`match` is the one stage that is NOT purely file-based (job_dir in, job_dir
out): it needs the session's class_session_id/course enrollment/gallery
vectors from the live DB, and writes detected_cluster/cluster_match rows
directly rather than a job_dir file -- see pipeline/match.py's module
docstring for why its DB access is deferred-imported rather than module-level.
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
from pipeline.align import run_align_stage
from pipeline.cluster import run_cluster_stage
from pipeline.detect import detect_all_frames
from pipeline.embed import run_embed_stage
from pipeline.extract import extract_frames
from pipeline.match import run_match_stage
from pipeline.params import PipelineParams
from pipeline.quality import run_quality_stage

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
    "quality": [
        "min_face_px", "max_abs_yaw_deg", "max_abs_pitch_deg", "blur_laplacian_min",
        "brightness_min", "brightness_max",
        "quality_weight_size", "quality_weight_blur", "quality_weight_frontality",
    ],
    "align": ["embed_input_size"],
    "embed": ["embed_batch_size"],
    "cluster": [
        "cluster_eps", "cluster_min_samples", "cluster_merge_distance_factor", "temporal_coherence_enabled",
        "temporal_overlap_min_fraction", "cluster_split_frame_span_fraction",
        "cluster_split_tightness_max", "cluster_split_eps_factor",
    ],
    "match": [
        "match_threshold", "match_margin_min", "uncertain_band",
        "session_health_poor_coverage_percent", "session_health_fair_coverage_percent",
        "session_health_poor_mean_similarity",
    ],
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
    """Phase 2's stub, still used for embed/cluster/match until Phases 5-6
    replace them one at a time.
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


def _run_quality_stage(stage_dir: Path, detections_parquet: Path, frame_dir: Path, params: PipelineParams, job_id: int) -> int:
    logger.info("job %s: stage 'quality' running (detections=%s)", job_id, detections_parquet)
    summary = run_quality_stage(detections_parquet, frame_dir, stage_dir, params)
    logger.info(
        "job %s: stage 'quality' done -- %s input, %s accepted, %s rejected (%s)",
        job_id, summary.input_count, summary.accepted_count, summary.rejected_count, summary.reject_reason_counts,
    )
    return summary.accepted_count


def _run_align_stage(stage_dir: Path, quality_parquet: Path, frame_dir: Path, params: PipelineParams, job_id: int) -> int:
    logger.info("job %s: stage 'align' running (quality=%s)", job_id, quality_parquet)
    result = run_align_stage(quality_parquet, frame_dir, stage_dir, params)
    logger.info("job %s: stage 'align' done -- %s crop(s) aligned", job_id, result.count)
    return result.count


def _run_embed_stage(stage_dir: Path, aligned_npy: Path, params: PipelineParams, model_dir: Path, job_id: int) -> int:
    logger.info("job %s: stage 'embed' running (aligned=%s)", job_id, aligned_npy)
    result = run_embed_stage(aligned_npy, stage_dir, params, model_dir)
    logger.info("job %s: stage 'embed' done -- %s crop(s) embedded", job_id, result.count)
    return result.count


def _run_cluster_stage(
    stage_dir: Path, embeddings_npy: Path, aligned_npy: Path, aligned_index_parquet: Path,
    quality_parquet: Path, params: PipelineParams, job_id: int,
) -> int:
    logger.info("job %s: stage 'cluster' running (embeddings=%s)", job_id, embeddings_npy)
    summary = run_cluster_stage(embeddings_npy, aligned_npy, aligned_index_parquet, quality_parquet, stage_dir, params)
    logger.info(
        "job %s: stage 'cluster' done -- %s clusters, %s noise, %s merges, %s split decisions",
        job_id, summary.cluster_count, summary.noise_count, summary.merge_count, len(summary.split_log),
    )
    return summary.cluster_count


def _run_match_stage(
    cluster_summary_parquet: Path, class_session_id: int, processing_job_id: int, params: PipelineParams, job_id: int,
) -> int:
    logger.info(
        "job %s: stage 'match' running (class_session_id=%s, cluster_summary=%s)",
        job_id, class_session_id, cluster_summary_parquet,
    )
    summary = run_match_stage(cluster_summary_parquet, class_session_id, params, processing_job_id)
    logger.info(
        "job %s: stage 'match' done -- %s clusters (%s confident, %s uncertain, %s unmatched), "
        "session_health=%s, coverage=%.1f%%",
        job_id, summary.cluster_count, summary.confident_count, summary.uncertain_count, summary.unmatched_count,
        summary.session_summary.session_health, summary.session_summary.coverage_percent,
    )
    return summary.cluster_count


def run_all_stages(
    job_dir: Path,
    params: PipelineParams,
    job_id: int,
    on_stage_start: Callable[[str], None] | None = None,
    video_path: Path | None = None,
    model_dir: Path | None = None,
    class_session_id: int | None = None,
    processing_job_id: int | None = None,
) -> None:
    """The actual stage loop: no database, no RQ, no network -- just the
    filesystem under `job_dir`, `params`, and (for extract/detect/embed)
    `video_path`/`model_dir`. Independently testable per non-negotiable
    rule #1; `process_session` below is a thin DB-aware wrapper around this.

    quality/align/cluster need no extra arguments beyond `job_dir` -- they
    read the previous stage's own output straight from the filesystem
    (`job_dir/detect/detections.parquet`, `job_dir/quality/quality.parquet`,
    `job_dir/align/aligned.npy`, `job_dir/embed/embeddings.npy`), the same
    convention every stage here follows: read the previous stage's
    directory, write your own. `embed` is the exception among the "no extra
    args" group because, like detect, it needs to load an ONNX model.

    `video_path`/`model_dir` are only required if the 'extract'/'detect'/
    'embed' stages actually need to run (i.e. aren't already cached);
    `class_session_id`/`processing_job_id` are only required if 'match'
    actually needs to run -- tests that pre-seed those stages' manifests
    can omit the corresponding argument, same as before Phase 3 added real
    stages here.

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
        elif stage == "quality":
            item_count = _run_quality_stage(
                stage_dir, job_dir / "detect" / "detections.parquet", job_dir / "extract", params, job_id
            )
        elif stage == "align":
            item_count = _run_align_stage(
                stage_dir, job_dir / "quality" / "quality.parquet", job_dir / "extract", params, job_id
            )
        elif stage == "embed":
            if model_dir is None:
                raise ValueError("run_all_stages: 'embed' stage needs model_dir, none given")
            item_count = _run_embed_stage(stage_dir, job_dir / "align" / "aligned.npy", params, model_dir, job_id)
        elif stage == "cluster":
            item_count = _run_cluster_stage(
                stage_dir,
                job_dir / "embed" / "embeddings.npy",
                job_dir / "align" / "aligned.npy",
                job_dir / "align" / "aligned_index.parquet",
                job_dir / "quality" / "quality.parquet",
                params, job_id,
            )
        elif stage == "match":
            if class_session_id is None or processing_job_id is None:
                raise ValueError(
                    "run_all_stages: 'match' stage needs class_session_id and processing_job_id, none given"
                )
            item_count = _run_match_stage(
                job_dir / "cluster" / "cluster_summary.parquet", class_session_id, processing_job_id, params, job_id,
            )
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
            class_session_id=row["class_session_id"],
            processing_job_id=job_id,
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
