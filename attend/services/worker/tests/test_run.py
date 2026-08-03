"""Phase 2 deliverable 7: "params_hash invalidation cascade" and
"orchestrator skips completed stages." Both exercised on run_all_stages,
which touches only the filesystem (a tmp_path), never a database -- see its
docstring for why that split exists.

Phase 3 made 'extract' and 'detect' real (ffmpeg / SCRFD) instead of stubs;
Phase 4 made 'quality' and 'align' real too. Tests exercising the FULL stage
loop monkeypatch pipeline.run.extract_frames and
pipeline.run.detect_all_frames with fakes (quality/align are left REAL,
since they're cheap pandas/numpy operations over whatever detect wrote --
this file is about orchestration logic (hashing, skip-if-complete), not
about whether ffmpeg or a real detector model works, which needs Mac-side
Docker with real media/models anyway.
"""

import pandas as pd
import pytest
from dataclasses import replace
from pathlib import Path

from pipeline.detect import DETECTION_COLUMNS
from pipeline.extract import FrameManifest
from pipeline.params import PipelineParams
from pipeline.run import STAGE_ORDER, compute_stage_hashes, is_stage_complete, run_all_stages


@pytest.fixture(autouse=True)
def _fake_extract_and_detect(monkeypatch):
    """Every run_all_stages call in this file goes through the full stage
    loop, including 'extract'/'detect' -- fake both out so tests don't need
    a real video file or a real ONNX model directory, only what's under
    test here: the orchestration/caching logic around them. 'quality'/
    'align' are left real (Phase 4): fake_detect_all_frames writes a
    correctly-shaped but EMPTY detections.parquet (0 rows, full column
    schema -- see DETECTION_COLUMNS), so the real quality/align stages run
    against a genuinely empty, faceless "video" and produce a clean 0-crop
    result instead of erroring on a missing/malformed file.
    """

    def fake_extract_frames(video_path, out_dir, fps):
        out_dir.mkdir(parents=True, exist_ok=True)
        return FrameManifest(frame_dir=out_dir, frame_count=0, fps=fps, source_width=0, source_height=0)

    class _FakeDetectionSummary:
        total_frames = 0
        total_detections = 0
        detections_per_frame_mean = 0.0
        detections_per_frame_min = 0
        detections_per_frame_max = 0
        face_width_histogram: dict = {}
        parquet_path = None

    def fake_detect_all_frames(frame_dir, out_dir, params, model_dir, fps):
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=DETECTION_COLUMNS).to_parquet(out_dir / "detections.parquet")
        return _FakeDetectionSummary()

    monkeypatch.setattr("pipeline.run.extract_frames", fake_extract_frames)
    monkeypatch.setattr("pipeline.run.detect_all_frames", fake_detect_all_frames)


FAKE_VIDEO_PATH = Path("/tmp/fake-video-for-tests.mp4")
FAKE_MODEL_DIR = Path("/tmp/fake-model-dir-for-tests")


def test_changing_a_downstream_param_only_invalidates_that_stage_onward():
    base = PipelineParams()
    changed = replace(base, cluster_eps=0.50)  # a "cluster" stage field

    base_hashes = compute_stage_hashes(base)
    changed_hashes = compute_stage_hashes(changed)

    upstream_stages = ["extract", "detect", "quality", "align", "embed"]
    downstream_stages = ["cluster", "match"]

    for stage in upstream_stages:
        assert base_hashes[stage] == changed_hashes[stage], f"{stage} should NOT be invalidated"
    for stage in downstream_stages:
        assert base_hashes[stage] != changed_hashes[stage], f"{stage} SHOULD be invalidated"


def test_changing_an_upstream_param_invalidates_everything_from_there_on():
    base = PipelineParams()
    changed = replace(base, detector_score_min=0.75)  # a "detect" stage field

    base_hashes = compute_stage_hashes(base)
    changed_hashes = compute_stage_hashes(changed)

    assert base_hashes["extract"] == changed_hashes["extract"]  # before detect: untouched
    for stage in ["detect", "quality", "align", "embed", "cluster", "match"]:  # detect onward: all change
        assert base_hashes[stage] != changed_hashes[stage], f"{stage} SHOULD be invalidated"


def test_run_all_stages_skips_already_complete_stages(tmp_path):
    params = PipelineParams()
    job_dir = tmp_path / "job-1"

    started_first_run: list[str] = []
    run_all_stages(
        job_dir, params, job_id=1, on_stage_start=started_first_run.append,
        video_path=FAKE_VIDEO_PATH, model_dir=FAKE_MODEL_DIR,
    )
    assert started_first_run == STAGE_ORDER
    for stage in STAGE_ORDER:
        assert is_stage_complete(job_dir / stage, compute_stage_hashes(params)[stage])

    # Second run with IDENTICAL params: every stage should be seen as
    # already complete. on_stage_start still fires (the orchestrator always
    # checks each stage), but no stage work / manifest rewrite should be
    # needed -- verified indirectly by checking the manifests' completed_at
    # timestamps don't move.
    import json
    first_run_completed_at = {
        stage: json.loads((job_dir / stage / "manifest.json").read_text())["completed_at"]
        for stage in STAGE_ORDER
    }

    started_second_run: list[str] = []
    run_all_stages(
        job_dir, params, job_id=1, on_stage_start=started_second_run.append,
        video_path=FAKE_VIDEO_PATH, model_dir=FAKE_MODEL_DIR,
    )
    assert started_second_run == STAGE_ORDER

    second_run_completed_at = {
        stage: json.loads((job_dir / stage / "manifest.json").read_text())["completed_at"]
        for stage in STAGE_ORDER
    }
    assert first_run_completed_at == second_run_completed_at, "no stage should have re-run"


def test_run_all_stages_reruns_only_invalidated_stages_after_param_change(tmp_path):
    job_dir = tmp_path / "job-2"

    base_params = PipelineParams()
    run_all_stages(job_dir, base_params, job_id=2, video_path=FAKE_VIDEO_PATH, model_dir=FAKE_MODEL_DIR)

    import json
    before = {
        stage: json.loads((job_dir / stage / "manifest.json").read_text())
        for stage in STAGE_ORDER
    }

    changed_params = replace(base_params, match_threshold=0.5)  # "match" stage only
    run_all_stages(job_dir, changed_params, job_id=2, video_path=FAKE_VIDEO_PATH, model_dir=FAKE_MODEL_DIR)

    after = {
        stage: json.loads((job_dir / stage / "manifest.json").read_text())
        for stage in STAGE_ORDER
    }

    for stage in ["extract", "detect", "quality", "align", "embed", "cluster"]:
        assert before[stage]["completed_at"] == after[stage]["completed_at"], f"{stage} should not have re-run"
    assert before["match"]["params_hash"] != after["match"]["params_hash"]


def test_run_all_stages_requires_video_path_when_extract_not_cached(tmp_path):
    with pytest.raises(ValueError, match="video_path"):
        run_all_stages(tmp_path / "job-3", PipelineParams(), job_id=3, model_dir=FAKE_MODEL_DIR)


def test_run_all_stages_requires_model_dir_when_detect_not_cached(tmp_path):
    with pytest.raises(ValueError, match="model_dir"):
        run_all_stages(tmp_path / "job-4", PipelineParams(), job_id=4, video_path=FAKE_VIDEO_PATH)
