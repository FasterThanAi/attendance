# Attend

A classroom attendance system: a teacher records a ~60s 4K video panning
across the room, the system detects and clusters faces, matches them against
an enrolled student gallery, and produces a draft attendance list for the
teacher to confirm. See `Attend-Classroom-Video-Attendance-Roadmap.pdf` (in
the parent folder) for the full 11-phase engineering plan this repo follows.

This repo currently implements **Phase 0 (Foundations and consent)**,
**Phase 1 (Enrollment and gallery construction)**,
**Phase 2 (Upload and orchestration)**, and
**Phase 3 (Frames and detection)**.

## What's here

```
attend/
  apps/web/              Next.js frontend (placeholder, starts Phase 2)
  services/api/          FastAPI backend
    app/
      main.py            entrypoint, request-id middleware, /health
      config.py           all env-driven settings (one place, per the brief)
      db/models.py        full 16-table schema
      db/session.py        async engine/session
      routers/health.py
      schemas/            typed Pydantic request/response models
      services/consent.py  assert_consent_valid() -- the consent gate
    alembic/              async-configured migrations
    tests/               pytest + in-memory SQLite fixture
  services/worker/        RQ worker
    pipeline/
      params.py            PipelineParams -- every tunable, one place
      extract.py           ffmpeg frame sampling
      detect.py            SCRFD face detection: Phase 1's non-tiled path,
                            plus Phase 3's tiled path (compute_tile_grid,
                            non_max_suppression, detect_faces_tiled,
                            detect_all_frames -> detections.parquet)
      align.py              5-point ArcFace alignment to 112x112
      quality.py            blur/brightness/yaw/pitch (Phase 1 subset; Phase 4
                            adds the full tunable-weight composite gate)
      embed.py              ArcFace r100 embedding via onnxruntime
      run.py                orchestrator -- extract/detect are now REAL
                            (Phase 3); quality/align/embed/cluster/match
                            still stubs (Phases 4-6)
      cluster.py, match.py  still stubs (Phases 5-6)
    db.py                  worker's own sync DB access (reflects tables live --
                          see its docstring for why this isn't shared with api)
    consent.py             sync consent gate (mirrors app/services/consent.py)
    enrollment.py           the Phase 1 job: pose_bucket, process_enrollment_
                          video (pure), enroll_student (DB-writing orchestrator)
  docker-compose.yml      postgres16 + redis7 + api + worker
  docs/CONSENT.md         plain-English consent form (Tamil section removed --
                          not needed for this project's students)
  eval/scripts/
    gallery_sanity.py      run this after enrolling students -- checks whether
                          your enrollment data can work at all before you
                          build anything else on top of it
    draw_detections.py     Phase 3: draws detection boxes + face_width_px
                          onto sampled frames from a job's detections.parquet,
                          so you can eyeball tiling/NMS/coverage
```

## How to verify (Mac)

Prerequisites: Docker Desktop, and that's actually it for this phase --
everything else runs inside containers.

```bash
cd attend
cp .env.example .env        # fill in S3_ACCESS_KEY/S3_SECRET_KEY if you've
                             # already got MinIO or Supabase Storage set up;
                             # empty is fine for now, storage isn't used until
                             # Phase 2
docker-compose up --build
```

Expected: four containers come up (postgres, redis, api, worker), with
postgres and redis reporting healthy before api/worker start.

In a second terminal, apply the migration and hit the health endpoint:

```bash
docker-compose exec api alembic upgrade head
curl http://localhost:8000/health
```

Expected `/health` response: `{"service":"attend-api","version":"0.1.0","environment":"development","database":{"ok":true,...},"redis":{"ok":true,...}}`

Run the test suite (uses an in-memory SQLite DB, not the Postgres container,
so it's fast and needs no setup beyond the installed requirements):

```bash
docker-compose exec api pytest -v
```

Expected: all tests in `tests/test_models.py` (one per table) and
`tests/test_consent.py` (consent gate) pass.

## Definition of done (from the roadmap, Phase 0)

- [x] `docker-compose up` starts cleanly, with healthchecks
- [x] `alembic upgrade head` applies (initial migration covers all 16 tables)
- [x] `/health` returns 200 with db + redis connectivity
- [ ] You have a signed consent form template — `docs/CONSENT.md` is the
      template; it still needs your institution's name/contact filled in,
      the Tamil section translated by a fluent speaker, and your supervisor
      to actually see it. That last part is on you, not something I can do.

## What I couldn't verify from here (Phase 0)

This was built and syntax-checked (`python -m py_compile` on every file,
`docker-compose.yml` parsed and structurally checked) in a sandbox with no
Docker and no PyPI access, so it has **not** been run end-to-end against a
real Postgres container yet. Run the "How to verify" steps above on your Mac
before treating Phase 0 as done — if `docker-compose up` or `alembic upgrade
head` surface an issue, tell me the error and I'll fix it directly.

## Phase 1: how to verify

The requirements files changed (worker gained numpy/opencv/onnxruntime/
insightface/psycopg2; api gained `rq`), so rebuild before testing:

```bash
docker-compose exec api alembic upgrade head   # applies 0002_enrollment_additions
docker-compose up --build
```

Run the worker's own test suite (the pure-logic tests -- these don't need a
real ONNX model or a real video):

```bash
docker-compose exec worker pytest -v
```

Expected: `test_align.py`, `test_pose.py`, `test_quality.py`, `test_embed.py`,
`test_consent.py` all pass. I ran the equivalent logic manually in my sandbox
(numpy/opencv were available there even without full pytest) and confirmed
correct behavior for alignment, blur/brightness/pose-estimation math, and
embedding normalisation -- so this is lower-risk than Phase 0's endpoints,
but still not run through the real ONNX models.

Then try a real enrollment end to end: create a student + consent row, record
yourself turning your head for 5 seconds, and POST it:

```bash
curl -X POST http://localhost:8000/students/1/enrollment -F "video=@your_clip.mp4"
curl http://localhost:8000/students/1/enrollment          # check status after a minute
python eval/scripts/gallery_sanity.py                       # after a few students are enrolled
```

## Phase 1 status: verified on real hardware

Confirmed on your Mac: insightface buffalo_l loads in Docker, a real 5s
enrollment video produced 8 embeddings across 3 poses (3/3/2), and
`gallery_sanity.py` showed 0.804 within-student similarity (target: >0.6),
with all worker tests passing. Both flagged unknowns turned out to be real
and were fixed: the recognition `get_feat` call needed a list of crops
rather than a batched array, and both model-loader functions needed a
fallback to insightface's actual default download path.

## Phase 2: what's new

- `POST/GET /sessions` -- minimal class-session creation. Not an explicit
  roadmap deliverable; added because upload needs a real `class_session_id`
  to attach to and nothing before this point created one.
- `POST /uploads`, `PUT /uploads/{id}/chunks/{n}`, `GET /uploads/{id}`,
  `POST /uploads/{id}/complete` -- resumable 5MB chunked upload. Chunk state
  lives on disk under `/data/jobs/uploads/{upload_id}/`, not in the DB.
  `/complete` assembles, validates with ffprobe (duration 20s-5min, shorter
  side >=1080px), creates the `video_upload` row, then runs the pre-flight
  check.
- `services/worker/pipeline/preflight.py` -- sharpness, backlighting, pan
  detection/speed, face yield, coverage, on 18 evenly-sampled frames. Runs as
  a worker RQ job that the api polls synchronously for up to 28 seconds, so
  ML code stays out of the api image but the teacher still gets an answer
  before leaving the classroom.
- `services/worker/pipeline/run.py` -- `process_session(job_id)`, the real
  orchestrator. Stages (`extract, detect, quality, align, embed, cluster,
  match`) are stubs per the Phase 2 spec (logging + an empty manifest);
  Phases 3-6 fill them in one at a time. The manifest/invalidation logic is
  real: each stage's cache key is a hash chained from its own relevant
  params plus every upstream stage's hash, so changing e.g. `match_threshold`
  only invalidates `match` (not `extract` through `embed`), while changing
  `detector_score_min` invalidates `detect` onward.
- `POST /sessions/{id}/process`, `GET /jobs/{id}` -- enqueue + status.
- `apps/web` -- Next.js 14 App Router scaffold with a working `/record`
  upload flow: 5MB chunking, exponential backoff per chunk, and resume via
  `localStorage` (keyed by filename+size+lastModified) so re-selecting the
  same file after closing the tab picks up from the server's reported
  progress instead of restarting.

## Phase 2: how to verify

```bash
docker-compose exec api alembic upgrade head   # no new migration this phase, just confirming current
docker-compose up --build                       # api gained ffmpeg + rq; worker unchanged deps

docker-compose exec worker pytest -v            # test_run.py: invalidation cascade, skip-if-complete
docker-compose exec api pytest -v               # test_upload_service.py: chunk idempotency, resume, missing-chunk rejection
```

I ran both of these test suites' underlying logic manually against the real
code in my sandbox (stubbing out sqlalchemy/pydantic import-time
dependencies, since neither installs here) and confirmed correct behavior:
the hash-chaining cascade genuinely only invalidates a changed stage and
whatever comes after it, and the chunked-upload service correctly reports
partial progress and rejects missing chunks. What I could NOT verify: any
of this running through actual HTTP requests, a real Postgres, real RQ
workers picking up jobs, or the frontend actually compiling (no npm registry
access here either -- see `apps/web/README.md`).

Try the real end-to-end flow:

```bash
# create a session (need an existing course_id/teacher_id from your DB)
curl -X POST http://localhost:8000/sessions -H "Content-Type: application/json" \
  -d '{"course_id": 1, "teacher_id": 1, "scheduled_at": "2026-08-04T09:00:00Z", "room": "204"}'

cd apps/web && npm install && cp .env.local.example .env.local && npm run dev
# open http://localhost:3000/record, set the session id, upload a video
```

## Phase 3: what's new

- `pipeline/params.py` -- 4 new tiling fields: `tile_trigger_long_side_px`
  (2000), `tile_size_px` (1280), `tile_overlap_px` (256),
  `nms_iou_threshold` (0.4). Added to `run.py`'s `STAGE_PARAM_FIELDS["detect"]`
  and api's `DEFAULT_PIPELINE_PARAMS` too -- verified programmatically that
  the two stay in sync (see "How to verify" below).
- `pipeline/detect.py` -- the actual tiled-detection deliverable:
  - `compute_tile_grid(width, height, tile_size, overlap)` -- splits a frame
    into overlapping tiles, with the last tile on each axis pinned to the
    frame's far edge so there's never an uncovered strip.
  - `non_max_suppression(detections, iou_threshold)` -- merges duplicate
    detections of the same face coming from adjacent tiles, or from the tile
    pass vs. the whole-frame-downscaled pass.
  - `detect_faces_tiled(image, model, params)` -- the full strategy: if a
    frame's long side exceeds `tile_trigger_long_side_px`, tile it and detect
    per-tile at native resolution (mapping tile-local coordinates back to
    frame space), PLUS one whole-frame pass downscaled to `tile_size_px` (to
    catch large front-row faces a tile boundary might cut through), then
    merge everything with NMS. Small frames (enrollment selfies, pre-flight
    samples) skip tiling entirely.
  - `detect_all_frames(frame_dir, out_dir, params, model_dir, fps)` -- runs
    tiled detection over every extracted frame using a
    `multiprocessing.Pool` (up to 4 workers, one detector load per worker
    process via an initializer -- loading a fresh ONNX session per frame is
    "the single most common way to make this stage 10-20x slower than it
    needs to be"), writes `detections.parquet` (columns: `frame_index`,
    `frame_timestamp_s`, `det_id`, `x1/y1/x2/y2`, `score`, 10 landmark
    columns, `face_width_px`).
- `services/worker/requirements.txt` -- added `pandas`/`pyarrow` for the
  parquet output.
- `eval/scripts/draw_detections.py` -- standalone debug script: draws boxes +
  `face_width_px`/score labels onto a sample of frames from a job's
  `detections.parquet`, so you can eyeball whether tiling actually found the
  back row, whether NMS left duplicate boxes on one face, and whether tile
  seams left gaps.
- `pipeline/run.py` -- `extract` and `detect` are now REAL stages, not stubs.
  `process_session` looks up the job's `video_upload.storage_uri` and passes
  it (plus `settings.insightface_home` as the model directory) into
  `run_all_stages`, which now raises loudly if either stage needs to run but
  wasn't given a `video_path`/`model_dir` (rather than silently doing
  nothing). `quality`, `align`, `embed`, `cluster`, `match` are still stubs.

## Phase 3: how to verify

```bash
docker-compose exec worker pip install -r requirements.txt  # picks up pandas/pyarrow
docker-compose up --build

docker-compose exec worker pytest -v tests/test_detect_tiling.py tests/test_run.py
```

Expected: `test_detect_tiling.py` (tile-grid coverage/overlap, NMS merge
across a tile boundary, tile-local-to-frame coordinate mapping) and
`test_run.py` (hash-invalidation cascade, skip-if-complete, plus two new
guard-clause tests for missing `video_path`/`model_dir`) all pass.

I ran the equivalent logic directly in my sandbox (real numpy/opencv, a fake
detector standing in for the real SCRFD ONNX model, and stubbed
sqlalchemy/config/db/pytest modules for `test_run.py`'s imports) and
confirmed all of it: the tile grid has no gaps and the requested overlap;
NMS correctly merges two overlapping detections into the higher-scoring one
and leaves distinct ones alone; a face placed near the far edge of a 3000x1800
synthetic frame is detected once (tile pass + whole-frame pass merged) with
its box within a few pixels of ground truth; and `run_all_stages` still only
invalidates the right stages after a param change, now with the real
extract/detect calls monkeypatched out. I also confirmed
`DEFAULT_PIPELINE_PARAMS` (api) and `PipelineParams` (worker) have exactly
matching field sets.

What I could NOT verify: real SCRFD tiled detection against an actual 4K
classroom video (no insightface/onnxruntime installed here, and no real
footage) -- `draw_detections.py` is exactly the tool to eyeball that once you
run a real upload through Docker. Also unverified: `multiprocessing.Pool`
behavior under real load (process count, pickling `PipelineParams` across
worker processes), and the parquet write/read round-trip with real pandas
(pandas import itself works in my sandbox, but I haven't run
`detect_all_frames` end-to-end against real frames).

## Next: Phase 4

Quality gate and alignment, wired into `run.py`'s `quality`/`align` stages
for classroom video (both already exist and are proven correct for
enrollment's use in Phase 1 -- Phase 4 is about applying them to the
`detections.parquet` this phase produces, plus building the full
tunable-weight composite quality score the roadmap describes, beyond
Phase 1's simpler `simple_quality_score`). Say "start phase 4" when Phase 3
is verified on your machine.
