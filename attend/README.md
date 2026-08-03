# Attend

A classroom attendance system: a teacher records a ~60s 4K video panning
across the room, the system detects and clusters faces, matches them against
an enrolled student gallery, and produces a draft attendance list for the
teacher to confirm. See `Attend-Classroom-Video-Attendance-Roadmap.pdf` (in
the parent folder) for the full 11-phase engineering plan this repo follows.

This repo currently implements **Phase 0 (Foundations and consent)**,
**Phase 1 (Enrollment and gallery construction)**,
**Phase 2 (Upload and orchestration)**,
**Phase 3 (Frames and detection)**,
**Phase 4 (Quality gate and alignment)**, and
**Phase 5 (Embedding and clustering)**.

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
      align.py              5-point ArcFace alignment to 112x112 (Phase 1's
                            align_face); Phase 4's align_crops batches this
                            over a whole video into aligned.npy (memmap) +
                            aligned_index.parquet
      quality.py            Phase 1's per-crop blur/brightness/yaw/pitch
                            helpers, plus Phase 4's score_detections (the
                            full tunable-weight composite gate + accept/
                            reject rules over a whole detections.parquet)
      embed.py              ArcFace r100 embedding via onnxruntime: Phase 1's
                            per-crop embed_batch, plus Phase 5's batch
                            embed_aligned (embeddings.npy memmap, throughput
                            logging)
      cluster.py             Phase 5: DBSCAN over embeddings (cosine metric),
                            quality-weighted cluster representatives,
                            per-cluster diagnostics, best-crop selection,
                            and a temporal-coherence merge/split post-pass
                            exploiting the fact that the camera pans
      run.py                orchestrator -- extract/detect (Phase 3),
                            quality/align (Phase 4), and embed/cluster
                            (Phase 5) are now REAL; only match is still a
                            stub (Phase 6)
      match.py               still a stub (Phase 6)
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
    contact_sheet.py        Phase 4: grid image of N accepted + N rejected
                          crops (labelled with quality score / reject_reason)
                          from a job's quality.parquet, to eyeball whether
                          the quality gate is calibrated sensibly
    cluster_report.py       Phase 5: cluster count/noise, size + tightness
                          distributions, and one contact-sheet-per-cluster
                          image (up to 12 members) -- the main learning
                          artifact of this phase
    sweep_cluster.py         Phase 5: re-runs ONLY clustering across a
                          cluster_eps x cluster_min_samples grid, reusing
                          cached embeddings.npy (never re-running detect/
                          embed); reports cluster count and purity if you
                          supply a ground-truth CSV
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
  nothing).
- **Gap found and fixed while building Phase 4**: the roadmap's own
  detections.parquet column spec includes `tile_origin_x`/`tile_origin_y`,
  which the original Phase 3 build omitted. Added a `tile_origin` field to
  `Detection`, set on tile-pass detections and left `None` (written as a
  `-1` sentinel) on whole-frame-pass detections. Also hardened
  `detect_all_frames` to always write the full column schema even when a
  video produces zero detections (previously `pd.DataFrame([])` had no
  columns at all, which would have broken Phase 4's `quality` stage on a
  genuinely faceless video).

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

## Phase 4: what's new

- `pipeline/params.py` -- 3 new composite-quality-score weight fields:
  `quality_weight_size` (0.4), `quality_weight_blur` (0.3),
  `quality_weight_frontality` (0.3). Added to `run.py`'s
  `STAGE_PARAM_FIELDS["quality"]` and api's `DEFAULT_PIPELINE_PARAMS` too --
  verified programmatically that `PipelineParams` and `DEFAULT_PIPELINE_PARAMS`
  still have exactly matching field sets.
- `pipeline/quality.py` -- the actual quality-gate deliverable:
  - `composite_quality_score(face_width_px, blur, yaw_deg, pitch_deg, params)`
    -- a WEIGHTED combination of normalised size/blur/frontality (frontality
    considers both yaw and pitch, each normalised against its own
    accept/reject bound, unlike Phase 1's yaw-only `simple_quality_score`,
    which now shares the same `FACE_WIDTH_NORM_PX`/`BLUR_NORM` constants so
    the two scoring functions can't silently drift apart).
  - `score_detections(detections_df, frame_dir, params) -> QualityResult` --
    for every detection: crop it out of its source frame (frames cached per
    `frame_index`, not re-read per detection), compute blur/brightness/pose/
    composite score, and apply the 6-rule reject gate from the roadmap
    (`low_detector_score`, `too_small`, `too_blurred`, `yaw_too_extreme`,
    `pitch_too_extreme`, `bad_brightness`, plus an `invalid_crop` rule of my
    own for a bbox that falls outside the frame). Every row is kept,
    rejected or not, with a `reject_reason` -- per the prompt, "Phase 7
    needs to analyse what was thrown away."
  - `run_quality_stage(...)` -- I/O wrapper: reads `detections.parquet`,
    writes `quality.parquet`, logs accepted/rejected counts by reason plus
    the accepted-crop score distribution.
- `pipeline/align.py` -- `align_crops(accepted_df, frame_dir, out_dir, params)
  -> AlignedSet`: aligns every accepted crop via the same `align_face` Phase 1
  already proved correct, writes `aligned.npy` as a single self-describing
  `(N, 112, 112, 3)` uint8 memmap (via `np.lib.format.open_memmap`, so later
  phases can `np.load(path, mmap_mode="r")` without needing N/dtype/shape
  passed separately) and `aligned_index.parquet` mapping each row back to
  `det_id`/`frame_index`/`quality_score`. `run_align_stage(...)` is the I/O
  wrapper `run.py` actually calls.
- `pipeline/run.py` -- `quality` and `align` are now REAL stages too. Neither
  needs `video_path`/`model_dir`; they just read the previous stage's output
  straight off the filesystem (`job_dir/detect/detections.parquet`,
  `job_dir/quality/quality.parquet`), the same convention every stage here
  follows.
- `eval/scripts/contact_sheet.py` -- Phase 4's own definition of done,
  verbatim: "manually inspecting 30 accepted and 30 rejected crops confirms
  the gate is making sensible calls." This script samples 30+30 by default,
  crops them from the source frames, and writes one labelled grid image.
- **Gap found and fixed from Phase 3** (see above): `tile_origin_x`/
  `tile_origin_y` columns were missing from `detections.parquet`; also
  hardened `detect_all_frames` against a zero-detection video producing a
  columnless (and thus downstream-breaking) DataFrame.

## Phase 4: how to verify

```bash
docker-compose up --build

docker-compose exec worker pytest -v tests/test_quality_gate.py tests/test_align_crops.py tests/test_run.py
```

Expected: `test_quality_gate.py` (each of the 6 reject rules fires on a
crafted input, rejected rows are kept not dropped, blur is scale-invariant
across two crop sizes, composite score weighting), `test_align_crops.py`
(memmap shape/dtype, byte-identical to `align_face`'s own output, index
parquet mapping, empty-input handling), and `test_run.py` (still-correct
hash-invalidation cascade with quality/align now real) all pass.

I ran the equivalent logic directly in my sandbox: real numpy/opencv/pandas
for the quality-gate and alignment tests (all 15 pass against real synthetic
JPEGs and landmark geometry, no mocking needed since `score_detections`
doesn't touch parquet itself). For anything touching actual parquet
read/write (`align_crops`'s index file, `run_quality_stage`,
`run_align_stage`, and `test_run.py`'s full stage loop with quality/align
now real), I don't have pyarrow installed here (no PyPI access, same as
every prior phase) -- I verified the LOGIC by temporarily swapping
`DataFrame.to_parquet`/`pd.read_parquet` for a pickle-based equivalent in my
own verification harness only (never in the shipped code, which still calls
real `to_parquet`/`read_parquet` throughout) and confirmed all of it: the
memmap's row 0 is byte-identical to `align_face` run on the same re-read
(post-JPEG) frame, the index parquet correctly maps row index to `det_id`,
an all-rejected video produces a clean 0-crop `aligned.npy` instead of
crashing, and the full `run_all_stages` loop (extract/detect faked, quality/
align real) still passes all 6 orchestration tests. I also re-ran every
Phase 1-3 worker test to confirm nothing regressed: `test_align` (1),
`test_pose` (11), `test_quality` (3), `test_embed` (3), `test_detect_tiling`
(7) -- 25 tests, all still pass alongside Phase 4's new `test_quality_gate`
(13) and `test_align_crops` (2).

What I could NOT verify: real parquet I/O with actual pyarrow (needs your
Mac's `pip install -r requirements.txt`), and manually inspecting real
accepted/rejected crops from real classroom footage --
`contact_sheet.py` is exactly the tool for that once you have a real job to
point it at. Also worth checking on your end: the roadmap's expectation that
70-80% of detections get rejected and 3000-6000 crops survive from a
90-student video -- I have no real footage to check those numbers against.

## Phase 5: what's new

- `pipeline/params.py` -- 4 new temporal-coherence tuning fields, all
  flagged as ASSUMPTIONS since the roadmap gives numbers for some of these
  and not others: `temporal_overlap_min_fraction` (0.3 -- roadmap says
  "overlap substantially" with no number), `cluster_split_frame_span_fraction`
  (0.6 -- this one IS the roadmap's own number, "roughly 60 percent of the
  total video"), `cluster_split_tightness_max` (0.5 -- "poor tightness" has
  no number given), `cluster_split_eps_factor` (0.7 -- how much tighter the
  split-attempt DBSCAN pass's eps is). Synced into `run.py`'s
  `STAGE_PARAM_FIELDS["cluster"]` and api's `DEFAULT_PIPELINE_PARAMS`.
- `pipeline/embed.py` -- `embed_aligned(aligned_memmap, out_dir, params,
  model_dir)`: batches through the singleton ArcFace model
  `embed_batch_size` crops at a time, asserts BGR uint8 (112,112,3) input
  (the single most common silent bug per the prompt), writes
  `embeddings.npy` as a self-describing `(N,512)` float32 memmap, logs
  crops/sec throughput. `run_embed_stage(...)` is the I/O wrapper.
- `pipeline/cluster.py` -- new file, the intellectual core of the project:
  - `cluster_embeddings(embeddings, quality_df, params) -> ClusterResult`:
    DBSCAN with `metric="cosine"`; noise (`-1`) kept, never discarded; for
    each cluster, a QUALITY-WEIGHTED mean representative (documented
    rationale: a plain mean lets however many bad crops exist pull the
    representative away from the good evidence in proportion to their
    count, not their reliability -- weighting by Phase 4's own quality score
    is the noise-averaging payoff Section 2.2 describes); per-cluster
    diagnostics (member count, mean quality, intra-cluster mean cosine
    similarity as "tightness," frame range); best_crop = highest-quality
    member.
  - The temporal-coherence post-pass (toggleable via
    `temporal_coherence_enabled`, already existed as a placeholder field):
    a merge step (union-find over cluster pairs whose representatives are
    close AND whose frame ranges overlap substantially) and a split step
    (a cluster spanning too much of the video with poor tightness gets a
    second, tighter-eps DBSCAN pass restricted to its own members; logged
    either way, including "flagged but couldn't actually split it").
  - `run_cluster_stage(...)`: the I/O wrapper. Cross-checks that
    `quality.parquet`'s accepted-row `det_id` order matches
    `aligned_index.parquet`'s before trusting embeddings' row order to line
    up with quality scores/frame indices -- fails loudly instead of silently
    misattributing a quality score to the wrong crop if the two stages are
    ever out of sync. Writes `clusters.parquet`, `cluster_summary.parquet`,
    and one best-crop JPEG per cluster.
- `pipeline/run.py` -- `embed` and `cluster` are now real stages. Only
  `match` (Phase 6) remains a stub.
- `eval/scripts/cluster_report.py` -- cluster/noise counts, size and
  tightness distributions, and a contact-sheet image per cluster (up to 12
  members) -- the roadmap's own framing: "the main learning artifact of
  this phase... Looking at them is how you will understand what your
  system is actually doing."
- `eval/scripts/sweep_cluster.py` -- sweeps `cluster_eps` (0.30-0.55, step
  0.02) x `cluster_min_samples` ({2,3,4,5}), reusing the SAME
  `embeddings.npy` for all 20 grid points (never re-running detect/embed --
  that's the entire point of Phase 2's stage caching). Reports cluster
  count/noise, and purity if you supply a `det_id,true_label` ground-truth
  CSV.
- **A real bug found and fixed while testing this phase, not sandbox-specific**:
  `quality_df[quality_df["accepted"]]` (used in `align.py`, `cluster.py`,
  `contact_sheet.py`, and `sweep_cluster.py`) silently returns a
  **columnless** DataFrame -- not just zero rows -- whenever `quality_df` is
  genuinely empty (0 detections), because an empty `"accepted"` column
  round-trips as `object` dtype rather than `bool`, and pandas boolean-masks
  an object-dtype column differently. This would have crashed the `cluster`
  stage with a confusing `KeyError: 'det_id'` on any real video that
  produces zero detections. Fixed everywhere by casting
  `.astype(bool)` before the mask. This is a genuine pandas edge case, not a
  sandbox/pyarrow artifact -- I found it because my own `test_run.py`
  verification exercises exactly this all-empty path.

## Phase 5: how to verify

```bash
docker-compose up --build

docker-compose exec worker pytest -v tests/test_cluster.py tests/test_embed_aligned.py tests/test_run.py
```

Expected: `test_cluster.py` (three well-separated Gaussian blobs -> exactly
3 clusters; noise kept not discarded; quality weighting measurably shifts
the representative toward high-quality members, both via the pure
`_weighted_mean_vector` function and end-to-end through
`cluster_embeddings`; the merge rule fires on constructed overlapping
clusters and does NOT fire when frame ranges don't overlap), `test_embed_
aligned.py` (correct memmap shape/dtype/normalisation, batching actually
splits into `embed_batch_size`-sized chunks, empty input, BGR/shape
assertion), and `test_run.py` (hash-invalidation cascade still correct with
embed/cluster now real) all pass.

I ran the equivalent logic directly in my sandbox: 61 tests total across
every worker test file pass, including all of Phase 5's new ones. Since
neither `scikit-learn` nor `pyarrow` are installed here (no PyPI access,
same constraint every phase), I verified `cluster.py`'s actual logic against
a from-scratch DBSCAN implementation (matching sklearn's documented
semantics: cosine distance, core points via neighbor count including self,
density-reachable expansion, border-point reassignment) written only for
this verification, never shipped -- and against the same pickle-based
parquet stand-in used in Phase 4's verification. This is exactly how I found
the `accepted` boolean-mask bug described above: my hand-rolled DBSCAN
correctly returned all-noise on the empty embeddings `test_run.py` feeds it,
which then hit the real (unmocked) `cluster.py` code path and surfaced the
bug immediately.

What I could NOT verify: real `scikit-learn` DBSCAN's actual clustering
behavior (my stand-in matches its documented semantics but is not the same
code), real ArcFace embedding throughput/quality, and -- the thing that
actually matters here -- whether clustering a real classroom video lands
within the roadmap's "within roughly 20 percent of the true headcount"
target. `cluster_report.py`'s per-cluster contact sheets are the tool for
judging that once you have a real job to point it at; `sweep_cluster.py`
is the tool for retuning `cluster_eps`/`cluster_min_samples` if the first
attempt is off.

## Phase 6: what's new

- `pipeline/match.py` -- new logic, split pure/DB exactly like
  `enrollment.py`'s `process_enrollment_video`/`enroll_student`:
  - `match_clusters(cluster_reps, gallery, params)`: cosine similarity matrix
    (one matrix multiply, both sides already L2-normalised) then the
    **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) for a
    globally-optimal one-to-one cluster-to-student assignment. NOT greedy
    top-1: two clusters can each have their single highest similarity
    pointed at the same gallery student (two people who look alike, or a
    noisy crop embedding slightly toward a third identity) -- greedy would
    double-book that student and stick the other cluster with whatever's
    left, even when a better global arrangement exists. `runner_up_similarity`
    is the best similarity to any OTHER gallery entry for that cluster, not
    the second-best row in the overall solution.
  - Three-band decision (`_decide_band`), verbatim from the roadmap:
    `margin = similarity - runner_up_similarity`; CONFIDENT if
    `similarity >= match_threshold AND margin >= match_margin_min`;
    UNCERTAIN if `similarity >= match_threshold - uncertain_band`;
    UNMATCHED otherwise.
  - `build_session_summary(...)`: proposed_present / needs_review /
    proposed_absent / unrecognised_clusters, `coverage_percent`,
    `mean_confident_similarity`, and a `session_health` rating (good/fair/poor
    -- ASSUMPTION thresholds, see `params.py`, since the roadmap names the
    three poor-health conditions but not numeric cutoffs). Hard invariant,
    enforced by raising (never silently routed around): `proposed_present +
    needs_review + proposed_absent == total_enrolled`, exactly.
  - `run_match_stage(...)`: the DB-aware orchestrator. Inserts one
    `detected_cluster` row per cluster first (so its DB-generated id can be
    used directly as `ClusterMatch.cluster_id` -- no separate id-mapping
    step), queries ONLY this course's `enrollment` for the gallery (never
    the whole institution), calls the pure functions above, inserts
    `cluster_match` rows, reads `video_upload.preflight_status_json` for
    `session_health`'s warnings input, and persists
    `class_session.draft_summary_json` + flips status to
    `awaiting_review`. Does **not** write `attendance_record` rows --
    turning a draft into committed attendance is Phase 8's job.
  - sqlalchemy/config/db imports are deferred (function-local, not
    module-level) specifically so `match_clusters`/`build_session_summary`/
    `build_gallery_matrix` stay importable and unit-testable without the
    full worker DB stack installed -- see the module docstring.
- Two gaps found while building this phase, both fixed:
  - `cluster_summary.parquet` (Phase 5) never actually stored the cluster's
    representative vector, even though Phase 6's own integration contract
    says "Consumes: cluster_summary.parquet with representative vectors."
    Added a `representative_vector` bytes column.
  - Phase 2's pre-flight check result was only ever returned in the
    upload-complete HTTP response, never stored -- but `session_health`
    needs it long after that response was sent. Added
    `video_upload.preflight_status_json`, persisted in
    `routers/upload.py`'s `complete_upload`.
- `services/api/app/db/models.py` -- `DetectedCluster`/`ClusterMatch`/the
  three related enums were already fully specified back in Phase 0 (a
  pleasant surprise); only 2 new nullable columns needed:
  `video_upload.preflight_status_json`, `class_session.draft_summary_json`.
  Migration `0003_phase6_additions.py`.
- `services/worker/db.py` -- reflected-table list extended with
  `enrollment`, `class_session`, `course`, `detected_cluster`,
  `cluster_match`.
- `pipeline/run.py` -- `match` is now a real stage, the last one in
  `STAGE_ORDER`; every stage is now real, none are stubs. `match` needs
  `class_session_id`/`processing_job_id` (new `run_all_stages` parameters,
  same "raise loudly if the stage needs to run and the argument is missing"
  pattern as `video_path`/`model_dir`).
- `GET /sessions/{id}/draft` (`routers/session.py` +
  `schemas/session_draft.py`) -- the session summary plus all three bands,
  each entry carrying student name/roll number/similarity/cluster best-crop
  URI/enrollment photo URI. Never a raw embedding vector anywhere in the
  response, per the roadmap's rule for this endpoint. 409 if the match
  stage hasn't run yet (`draft_summary_json IS NULL`).
- `eval/scripts/match_report.py` -- counts per band, a similarity
  distribution per band, and the ten lowest-margin CONFIDENT matches (the
  ones closest to being reclassified as uncertain if the threshold moved).
  Talks to Postgres directly (plain `psycopg2`, same pattern as
  `gallery_sanity.py`) rather than reading a job_dir file, because match is
  the one pipeline stage whose output lives in the DB, not the filesystem.

## Phase 6: how to verify

```bash
docker-compose up --build

docker-compose exec worker pytest -v tests/test_match.py tests/test_run.py
```

Expected: `test_match.py` -- Hungarian assignment beats greedy on a
constructed matrix where a naive top-1 greedy would double-book a gallery
student (both tests assert the OPPOSITE, better assignment plus a strictly
higher total similarity); band boundaries exact at the threshold/margin/
uncertain-band values; `match_clusters` never returns a student id outside
the gallery dict it was given; the session-summary invariant holds on a
realistic mixed-band scenario and raises `ValueError` on a deliberately
corrupted `matches` list (same student both confident and uncertain --
something real Hungarian output can never produce, but the function must
still refuse it). `test_run.py` -- the `match` stage is now real in the
orchestration/caching tests too (fully faked out via
`monkeypatch.setattr("pipeline.run.run_match_stage", ...)`, same treatment
as `extract_frames`/`detect_all_frames`, since unlike every other stage it
needs a live DB this filesystem-only test file deliberately has none of);
two new tests confirm it raises loudly if `class_session_id` or
`processing_job_id` is missing when the stage actually needs to run.

I ran the pure-function tests directly in my sandbox: all 15 pass. Neither
`scipy` nor `sqlalchemy` are installed here (no PyPI access, same constraint
every phase), so I wrote a small, genuinely-correct (brute-force, not
approximate) `linear_sum_assignment` stand-in for verification only --
never shipped, the real code imports actual `scipy.optimize`. To keep
`match_clusters`/`build_session_summary` testable at all without
`sqlalchemy` installed, I moved the DB-touching imports (`sqlalchemy`,
`config`, `db`) to be function-local inside `run_match_stage` rather than
module-level -- this is a real design improvement, not just a sandbox
workaround: it means the pure decision logic can be imported and tested in
any environment with numpy/scipy, without dragging in Postgres client
libraries it never needs. I also re-ran every other sandbox-runnable test
file (`test_align.py`, `test_detect_tiling.py`, `test_quality.py`,
`test_placeholder.py`) to confirm the `db.py`/`run.py` edits didn't break
anything import-level; all still pass.

What I could NOT verify: `run_match_stage` itself (needs a real Postgres
with the `enrollment`/`course`/`class_session` tables populated), the
`GET /sessions/{id}/draft` endpoint (needs a real async DB session and
FastAPI test client -- `fastapi`/`sqlalchemy`/`asyncpg` aren't installed
here), and `match_report.py`'s DB query itself (verified its
histogram/margin-sorting *logic* against synthetic rows with `psycopg2`
stubbed out, not against a real `cluster_match` table). All of these are
exactly the kind of thing your real Docker/Postgres run is positioned to
confirm that I'm not.

## Phase 7: what's new

This phase is different in kind from Phases 1-6: it's an evaluation harness,
not more pipeline code, and its central deliverable (a real accuracy number
with failure modes characterised) depends entirely on real recorded video
and real hand-labelled ground truth that only exists once you run this on
your machine. What follows is the tooling; the report itself
(`docs/EVALUATION.md`) is a filled-in-by-you template, not a result.

- `eval/scripts/eval_lib.py` -- new shared module every other Phase 7 script
  builds on, same pure/impure split as `pipeline/match.py`:
  - Pure: `load_truth_csv`/`load_all_truth` (with a `_coerce_bool_column`
    guard against a real pandas trap -- a hand-edited truth.csv with
    `"TRUE"`/`"1"`/`"yes"` instead of clean `"True"`/`"False"` would
    otherwise silently read as all-present, since a non-empty STRING
    `"False"` is truthy in Python; `.astype(bool)` alone doesn't catch
    this), `compute_confusion`/`precision_recall_f1`/`accuracy`,
    `stratified_breakdown` (one row per group column x value, not a full
    cross-product -- an 8-session dataset fragments into meaningless cells
    otherwise), `predicted_present_from_matches` (CONFIDENT-only counts as
    "system says present" -- documented once, here, since every script
    depends on this scoring rule), `sweep_match_threshold` (re-runs
    `match_clusters` at many thresholds against the SAME cached cluster
    representatives), `pipeline_yield_for_session`, `clustering_quality`
    (purity/over-split/merge rate from an optional, this-project's-own
    `cluster_labels.csv` spot-labelling format -- the roadmap asks for this
    metric but doesn't specify a label file format beyond "support partial
    labelling"), and two ablation-only helpers: `plain_mean_representative`
    (quality-weighting-off) and `majority_vote_predicted_present`
    (clustering-off -- an ASSUMPTION vote-count threshold reusing
    `cluster_min_samples`, documented in its own docstring).
  - Impure: `fetch_session_context` (a plain-psycopg2 lookup, same
    standalone style as `gallery_sanity.py`/`match_report.py`, turning a
    `class_session_id` into enrolled students/gallery vectors/job_dir) and
    `fetch_gallery_photo_uris`. Both defer their `psycopg2` import to
    function-local, mirroring `pipeline/match.py`'s `run_match_stage` --
    keeps every pure function above importable/testable without a DB
    driver installed at all.
  - CONVENTION this phase introduces: `eval/datasets/{session_id}/` -- the
    directory name IS the `class_session_id`, as a string.
- `eval/scripts/label_session.py` -- Phase 7 deliverable 1's tool.
  Keyboard-only (space to play/pause, y/n to mark present/absent and
  auto-advance, `[`/`]`/`1`-`3`/`g`/`e` for row/seat/glasses/notes,
  `b`/`p` to go back or jump to a roll number), writes
  `eval/datasets/{class_session_id}/truth.csv` after every single
  keystroke that records a decision -- an interrupted 90-student labelling
  session never loses progress. Resumes correctly if re-opened (loads
  already-labelled rows back in) and never writes a false default for a
  student you haven't gotten to yet.
- `eval/scripts/evaluate.py` -- Phase 7 deliverable 2's core harness:
  student-level precision/recall/F1/accuracy (FP and FN always reported
  separately, never folded together), pipeline yield (mean detections per
  present student, zero-accepted-crop fraction -- "the most important
  single diagnostic," per the roadmap), clustering quality, and the
  mandatory stratified breakdown by row_number/seat_position/wears_glasses.
  Also runs the ablation study (`--ablation
  {none,temporal-coherence-off,quality-weighting-off,clustering-off}`),
  entirely from cached embeddings, no re-detection needed. The other two
  ablations the roadmap asks for (tiled detection off, quality gating off)
  change what gets DETECTED/ACCEPTED in the first place, not just how
  accepted crops get clustered/matched -- those need a real pipeline
  re-run at modified `PipelineParams` first; `evaluate.py`'s docstring
  says so plainly rather than faking a shortcut.
- `eval/scripts/sweep_threshold.py` -- Phase 7 deliverable 3. Re-runs ONLY
  `match_clusters` across `match_threshold` 0.25-0.60 step 0.01 per
  session, aggregates confusion counts (summed, not averaged, across
  sessions of different sizes), writes a precision-recall curve and an
  FP/FN-vs-threshold plot (`matplotlib`, added to
  `services/worker/requirements.txt`), and a printed table. Deliberately
  does NOT recommend the F1-maximising threshold as the answer -- F1
  treats a false positive and a false negative as equally costly, and the
  roadmap is explicit that they aren't ("false positives break the
  anti-proxy claim, false negatives break teacher trust"). That choice,
  and the written paragraph justifying it, is a human judgment call this
  script hands back to you, not something it can make up.
- `eval/scripts/failure_gallery.py` -- Phase 7 deliverable 4. One JPEG page
  per false negative/false positive: up to 4 enrollment photos, up to 8
  crops from whichever cluster the Hungarian algorithm actually linked
  that student to (even an UNCERTAIN or UNMATCHED linkage still carries a
  `student_id` in `pipeline.match`'s output -- only a truly unassigned
  student's page says so explicitly, "zero linkable evidence"), and the
  decision/similarity/runner-up numbers.
- `docs/EVALUATION.md` -- the written-report skeleton (dataset description,
  method, metrics table, stratified breakdown, threshold-selection
  reasoning, three largest failure modes, limitations, ablation table),
  structured exactly per the roadmap's required sections, with every
  numeric section explicitly marked `PENDING` and the exact command to run
  to fill it in -- not fabricated placeholder numbers.
- `pipeline/params.py` -- `match_threshold`'s comment now explicitly flags
  it as an uncalibrated first-pass guess and points at
  `sweep_threshold.py`/`docs/EVALUATION.md` section 5 as where the real
  value comes from.

## Phase 7: how to verify

Unlike every prior phase, there is no "run these against real footage and
confirm N/N pass" step here, because this phase's whole point is that N/N
doesn't exist until you provide 8+ real labelled sessions. What IS
verifiable without that:

```bash
docker-compose exec worker pytest -v eval/scripts/tests/test_eval_lib.py
```

Expected: all of `eval_lib.py`'s pure functions pass -- confusion-matrix
counting, precision/recall/F1 (including the zero-positives edge case),
the stratified breakdown genuinely surfacing a row-1-vs-row-6 gap an
aggregate would hide, the CONFIDENT-only scoring rule, the threshold
sweep's recall dropping as the threshold rises on a constructed example,
`rows_for_session` correctly leaving an unmapped roll_number as NaN rather
than guessing, the zero-accepted-crop "unrecoverable failure" flag, and the
majority-vote ablation's vote-counting.

I ran the equivalent logic directly in my sandbox: all 16 tests pass.
Beyond the unit tests, I built synthetic job_dir fixtures (a fake
`cluster_summary.parquet`/`embeddings.npy`/`quality.parquet`/
`aligned_index.parquet`, using the same pickle-based parquet stand-in as
every prior phase's sandbox verification, since `pyarrow` isn't installed
here) and a synthetic 4-student `truth.csv`, then ran `evaluate.py`'s full
`run()` end-to-end against them (with `fetch_session_context`
monkeypatched to skip the real DB call) across all four ablation modes,
`sweep_threshold.py`'s full sweep (confirmed it writes two valid, non-empty
PNGs and a table whose numbers exactly match `evaluate.py`'s baseline
figures for the same fixture), and `failure_gallery.py` (confirmed it
correctly identified the one constructed false-negative student and wrote
a valid, readable JPEG page for them). All of this exercised real,
un-mocked scoring/aggregation/plotting logic -- only the DB queries
themselves were stubbed out, exactly the same boundary every prior phase's
sandbox verification has had (no Postgres/real video available here).
`sklearn.cluster.DBSCAN` and `scipy.optimize.linear_sum_assignment` were
exercised via the same from-scratch stand-ins built for Phase 5/6's
verification (not shipped -- the real code imports the genuine libraries,
pinned in `requirements.txt`).

What I could NOT verify, and what only your machine can: `label_session.py`
against a real video file and a real Postgres roster (needs `cv2.VideoCapture`
on a real file and a real DB connection -- I exercised its save/load
round-trip logic directly instead, including the same bool-coercion trap
`eval_lib.py` guards against); anything about real accuracy numbers,
real failure modes, or a real threshold choice, since those don't exist
without real recorded, labelled classroom sessions. `docs/EVALUATION.md`
is scaffolding, not a result, until you've run the tools above against your
own 8+ sessions.

## Next: Phase 8

Teacher review UI and the append-only attendance commit flow -- turning a
session's draft roster (`GET /sessions/{id}/draft`, Phase 6) plus the
`match_threshold` you just calibrated into `AttendanceRecord` rows a
teacher can confirm or override, never mutating a past record (non-negotiable
rule #4: corrections are new rows with `supersedes_id`, not updates). Say
"start phase 8" when Phase 7 is verified on your machine.
