# Attend — evaluation report

**Status: template. No real recorded/labelled classroom session has gone
through this document yet.** Every numbered section below states exactly
what to run and what to paste in once you have real sessions labelled — see
[How to fill in this report](#how-to-fill-in-this-report) at the bottom.
Nothing in this file should be read as a real result until every `PENDING`
marker below is replaced with an actual number from a real run.

This report exists to answer one question honestly, for a sceptical
reviewer: does this system actually work, and if not everywhere, exactly
where does it not? Per the roadmap's own framing, the difference between
"it works, about 95 percent" and a precise, stratified, failure-mode-aware
number is the entire difference in seniority this phase is meant to
demonstrate.

## 1. Dataset description

`PENDING`. Fill in once you have labelled sessions under `eval/datasets/`:

- Number of sessions: `PENDING` (roadmap's definition of done requires at
  least 8).
- Number of distinct students across all sessions: `PENDING`.
- Per-session: date, course, room, approximate class size, camera/pan
  conditions worth noting (lighting, backlighting, how fast the pan was).
- Conditions deliberately varied across sessions (if any): time of day,
  room, whether students were told in advance, seasonal clothing (masks,
  hats, sunglasses), etc. -- the more real-world variation the dataset
  covers, the more the headline numbers below actually mean.

## 2. Method

For each labelled session:

1. The session's video is processed through the full pipeline (extract →
   detect → quality → align → embed → cluster → match), producing cached
   artifacts under that session's `job_dir` (`embeddings.npy`,
   `quality.parquet`, `cluster_summary.parquet`, etc.).
2. `eval/scripts/label_session.py` produces `eval/datasets/{class_session_id}/truth.csv`
   -- one row per enrolled student, hand-labelled while watching the actual
   video (present/absent, row number, seat position, glasses, notes).
3. `eval/scripts/evaluate.py` re-runs matching in-memory against the cached
   cluster representatives at the PipelineParams in `pipeline/params.py`,
   and scores the result against `truth.csv`.
4. A decision counts as "system says present" only if it's `CONFIDENT` (see
   `eval_lib.py`'s module docstring for why `UNCERTAIN`/`UNMATCHED` both
   count as "system says absent" for these metrics -- crediting a
   pending-review guess would overstate what the automatic system achieved
   on its own).

## 3. Headline metrics

`PENDING` -- run:

```bash
DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \
    python eval/scripts/evaluate.py --job-data-dir /data/jobs --datasets-dir eval/datasets
```

and paste the `STUDENT-LEVEL PRESENCE` and `PIPELINE YIELD` blocks it prints,
in this shape:

| metric | value |
|---|---|
| precision | `PENDING` |
| recall | `PENDING` |
| F1 | `PENDING` |
| accuracy | `PENDING` |
| true positives | `PENDING` |
| false positives (proxy hole) | `PENDING` |
| false negatives (trust killer) | `PENDING` |
| true negatives | `PENDING` |
| mean detections per present student | `PENDING` |
| fraction of present students with zero accepted crops (unrecoverable) | `PENDING` |

## 4. Stratified breakdown

`PENDING` -- copied from the same `evaluate.py` run's `STRATIFIED BREAKDOWN`
block. This section is mandatory, not optional (roadmap, verbatim): an
aggregate number can hide a system that only works in the front row.

| group | value | n | precision | recall | f1 | fp | fn |
|---|---|---|---|---|---|---|---|
| row_number | 1 | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| row_number | ... | | | | | | |
| seat_position | left | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| seat_position | centre | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| seat_position | right | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| wears_glasses | True | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |
| wears_glasses | False | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` |

**Observations:** `PENDING` -- once filled in, call out explicitly whether
recall (or precision) drops meaningfully in any stratum, e.g. "recall in
row 5+ is 0.XX vs 0.YY overall" -- don't just present the table, say what it
means.

## 5. Threshold selection

`PENDING` -- run:

```bash
DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \
    python eval/scripts/sweep_threshold.py --job-data-dir /data/jobs \
    --datasets-dir eval/datasets --out-dir eval/reports
```

This produces `eval/reports/precision_recall_curve.png`,
`eval/reports/fp_fn_vs_threshold.png`, and a printed table. Paste the table
here, embed both images, and then write the operating-point decision as an
actual paragraph -- not just a number. The roadmap's framing to work from:
false positives break the anti-proxy claim (a student who wasn't there gets
marked present -- this is what makes the whole system worthless as an
attendance record), false negatives break teacher trust (a student who WAS
there gets marked absent, and if that happens often enough teachers stop
trusting -- and stop using -- the system). Decide which failure mode this
specific deployment can tolerate less, given who's using it and what it's
for, and justify the chosen `match_threshold` against that, not against
whichever value happens to maximise F1 (`sweep_threshold.py` deliberately
does not recommend the F1-maximising point for this reason -- see its
docstring).

`PENDING`: chosen `match_threshold` = `___`, reasoning:

> `PENDING -- write the actual justification paragraph here once you have real curves to look at.`

Once decided, update `pipeline/params.py`:

```python
match_threshold: float = 0.XX  # Phase 7 calibration: YYYY-MM-DD, against
                                # eval/datasets/{...} (N sessions, M students).
                                # Chose this value because <reasoning>.
```

## 6. Three largest failure modes

`PENDING` -- run:

```bash
DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \
    python eval/scripts/failure_gallery.py --job-data-dir /data/jobs \
    --datasets-dir eval/datasets --out-dir eval/reports/failures
```

Look through the generated pages (one per false negative/false positive)
and identify the three most common root causes. For each: name it, say how
many of the total false negatives/positives it explains, and embed one or
two example pages as evidence. Categories to watch for, based on what this
pipeline can plausibly get wrong (not a prediction of what you'll actually
find, just where to look first): back-row students whose crops never pass
the quality gate at all (`zero_accepted_crop_fraction` from section 3 is
the leading indicator); a real person over-split into two clusters, each
too thin on its own to cross `match_threshold` confidently; two students
who look similar enough that the Hungarian assignment picks the wrong one
of the pair; a face at an extreme yaw/pitch that gets rejected by the
quality gate even though a human would still recognise it easily.

1. `PENDING`: name, count/fraction of failures, example page(s).
2. `PENDING`
3. `PENDING`

## 7. Limitations

Known, structural limitations of this evaluation and this system, stated
plainly rather than glossed over:

- **`pipeline_yield_for_session`'s crop-attribution is assignment-based, not
  ground-truth-based.** "Detections attributed to a student" means
  whichever cluster the Hungarian algorithm assigned them, at ANY
  confidence band -- a student confidently mis-assigned someone else's
  cluster would show a nonzero count here even though none of those crops
  are actually theirs. This can only be fully corrected with per-crop
  ground truth, which this project's `truth.csv` format (per-student, not
  per-crop) does not capture.
- **Clustering-quality metrics (purity/over-split/merge rate) depend on
  optional, hand-maintained `cluster_labels.csv` spot-labelling** -- not a
  roadmap-specified format, this project's own addition (see
  `eval_lib.clustering_quality`'s docstring) -- and are only as complete as
  whatever fraction of clusters you've had time to manually check against
  `cluster_report.py`'s contact sheets.
- **The "clustering off" and "quality weighting off" ablations only affect
  matching against an unchanged set of embeddings** -- they do not, and
  cannot, tell you what would happen if tiling or quality-gating had been
  off from the start, since those two DO change what gets embedded at all
  (see the ablation table's own notes in section 8).
- **Ground truth itself has a ceiling of accuracy**: it's produced by one
  human watching the same video the system watched, which has its own
  failure modes (a labeller can miss someone in a packed back row too).
  `notes` in `truth.csv` is the place to flag any label you weren't fully
  confident about.
- `PENDING`: add anything else that comes up once real data is in front of
  you -- this section should grow, not shrink, as you learn more about
  where the system actually breaks.

## 8. Ablation study

`PENDING` -- run `evaluate.py` once per ablation and fill in the table:

```bash
for ablation in none temporal-coherence-off quality-weighting-off clustering-off; do
    DATABASE_URL=postgresql://attend:attend@localhost:5432/attend \
        python eval/scripts/evaluate.py --job-data-dir /data/jobs \
        --datasets-dir eval/datasets --ablation "$ablation"
done
```

| ablation | precision | recall | F1 | note |
|---|---|---|---|---|
| none (baseline) | `PENDING` | `PENDING` | `PENDING` | full pipeline as shipped |
| tiled detection off | `PENDING` | `PENDING` | `PENDING` | **requires a separate real pipeline re-run** at `tile_trigger_long_side_px` set above the frame's long side (forces the single-full-frame detection path) -- not derivable from cached artifacts, since it changes what gets detected in the first place. Re-run `POST /sessions/{id}/process` with that param override, then point `evaluate.py` at the resulting job. |
| quality gating off | `PENDING` | `PENDING` | `PENDING` | **also requires a real re-run**, with `detector_score_min=0`, `min_face_px=0`, `blur_laplacian_min=0`, `max_abs_yaw_deg=180`, `max_abs_pitch_deg=180`, `brightness_min=0`, `brightness_max=255` -- same reason as above. |
| clustering off (majority vote) | `PENDING` | `PENDING` | `PENDING` | `--ablation clustering-off`; computed from cached embeddings, no re-run needed |
| temporal coherence off | `PENDING` | `PENDING` | `PENDING` | `--ablation temporal-coherence-off`; computed from cached embeddings, no re-run needed |
| quality weighting off | `PENDING` | `PENDING` | `PENDING` | `--ablation quality-weighting-off`; computed from cached embeddings, no re-run needed |

**Observations:** `PENDING` -- once filled in, say which single change
contributes the most, since that's "the most interesting result the
project will produce" (roadmap, verbatim) and deserves a real sentence, not
just a table row.

---

## How to fill in this report

In order:

1. Record and label at least 8 sessions (`label_session.py`).
2. Run `evaluate.py` for the baseline -> sections 1, 3, 4.
3. Run `sweep_threshold.py`, look at the plots, decide, update
   `pipeline/params.py` -> section 5.
4. Re-run `evaluate.py` at the newly-chosen threshold (if different from
   default) so sections 3/4 reflect the value you actually shipped.
5. Run `failure_gallery.py`, read through the pages -> section 6.
6. Run `evaluate.py --ablation ...` for the three cached-artifact ablations;
   for the two that need a real pipeline re-run, do that re-run first ->
   section 8.
7. Fill in section 7 with whatever limitations you actually ran into, not
   just the ones listed here.
