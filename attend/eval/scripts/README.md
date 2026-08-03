# eval/scripts

Debug/tuning utilities (Phases 3-5) and the Phase 7 evaluation harness.

- `draw_detections.py`, `contact_sheet.py`, `cluster_report.py`,
  `sweep_cluster.py`, `match_report.py` -- per-phase debug tools. See each
  file's own docstring for what it's for.
- `gallery_sanity.py` -- Phase 1: within/between-student embedding
  similarity sanity check.
- `eval_lib.py` -- Phase 7's shared library: truth.csv loading, confusion
  matrix/precision/recall/F1, the stratified breakdown, the match-threshold
  sweep helper, and the DB lookup every other Phase 7 script uses. See its
  module docstring for the pure/impure split and the CONFIDENT-only scoring
  rule every script here follows.
- `label_session.py` -- Phase 7 deliverable 1: keyboard-driven ground-truth
  labelling tool. Writes `eval/datasets/{class_session_id}/truth.csv`.
- `evaluate.py` -- Phase 7 deliverable 2: the core metrics harness,
  including the ablation study.
- `sweep_threshold.py` -- Phase 7 deliverable 3: `match_threshold`
  calibration sweep (precision-recall curve, FP/FN-vs-threshold plot).
- `failure_gallery.py` -- Phase 7 deliverable 4: one diagnostic page per
  false negative/false positive.
- `tests/test_eval_lib.py` -- unit tests for `eval_lib.py`'s pure functions.

See `docs/EVALUATION.md` for the report these scripts feed into, and the
roadmap PDF for the full Phase 7 spec.
