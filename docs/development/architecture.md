# digital-metrics architecture (maintainer reference)

Agent and maintainer reference; entry point is `AGENTS.md`.

## Contents

- [Subpackage rules](#subpackage-rules)
- [Module map](#module-map)
- [Tests](#tests)
- [Scripts](#scripts)
- [Fixtures and root files](#fixtures-and-root-files)

## Subpackage rules

See `AGENTS.md` § Architecture (subpackage layout, `__init__.py` re-exports, shared foundation modules, lazy extras).

## Module map

```
src/
  digital_metrics/
    __init__.py       # public API (re-exports from the subpackages below)
    types.py          # Pydantic models: PredictMatch, Metrics, DetectionMetrics
    ci.py             # Wilson confidence interval (foundation; types depends on it)
    validation.py     # validate_dataframes: shared GT/preds column + label checks
    translit.py       # Cyrillic→Latin schemes + restore_labels: map transliterated model labels onto the GT vocabulary
    grouping.py       # image_row_indices: per-image positional row grouping (perf helper, shared)
    config.py         # ScoringConfig/PreprocessConfig/InferenceConfig (optional grouped Evaluation args)
    evaluation.py     # Evaluation orchestrator: data/IO + dispatch to a scoring engine
    fixed.py          # evaluate_fixed / FixedEvaluationResult: score supplied fixed/AP populations (no IO)
    comparison.py     # paired statistics: compare_paired, mcnemar_recall, bootstrap_*_delta, adjust_benjamini_hochberg
    calibration.py    # ConfidenceCalibrator: threshold selection (delegated by Evaluation)
    engines/          # pluggable scoring engines selected by Evaluation's backend
      __init__.py     # re-exports: ScoringEngine, ScoringInputs, EvaluationResult, Native/BackendEngine, compute_metrics_from_matches
      base.py         # ScoringEngine protocol + ScoringInputs/EvaluationResult dataclasses
      native.py       # NativeEngine: match → calibrate → slice → metrics/mAP/kappa/CM
      backend.py      # BackendEngine: external library scoring + adapt onto native Metrics
    matching/         # box matching: geometry → assignment → records
      __init__.py     # re-exports: compute_iou_matrix, find_duplicates_bboxes, match_boxes, MatchingStrategy, assign_*
      iou.py          # IoU matrix computation
      assignment.py   # pure box-assignment kernels (greedy/iou_prior/hungarian) on IoU matrices
      matching.py     # box matching → PredictMatch records; wraps assignment kernels
    scoring/          # metric computations from matches/boxes
      __init__.py     # re-exports: compute_map/compute_ap/APMethod, compute_kappa,
                      #   find_best_*/slice_by_conf/ConfidenceOptimization, get_confusion_matrix/get_confusions,
                      #   PRObservation/build_pr_observations
      ap.py           # AP / mAP computation; APMethod + MatchingStrategy options; reuses assignment kernels
      confidence.py   # best-confidence search: per-class + global (YOLO-style) thresholds
      kappa.py        # Cohen's kappa (pixel-mask method)
      confusion.py    # confusion matrix helpers (native, match-record based)
    preprocess/       # predictions preprocessing
      __init__.py     # re-exports: filter_by_confidence, apply_nms, PredictionPreprocessor
      nms.py          # confidence filter + custom NMS
      preprocessor.py # PredictionPreprocessor: conf filter + NMS (delegated by Evaluation)
    reporting/        # output artifacts
      __init__.py     # re-exports: get_dashboards, plot_confidence_intervals, render_evaluation, RenderedEvaluation, RenderedPlot
      dashboard.py    # Excel export + CI plots
    tracking/         # optional ClearML experiment tracking (separate layer, lazy clearml import)
      __init__.py     # re-exports: ClearMLTracker, summarize_metrics
      clearml_tracker.py # ClearMLTracker: mirror an Evaluation's scalars/artifacts/plots/logs into a ClearML Task
    backends/         # optional, torch-backed metric backends (lazy torch import)
      __init__.py     # re-exports: compute_detection_metrics/Backend, compute_ultralytics_*/find_ultralytics_confidence, compute_torchmetrics_metrics/find_torchmetrics_confidence, YoloMetrics
      external.py     # single entry point: compute_detection_metrics(backend=...)
      operating_points.py # shared raw-count / realizable-threshold calibration kernels
      ultralytics_metrics.py  # optional YOLO-comparable metrics (ap_per_class) + confusion matrix + calibration
      torchmetrics_metrics.py # optional COCO mAP via torchmetrics MeanAveragePrecision + calibration
    inference/        # optional YOLO inference (lazy torch import)
      __init__.py     # re-exports: predict_on_images, ImageNameMode
      yolo_predict.py # YOLO inference helper (Evaluation.predict_to_dataframe)
```

## Tests

```
tests/
  conftest.py
  test_iou.py
  test_matching.py
  test_ap.py
  test_ci.py
  test_evaluation.py
  test_nms.py
  test_preprocessor.py          # PredictionPreprocessor: enabled flag, conf/NMS, no-op, no-mutate
  test_confidence.py
  test_confidence_calibrator.py # ConfidenceCalibrator: leak check, dispatch, warning, parity
  test_engines.py               # NativeEngine/BackendEngine: run result, resolve split, selection
  test_config.py                # grouped config objects: defaults, grouped==flat, precedence
  test_dashboard.py            # get_dashboards / plot_confidence_intervals output
  test_external_metrics.py     # dispatcher; ValueError path runs without extras
  test_evaluation_backend.py   # backend selection + DetectionMetrics→Metrics adapter + calibration
  test_ultralytics_metrics.py  # optional; skipped unless `ultralytics` is installed
  test_torchmetrics_metrics.py # optional; skipped unless `torchmetrics` is installed
  test_torchmetrics_calibration.py # torch-free curve helpers + optional when-installed calibration
  test_yolo_predict.py         # predict_to_dataframe: torch-free helpers, image_path/ImportError guards
  test_translit.py             # transliterate/normalize/index/restore_labels + Evaluation wiring (zero-metrics without the flag, perfect with it)
  test_clearml_tracker.py      # ClearMLTracker: summarize_metrics nanmean + scalar/artifact/plot/loguru dispatch via fake Task (torch/clearml-free)
  __init__.py
  test_backend_operating_oracles.py  # compare retained detections with actual backend matchers, not AP envelopes
  test_local_release.py  # local release acceptance tests (real Git, PSR, uv and installed hooks)
  test_operating_point_scaling.py  # calibration work follows changed images rather than every global score
  test_review_regressions.py  # public regressions from the October 2026 independent review
  test_shared_architecture.py  # library-boundary import contract plus shared-doc link check
  test_shared_extraction.py  # scientific/API/artifact contracts for the additive shared extraction
  test_shared_legacy_characterization.py  # legacy API observations captured before extraction, 2026-10-12
  fixtures/shared_ap_baseline.json
```

## Scripts

Comparison scripts all read the fixture `test` split; external backends are skipped when their extra is absent.

- `scripts/eval.py` — local evaluation script (see `AGENTS.md` § Local evaluation).
- `scripts/profile_backends.py` — time + cProfile native vs ultralytics vs torchmetrics (`--calibrated`, `--gt`/`--preds`/`--only`).
- `scripts/compare_backends.py` — all three ways side by side (`Evaluation` vs `ultralytics` vs `torchmetrics`): mean summary + per-class P/R/F1 and mAP.
- `scripts/compare_ultralytics.py` — `Evaluation` mAP vs the `ultralytics` backend.
- `scripts/compare_ultralytics_prf1.py` — `Evaluation` P/R/F1 vs the `ultralytics` backend.
- `scripts/plot_prf1_vs_map.py` — writes `docs/prf1_means.png` and `docs/prf1_pr_curve.png` (gitignored, historical comparison figures; not linked from the docs). The maintained synthetic illustration `docs/prf1_operating_points.svg` is reproduced by `scripts/create_review_examples.py`; AP is backend-specific.
- `scripts/create_review_examples.py` — creates the tiny synthetic evaluation fixtures and the illustrative SVG figure.
- `scripts/eval_identity.py` — identity sanity check (predictions == GT → all metrics 1.0).
- `scripts/local_release.py` — local semantic version commits, tags, changelog and hooks (see `docs/local-release.md`).
- `scripts/quickstart_from_weights.py` — onboarding example that evaluates from YOLO weights; run with `uv run --extra ultralytics`.
- `scripts/sanity_corrupt_labels.py` — corrupts about half of the labels and expects the F1 optimiser to recover.

## Fixtures and root files

```
fixtures/
  ground_truths_all.csv     # synthetic GT example (val/test and an empty image)
  predicts_all.csv          # synthetic predictions
  eval_metrics.csv          # per-class metrics output (generated by eval.py)
  eval_confusion_matrix.csv # confusion matrix output
  eval_best_confidences.json # val-calibrated thresholds per class
  README.md                 # describes the synthetic example data
AGENTS.md                   # shared agent instructions
CLAUDE.md                   # imports AGENTS.md
pyproject.toml
uv.lock
.pre-commit-config.yaml
CHANGELOG.md                # generated
README.md / README.ru.md
docs/
templates/CHANGELOG.md.j2
```
