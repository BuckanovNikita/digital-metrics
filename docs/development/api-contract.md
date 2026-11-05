# Public API contract (must survive any refactor)

## Contents

- [Evaluation](#evaluation)
- [Result models and preprocessing](#result-models-and-preprocessing)
- [Type aliases and scoring](#type-aliases-and-scoring)
- [External backends](#external-backends)
- [ClearML tracking](#clearml-tracking)
- [Fixed evaluation, paired statistics and rendering](#fixed-evaluation-paired-statistics-and-rendering)

All of the following must exist after any refactor. Entry point: `AGENTS.md`.

Parameters after the first positional arguments are keyword-only where the code declares `*`
(`Evaluation` after `iou_threshold`, `evaluation(...)` after `split`, `predict_to_dataframe` after
`weights`, `compute_detection_metrics` after `preds_df`, `ClearMLTracker` after `task`); preserve that.

## Evaluation

- `Evaluation(preds_df, split_df, iou_threshold, preprocess, skip_cohen_kappa,
  matching_strategy, preprocess_preds_conf_threshold,
  preprocess_preds_nms_containment_threshold, preprocess_preds_nms_iou_threshold,
  ap_method, confidence_optimization, weights_path, backend, predict_kwargs,
  transliterated_labels, translit_match_cutoff)` —
  `preds_df` may be `None` (an empty placeholder is created) to predict first;
  `weights_path` is an optional YOLO weights path. `backend` (`None` = native;
  `"ultralytics"` / `"torchmetrics"`) makes `Evaluation` the single entry point:
  when set, the call scores the split with that external library over
  `_raw_preds_df` instead of the native pipeline (see `evaluation.detection_metrics`
  below). `predict_kwargs` (`dict | None`) is forwarded to Ultralytics'
  `model.predict` when predictions are auto-generated from `weights_path`
  (e.g. `{"conf": 0.25, "imgsz": 1280, "half": True}`); ignored when `preds_df` is
  given. The constructor also accepts three optional **grouped config** objects —
  `scoring` (`ScoringConfig`), `preprocessing` (`PreprocessConfig`), `inference`
  (`InferenceConfig`) — as a tidier alternative to the flat kwargs above. They are
  purely additive: every flat kwarg still works, and when a group is passed it
  supplies that whole group and takes precedence over its corresponding flat kwargs
  (defaults mirror the flat defaults). `backend` stays a flat top-level arg
- `ScoringConfig` / `PreprocessConfig` / `InferenceConfig` exported from `digital_metrics`
  (defined in `config.py`) — `ScoringConfig(iou_threshold, matching_strategy,
  ap_method, confidence_optimization, skip_cohen_kappa)`; `PreprocessConfig(dedup_gt,
  conf_threshold, nms_containment_threshold, nms_iou_threshold)` (`dedup_gt` ← the
  flat `preprocess`); `InferenceConfig(weights_path, predict_kwargs)`
- `evaluation(split, find_best_confs, calibration_split)` — main call. When
  `preds_df` was `None`, the first run generates predictions from `weights_path`
  over just the splits it will use — the evaluation split plus `calibration_split`
  (or every image when `split="all"`), via `_splits_to_predict`; if `preds_df` is
  `None` and no `weights_path` was given it raises `ValueError`. After confidence
  optimisation it logs a warning for any class whose chosen threshold equals the
  minimum prediction confidence (the cut keeps every detection, so optimisation
  had no effect — e.g. identical pred/GT boxes)
- `evaluation.predict_to_dataframe(weights, split, conf, iou, imgsz, batch, device,
  image_name, **model_kwargs)` — optional YOLO inference: runs Ultralytics
  ``weights`` over the images in ``split_df['image_path']`` and stores predictions
  in the standard schema as ``preds_df``/``_raw_preds_df`` (``image_name`` = last
  path part). `split` accepts a single split, a list of splits (e.g.
  `["test", "val"]`), or `None` (every image); selecting splits needs a `"split"`
  column. `**model_kwargs` are forwarded verbatim to `model.predict` (e.g. `half`,
  `augment`, `max_det`, `classes`). Requires the `ultralytics` extra (lazy import,
  `ImportError` with hint); raises `ValueError` when `split_df` has no `image_path`
  column (or no `"split"` column when `split` is given). Row-building helpers in
  `inference/yolo_predict.py` (`predict_on_images`, `_detection_rows`) are
  torch-free (`predict_on_images` also accepts `batch` + `**model_kwargs`).
  `batch` (default 16) is the chunk size fed to each `model.predict` call and the
  chunk size used by the wrapper. Lower `batch` or `imgsz` to reduce work per
  prediction call. `batch` also flows through `predict_kwargs`. On success or
  failure the wrapper releases owned results before `_release_gpu_memory`
  drops predictor references, moves the model to CPU and requests CUDA cache
  cleanup. Object-lifetime tests establish release ordering; they do not measure
  reserved-memory reduction or guarantee allocator behavior.
- Input validation raises `ValueError` for missing columns, nonnumeric/nonfinite
  confidences outside [0,1], malformed labelled boxes (missing or nonfinite
  coordinates, inverted corners or zero area), reserved background labels,
  duplicate/missing/sentinel row indices, normalized image-ID collisions and
  image ownership across splits. Only fully unlabelled empty-box GT rows are
  placeholders. Validation occurs before filtering, suppression and unknown-label
  dropping. Unknown prediction classes are warned about and dropped consistently
  from native and backend evaluation.
- `transliterated_labels=True` (+ `translit_match_cutoff`, default `0.8`) —
  handles models trained on a transliterated conversion of the dataset
  (`"Грязь на основе"` → `"Gryaz_na_osnove"`), which would otherwise score all
  zeros. `Evaluation._restore_transliterated_labels` runs before
  `_drop_unknown_pred_classes` (native *and* backend paths) and rewrites the
  labels in both `preds_df` and `_raw_preds_df`. Reverse transliteration is
  ambiguous, so matching goes forward off the GT: `translit.py` transliterates
  each GT label with three schemes (`common`/`gost`/`icao`, plus a `ё`→`е` fold),
  normalises (lower-case, non-alphanumerics dropped) and indexes them; a pred
  label hits by exact key, then by `difflib` fuzzy match at the cutoff; no close
  match → warning listing the labels, left unchanged. Ambiguous normalized keys
  remain unresolved, including through fuzzy fallback. Public helpers exported
  from `digital_metrics`: `transliterate(text, scheme)`, `restore_labels(labels,
  gt_labels, cutoff)`, `TRANSLIT_SCHEMES`
- `evaluation.compute_metrics_ultralytics(split)` /
  `evaluation.compute_metrics_torchmetrics(split)` — score `split` with that
  external backend over `_raw_preds_df` and return `dict[str, DetectionMetrics]`
  (auto-predicting first via `weights_path` when needed); require the matching
  optional extra (lazy import, `ImportError` with hint)
- `evaluation.metrics` — `dict[str, Metrics]`. In `backend` mode this holds the
  backend's numbers adapted onto native `Metrics`. Explicit thresholds provide
  observed TP/FP/FN and Wilson intervals; unthresholded or missing summaries have
  `counts_observed=False` and unavailable intervals (`cohen_kappa = -1`).
- `evaluation.detection_metrics` — `dict[str, DetectionMetrics]`; the raw external
  backend output, populated only in `backend` mode (empty otherwise)
- `evaluation.cm`, `evaluation.class_labels` — populated natively and by the
  `"ultralytics"` backend (Ultralytics' own confusion-matrix logic); `None` for the
  `"torchmetrics"` backend, where `get_dashboards` then skips the CM sheet
- `evaluation.best_confidences` — `dict[str, float]` (selected per-class thresholds)
- `evaluation.unfiltered_matches` — matches before confidence slicing
- `evaluation._raw_preds_df` — unpreprocessed predictions (passed to `compute_map`)
- `evaluation.get_dashboards(save_to_excel, path, save_confusion_matrix)`
- `evaluation.plot_confidence_intervals(metric, confidence_level, save_path, figsize)`
- `evaluation.get_topk_confusions(main_class, k)`
- `evaluation.get_dfs_visualization(find_best_confs, apply_thresholds)` — returns
  `(gt_df, preds_df)` each with a `predict_type` column (preds: `TP`/`FP`, GT:
  `TP`/`FN`), annotated from `unfiltered_matches`. `apply_thresholds=True` folds the
  per-class `best_confidences` in (mirroring `slice_by_conf`): predictions below
  their class threshold become `filtered`, and a GT detected only by a filtered
  prediction turns `FN`

## Result models and preprocessing

- `Metrics` fields: `tp, fp, fn, confidence, ap50, ap75, ap50_95, cohen_kappa,
  precision, recall, f1_score, perebrak, nedobrak, *_ci_lower, *_ci_upper`
  — `ap50/ap75/ap50_95` are `float | nan` (NaN when class absent from split)
- `PredictMatch` with `type` computed field and an optional `iou` field (IoU
  with the associated GT box; `None` for `background` FPs and FNs)
- `preprocess.filter_by_confidence(preds_df, threshold)` → filtered DataFrame
- `preprocess.apply_nms(preds_df, same_class_containment_threshold, cross_class_iou_threshold)`
  → DataFrame with suppressed rows removed

## Type aliases and scoring

- `APMethod = Literal["interp", "continuous"]` exported from `digital_metrics`
- `MatchingStrategy = Literal["greedy", "hungarian", "iou_prior"]` exported from `digital_metrics`
- `ConfidenceOptimization = Literal["per_class", "global"]` exported from `digital_metrics`
  — `"global"` selects a single YOLO-style threshold (max mean per-class F1) shared
  by all classes; `"per_class"` (default) tunes one threshold per class.
  Both pick thresholds only at *realizable* operating points (confidence
  tie-group boundaries), so the optimised F1 equals the F1 actually obtained when
  the threshold is applied via `slice_by_conf` — tied confidences cannot produce
  a non-achievable mid-tie optimum.
- `compute_map(gt_df, preds_df, metrics, split_image_names, method, strategy)` —
  all three strategies supported including `"hungarian"`

## External backends

- `compute_detection_metrics(gt_df, preds_df, backend, classes, split_image_names)`
  and `Backend = Literal["ultralytics", "torchmetrics"]` exported from `digital_metrics` —
  single entry point dispatching to the two external backends; raises `ValueError`
  on an unknown backend (before any heavy import)
- `DetectionMetrics` exported from `digital_metrics` — shared per-class result model
  (`precision/recall/f1/ap50/ap75/ap50_95`) returned by both backends;
  `YoloMetrics` is kept as a backward-compatible alias of it
- `compute_ultralytics_metrics(gt_df, preds_df, classes, split_image_names,
  conf_threshold)` exported from `digital_metrics` — optional YOLO-comparable P/R/F1/AP
  via Ultralytics' own `ap_per_class`. `conf_threshold` (`None` | `float` |
  `dict[str, float]`) reports retained counts and P/R/F1 at an explicit threshold
  (e.g. calibrated on val), otherwise the native backend summary; AP is always
  over the full curve. Requires the `ultralytics` extra (lazy import, raises
  `ImportError` with an install hint when missing)
- `find_ultralytics_confidence(gt_df, preds_df, classes, split_image_names, mode)`
  exported from `digital_metrics` — calibration helper using whole-image retained
  detections. `mode="global"` returns the exact max-macro-F1 scalar threshold;
  `"per_class"` returns a dictionary with deterministic coordinate-local macro-F1
  refinement for mixed-class ties, without a joint global-optimum claim. Pair with
  `compute_ultralytics_metrics(conf_threshold=...)`. Requires the `ultralytics` extra
- `compute_ultralytics_confusion_matrix(gt_df, preds_df, classes,
  split_image_names, conf, iou_thres)` exported from `digital_metrics` — optional
  confusion matrix via a numpy port of Ultralytics'
  `ConfusionMatrix.process_batch`; returns `(matrix, labels)` with shape
  `(nc+1, nc+1)` in sklearn row=GT/col=pred orientation; requires the
  `ultralytics` extra (lazy import, raises `ImportError` with an install hint)
- `compute_torchmetrics_metrics(gt_df, preds_df, classes, split_image_names,
  conf_threshold)` exported from `digital_metrics` — optional general COCO mAP via
  torchmetrics' `MeanAveragePrecision`. `conf_threshold` (`None` | `float` |
  `dict[str, float]`) reports observed retained counts and P/R/F1 with COCO
  matching, or the unthresholded backend summary when None. AP uses the full
  predictions. Requires the `torchmetrics` extra (lazy import with install hint).
- `find_torchmetrics_confidence(gt_df, preds_df, classes, split_image_names, mode)`
  exported from `digital_metrics` — torchmetrics calibration helper mirroring
  `find_ultralytics_confidence`: realizable F1-optimal confidence with COCO matching.
  `mode="global"` returns a `float`, `"per_class"` a `dict[str, float]`; pair with
  `compute_torchmetrics_metrics(conf_threshold=...)`. Requires the `torchmetrics` extra

## ClearML tracking

- `ClearMLTracker(task, project_name, task_name, output_uri, attach_logs,
  log_level, **task_init_kwargs)` and `summarize_metrics(metrics)` exported from
  `digital_metrics` (defined in `tracking/clearml_tracker.py`) — standalone ClearML
  tracking layer over a finished `Evaluation`. `log_evaluation(evaluation, *,
  iteration, artifacts_dir, save_to_excel, save_confusion_matrix)` mirrors
  scalars + a per-class table + `mean_*` single values, artifacts (dashboard
  DataFrames, `best_confidences`, confusion matrix, written `.xlsx` files), the CI
  PNGs + confusion-matrix plot, and (via `attach_loguru`) run logs. Individual
  layers `log_scalars` / `log_artifacts` / `log_plots` / `attach_loguru` /
  `detach_loguru` / `close`; context-manager closes the task. `clearml` is an
  optional extra imported lazily and **only** when the tracker creates its own
  task (an injected `task=` needs no extra); `summarize_metrics` is torch/ClearML-free

## Fixed evaluation, paired statistics and rendering

- 0.7.0 additive exports: `evaluate_fixed`, `FixedEvaluationResult`; `compare_paired`,
  `mcnemar_recall`, `bootstrap_precision_delta`, `bootstrap_recall_delta`,
  `adjust_benjamini_hochberg`; `BHResult`, `PairedClassResult`, `PairedComparisonResult`,
  `TestResult`; `render_evaluation`, `RenderedEvaluation`, `RenderedPlot`; `PRObservation`,
  `build_pr_observations` — exported from `digital_metrics`; contracts in
  [`docs/shared-evaluation.md`](../shared-evaluation.md).
