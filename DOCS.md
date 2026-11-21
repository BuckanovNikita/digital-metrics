# digital-metrics reference

Read on demand. Binding rules are in [AGENTS.md](AGENTS.md); user guides are in [docs/](docs/README.md).

## Architecture and data flow

`Evaluation(preds_df, split_df, ...)` validates both DataFrames, optionally restores transliterated
labels, drops unknown prediction classes, preprocesses predictions (confidence filter + custom NMS
into `preds_df`, keeping `_raw_preds_df` untouched), then dispatches to a scoring engine:

- `NativeEngine`: match -> calibrate thresholds on `calibration_split` -> slice by confidence ->
  per-class `Metrics`, mAP (over `_raw_preds_df`), Cohen's kappa, confusion matrix.
- `BackendEngine` (`backend="ultralytics"|"torchmetrics"`): external library scoring over
  `_raw_preds_df`, raw result in `detection_metrics`, adapted onto native `Metrics`.

Reporting (`get_dashboards`, `plot_confidence_intervals`, `render_evaluation`) and the optional
`ClearMLTracker` consume a finished `Evaluation`. `evaluate_fixed` scores supplied populations
without IO; `comparison.py` provides paired statistics (contracts: [docs/shared-evaluation.md](docs/shared-evaluation.md)).

## Module map

```
src/digital_metrics/
  __init__.py      flat public API          types.py      PredictMatch, Metrics, DetectionMetrics
  ci.py            Wilson interval          validation.py validate_dataframes (columns, labels, boxes)
  translit.py      Cyrillic->Latin schemes, restore_labels
  grouping.py      image_row_indices (per-image row grouping)
  config.py        ScoringConfig / PreprocessConfig / InferenceConfig
  evaluation.py    Evaluation orchestrator  calibration.py ConfidenceCalibrator
  fixed.py         evaluate_fixed, FixedEvaluationResult
  comparison.py    compare_paired, mcnemar_recall, bootstrap_*_delta, adjust_benjamini_hochberg
  engines/         base.py (ScoringEngine, ScoringInputs, EvaluationResult), native.py, backend.py
  matching/        iou.py, assignment.py (greedy/iou_prior/hungarian kernels), matching.py (match_boxes)
  scoring/         ap.py, confidence.py (best-threshold search), kappa.py (pixel masks), confusion.py
  preprocess/      nms.py (filter_by_confidence, apply_nms), preprocessor.py
  reporting/       dashboard.py (Excel, CI plots, render_evaluation)
  tracking/        clearml_tracker.py (ClearMLTracker, summarize_metrics)
  backends/        external.py (dispatcher), operating_points.py (shared calibration kernels),
                   ultralytics_metrics.py, torchmetrics_metrics.py
  inference/       yolo_predict.py (predict_on_images, ImageNameMode)
```

Scripts: `eval.py` (end-to-end on fixtures), `eval_identity.py` (preds == GT -> all 1.0),
`sanity_corrupt_labels.py`, `compare_backends.py`, `compare_ultralytics.py`,
`compare_ultralytics_prf1.py`, `profile_backends.py` (`--calibrated`, `--gt/--preds/--only`),
`plot_prf1_vs_map.py` (gitignored PNGs), `create_review_examples.py` (synthetic fixtures and
`docs/prf1_operating_points.svg`), `quickstart_from_weights.py` (needs the ultralytics extra),
`local_release.py`.

## Domain terms

- **Split**: `train`/`val`/`test` value in the GT `split` column; an image belongs to one split.
- **TP**: IoU >= threshold, correct label, GT not yet matched. **FP**: no GT matched (records the
  GT label for a cross-class match, else `background`). **FN**: unmatched GT.
- **Precision** TP/(TP+FP); **Recall** TP/(TP+FN); **F1** 2PR/(P+R); **CI** Wilson interval.
- **perebrak** = 1 - precision; **nedobrak** = 1 - recall (Russian domain terms).
- **mAP50 / mAP75 / mAP50-95**: AP at IoU 0.50, 0.75, and the mean over 0.50:0.05:0.95.
- **Operating point**: a realizable confidence threshold at a tie-group boundary.
- **Placeholder row**: a fully unlabelled empty-box GT row marking an image with no objects.
- **counts_observed**: whether TP/FP/FN in a backend-adapted `Metrics` are exact.

## DataFrame schema

Shared: `image_name` (str), `instance_label` (str), `bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`,
`bbox_y_br` (float). GT only: `split`; `image_path` (only for `predict_to_dataframe`);
`image_width`, `image_height` (only when `skip_cohen_kappa=False`). Predictions only: `confidence` in [0,1].

## Metric internals

- **mAP**: predictions sorted by confidence per class; each image's IoU matrix is computed once per
  class (`_precompute_image_matches`) and reused across the ten thresholds; the assignment kernel
  runs per threshold; cumulative TP/FP form the P-R curve. AP is NaN for classes with no GT in
  the evaluated split, so class averages use `nanmean`.
- **AP `interp`** (Evaluation default): 101-point COCO interpolation with Ultralytics sentinels
  (`mpre[0]=1.0`, `mrec[-1]=recall[-1]+1e-4`), `scipy.integrate.trapezoid`; 0.0 on empty recall.
- **AP `continuous`**: VOC 2010+ rectangle area with `(0,0)`/`(1,0)` sentinels and a right-to-left
  precision envelope.
- **Greedy**: predictions by confidence; each takes the highest-IoU unmatched GT; same label -> TP,
  else FP with the GT label; below threshold -> FP `background`.
- **IoU-prior**: all same-label pairs with IoU >= threshold sorted by IoU, assigned greedily;
  confidence plays no role. In `compute_map` results keep confidence-sorted order.
- **Hungarian**: `scipy.optimize.linear_sum_assignment` on negative IoU, O(N^3) per image.
- **Calibration**: `per_class` tunes one threshold per class; `global` picks one max-mean-F1
  threshold. A warning is logged when a chosen threshold equals the minimum prediction confidence.
- **Ultralytics backend**: numpy port of `BaseValidator.match_predictions` + `box_iou`, then
  Ultralytics `ap_per_class`. Unthresholded P/R/F1 are read off the smoothed 1000-point curve at
  the global max-mean-F1 point (IoU 0.50). CM via a port of `ConfusionMatrix.process_batch`
  (conf 0.25, IoU 0.45), transposed to row=GT/col=pred. `model.val()` parity needs identical scope,
  order, preprocessing and version.
- **TorchMetrics backend**: `MeanAveragePrecision` (pycocotools) `map_50/map_75/map`; P/R/F1 from
  the IoU-0.50 101-point curve at the per-class max-F1 recall. No CM (`cm is None`). Float32 boxes
  and stable float32 score order; retention compares original confidences.
- **Backend calibration** (`calibration_split`): thresholds chosen on the calibration split, then
  eval-split P/R/F1 re-matched on retained detections; AP stays over all predictions. IoUs are
  cached; only groups changed by an observed-score event update. Global and COCO per-class searches
  are exact (highest equal optimum). Mixed-class Ultralytics per-class calibration uses
  deterministic coordinate ascent on realized macro-F1 (local optimum only; ties accept only
  higher thresholds; class order follows the vocabulary). Without `calibration_split` a backend picks
  its own operating point on the eval split; `find_best_confs` does not apply; only classes with GT
  are scored.
- **Transliteration**: GT labels transliterated forward with `common`/`gost`/`icao` (+ `ё`->`е`),
  normalised (lower-case, alphanumerics only) and indexed; a prediction label matches by exact key,
  then `difflib` at `translit_match_cutoff` (0.8). Ambiguous keys stay unresolved; misses are warned.
- **Inference**: `predict_to_dataframe` chunks by `batch` (default 16); afterwards it releases
  results, drops predictor references, moves the model to CPU and requests CUDA cache cleanup.

## Public API contract

All must exist after any refactor; keyword-only after `iou_threshold` (`Evaluation`), `split`
(`evaluation(...)`), `weights` (`predict_to_dataframe`), `preds_df` (`compute_detection_metrics`),
`task` (`ClearMLTracker`).

- `Evaluation(preds_df, split_df, iou_threshold, preprocess, skip_cohen_kappa, matching_strategy,
  preprocess_preds_conf_threshold, preprocess_preds_nms_containment_threshold,
  preprocess_preds_nms_iou_threshold, ap_method, confidence_optimization, weights_path, backend,
  predict_kwargs, transliterated_labels, translit_match_cutoff, scoring, preprocessing, inference)`.
  `preds_df=None` predicts from `weights_path` over only the needed splits (`ValueError` without
  weights). A grouped config replaces its whole group of flat kwargs; `backend` stays flat.
- `ScoringConfig(iou_threshold, matching_strategy, ap_method, confidence_optimization,
  skip_cohen_kappa)`, `PreprocessConfig(dedup_gt, conf_threshold, nms_containment_threshold,
  nms_iou_threshold)` (`dedup_gt` <- flat `preprocess`), `InferenceConfig(weights_path, predict_kwargs)`.
- `evaluation(split, find_best_confs, calibration_split)`; `predict_to_dataframe(weights, split,
  conf, iou, imgsz, batch, device, image_name, **model_kwargs)` (`ImportError` without the extra,
  `ValueError` without `image_path`/`split` columns); `compute_metrics_ultralytics(split)`,
  `compute_metrics_torchmetrics(split)`.
- Attributes: `metrics` (`dict[str, Metrics]`), `detection_metrics` (backend mode only), `cm`,
  `class_labels` (`None` for torchmetrics; dashboards skip the CM sheet), `best_confidences`,
  `unfiltered_matches`, `_raw_preds_df`.
- Methods: `get_dashboards(save_to_excel, path, save_confusion_matrix)`,
  `plot_confidence_intervals(metric, confidence_level, save_path, figsize)`,
  `get_topk_confusions(main_class, k)`, `get_dfs_visualization(find_best_confs, apply_thresholds)`
  (adds `predict_type`: TP/FP/`filtered` for preds, TP/FN for GT).
- Validation raises `ValueError` for missing columns, non-finite or out-of-range confidences,
  malformed labelled boxes, reserved background labels, duplicate/missing/sentinel indices,
  normalized image-ID collisions and images owned by several splits.
- `Metrics`: `tp, fp, fn, confidence, ap50, ap75, ap50_95, cohen_kappa, precision, recall,
  f1_score, perebrak, nedobrak, *_ci_lower, *_ci_upper`, `counts_observed`; backend-adapted
  unthresholded rows use `cohen_kappa = -1`. `PredictMatch` has computed `type` and optional `iou`.
- Aliases: `APMethod`, `MatchingStrategy`, `ConfidenceOptimization` (`per_class`|`global`),
  `Backend` (`ultralytics`|`torchmetrics`); `YoloMetrics` aliases `DetectionMetrics`.
- Subpackage functions: `preprocess.filter_by_confidence`, `preprocess.apply_nms(preds_df, same_class_containment_threshold,
  cross_class_iou_threshold)`, `scoring.compute_map(gt_df, preds_df, metrics, split_image_names, method,
  strategy)`, `matching.match_boxes(gt_df, preds_df, iou_threshold, strategy, split_image_names)`.
- Top-level functions: `compute_detection_metrics(gt_df, preds_df, backend, classes, split_image_names)` (unknown
  backend -> `ValueError` before heavy imports), `compute_ultralytics_metrics` /
  `compute_torchmetrics_metrics(..., conf_threshold: None|float|dict)`,
  `find_ultralytics_confidence` / `find_torchmetrics_confidence(..., mode)`,
  `compute_ultralytics_confusion_matrix(..., conf, iou_thres)` -> `(nc+1, nc+1)` matrix and labels,
  `transliterate(text, scheme)`, `restore_labels(labels, gt_labels, cutoff)`, `TRANSLIT_SCHEMES`.
- ClearML: `ClearMLTracker(task, project_name, task_name, output_uri, attach_logs, log_level,
  **task_init_kwargs)` with `log_evaluation(evaluation, *, iteration, artifacts_dir, save_to_excel,
  save_confusion_matrix)`, `log_scalars`, `log_artifacts`, `log_plots`, `attach_loguru`,
  `detach_loguru`, `close`, context manager; `clearml` imported only when the tracker creates its
  task. `summarize_metrics(metrics) -> (df, means)` is ClearML-free.
- 0.7.0 shared exports: `evaluate_fixed`, `FixedEvaluationResult`, `compare_paired`,
  `mcnemar_recall`, `bootstrap_precision_delta`, `bootstrap_recall_delta`,
  `adjust_benjamini_hochberg`, `BHResult`, `PairedClassResult`, `PairedComparisonResult`,
  `TestResult`, `render_evaluation`, `RenderedEvaluation`, `RenderedPlot`, `PRObservation`,
  `build_pr_observations`.

## Operations

- **Local evaluation**: `uv run python scripts/eval.py` loads `fixtures/ground_truths_all.csv` and
  `fixtures/predicts_all.csv`, preprocesses (conf 0.1, containment 0.9, NMS IoU 0.6), calibrates on
  `val`, evaluates `test` at IoU 0.3 with `greedy`/`continuous`, and rewrites `fixtures/eval_*`.
- **Release recovery**: attempt state is `digital-metrics-release.json` under the Git directory; a
  shared `digital-metrics-release.lock` directory serialises worktrees. Inspect `owner.json` and
  confirm the process stopped before removing a stale lock. Details: [docs/local-release.md](docs/local-release.md).
- **Live ClearML**: deploy this project's own ClearML into its own namespace (prefix such as
  `digital-metrics-it-<id>`, never a protected name), with a `ResourceQuota`, `LimitRange`, default
  storage class and teardown; reach it by `kubectl port-forward`. Namespaces are deleted 48 hours
  after creation or after a restart, so tooling must recreate them. Request `nvidia.com/gpu: 1` only
  when needed. Protected names: `clearml`, `mongodb`, `ingress-nginx`, `infra-janitor`, `default`,
  `kube-*`. Add the `LimitRange` because a CPU or memory quota rejects pods without requests. Never
  use the dedicated ClearML or FiftyOne MongoDB services (they need an owner-issued bundle, none is
  issued) or this project's retired shared ClearML key pair. Platform rules belong to the k8s-infra
  repository's AGENTS.md and DOCS.md.

## History

The full pre-compression documents are in git at commit 2a3bd54; read one with
`git show 2a3bd54:<path>`. Removed paths: `INFRA_MIGRATION.md`, `docs/agents/issue-tracker.md`,
`docs/agents/triage-labels.md`, `docs/development/api-contract.md`,
`docs/development/architecture.md`, `docs/development/metric-internals.md`; `AGENTS.md` was rewritten.
