# Metric internals (maintainer reference)

## Contents

- [Standard metrics (YOLO-compatible)](#standard-metrics-yolo-compatible)
- [mAP](#map)
- [AP Methods (`APMethod`)](#ap-methods-apmethod)
- [What is NOT identical to YOLO](#what-is-not-identical-to-yolo)
- [External metrics backends (optional, single entry point)](#external-metrics-backends-optional-single-entry-point)
- [ClearML tracking (optional, separate layer)](#clearml-tracking-optional-separate-layer)
- [Box Matching — Three Strategies](#box-matching--three-strategies)

Invariants that every change must respect are summarised in `AGENTS.md` § Invariants; this file is the full reference.

## Standard metrics (YOLO-compatible)

- **IoU** — `intersection_area / union_area`
- **TP** — IoU ≥ threshold, correct label, GT not yet matched (one match per GT)
- **FP** — no GT matched (IoU below threshold, or all candidates already taken)
- **FN** — GT box with no matching prediction
- **Precision** — `TP / (TP + FP)`
- **Recall** — `TP / (TP + FN)`
- **F1** — `2 * P * R / (P + R)`
- **perebrak** — `1 - precision` (domain term; false-positive rate)
- **nedobrak** — `1 - recall` (domain term; miss rate)
- **CI** — Wilson interval on precision / recall

## mAP

mAP is computed **independently** from the per-threshold matching used for P/R/F1.
It uses `_raw_preds_df` (unpreprocessed predictions) and runs its own inner matching
loop for each IoU threshold, mirroring the Ultralytics two-path design:

- Sort all predictions by confidence descending (globally per class).
- Compute each image's pred↔GT IoU matrix once per class (`_precompute_image_matches`)
  and reuse it across all ten thresholds — the IoU is threshold-independent, so this
  avoids ~10x redundant IoU work (the assignment kernel still runs per threshold).
- For each IoU threshold in `[0.50, 0.55, …, 0.95]` (10 values):
  - Match using the configured `strategy` (greedy, iou_prior, or hungarian).
  - Accumulate cumulative TP/FP → precision-recall curve.
- **AP** — area under P-R curve, method configurable (see AP Methods below).
- **mAP50** — AP at IoU = 0.50
- **mAP75** — AP at IoU = 0.75
- **mAP50-95** — mean of AP over all 10 thresholds

Classes with **no GT instances in the evaluated split** receive `float("nan")`
for `ap50`, `ap75`, and `ap50_95` — not `0.0`. This allows `nanmean` to correctly
exclude absent classes from averages.

**Preprocessing split**: confidence filtering and NMS are applied to `self.preds_df`
(used for P/R/F1/CM), but `compute_map` always receives `self._raw_preds_df`
(unfiltered original predictions), matching the Ultralytics design.

## AP Methods (`APMethod`)

Two AP integration methods are available via `ap_method=` on `Evaluation`. The
`Evaluation` constructor defaults to `"interp"` (YOLO-like); the lower-level
`compute_map` still defaults to `"continuous"`.

- **`"interp"`** (Evaluation default) — 101-point COCO interpolation,
  Ultralytics-compatible sentinels (`mpre[0] = 1.0`, `mrec[-1] = recall[-1] + 1e-4`),
  integrates with `scipy.integrate.trapezoid` over 101 equally-spaced recall points. Returns
  0.0 on empty recall.
- **`"continuous"`** — VOC 2010+ rectangle-area integration. Prepends `(0, 0)`
  and appends `(1, 0)` sentinels, right-to-left precision envelope, sums rectangle
  areas at recall change points.

The integration methods can produce different AP; compare their results on the
same inputs rather than assuming parity.

## What is NOT identical to YOLO

Precision, recall, F1, and the confusion matrix are computed from `match_boxes`
which runs once at a single IoU threshold with optional confidence filtering and
label-aware TP classification. YOLO does not expose per-threshold P/R/F1 in the
same way. These metrics are intentionally custom. Do not try to make them
numerically match YOLO's console output.

## External metrics backends (optional, single entry point)

`backends/external.py` exposes one dispatcher for library-backed metrics:

```python
Backend = Literal["ultralytics", "torchmetrics"]

def compute_detection_metrics(
    gt_df, preds_df, *, backend="ultralytics", classes=None, split_image_names=None,
) -> dict[str, DetectionMetrics]:
    ...
```

Both backends return `dict[str, DetectionMetrics]` (per-class
`precision/recall/f1/ap50/ap75/ap50_95`), scored only on classes with at least one
GT box. Each backend is a heavy **optional extra** (both pull in `torch`),
imported lazily; the core install stays torch-free. The underlying functions
(`compute_ultralytics_metrics`, `compute_torchmetrics_metrics`) are also public
and callable directly.

`Evaluation` is the unified entry point over these backends: pass
`backend="ultralytics"`/`"torchmetrics"` (or call
`evaluation.compute_metrics_ultralytics(split)` /
`compute_metrics_torchmetrics(split)` directly). In `backend` mode the call scores
the split over `_raw_preds_df`, stores the raw result on `detection_metrics`, and
adapts it onto native `Metrics` for dashboards. Explicit thresholds carry exact
counts and Wilson intervals; unthresholded or missing summaries have unavailable
counts/intervals (`counts_observed=False`). By
default a backend self-selects its operating point on the eval split;
`find_best_confs` does not apply. Passing `calibration_split` enables proper
"calibrate on val, report on test" for **both** backends: realizable
thresholds are selected on the calibration split (`find_ultralytics_confidence` /
`find_torchmetrics_confidence`, per `confidence_optimization` mode) and
`compute_ultralytics_metrics(..., conf_threshold=)` /
`compute_torchmetrics_metrics(..., conf_threshold=)` reads the eval split's P/R/F1
at that confidence by re-matching retained detections. AP stays over the full
prediction set. Calibration caches IoUs and updates only groups changed by an
observed-score event, preserving complete ties. Scalar global and independent
COCO per-class searches are exact, selecting the highest equal optimum.
Mixed-class Ultralytics per-class calibration refines the actual threshold
dictionary with deterministic coordinate ascent on realized macro-F1, holding
other thresholds fixed and repeating until no coordinate improves. Equal
objectives accept only higher thresholds. It establishes a local coordinate
optimum, without an exhaustive joint optimality claim; class order follows the
vocabulary and candidates are own-class scores plus the highest dataset score.
COCO groups by image/class; Ultralytics rematches the complete affected image
in original mixed-class row order, preserving equal-IoU tie behavior.
TorchMetrics uses float32 boxes and stable float32 score order,
while retention compares original confidence values.
The `"ultralytics"` backend also fills `cm` / `class_labels` via
`compute_ultralytics_confusion_matrix` (Ultralytics' `ConfusionMatrix.process_batch`,
ported; conf 0.25 / IoU 0.45, transposed to the sklearn row=GT/col=pred convention);
`"torchmetrics"` has no CM (`cm` is `None`).

- **`"ultralytics"`** (`backends/ultralytics_metrics.py`, the `ultralytics` extra: `uv sync --extra ultralytics`; user-facing install name `pip install digital-metrics[ultralytics]`) — YOLO-comparable. Boxes go through a faithful
  per-threshold re-match (`_match_predictions`, a numpy port of
  `BaseValidator.match_predictions`) and Ultralytics' `box_iou`, then Ultralytics'
  own `ap_per_class`. Equivalence to `model.val()` requires identical prediction/GT
  scope and order, preprocessing and backend version. P/R/F1 are read off the
  smoothed 1000-point P-R curve at the single global max-mean-F1 threshold (IoU
  0.50) for unthresholded summaries. Native assignment can differ; see the
  backend-specific comparison contracts in `docs/why_prf1_differs.md`.
- **`"torchmetrics"`** (`backends/torchmetrics_metrics.py`, the `torchmetrics` extra: `uv sync --extra torchmetrics`; user-facing install name `pip install digital-metrics[torchmetrics]`) — general COCO mAP from torchmetrics'
  `MeanAveragePrecision` (pycocotools backend). AP is torchmetrics' own
  `map_50`/`map_75`/`map` per class. torchmetrics is AP-native with no headline
  P/R/F1, so we derive them the YOLO way: off its `extended_summary` IoU-0.50
  101-point P-R curve at the per-class max-F1 recall point. Calibration uses the
  raw retained detections to report realizable P/R/F1 at explicit thresholds,
  with exact TP/FP/FN and observed-score calibration tie groups.

`YoloMetrics` is kept as a backward-compatible alias of `DetectionMetrics` (same
fields); `compute_ultralytics_metrics` still returns those objects.

## ClearML tracking (optional, separate layer)

`tracking/clearml_tracker.py` is a **standalone** layer sitting on top of a
finished `Evaluation` — the core evaluation code knows nothing about ClearML.
`clearml` is an optional extra (the `clearml` extra: `uv sync --extra clearml`; user-facing install name `pip install digital-metrics[clearml]`), imported
lazily; the core install stays torch/ClearML-free.

```python
from digital_metrics import ClearMLTracker

with ClearMLTracker(project_name="detector", task_name="run-42") as tracker:
    evaluation("test", calibration_split="val")
    tracker.log_evaluation(evaluation)   # scalars + artifacts + plots
```

`ClearMLTracker(task=None, *, project_name, task_name, output_uri, attach_logs,
log_level, **task_init_kwargs)` — pass an existing `Task` or let it `Task.init`
one. `clearml` is imported **only** when it creates the task, so an injected
`task` (tests) needs no extra. `attach_logs=True` (default) installs a `loguru`
sink forwarding run logs to the ClearML console. `log_evaluation(evaluation, *,
iteration, artifacts_dir, save_to_excel, save_confusion_matrix)` runs
`get_dashboards` once and mirrors four things:

- **Scalars** — per-class P/R/F1/mAP as scalar plots, the per-class metrics table,
  and headline means (`mean_*`, nan-aware for AP) as single values.
- **Artifacts** — the analyst/production dashboard DataFrames, `best_confidences`,
  the confusion matrix, and the Excel files `get_dashboards` wrote.
- **Plots** — the four CI PNGs as images + the confusion matrix as a CM plot.
- **Logs** — via the loguru sink.

`summarize_metrics(metrics) -> (df, means)` is the torch/ClearML-free helper
(per-class DataFrame + nan-aware means) it uses; also public. The layer methods
(`log_scalars` / `log_artifacts` / `log_plots` / `attach_loguru` /
`detach_loguru` / `close`) can be called individually.

## Box Matching — Three Strategies

`matching/matching.py` exposes three strategies behind a common interface:

```python
MatchingStrategy = Literal["greedy", "hungarian", "iou_prior"]

def match_boxes(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    iou_threshold: float,
    strategy: MatchingStrategy = "greedy",
    split_image_names: list[str] | None = None,
) -> dict[str, list[PredictMatch]]:
    ...
```

`split_image_names` is an internal parameter populated by `Evaluation._call`
from `gt_df["image_name"].unique()`. External callers can pass `None`.

Note: `match_boxes` (and `compute_map`) default to `"greedy"`, but the
`Evaluation` constructor defaults to the YOLO-like `"iou_prior"`.

### Greedy (YOLO-style)

1. Sort predictions by confidence descending.
2. For each prediction, find the highest-IoU unmatched GT.
3. If IoU ≥ threshold → GT consumed; labels match → TP, else → FP with GT label.
4. Otherwise → FP with `gt_label="background"`.
5. Any unmatched GT → FN.

Used for P/R/F1/CM and for the mAP inner loop when selected as the strategy.

### IoU-Prior (Evaluation default, Ultralytics non-scipy style)

1. Find all pred-GT pairs where IoU ≥ threshold **and** labels match.
2. Sort by IoU descending.
3. Assign greedily: each pred and each GT matched at most once; highest-IoU pair wins.
4. Unmatched preds → FP (cross-class closest GT recorded for CM if IoU ≥ threshold
   and labels differ; otherwise `"background"`).
5. Unmatched GTs → FN.

Key difference from greedy: **confidence plays no role in pairing**. A lower-confidence
pred with better IoU wins the GT. Different assignment can change recall and AP.

**In `compute_map`**: since the loop is already per-class, label matching is
automatic — iou_prior simply sorts all pairs with IoU ≥ threshold by IoU and
assigns greedily, keeping the result in confidence-sorted index order for the P-R curve.

### Hungarian (globally optimal)

Uses `scipy.optimize.linear_sum_assignment` on the negative IoU matrix.
Geometry-first, confidence-independent. More expensive: O(N³) per image.

**Supported in `compute_map`** — runs `linear_sum_assignment` per image then maps
results back to confidence-sorted index order for the P-R curve.
