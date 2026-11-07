# digital-metrics agent instructions

Object detection evaluation library. Computes detection metrics (precision, recall, mAP50/75/50-95,
F1, Cohen's kappa, confidence intervals) from pandas DataFrames of ground-truth and predictions.
Output: per-class `Metrics` objects, confusion matrix, Excel dashboards, CI plots.

## Reference files (read when needed)

| Read when | File |
|---|---|
| Changing module layout, adding a module or subpackage, or touching imports | [docs/development/architecture.md](docs/development/architecture.md) |
| Changing matching, AP/mAP, confidence calibration, external backends or the ClearML layer, or explaining metric differences | [docs/development/metric-internals.md](docs/development/metric-internals.md) |
| Changing any public name, signature, default, attribute or exception, or doing any refactor (mandatory) | [docs/development/api-contract.md](docs/development/api-contract.md) |
| Release, changelog or hook recovery | [docs/local-release.md](docs/local-release.md) |
| Fixed-population, paired-statistics or rendering APIs | [docs/shared-evaluation.md](docs/shared-evaluation.md) |
| Why the 2026-10-04 contracts exist | [docs/review-remediation-2026-10-04.md](docs/review-remediation-2026-10-04.md) |

## Environment

- **Python**: 3.11+; **package manager**: `uv` with lockfile (`uv.lock`); **virtual environment**: `uv venv` (`.venv/`)

```bash
uv venv
uv sync          # install from lockfile
uv sync --locked --dev   # fresh worktree setup (no .venv yet)
uv add <pkg>     # add dependency (updates pyproject.toml + uv.lock)
uv run pytest    # run tests inside venv

uv sync --extra ultralytics    # optional YOLO-comparable metrics backend (heavy: torch)
uv sync --extra torchmetrics   # optional COCO-mAP metrics backend (heavy: torch)
uv sync --extra clearml        # optional ClearML experiment tracking layer (torch-free)
```

Never use `pip` directly. Always use `uv`. Develop optional backends with `uv sync --extra <name>`;
`pip install` of the package extras appears only in user-facing install hints.

## Checks

Run `uv run --locked --no-sync prek run --all-files`; `.pre-commit-config.yaml` owns the verification tool inventory.

Recommended, not hooked (formatting and coverage are not enforced; the repo has no CI):

```bash
uv run ruff format .                              # format
uv run ruff check . --fix                         # lint + autofix
uv run pytest --cov=src/digital_metrics tests/    # with coverage
```

Ruff config is in `pyproject.toml`. mypy runs in strict mode.

The library-boundary import contract is enforced by `tests/test_shared_architecture.py` (see
`docs/plans/shared-extraction/architecture.json`): the seven scoped modules under `src/digital_metrics/`
(`__init__.py`, `fixed.py`, `comparison.py`, `scoring/ap.py`, `scoring/__init__.py`,
`reporting/dashboard.py`, `reporting/__init__.py`) must not import any module whose top-level name is
`clearml_yolo`, `hydra` or `clearml`.

## Code style

- **Simple over clever**: readable code is preferred over micro-optimisations. Optimise only where profiling shows a bottleneck.
- **Explicit types**: all public functions must have full type annotations.
- **Small, focused functions**: each function does one thing.
- **No dead code**: remove commented-out blocks, unused imports, TODO stubs.
- **Logging**: library code under `src/` uses `loguru`; no bare `print()` calls. Scripts in `scripts/`
  keep their existing printed summary output.
- **No mixed languages in logs/comments**: use English throughout.

## External library documentation

**Always use context7** to pull up-to-date documentation before writing or changing code that touches a
third-party library (e.g. `ultralytics`, `torchmetrics`, `clearml`, `pandas`, `pydantic`). Do not rely on
memory for external APIs — resolve the library and fetch the relevant docs via context7 first, then
implement against what the current version actually exposes.

## Architecture

Modules are grouped into subpackages by role. Each subpackage's ``__init__.py`` re-exports its public
names, so internal and external code imports from the subpackage (e.g. ``from .scoring import
compute_map``), and the top-level ``digital_metrics/__init__.py`` keeps the public API flat
(``from digital_metrics import ...``).
``types.py``, ``ci.py`` and ``validation.py`` stay at the top level as the shared
foundation (``types`` depends on ``ci``), alongside the ``evaluation`` orchestrator,
the ``calibration`` collaborator it delegates threshold selection to, and the
``engines`` it dispatches scoring to (``NativeEngine`` / ``BackendEngine``).

The core install stays torch-free and ClearML-free. `backends/`, `inference/` and `tracking/` import
their extras lazily.

Full module map, tests, scripts and fixtures: `docs/development/architecture.md`.

## DataFrame schema

Both DataFrames share these columns:

| Column | Type | Description |
|---|---|---|
| `image_name` | str | Unique image identifier |
| `instance_label` | str | Class name |
| `bbox_x_tl` | float | Bounding box top-left x |
| `bbox_y_tl` | float | Bounding box top-left y |
| `bbox_x_br` | float | Bounding box bottom-right x |
| `bbox_y_br` | float | Bounding box bottom-right y |
| `split` | str | `train` / `val` / `test` (GT only) |
| `confidence` | float | Detection score (predictions only) |
| `image_path` | str | Full path to the image file (GT only; required only by `Evaluation.predict_to_dataframe`) |
| `image_width` | int | Image width in pixels (GT only; required only when `skip_cohen_kappa=False`) |
| `image_height` | int | Image height in pixels (GT only; required only when `skip_cohen_kappa=False`) |

## Invariants

- mAP is computed independently of the P/R/F1 matching. Confidence filtering and NMS apply to
  `preds_df` (P/R/F1/CM), but `compute_map` always receives `_raw_preds_df` (unfiltered), matching the
  Ultralytics two-path design.
- Classes with **no GT instances in the evaluated split** receive `float("nan")` for `ap50`, `ap75`, and
  `ap50_95` — not `0.0`. This allows `nanmean` to correctly exclude absent classes from averages.
- `Evaluation` defaults to `matching_strategy="iou_prior"` and `ap_method="interp"`; `match_boxes` and
  `compute_map` default to `"greedy"`, and `compute_map` to `"continuous"`.
- Precision, recall, F1, and the confusion matrix are computed from `match_boxes` which runs once at a
  single IoU threshold with optional confidence filtering and label-aware TP classification. YOLO does not
  expose per-threshold P/R/F1 in the same way. These metrics are intentionally custom. Do not try to make
  them numerically match YOLO's console output.
- AP is backend-specific; native assignment can differ from backends (`docs/why_prf1_differs.md`).
- Explicit backend thresholds carry exact TP/FP/FN and Wilson intervals; unthresholded or missing
  (reconstructed) summaries have `counts_observed=False` and unavailable (NaN) Wilson intervals.
- Every public name in `docs/development/api-contract.md` must exist after any refactor.

## Tests

- Use `pytest` + `pytest-cov` (run: see Checks).
- Place fixtures in `tests/conftest.py` and data fixtures in `tests/fixtures/`.
- Cover:
  - IoU edge cases (perfect overlap, no overlap, partial)
  - Greedy matching: hand-computed TP/FP/FN on `tiny_dataset`
  - IoU-prior matching: same `tiny_dataset` counts; discriminating fixture that
    verifies lower-conf/higher-IoU pred wins the GT (opposite of greedy)
  - Hungarian matching: same `tiny_dataset`, verify total TP ≥ greedy TP
  - AP on trivial cases (perfect detector → 1.0; zero-precision → 0.0)
  - Both AP methods (continuous + interp) parametrised; interp empty-recall guard
  - `compute_map` strategy divergence: single GT, two competing preds —
    greedy AP=1.0, iou_prior AP=0.5
  - CI bounds within [0,1], lower ≤ estimate ≤ upper
  - Full `Evaluation` round-trip: check `metrics` keys, `cm` shape,
    and that `ap50` field is populated for each class
  - NMS: `filter_by_confidence` drops rows below threshold; `apply_nms`
    suppresses same-class containment and cross-class IoU duplicates


## Local evaluation

`scripts/eval.py` runs a full end-to-end evaluation against the fixture data:

```bash
uv run python scripts/eval.py
```

What it does:
1. Loads `fixtures/ground_truths_all.csv` and `fixtures/predicts_all.csv`
2. Applies predictions preprocessing (confidence filter + NMS) — affects P/R/F1/CM only
3. Calibrates confidence thresholds on the `val` split
4. Evaluates on the `test` split
5. Writes results to `fixtures/` and prints a per-class summary table

Current settings in the script (`matching_strategy`/`ap_method` are pinned to the documented baseline
`greedy`/`continuous`; the `Evaluation` defaults are YOLO-like `iou_prior`/`interp`):
- `iou_threshold=0.3`
- `matching_strategy="greedy"`, `ap_method="continuous"` (pinned)
- `preprocess_preds_conf_threshold=0.1`
- `preprocess_preds_nms_containment_threshold=0.9`
- `preprocess_preds_nms_iou_threshold=0.6`

The checked-in input CSVs are small synthetic examples, described in [fixtures/README.md](fixtures/README.md).
They do not reproduce the historical large-dataset output files. Original source data are unavailable and
their provenance is not asserted. Running this script overwrites the generated output CSV/JSON files with
current synthetic-example results.

## Versioning and release

Local releases use Python Semantic Release and Conventional Commits on `main`. `fix:` / `perf:` bump
PATCH; `feat:` bumps MINOR. Breaking changes (`!` or `BREAKING CHANGE:`) bump MINOR while the package is
below 1.0. Other commit categories do not trigger releases by themselves. Install the local hooks with
`uv run --locked --no-sync python scripts/local_release.py --install-hooks`.

The release helper updates `pyproject.toml`, the project's version in `uv.lock` and `CHANGELOG.md`, runs
the configured checks, commits the metadata and creates an annotated `vX.Y.Z` tag. It never pushes,
publishes or builds artifacts. See [local release instructions](docs/local-release.md) for manual
invocation, changelog regeneration and recovery. Preserve `docs/changelog-history.md`, the curated
changelog through 0.5.3; newer entries are generated automatically.

## Review contracts (dated 2026-10-04)

`docs/review-remediation-2026-10-04.md` is the dated ledger of the input, backend, lifecycle and reporting
boundary decisions; the live contracts it introduced are stated under Invariants, Local evaluation and
`docs/development/api-contract.md`.
