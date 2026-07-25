# Forecasting bootstrap reference — schema

The Track-3 bootstrap reference is two files:

```
forecasting/bootstrap/draws.parquet      # zstd
forecasting/bootstrap/draws.meta.json    # provenance sidecar
```

## `draws.parquet`

One row per `(reduction, model, scope, metric, draw)` — a single long frame
holding the per-draw values for all three reductions.

| column | type | description |
|---|---|---|
| `reduction` | string (dict) | which reduction the row belongs to: `skill`, `rank`, or `fairness` |
| `model` | string (dict) | method identifier (10 values; see `draws.meta.json:methods`) |
| `scope` | string (dict) | the headline scope (see below) |
| `metric` | string (dict) | scored metric for `rank` rows (`mae` / `auroc` / `overall`); empty `""` for `skill` and `fairness` |
| `draw` | int32 | bootstrap-draw index in `[0, n_boot)` |
| `value` | float32 | the per-draw value of this reduction for `(model, scope[, metric])` |

### `scope` values by reduction

- **skill**: `channel_0_score`..`channel_18_score`, `sleep_score`, `workout_score`,
  `activity_score`, `physiology_score`, `overall_score`.
- **rank**: `channel_<i>`, `sleep`, `workout`, `activity`, `physiology`, `overall`
  (paired with `metric` ∈ `mae` / `auroc` / `overall`).
- **fairness**: `age_group`, `sex`, `overall`, the 4 sensor categories
  (`activity` / `physiology` / `sleep` / `workout`), and `channel_<i>`.

### `value` semantics

- **skill** — paired skill score `1 − exp(mean_task log R)` vs `seasonal_naive`,
  per draw (resampled-user cohort). `R` is the clipped per-task error ratio;
  continuous error = MAE, binary error = `max(1 − AUROC, 0.005)`.
- **rank** — cross-method average rank for the draw (lower error → rank 1),
  meaned over the resampled cohort.
- **fairness** — MAPD disparity-ratio fair skill score for the draw:
  per-task disparity `D = mean(|E_g − E_g'|)` over subgroup pairs, ratio vs the
  baseline's `D`, clipped, geomean-averaged across tasks (category-balanced),
  macro-averaged across attributes.

Resamples are **paired** across methods (one shared `boot_idx` matrix, `seed=42`),
so per-draw cross-method comparisons (skill ratios, ranks) are valid.

### Reducing the fairness draws — two footguns

The published `S_fair` values (paper Table 4, leaderboard) are **point estimates
on the full cohort**, which this file does **not** contain — it holds only the
1000 resampled draws. Reduce them carefully:

- **Do not report the draw mean as the point estimate.** The disparity ratio is
  skewed and downward-biased, so its bootstrap mean sits well below the point.
  For `AutoETS`/`overall` the draw mean is ≈ −2.17 versus a point of ≈ −3.04.
  Recover the point with
  `compute_fair_skill_scores_from_errors(err, demo, baseline_method="seasonal_naive")`
  on the per-user substrate (see [`../SCHEMA.md`](../SCHEMA.md)).
- **Percentile intervals are biased low; the published intervals are BCa.** BCa
  needs the draws **plus** a leave-one-user-out jackknife of the point flow
  (`bootstrap_fair_skill_score(..., bca=True)` returns both). The difference
  changes conclusions: `DLinear`/`overall` has a percentile interval of
  `(−0.052, +0.235)` — spanning zero — but a BCa interval of `(+0.113, +0.322)`,
  which does not.

> **Fairness rows regenerated 2026-07-25.** The originals were computed with the
> legacy **max-min** disparity primitive while the paper and leaderboard use
> **MAPD**; they now agree. `sex` rows are bit-identical either way (MAPD ≡
> max-min for a 2-level attribute); `age_group`, `overall`, the categories and
> the per-channel scopes changed. `skill` and `rank` rows were not touched. See
> `draws.meta.json:fairness_rows_regenerated`.

## `draws.meta.json`

```jsonc
{
  "n_boot": 1000,
  "seed": 42,
  "ci_level": 0.95,
  "splits": ["test"],
  "baseline": "seasonal_naive",
  "methods": ["seasonal_naive", "autoARIMA", ...],   // 10 entries
  "continuous_metrics": ["mae"],
  "binary_metrics": ["auroc"],
  "age_bins": [18, 30, 40, 50, 60],
  "reductions": ["skill", "rank", "fairness"],
  "within_user_aggregation": "micro",
  "aggregation_unit": "user",
  "n_rows": 0,
  "git_commit": "...",
  "timestamp": "..."
}
```

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-3 baseline for skill / fairness: `seasonal_naive`.
- Fairness disparity primitive: **MAPD** (mean absolute pairwise difference);
  for a 2-level attribute (`sex`) this equals the historical max-min.
- Format: single Parquet, dictionary-encoded categoricals, `float32` value,
  `int32` draw index, `zstd` compression.

## Tracks

| dir | track | status |
|---|---|---|
| `imputation/bootstrap/` | Track 2 — Imputation | live |
| `forecasting/bootstrap/` | Track 3 — Forecasting (above) | live |
| `downstream/bootstrap/` | Track 1 — Outcome Prediction | added later |
