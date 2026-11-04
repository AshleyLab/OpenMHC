# Downstream bootstrap reference — schema

The Track-1 (outcome prediction) bootstrap reference is two files:

```
downstream/bootstrap/draws.parquet      # zstd
downstream/bootstrap/draws.meta.json    # provenance sidecar
```

This is the companion to the per-method `downstream/<method>.parquet` substrate
(see `../SCHEMA.md`). The substrate is the raw per-user pairs; this is the
**Phase-1 per-draw error frame** the skill / rank CIs reduce from, so a
consumer can recompute the leaderboard intervals without re-running the (paired,
1000-draw) bootstrap over the pairs.

## `draws.parquet`

One row per `(method, task, subgroup_attr, subgroup_value, draw)` — the per-draw
error `E` only. Unlike Tracks 2/3, the ratio / rank / skill reductions are **not**
precomputed here; Phase-2 derives all three from `E` (every metric is paired vs the
`linear` baseline using the same draw indices).

| column | type | description |
|---|---|---|
| `method` | string | method identifier (8 values; see `draws.meta.json:methods`) |
| `task` | string | benchmark task name (one of the 32 `BENCHMARK_TASKS`) |
| `task_type` | string | `binary`, `ordinal`, or `regression` |
| `domain` | string | task domain: `Demographics`, `Medical conditions`, `Body metrics and biomarkers`, `Mental well-being`, `Sleep and lifestyle` |
| `subgroup_attr` | string | `all` |
| `subgroup_value` | string | `all` |
| `draw` | int | `-1` for the point estimate (full cohort, no resampling), else the bootstrap-draw index in `[0, n_boot)` |
| `E` | float32 | per-draw error `E = 1 − metric` for this cell, evaluated on the resampled cohort |

### Value semantics

- **`E = 1 − metric`**, where the metric is the task's primary cohort-level score:
  binary = **AUPRC**, ordinal = **Spearman ρ**, regression = **Pearson r**. Lower
  `E` is better, so the paired skill score `S = 1 − geomean_task(E_method / E_linear)`
  (domain-balanced, clipped) and the cross-method rank are well-defined per draw.
- **`draw = -1`** is the point estimate (the metric on the full test cohort); draws
  `0 … n_boot−1` are the paired bootstrap resamples.
- **Paired resamples** — for each task the same `seed=42` resample indices are reused
  across all methods, so per-draw cross-method comparisons (skill ratios, ranks) are
  valid.
- No NaN-filling: a `(task, draw)` cell with an undefined metric simply has no rows.
- Phase 2 filters on `subgroup_attr == "all"`, so draws holding `age_group` and `sex`
  rows from before `S_fair` was retired still work.

## `draws.meta.json`

```jsonc
{
  "n_boot": 1000,
  "seed": 42,
  "baseline": "linear",                                 // skill baseline
  "methods": ["linear", "multirocket", "lsm2", "toto",
              "chronos2", "xgboost", "wbm", "gru_d"],   // 8 entries (incl. the baseline)
  "n_tasks": 32
}
```

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-1 baseline for skill: `linear`.
- Skill is **domain-balanced macro** (mean over the 5 domains' geomean ratios).
- Format: single Parquet, dictionary-encoded string columns, `float32` `E`, `zstd`
  compression.

## Tracks

| dir | track | status |
|---|---|---|
| `imputation/bootstrap/` | Track 2 — Imputation | live |
| `forecasting/bootstrap/` | Track 3 — Forecasting | live |
| `downstream/bootstrap/` | Track 1 — Outcome Prediction (above) | live |
