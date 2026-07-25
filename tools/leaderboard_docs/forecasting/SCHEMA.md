# Forecasting submission substrate — schema

This is the schema of the file a **submitter uploads** for the Track-3
(forecasting) leaderboard: one per-method, per-user metric parquet plus a small
display sidecar.

```
forecasting/<method>.parquet      # per-user substrate (this file)
forecasting/<method>.meta.json    # display sidecar (below)
```

The parquet is produced by the evaluation run — pass both `output_dir=` and
`method_name="<method>"` to `openmhc.evaluate_forecasting` and it writes
`per_user_errors.parquet` with the `model` column set to `<method>`. Upload that
file verbatim (renamed to `<method>.parquet`); do not hand-author it. The
maintainers concatenate every `forecasting/*.parquet` and recompute paired skill,
fair skill, and average rank against the `seasonal_naive` baseline during
ingestion, so the columns and dtypes below must match exactly.

> **`method_name` is required and must equal your `<method>` stem.** It defaults
> to `"custom"`, and ingestion **groups rows by the `model` column** — so a
> submission left at the default collides with every other default submission and
> is scored under the wrong identity. This is the `method_name` argument to
> `evaluate_forecasting` (which sets the parquet column), *not* the cosmetic
> `to_submission_yaml(method_name=...)` display name. It must differ from the
> baseline name `seasonal_naive`.

> For the separate **bootstrap reference** frame (per-draw CIs, not the
> submission file), see [`bootstrap/SCHEMA.md`](bootstrap/SCHEMA.md).

## `<method>.parquet`

One row per `(model, group, metric, channel_idx, channel_name, user_id)`. Single
Parquet, dictionary-encoded string columns, **`float64`** metric value, `zstd`
compression. Unlike Track 2 (which stores the error `E_per_user`), the forecasting
substrate stores the **raw** per-user metric value, so one file serves all three
reducers — skill and fairness convert it to an error on load, rank uses it
directly.

| column | type | description |
|---|---|---|
| `model` | string (dict) | your method identifier; **must equal the `<method>` filename stem** — set it via `evaluate_forecasting(method_name="<method>")` (defaults to `"custom"`) |
| `group` | string (dict) | `continuous` (`ch 0`–`6`, scored on `mae`) or `binary` (`ch 7`–`18`, scored on `auroc`) |
| `metric` | string (dict) | scored metric: `mae` (continuous) or `auroc` (binary) |
| `channel_idx` | int16 | sensor channel index (0–18); see [Channel categories](#channel-categories) |
| `channel_name` | string (dict) | human-readable channel name |
| `user_id` | string (dict) | participant id from the canonical split |
| `metric_value` | float64 | RAW per-user metric, micro-pooled over the user's forecast windows (`Σcell / Σcount`) |
| `n_values` | int64 | finite horizon-cell count behind `metric_value` |

### Channel categories

The headline `overall` scope is **category-balanced** — each category is weighted
once regardless of how many channels it holds (so the 10 `workout` channels do not
outvote the 2 `sleep` ones). Categories partition `group`: `continuous` =
`activity` + `physiology`, `binary` = `sleep` + `workout`.

| category | `channel_idx` |
|---|---|
| `activity` | 0, 1, 2, 3, 4 |
| `physiology` | 5, 6 |
| `sleep` | 7, 8 |
| `workout` | 9–18 |

Source of truth: `CATEGORY_SCOPES` in `src/forecasting_evaluation/metrics/metric_spec.py`.

### Value semantics

- **`metric_value`** — the raw per-user metric, *before* any cross-user reduction
  and *before* any error conversion. Continuous channels: per-user MAE. Binary
  channels: per-user AUROC. Stored at **float64** for byte-exact reproduction of
  the published aggregates.
- **On load the maintainers convert to an error**: continuous `error =
  metric_value` (MAE, lower is better); binary `error = max(1 − auroc, 0.005)`
  (the `0.005` floor keeps perfect-AUROC users finite). Rank uses `metric_value`
  directly. Skill is the paired geomean of clipped per-user error ratios vs the
  baseline.
- **Structurally-absent cells are omitted, not NaN-filled** — a `(channel,
  metric, user)` with no finite horizon cells simply has no row; the grid is not a
  full cartesian product. The shipped Seasonal-Naive baseline
  (`src/openmhc/data/baselines/forecasting_seasonal_naive_per_user_errors.parquet`)
  is the canonical shape your submission should mirror.

### No subgroup rows — fairness is joined server-side

Unlike Track 2, the forecasting substrate is keyed by `user_id` only and carries
**no `subgroup_attr` / `subgroup_value` columns**. Submitters do **not** ship
subgroup rows — `evaluate_forecasting` emits exactly the columns above — and the
fairness skill score is joined at scoring time.

That join needs **no private data**: every forecasting user also appears in
`imputation/locf.parquet`, which carries per-user `age_group` / `sex` subgroup
rows under the canonical 18/30/40/50/60 age bins. Reusing that partition gives
forecasting fairness the identical user→subgroup mapping as imputation, with no
extra artifact. This is exactly what the live leaderboard does, and it
reproduces the published `S_fair` point estimates:

```python
demo = {}                                     # {user_id: {age_group, sex}}
imp = pd.read_parquet(
    hf_hub_download(REPO, "imputation/locf.parquet", repo_type="dataset"),
    columns=["subgroup_attr", "subgroup_value", "user_id"])
for attr in ("age_group", "sex"):
    sub = imp[imp.subgroup_attr.astype(str) == attr][
        ["user_id", "subgroup_value"]].drop_duplicates()
    for uid, val in sub.itertuples(index=False):
        demo.setdefault(str(uid), {})[attr] = str(val)

err  = to_error_df(substrate, user_col="user_id")   # canonical error conversion
fair = compute_fair_skill_scores_from_errors(
    err, demo, baseline_method="seasonal_naive")
fair[fair.scope == "overall"]                       # == published S_fair points
```

> **Use the `_from_errors` entry point.** Track 3 passes **per-user** errors plus
> a `demographics` dict; Track 2 passes **per-cell mean** errors to
> `compute_fair_skill_scores`. Mixing the two up returns plausible but wrong
> numbers with no error raised. Convert with `to_error_df` rather than by hand.

## `<method>.meta.json`

Tiny display sidecar the leaderboard reads to render the row:

```jsonc
{
  "display_name": "My Forecaster",   // shown in the leaderboard
  "type": "Deep Learning",           // category label
  "submitter": "Stanford CS",        // lab / team
  "subtrack": "other",               // Track 3 has no single-day/long-context split
  "fallback_rate": 0.0               // Seasonal-Naive substitution fraction (see below)
}
```

### `fallback_rate`

Scalar in `[0, 1]`. The run's `overall_fallback_rate` (mirrors
`openmhc._results.ForecastingResults.overall_fallback_rate`): the fraction of
forecast cells the model emitted as non-finite (NaN), which the harness then
substituted with the **Seasonal-Naive** baseline before scoring.

- **`0.0`** — every forecast cell was finite; the harness never substituted.
- **`>0`** — the model returned NaN at that fraction of forecast cells and the
  harness filled them with Seasonal-Naive. Treat headline scores cautiously when
  `fallback_rate > ~5%`, since the metric is inflated with baseline performance
  on the substituted positions.

The field is added to the sidecar by `tools/upload_leaderboard_substrate.py` in
one of two ways:

1. Explicit: `--fallback-rate FLOAT`
2. Auto: `--results-json PATH` — the tool reads the run's `results.json` and
   extracts the top-level `overall_fallback_rate`.

Existing sidecar fields are preserved on update (the tool fetches the current
sidecar from HF and merges only the provided fields).

## Conventions

- Evaluated against the canonical split `sharable_users_seed42_2026` (`test`).
- Track-3 baseline for skill / fairness: `seasonal_naive` (server-side; do not
  submit it).
- Scored set: `mae` (continuous channels 0–6) + `auroc` (binary channels 7–18);
  ratio clip `[0.01, 100]`; within-user aggregation `micro`; unit `user`.
- Fairness disparity primitive: mean absolute pairwise difference (MAPD).
- Must use the standard evaluation protocol — canonical dataset version, split
  file, sample-index, and horizon (24 h).

## Uploaded with

Submitters open a PR on the dataset with `HfApi().upload_folder(...,
create_pr=True)` (see the repo README, "Submit to the Leaderboard").
Maintainers can use `tools/upload_leaderboard_substrate.py` in the
[code repo](https://github.com/AshleyLab/myheartcounts-dataset).
