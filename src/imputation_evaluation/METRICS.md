# Track 2 — Imputation Metric Specification

Canonical formulas and scope set for the OpenMHC imputation leaderboard.
All implementations live in
[`paper_metrics_core.py`](evaluation/paper_metrics_core.py) and
[`bootstrap_skill_rank.py`](evaluation/bootstrap_skill_rank.py); the run
scripts in [`scripts/paper_results/`](../../scripts/paper_results/) are
thin orchestrators.

## TL;DR

The leaderboard publishes two families of numbers per method per split:

| File | Quantity | Reducer | Code |
|---|---|---|---|
| `skill_scores_bootstrap.csv` | **Skill score** (regret vs. baseline) | paired per-user geomean | `compute_skill_scores` |
| `avg_rankings_bootstrap.csv` | **Average rank** (lower is better) | per-user rank, two-stage | `compute_average_rankings` |

Every cell carries `mean / se / ci_lo / ci_hi / n_boot` from a 1000-draw
cluster bootstrap over users.

## Notation

| Symbol | Meaning |
|---|---|
| `m` | method (e.g. `lsm2`, `linear`, `locf`) |
| `b` | baseline method (default: `locf`, see `BASELINE_CONTINUOUS`) |
| `s` | masking scenario (one of 6) |
| `c` | channel (one of 19, or one of 2 collapsed-binary categories) |
| `r = (s, c)` | task |
| `u` | user (cluster unit for the bootstrap) |
| `E_{m,r,u}` | per-task per-user error (defined below) |
| `R_{m,r}` | per-task paired ratio against the baseline |
| `S_{m, scope}` | skill score for one scope |
| `T_{m,r}` | per-task rank of method `m` on task `r` (averaged over users) |

## 1. Per-task per-user error E

Per-task per-user errors are produced upstream by
[`pair_aggregator.aggregate_pairs(..., return_per_user=True)`](evaluation/pair_aggregator.py).

> **Canonical artifact.** `per_user_errors.parquet` — one file per
> method, schema `PER_USER_ERRORS_PARQUET_COLUMNS` (see
> [`evaluation/bootstrap_skill_rank.py`](evaluation/bootstrap_skill_rank.py)).
> Emitted by
> [`evaluation/per_user_errors.py::build_per_user_errors`](evaluation/per_user_errors.py),
> which is the single canonical producer the public API, the
> `mhc-impute-eval` runner, and the paper Phase-0 driver all share.
> The file's content equals the per-(user, channel, cell) E this
> section defines — `_per_user_errors_for_cell` /
> `_per_user_collapsed_errors_for_cell` of the bootstrap path emit
> exactly the same values (parity vs the on-disk paper artifact is
> pinned by
> `scripts/paper_results/imputation/parity/parity_locf.py`).
> §8.2 covers the Phase B fast paths that consume it directly.

**Continuous channels (`ch_0..ch_6`, 7 channels)**

```
E_{m,r,u}  =  MAE  =  ( Σ_t |y_t − ŷ_t| ) / N_{r,u}
```

over the masked positions `(r, u)` (subset of `target_mask == 1`). MAE
matches Track 3 forecasting — the per-task ratio is invariant under MAE
vs RMSE vs nMAE, so the skill score doesn't depend on this choice
(`paper_metrics_core.py:99-107`).

**Per-channel binary channels (`ch_7..ch_18`, 12 channels)**

```
E_{m,r,u}  =  1 − AUC_{m,r,u}     (per-user pooled AUC, then floored)
E_{m,r,u}  ←  max(E_{m,r,u}, BINARY_ERROR_FLOOR)
```

The floor `ε = 0.005` (`bootstrap_skill_rank.py:80`) prevents a
perfect-AUC user from dividing the paired ratio by zero and being
dropped — matches forecasting Track 3.

**Collapsed-binary "channels" (Part D — `cat_collapsed:sleep`, `cat_collapsed:workouts`)**

To avoid the 10-workout-channels-vs-2-sleep-channels weight imbalance under
the per-task geomean, two synthetic tasks are added per scenario:

```
E_{m, s, cat_collapsed:K, u}  =  nanmean_{c ∈ K} (1 − AUC_{m, s, c, u})
```

with the same floor applied after the nanmean
(`pair_aggregator.py:401-419`,
`bootstrap_skill_rank.py:_per_method_cell_collapsed_errors`).

## 2. Per-task paired ratio R

```
r_{m,r,u}  =  clip( E_{m,r,u} / E_{b,r,u},  CLIP_LOWER,  CLIP_UPPER )
R_{m,r}    =  exp(  mean_u  log r_{m,r,u}  )                  (geomean over users)
```

Constants: `CLIP_LOWER = 0.01`, `CLIP_UPPER = 100.0`
(`paper_metrics_core.py:25-26`; same values aliased as
`SKILL_CLIP_LOWER` / `SKILL_CLIP_UPPER` in
`bootstrap_skill_rank.py:77-78`).

Users are dropped per task if `E_{b,r,u} ≤ 0` or non-finite; tasks
require `SKILL_MIN_PAIRS = 1` surviving user. The baseline-vs-self row
(`m = b`) is excluded — `R_{b,r} ≡ 1` by construction (see
`compute_per_task_paired_R` at
`bootstrap_skill_rank.py:888-985`).

## 3. Skill score S

**Per task (leaf scope `task:<s>:<c>`)** — degenerate single-task case:

```
S^{task}_{m, s, c}  =  1 − clip(R_{m, s, c}, CLIP_LOWER, CLIP_UPPER)
```

**Per aggregated scope** — geomean of clipped ratios over the scope's
tasks:

```
S_{m, scope}  =  1 − exp(  mean_{r ∈ scope}  log clip(R_{m,r}, CLIP_LOWER, CLIP_UPPER)  )
```

Higher is better; `S = 0` ⇔ method matches baseline; `S > 0` ⇔ method
beats baseline. Code: `compute_skill_scores(mode="paired")` at
`paper_metrics_core.py:143-341`.

The reducer additionally accepts `mode="pooled"` (legacy
`E_method / E_baseline` on user-macro E) but this is no longer the
leaderboard estimand; it stays reachable from
`compute_imputation_paper_metrics.py --skill-mode pooled`.

## 4. Average rank T

Two-stage form, mirrors forecasting Track 3
(`_average_rankings_per_user` at `paper_metrics_core.py:420-474`):

**Stage 1 — per-user rank, then user mean (`task_rank`)**

For each task `r = (s, c)`:

1. Build the `users × methods` matrix of E values.
2. Rank methods within each user (`method="average"`, ascending — lowest
   E → rank 1, ties averaged).
3. `task_rank_{m,r} = nanmean_u rank_{m,r,u}`

**Stage 2 — mean over tasks in scope**

```
T_{m, scope}  =  mean_{r ∈ scope}  task_rank_{m, r}
```

Per-user ranking is scale-free, so channels with different absolute MAE
magnitudes don't bias the cross-task mean.

**Per task (leaf scope `task:<s>:<c>`)** — degenerate single-task case
(emitted directly from Stage 1):

```
T^{task}_{m, s, c}  =  task_rank_{m, s, c}
```

## 5. Scope catalog

The same scope set is emitted by `compute_skill_scores` and
`aggregate_task_ranks_to_scopes`. Each row in the output CSVs is keyed
by `(method, scope, split)`.

### 5.1 Per-scenario scopes (6 rows per method per split)

All per-scenario scopes use the **3-level B.2 form**: L3 within-bucket
task mean → L2 mean over buckets-present within the scenario. ``n_tasks``
reports the number of buckets present in the scenario.

| scope | buckets present | n_tasks |
|---|---|---|
| `random_noise` | activity, physiology, sleep, workouts | 4 |
| `temporal_slice` | activity, physiology, sleep, workouts | 4 |
| `signal_slice` | activity, physiology, sleep, workouts | 4 |
| `sleep_gap` | activity, physiology (binary excluded by `EXCLUDE_BINARY_SCENARIOS`) | 2 |
| `workout_gap` | physiology (only `ch_5, ch_6` masked) | 1 |
| `intensity_failure` | physiology (only `ch_5, ch_6` masked) | 1 |

In single-bucket scenarios (`workout_gap`, `intensity_failure`) the
3-level form degenerates to the per-channel geomean over `ch_5, ch_6`,
so those values are numerically equivalent to a flat 2-task geomean.

### 5.2 Per-category scopes (4 rows per method per split)

Channel partition (`paper_metrics_core.py:48-53`):

| category | channels | what |
|---|---|---|
| `activity` | `ch_0..ch_4` | iPhone steps/distance/flights + Watch steps/distance |
| `physiology` | `ch_5..ch_6` | Watch heart rate + active energy |
| `sleep` | `ch_7..ch_8` | sleep asleep / inbed (binary) |
| `workouts` | `ch_9..ch_18` | 10 workout-type binary channels |

| scope | aggregation | n_tasks |
|---|---|---|
| `cat:activity` | log-space geomean over `{random_noise, temporal_slice, signal_slice}` × `{ch_0..ch_4}` per-task R | 15 |
| `cat:physiology` | log-space geomean over `{random_noise, temporal_slice, signal_slice}` × `{ch_5, ch_6}` per-task R | 6 |
| `cat:sleep` | log-space geomean over `{random_noise, temporal_slice, signal_slice}` × `cat_collapsed:sleep` per-scenario R | 3 |
| `cat:workouts` | log-space geomean over `{random_noise, temporal_slice, signal_slice}` × `cat_collapsed:workouts` per-scenario R | 3 |

All four are **structural-only** by design — `EXCLUDE_BINARY_SCENARIOS`
strips binary tasks from semantic scenarios, and continuous categories
deliberately mirror the same restriction so all four category scores
share the same scenario denominator.

**Binary `cat:sleep` / `cat:workouts`** read from the synthetic
`cat_collapsed:<cat>` channel keys in the upstream per-user dict
(`pair_aggregator.py:401-419`). Each per-user category E is the
arithmetic mean of `(1 − AUC_channel)` across that user's defined
channels in the category, after which the standard paired-ratio
machinery applies. The per-channel `cat:sleep` / `cat:workouts`
that previously took the geomean over individual binary channels
have been deleted — they overweighted workouts at 10× sleep by
construction.

**Continuous categories are not collapsed** (`activity` has 5
channels, `physiology` has 2 — they "already weight roughly fairly").
So there is no `cat_collapsed:activity` / `cat_collapsed:physiology`
synthetic channel; the bucket-source for `cat:activity` / `cat:physiology`
is the per-channel rows directly.

### 5.3 Cross-scenario scopes (2 rows per method per split)

Both use the same 3-level B.2 form as the per-scenario scopes,
with an additional **L1 log-space mean over scenarios in scope**.
``n_tasks`` reports the number of scenarios contributing.

| scope | scenarios in L1 | n_tasks |
|---|---|---|
| `semantic` | `{sleep_gap, workout_gap, intensity_failure}` | 3 |
| `overall` | all 6 scenarios | 6 |

`overall` is the headline skill / rank surfaced on the live leaderboard
(computed by the OpenMHC HF Space's `leaderboard_compute.py` from the
substrate parquets at `MyHeartCounts/OpenMHC-leaderboard-data:imputation/`).
The new `overall` is the 3-level form universally applied.

### 5.3.1 The universal 3-level B.2 form

All cross-scenario scopes (`semantic`, `overall`) and all per-scenario
scopes share one operator:

```
L3 (within bucket × scenario):  bucket_log_R[m, s, b] = mean over tasks(s, b)
                                                       of log(clip(R_task))
L2 (within scenario):           scenario_log_R[m, s]  = mean over buckets-present
                                                       of bucket_log_R[m, s, b]
L1 (across scenarios):          scope_log_R[m]        = mean over scenarios-in-scope
                                                       of scenario_log_R[m, s]
S_scope[m]                       = 1 − exp(scope_log_R[m])
```

Per-scenario scopes (`random_noise`, `temporal_slice`, `signal_slice`,
`sleep_gap`, `workout_gap`, `intensity_failure`) use only L3+L2 — there
is no L1 axis because there's only one scenario. `semantic` uses L1
over the 3 semantic scenarios; `overall` uses L1 over all 6 scenarios.

Each of the four sensor categories contributes once per scenario via L2,
regardless of how many constituent channels it has:

where `C = {activity, physiology, sleep, workouts}`, `K = |C ∩ buckets present|`,
and the per-bucket geomeaned ratio is

```
R_c  =  exp(  (1/n_c) · Σ_{r ∈ tasks(c)}  log(R_task_r)  )
```

— at the per-scenario grain. The **bucket source** is fixed
(`paper_metrics_core._bucket_log_R_per_scenario`):
- `activity` ← continuous per-channel rows for `ch_0..ch_4`.
- `physiology` ← continuous per-channel rows for `ch_5..ch_6`.
- `sleep` ← `cat_collapsed:sleep` synthetic channel key (one row per
  scenario, value `nanmean(1 − AUC_channel)` over `ch_7..ch_8`).
- `workouts` ← `cat_collapsed:workouts` synthetic channel key (one row
  per scenario, value `nanmean(1 − AUC_channel)` over `ch_9..ch_18`).
- Per-channel binary rows (`ch_7..ch_18` with `channel_type == "binary"`)
  are **dropped** — they reach the headline only via the collapsed rows
  so per-channel binary would double-count.

Why 3-level (and not flat): a flat geomean over the underlying ~38 tasks
would weight activity 6.7× heavier than sleep (20 vs. 3 tasks) and weight
scenarios proportional to their task counts. The 3-level form gives each
bucket equal voice within each scenario and each scenario equal voice in
the headline.

Rank side mirrors the skill side: L3 per-(scenario, bucket) mean of
`task_rank`; L2 mean over buckets-present per scenario; L1 arithmetic
mean over scenarios in scope (no log/exp because rank is linear).
``n_tasks`` reports the same axis it does on the skill side — buckets
for per-scenario scopes, scenarios for cross-scenario scopes.

### 5.4 Per-task leaf scopes (`task:<s>:<c>`)

One row per `(method, scenario, channel)` cell in §5.1 plus one per
`(method, scenario, cat_collapsed:K)` synthetic-channel cell:

```
68  per-channel task scopes
 +  6  collapsed-binary task scopes
= 74 leaf scopes per method per split
```

For these leaves `n_tasks = 1`,
`skill_score = 1 − clip(R)` (single-task case of the geomean formula),
`avg_rank = task_rank` (single-task case of the Stage-2 mean).

## 6. Per-scenario channel inventory (correctness contract)

The pair writer only writes rows for actually-masked positions, so each
scenario's task set is determined by its mask generator's
`mask_channels` argument
([`masking/*.py`](masking/)):

| Scenario | Mask channels | Evaluated channels |
|---|---|---|
| `random_noise` | random patches per channel | `ch_0..ch_18` |
| `temporal_slice` | contiguous blocks, all channels | `ch_0..ch_18` |
| `signal_slice` | whole channels per day | `ch_0..ch_18` |
| `sleep_gap` | all except `ch_7, ch_8` during sleep | `ch_0..ch_6`*, `ch_9..ch_18`*\*\* |
| `workout_gap` | `ch_5, ch_6` during workouts | `ch_5, ch_6` |
| `intensity_failure` | `ch_5, ch_6` when HR > threshold | `ch_5, ch_6` |

\* Binary `ch_9..ch_18` rows are produced but dropped by
`EXCLUDE_BINARY_SCENARIOS` before any aggregated scope — semantic
scenarios are continuous-only in the leaderboard.

This restriction is inherited automatically by `task:*` scopes; no
per-scenario allowlist exists in the kernel.

## 7. Subgroup (cohort) reporting

Per-user and draw files keep `subgroup_attr` and `subgroup_value`, holding the
constant `all`. A cohort view is a membership filter of the `all` rows: take each
participant's `age_group` and `sex` from `labels.user_demographics`, keep the
rows of the participants in the cohort, and run the skill and rank reducers on
them. Age and sex rows are no longer produced.

Files published before `S_fair` was retired also hold `age_group` and `sex`
rows. The reducers and table generators read only the `all` rows of such files,
so they need no conversion.

## 8. Bootstrap (cluster-on-user)

**Phase 1 — `bootstrap_imputation_draws.py`**

For each split, draw `n_boot = 1000` resample matrices `M ∈ N_users × n_boot`,
each column a multiset bootstrap over the canonical user set.
The matrix is **shared across all scenarios within the split** so
cross-scenario covariance is preserved
(`bootstrap_skill_rank.py:_seed_for_split:384-396`,
seeded by `SHA-256("seed|split")` so the draw is reproducible across
runs).

For each draw `(b)`:
- recompute per-task per-user paired ratios on the resampled cohort
- aggregate to `R^{(b)}_{m, r}` (geomean over users, §2)
- compute per-task per-user ranks → `task_rank^{(b)}_{m, r}` (Stage 1, §4)

For binary channels, the code precomputes the per-(user, channel) pooled
AUC matrix once per `(method, scenario)` and reuses it across the
per-channel `E`, paired-ratio `R`, collapsed-binary, and rank reducers
(`_per_user_auc_from_cell_stats`, wired in the Phase 1 loop at
`bootstrap_skill_rank.py:399-434, 1355-1363`).

Output: `bootstrap_draws.parquet` with one row per
`(method, scenario, split, channel, subgroup_attr, subgroup_value, draw)`
carrying `E`, `R`, `rank`. `subgroup_attr` and `subgroup_value` hold the
constant `all` (§7).

**Phase 2 — `aggregate_imputation_paper_metrics.py`**

Calls `compute_skill_scores(mode="paired")` and
`aggregate_task_ranks_to_scopes` once per draw, stacks the per-draw scope
tables, then for each `(method, scope, split)` cell summarizes across
draws:

```
mean        =  nanmean_b   X^{(b)}
SE          =  nanstd_b    X^{(b)}     (ddof=1)
ci_lo, ci_hi = percentile_b X^{(b)} at (α/2, 1 − α/2)  with  α = 0.05
n_boot      =  count of finite draws in this cell
```

(`_summarize` in `bootstrap.py`, reused by Phase 2.)

CI level is 0.95 by default (set in `sweep_methods.yaml:ci_level`).
Pairing via the shared resample matrix controls sampling covariance. Skill / rank
rows are percentile-only; a `bca_skill_rank` knob exists but is not implemented
(§S7).

### 8.1 Recomputing against a subset of methods

The Phase 0 → Phase 1 → Phase 2 boundary is well-defined:
`bootstrap_draws.parquet` is the canonical Phase-1 output (plus its sibling
`per_user_errors.parquet`, emitted by
`bootstrap_imputation_draws.py --per-user-errors`), and any subset of the
methods inside them can be re-aggregated in Phase 2 alone — no Phase 0
(per-method eval) or Phase 1 (resample) rerun needed.

The Phase 2 aggregator takes a `--method-filter` flag that pre-filters
the parquet rows before any reducer runs:

```bash
python scripts/paper_results/aggregate_imputation_paper_metrics.py \
  --draws  ${PAPER_OUT}/bootstrap_draws.parquet \
  --output-dir ${PAPER_OUT}/subset-A/ \
  --method-filter locf lsm2 lsm2_weekly_sparse linear brits \
  --baseline-method locf \
  --clip-lower 0.01 --clip-upper 100.0 \
  --ci-level 0.95
```

Or via the pipeline driver: set `method_filter: [locf, lsm2, …]` in the
sweep YAML and run `run_paper_pipeline.py --skip-eval --skip-phase1`.
The aggregator is dispatched with the filter automatically.

**Two invariants the math gives you:**

| Quantity | Depends on the pool? | Why |
|---|---|---|
| Skill score (vs. baseline) | **No** | Paired ratio `E_method / E_baseline` per task is independent of every other method in the parquet |
| Avg rank | **Yes** | `compute_average_rankings` ranks methods against each other per (scenario, channel, user); changing the pool changes every rank |
| Bootstrap SE / CI | follows the underlying quantity | Skill SE unchanged; rank SE depends on pool size |

So a subset rerun lets you re-cut ranks against a custom comparison
group while skill numbers stay bit-identical to the full-pool
run.

**Operational guardrail:**

1. **The baseline must be in the filter.** Skill pairs
   each method's E against the baseline's E per task; if the baseline
   isn't in the parquet rows after filtering, the inner merge produces
   no rows and the CSV gets empty skill rows for every
   method. The pipeline driver's `_method_filter_args` helper raises a
   `ValueError` before subprocess launch if the YAML's `method_filter`
   excludes `baseline_method`; the direct CLI does not enforce this, so
   include `locf` (or whatever `--baseline-method` is) explicitly.

### 8.2 Phase B fast paths — consuming `per_user_errors.parquet` directly

The canonical per-method `per_user_errors.parquet` artifact (§1) lets
both Phase 1 and the point-estimate flow skip the pair re-scan. Same
math; ~10× wall-time saving on a full 16-method run because the slow
step is reading per-channel pair files.

**Phase 1 (bootstrap) fast path** — `bootstrap_imputation_draws.py`
gains a `--per-user-errors-dir <dir>` flag that loads each method's
file from `<dir>/<method>.parquet` and feeds
[`compute_per_draw_errors_from_per_user_errors`](evaluation/bootstrap_skill_rank.py)
instead of the pair-loading path:

```bash
python scripts/paper_results/imputation/bootstrap_imputation_draws.py \
  --method-dirs configs/paper/bootstrap_method_dirs.json \
  --per-user-errors-dir <dir> \
  --methods locf mean linear brits \
  --output results/paper-fast/bootstrap_draws.parquet \
  --n-boot 1000 --seed 42 --splits test
```

`--method-dirs` is still required so manifests can reconstruct the
canonical user-id union (preserves `boot_idx` cohort ordering against
the legacy path; same seed → same draws). `--methods` selects the
**ranking pool** (avg-rank is taken across the selected methods) and
the **skill-score pool** (the baseline must be in the selection — same
guardrail as §8.1).

**Point-estimate fast path** —
`scripts/paper_results/compute_imputation_paper_metrics.py` gains a
`--per-user-errors <dir>` switch that bypasses `_gather_registry`'s
pair scan entirely:

```bash
python scripts/paper_results/compute_imputation_paper_metrics.py \
  --per-user-errors <dir> \
  --methods locf mean linear brits \
  --output-dir results/paper-fast-point/
```

Same `--methods` ranking-pool semantics.

**Public-API integration** — `openmhc.evaluate_imputation` accepts
`output_dir=` (persists `per_user_errors.parquet`) and
`baseline_errors=` (paired-R skill against a single frozen baseline,
typically the LOCF file shipped at
`src/openmhc/data/baselines/imputation_locf_per_user_errors.parquet`).
Single-method scope only; cross-method ranks still require running the
imputers separately and aggregating via the point CLI above. See
`src/openmhc/_evaluate.py::evaluate_imputation`'s docstring for the
full surface and the multi-method ranking pattern.

**Known byte-level divergences from the legacy pair-based bootstrap**
(§3 parity report):

- `per_user_errors.parquet` stores `E_per_user` in **float32**; the
  legacy bootstrap holds float64 until the final write. After
  reduction the per-draw E can differ at the float32 ULP scale
  (~1e-3 on large-magnitude `ch_6` ~10000 values; 4.9% of E rows
  affected). The Phase 2 aggregate CSVs absorb this into bootstrap
  noise (`§3` confirms `≤ 5 × SE`).
- The legacy `_per_method_cell_paired_collapsed_ratios` applies
  `BINARY_ERROR_FLOOR` **per-channel inside the nanmean** when
  computing collapsed-binary R; the per-user-errors substrate stores
  the **unfloored** value, so the `cat_collapsed:*` R of the two paths
  can differ on the `all` rows.
- Strict byte-parity on `bootstrap_draws.parquet` would require
  storing `E_per_user` in float64 (~doubles file size) and
  reconstructing the per-channel-floored cat E inside
  `_emit_draws_for_cell` (~30 lines). Headline numbers are
  unaffected today.

## 9. Constants (verified canonical values)

| Constant | Value | Location |
|---|---|---|
| `CLIP_LOWER` / `SKILL_CLIP_LOWER` | `0.01` | `paper_metrics_core.py:25`, `bootstrap_skill_rank.py:77` |
| `CLIP_UPPER` / `SKILL_CLIP_UPPER` | `100.0` | `paper_metrics_core.py:26`, `bootstrap_skill_rank.py:78` |
| `BASELINE_CONTINUOUS` / `BASELINE_BINARY` | `"locf"` | `paper_metrics_core.py:28-29` |
| `BINARY_ERROR_FLOOR` | `0.005` | `bootstrap_skill_rank.py:80` |
| `SKILL_MIN_PAIRS` | `1` | `bootstrap_skill_rank.py:79` |
| `EXCLUDE_BINARY_SCENARIOS` | `{sleep_gap, workout_gap, intensity_failure}` | `paper_metrics_core.py:42` |
| `SEMANTIC_SCENARIOS` | `{sleep_gap, workout_gap, intensity_failure}` | `paper_metrics_core.py:45` |
| `n_boot` | `1000` | `sweep_methods.yaml:37` |
| `seed` | `42` | `sweep_methods.yaml:38` |
| CI level | `0.95` | `sweep_methods.yaml:51` |
| Splits scored | `["test"]` | `sweep_methods.yaml:39` |

## 10. Worked example — `task:random_noise:ch_5` for one method

Inputs (illustrative, 3 users):

| user | E_{m, random_noise, ch_5, u} (MAE) | E_{b, random_noise, ch_5, u} (MAE) |
|---|---|---|
| u1 | 4.0 | 8.0 |
| u2 | 2.0 | 4.0 |
| u3 | 1.0 | 10.0 |

Per-user ratios (clip range `[0.01, 100]`, all in-bounds):

```
r_u1 = 4.0 / 8.0   = 0.50
r_u2 = 2.0 / 4.0   = 0.50
r_u3 = 1.0 / 10.0  = 0.10
```

Per-task R (geomean over users):

```
R = exp( (log 0.50 + log 0.50 + log 0.10) / 3 )
  = exp( (−0.693 + −0.693 + −2.303) / 3 )
  = exp( −1.230 )
  = 0.292
```

Per-task skill (task:* leaf scope):

```
S^{task}_{m, random_noise, ch_5}  =  1 − clip(0.292, 0.01, 100)
                                   =  1 − 0.292
                                   =  0.708
```

Per-scenario skill — geomean over `random_noise`'s 19 channels'
clipped Rs, then `1 − exp(mean(log R))`. The task:* leaf is the
single-task case where the mean collapses to one term.

Bootstrap: repeat the per-user join with each draw's resampled cohort,
get 1000 values of `S^{task,(b)}`, report
`mean / se / ci_lo / ci_hi`.

## §S7. BCa (bias-corrected & accelerated) CIs for skill / rank

The `--bca-skill-rank` flag on `aggregate_imputation_paper_metrics.py` is wired
through the CLI but the leave-one-user-out jackknife of the skill / rank point
flow is not implemented (each recompute would run `compute_per_task_paired_R` +
`compute_skill_scores` + `compute_average_rankings` once per user, ~N × per-iter
at N ≈ 5K users in the production cohort). Turning the flag on raises
`NotImplementedError`. The flag is OFF by default.

## See also

- [`README.md`](README.md) — public API + library usage
- [`evaluation/paper_metrics_core.py`](evaluation/paper_metrics_core.py) — pure-pandas reducers
- [`evaluation/bootstrap_skill_rank.py`](evaluation/bootstrap_skill_rank.py) — Phase 1 + Phase 2 kernels
- [`scripts/paper_results/run_paper_pipeline.py`](../../scripts/paper_results/run_paper_pipeline.py) — end-to-end driver
- [Forecasting Track 3 metric spec](../forecasting_evaluation/) — the parity target for the paired ratio / per-user rank reducers
