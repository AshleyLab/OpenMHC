"""Pure functions for the imputation paper's headline metrics.

Holds the constants (clip bounds, baselines, semantic-scenario set,
channel categories) and the deterministic transforms used to derive
**skill score**, **average rank**, and **baseline errors** from a long-format
``errors`` DataFrame.

Both the canonical script
``scripts/paper_results/compute_imputation_paper_metrics.py`` and the
bootstrap pipeline (``bootstrap_skill_rank.py``) import from this module so
the per-draw aggregation matches the published point estimates exactly.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

CLIP_LOWER = 1e-2
CLIP_UPPER = 100.0

BASELINE_CONTINUOUS = "locf"
BASELINE_BINARY = "locf"

N_CONTINUOUS = 7  # ch_0..ch_6
N_BINARY = 12  # ch_7..ch_18

# Semantic masking scenarios where binary channels are excluded
EXCLUDE_BINARY_SCENARIOS: set[str] = {"sleep_gap", "workout_gap", "intensity_failure"}

# Semantic scenarios are excluded from per-category scores
SEMANTIC_SCENARIOS: set[str] = {"sleep_gap", "workout_gap", "intensity_failure"}

# Channel categories for per-category skill scores
CHANNEL_CATEGORIES: dict[str, set[str]] = {
    "activity": {"ch_0", "ch_1", "ch_2", "ch_3", "ch_4"},
    "physiology": {"ch_5", "ch_6"},
    "sleep": {"ch_7", "ch_8"},
    "workouts": {f"ch_{i}" for i in range(9, 19)},
}

# Binary categories eligible for the collapsed-scope leaderboard rows.
# Each entry maps a category name to its ordered tuple of channel indices.
# Order matters for stable column layout in the precomputed per-(user, cat)
# matrix that drives the per-draw E for ``cat_collapsed:*`` and
# ``overall_binary_collapsed`` scopes — see ``bootstrap_skill_rank.py``'s
# ``_per_method_cell_collapsed_errors``. Continuous categories are
# deliberately NOT collapsed (decided in plan Part D): the unequal channel
# counts that motivate the binary collapse don't apply to continuous,
# whose seven channels already weight roughly fairly under the per-task
# geomean.
BINARY_CATEGORIES_ORDERED: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("sleep", (7, 8)),
    ("workouts", tuple(range(9, 19))),
)


def channel_category(channel: str) -> str | None:
    """Return the category name for a channel, or None if uncategorised."""
    for cat, channels in CHANNEL_CATEGORIES.items():
        if channel in channels:
            return cat
    return None


def is_collapsed_channel(channel: str) -> bool:
    """True for the synthetic ``cat_collapsed:*`` channel labels emitted by Part D."""
    return channel.startswith("cat_collapsed:")


# --------------------------------------------------------------------------
# 3-level B.2 helpers (skill + rank kernels)
# --------------------------------------------------------------------------


def _bucket_log_R_per_scenario(
    legacy_df: pd.DataFrame,
    collapsed_df: pd.DataFrame,
) -> pd.DataFrame:
    """L3 of the 3-level B.2 form: per-(method, scenario, bucket) mean of log(clip(R)).

    Averages log(clip(R)) over the bucket's constituent tasks in that
    scenario.

    Continuous legacy rows (category in {activity, physiology})
    contribute to their named bucket; collapsed rows contribute to
    sleep/workouts via the suffix of their synthetic
    ``cat_collapsed:<cat>`` channel key. Per-channel binary rows
    (ch_7..ch_18) are dropped — those reach the headline only via the
    collapsed rows so per-channel binary would double-count.

    Returns ``[method, scenario, bucket, bucket_log_R]`` with one row
    per present (method, scenario, bucket).
    """
    pieces: list[pd.DataFrame] = []
    cont = legacy_df[legacy_df["category"].isin({"activity", "physiology"})]
    if not cont.empty:
        pieces.append(cont.assign(bucket=cont["category"]))
    if not collapsed_df.empty:
        pieces.append(
            collapsed_df.assign(
                bucket=collapsed_df["channel"].str.split(":", n=1, expand=True)[1],
            )
        )
    if not pieces:
        return pd.DataFrame(columns=["method", "scenario", "bucket", "bucket_log_R"])
    buckets_df = pd.concat(pieces, ignore_index=True)
    return (
        buckets_df.groupby(["method", "scenario", "bucket"], observed=True)["clipped_ratio"]
        .apply(lambda r: float(np.mean(np.log(r.values))))
        .reset_index(name="bucket_log_R")
    )


def _scenario_log_R_per_method(bucket_log_R: pd.DataFrame) -> pd.DataFrame:
    """L2 of the 3-level B.2 form: per-(method, scenario) mean over buckets.

    Arithmetic mean over buckets-present of ``bucket_log_R``.

    Returns ``[method, scenario, log_R_scenario, n_buckets]`` with one
    row per present (method, scenario).
    """
    if bucket_log_R.empty:
        return pd.DataFrame(columns=["method", "scenario", "log_R_scenario", "n_buckets"])
    return (
        bucket_log_R.groupby(["method", "scenario"], observed=True)
        .agg(
            log_R_scenario=("bucket_log_R", "mean"),
            n_buckets=("bucket_log_R", "size"),
        )
        .reset_index()
    )


def _multi_scenario_skill(
    scen_log_R: pd.DataFrame,
    scenarios: Iterable[str],
    *,
    scope_label: str,
) -> list[dict]:
    """L1 of the 3-level B.2 form: per-method log-space mean over scenarios.

    Reduces ``log_R_scenario`` over a chosen set of scenarios.

    Emits one row per method with skill = ``1 − exp(mean of log_R_scenario
    over scenarios-present)``. ``n_tasks`` is the number of scenarios
    that actually contributed (≤ ``len(scenarios)``).
    """
    scenarios = set(scenarios)
    if scen_log_R.empty or not scenarios:
        return []
    in_scope = scen_log_R[scen_log_R["scenario"].isin(scenarios)]
    if in_scope.empty:
        return []
    out: list[dict] = []
    for method, grp in in_scope.groupby("method", observed=True):
        vals = grp["log_R_scenario"].to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        out.append(
            {
                "method": method,
                "scope": scope_label,
                "skill_score": 1.0 - float(np.exp(float(np.mean(vals)))),
                "n_tasks": int(vals.size),
            }
        )
    return out


def _scenario_rank_per_method(
    per_task: pd.DataFrame,
    has_n_users: bool,
) -> pd.DataFrame:
    """L2 of the 3-level rank form: per-(method, scenario) mean over buckets.

    Arithmetic mean over buckets-present of ``bucket_rank``.

    Stage L3 (per-bucket task mean) and L2 (mean over buckets) fold
    into a single groupby pair below because mean-of-means with equal
    inner weight equals the weighted mean of bucket-level means — but
    we materialize the L3 step explicitly so the bucket weights are
    equal regardless of how many tasks live in each bucket within a
    scenario.

    Returns ``[method, scenario, scenario_rank, n_buckets,
    (scenario_users)]``.
    """
    if per_task.empty:
        cols = ["method", "scenario", "scenario_rank", "n_buckets"]
        if has_n_users:
            cols.append("scenario_users")
        return pd.DataFrame(columns=cols)

    # Recover the bucket from the channel name alone: the per-task frames
    # produced by the rank kernel carry the channel type implicitly through
    # ``channel`` (per-channel ch_K) and the collapsed ``cat_collapsed:*``
    # synthetic key.
    df = per_task.copy()
    chan_str = df["channel"].astype(str)
    cat = chan_str.map(channel_category)

    def _bucket(ch: str, c: str | None) -> str | None:
        if ch.startswith("cat_collapsed:"):
            return ch.split(":", 1)[1]
        if c in {"activity", "physiology"}:
            return c
        return None  # per-channel binary → dropped

    df["bucket"] = [_bucket(ch, c) for ch, c in zip(chan_str, cat)]
    df = df[df["bucket"].notna()]
    if df.empty:
        cols = ["method", "scenario", "scenario_rank", "n_buckets"]
        if has_n_users:
            cols.append("scenario_users")
        return pd.DataFrame(columns=cols)

    # L3: per (method, scenario, bucket) mean of task_rank over tasks.
    l3_agg: dict = {"bucket_rank": ("task_rank", "mean")}
    if has_n_users:
        l3_agg["bucket_users"] = ("n_users", "max")
    bucket_lvl = (
        df.groupby(["method", "scenario", "bucket"], observed=True).agg(**l3_agg).reset_index()
    )

    # L2: per (method, scenario) mean over buckets present.
    l2_agg: dict = {
        "scenario_rank": ("bucket_rank", "mean"),
        "n_buckets": ("bucket_rank", "size"),
    }
    if has_n_users:
        l2_agg["scenario_users"] = ("bucket_users", "max")
    return bucket_lvl.groupby(["method", "scenario"], observed=True).agg(**l2_agg).reset_index()


def _multi_scenario_rank(
    scen_rank: pd.DataFrame,
    scenarios: Iterable[str],
    *,
    scope_label: str,
    has_n_users: bool,
) -> list[dict]:
    """L1 of the 3-level rank form: per-method mean over scenarios.

    Arithmetic mean of ``scenario_rank`` over a chosen set of scenarios.
    """
    scenarios = set(scenarios)
    if scen_rank.empty or not scenarios:
        return []
    in_scope = scen_rank[scen_rank["scenario"].isin(scenarios)]
    if in_scope.empty:
        return []
    out: list[dict] = []
    for method, grp in in_scope.groupby("method", observed=True):
        vals = grp["scenario_rank"].to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        row: dict = {
            "method": method,
            "scope": scope_label,
            "avg_rank": float(np.mean(vals)),
            "n_tasks": int(vals.size),
        }
        if has_n_users:
            row["n_users"] = int(grp["scenario_users"].max())
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Error extraction (registry-DataFrame variant)
# --------------------------------------------------------------------------


def extract_errors(
    df: pd.DataFrame,
    split: str,
    subgroup_attr: str = "all",
    subgroup_value: str = "all",
    continuous_metric: str = "MAE",
) -> pd.DataFrame:
    """Extract per-task error E for each method from a registry-shaped DF.

    Returns columns ``[method, scenario, channel, channel_type, E]``.

    ``continuous_metric`` defaults to ``"MAE"`` to match the forecasting
    track (Track 3 commit 79c8628). At per-channel scope the per-task skill
    ratio ``E_method / E_baseline`` is identical under MAE, RMSE, nMAE, and
    nRMSE — per-channel ``channel_std`` cancels exactly — so the skill score
    is unchanged. Switching to MAE keeps the leaderboard absolute-value
    columns aligned across the two tracks. Callers may still pass
    ``continuous_metric="RMSE"`` or ``"nRMSE"`` to read those columns if
    their registry DataFrame carries them.
    """
    mask = (
        (df["split"] == split)
        & (df["subgroup_attr"] == subgroup_attr)
        & (df["subgroup_value"] == subgroup_value)
        & (~df["channel"].str.contains("aggregate"))
    )
    sub = df[mask].copy()

    rows = []
    for _, row in sub.iterrows():
        ch_type = row["channel_type"]
        if ch_type == "binary" and row["scenario"] in EXCLUDE_BINARY_SCENARIOS:
            continue
        if ch_type == "continuous":
            e = row.get(continuous_metric)
        elif ch_type == "binary":
            auc = row.get("roc_auc")
            e = 1.0 - auc if pd.notna(auc) else np.nan
        else:
            continue
        if pd.notna(e):
            rows.append(
                {
                    "method": row["method"],
                    "scenario": row["scenario"],
                    "channel": row["channel"],
                    "channel_type": ch_type,
                    "E": float(e),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Skill score & average rank
# --------------------------------------------------------------------------


def compute_skill_scores(
    errors: pd.DataFrame,
    baseline_errors: pd.DataFrame | None = None,
    *,
    mode: str = "paired",
    clip_lower: float = CLIP_LOWER,
    clip_upper: float = CLIP_UPPER,
) -> pd.DataFrame:
    """Skill score per method × scope.

    Two input modes are supported. The previous silent switch on the ``R``
    column has been removed: the mode must be selected explicitly.

    1. ``mode="paired"`` (default — the leaderboard estimand). ``errors``
       must carry an ``R`` column (the per-task paired geometric-mean ratio
       at the per-user grain). Emit
       ``S = 1 − exp(nanmean(log(clip(R, clip_lower, clip_upper))))`` over
       tasks in scope. ``baseline_errors`` is ignored (R is already paired
       against the baseline). The R column is produced by either the
       point-flow helper
       :func:`bootstrap_skill_rank.compute_per_task_paired_R` or the
       bootstrap kernel
       :func:`bootstrap_skill_rank._per_method_cell_paired_ratios`.
    2. ``mode="pooled"`` (legacy opt-in). Form ``E_method / E_baseline`` per
       task from ``errors`` and ``baseline_errors``, clip, geomean. Not the
       leaderboard estimand; reachable from
       ``compute_imputation_paper_metrics.py --skill-mode pooled``.

    Emits these scopes (identical layout in both modes):

    Per-channel (the historical layout):
      - ``<scenario>`` (one per scenario)
      - ``cat:activity`` / ``cat:physiology`` / ``cat:sleep`` / ``cat:workouts``
        (structural scenarios only)
      - ``semantic`` (semantic scenarios only)
      - ``overall`` (all 19 per-channel tasks × all scenarios)

    Collapsed-binary (added by Part D — leaderboard reports side-by-side):
      - ``cat_collapsed:sleep`` / ``cat_collapsed:workouts`` (one geomean per
        category × scenario task set)
      - ``overall_binary_collapsed`` (7 continuous per-channel tasks + 2
        binary categories × scenarios — same task layout as ``overall``
        with the 12 binary channels replaced by 2 collapsed-category tasks)

    The collapsed scopes consume rows where ``channel`` starts with
    ``cat_collapsed:`` (produced upstream by
    ``bootstrap_skill_rank._per_method_cell_collapsed_errors``). The legacy
    per-channel scopes explicitly exclude those rows so they don't
    double-count the binary information.
    """
    if mode == "paired":
        if "R" not in errors.columns:
            raise ValueError(
                "compute_skill_scores: mode='paired' requires an 'R' column "
                "in ``errors``. Build R via "
                "``bootstrap_skill_rank.compute_per_task_paired_R`` "
                "(point-flow) or via ``compute_per_draw_errors`` (bootstrap), "
                "or pass mode='pooled' for the legacy E + baseline_errors path."
            )
        # R is pre-clipped at Phase 1, but re-clip here so the result is
        # invariant to whether callers pre-process.
        ratios = []
        for _, row in errors.iterrows():
            r = row["R"]
            if pd.isna(r) or r <= 0:
                continue
            clipped = float(np.clip(r, clip_lower, clip_upper))
            ratios.append(
                {
                    "method": row["method"],
                    "scenario": row["scenario"],
                    "channel": row["channel"],
                    "category": channel_category(row["channel"]),
                    "clipped_ratio": clipped,
                    "is_collapsed": is_collapsed_channel(row["channel"]),
                }
            )
    elif mode == "pooled":
        if baseline_errors is None:
            raise ValueError(
                "compute_skill_scores: mode='pooled' requires "
                "``baseline_errors`` to form the per-task E_method / "
                "E_baseline ratio."
            )
        bl = baseline_errors.set_index(["scenario", "channel"])["E"]
        ratios = []
        for _, row in errors.iterrows():
            key = (row["scenario"], row["channel"])
            if key not in bl.index:
                continue
            e_baseline = bl[key]
            if e_baseline <= 0 or np.isnan(e_baseline):
                continue
            ratio = row["E"] / e_baseline
            clipped = np.clip(ratio, clip_lower, clip_upper)
            ratios.append(
                {
                    "method": row["method"],
                    "scenario": row["scenario"],
                    "channel": row["channel"],
                    "category": channel_category(row["channel"]),
                    "clipped_ratio": clipped,
                    "is_collapsed": is_collapsed_channel(row["channel"]),
                }
            )
    else:
        raise ValueError(
            f"compute_skill_scores: unknown mode={mode!r}; expected 'paired' or 'pooled'."
        )

    ratio_df = pd.DataFrame(ratios)
    if ratio_df.empty:
        return pd.DataFrame(columns=["method", "scope", "skill_score", "n_tasks"])

    def _skill(log_ratios: np.ndarray) -> float:
        return 1.0 - float(np.exp(np.mean(log_ratios)))

    # Partition into legacy per-channel rows vs collapsed-binary rows.
    legacy_df = ratio_df[~ratio_df["is_collapsed"]]
    collapsed_df = ratio_df[ratio_df["is_collapsed"]]

    results = []

    # --- Universal 3-level B.2 (per-scenario / semantic / overall) ----
    # L3 (within bucket × scenario): per-(method, scenario, bucket) mean
    #   of log(clip(R_task)).
    # L2 (within scenario):           per-(method, scenario) mean over
    #   buckets-present of L3 values.
    # L1 (across scenarios):          per-method mean over scenarios-in-scope
    #   of L2 values. ``S_scope = 1 − exp(L1)``.
    #
    # Bucket sourcing:
    #   activity   ← continuous category=="activity" rows (ch_0..ch_4)
    #   physiology ← continuous category=="physiology" rows (ch_5..ch_6)
    #   sleep      ← cat_collapsed:sleep rows (Part D)
    #   workouts   ← cat_collapsed:workouts rows (Part D)
    # Per-channel binary rows (category in {sleep, workouts}, ch_7..ch_18)
    # are NOT consumed by per-scenario / semantic / overall — they reach
    # those scopes via the cat_collapsed:* rows. They still emit as
    # ``task:<sc>:ch_K`` leaves and as the per-channel ``cat:sleep`` /
    # ``cat:workouts`` scopes below.
    bucket_log_R_per_scen = _bucket_log_R_per_scenario(legacy_df, collapsed_df)
    scen_log_R = _scenario_log_R_per_method(bucket_log_R_per_scen)

    # Per-scenario scopes: one row per (method, scenario) present.
    for _, row in scen_log_R.iterrows():
        results.append(
            {
                "method": row["method"],
                "scope": row["scenario"],
                "skill_score": 1.0 - float(np.exp(float(row["log_R_scenario"]))),
                "n_tasks": int(row["n_buckets"]),  # buckets in this scenario
            }
        )

    # cat:<cat> (structural scenarios only, per-channel) — continuous
    # categories only. The per-channel binary cat:sleep / cat:workouts
    # are deliberately dropped: the collapsed variants below are the
    # equal-weighted binary categories that feed the headline.
    structural_cont = legacy_df[
        (~legacy_df["scenario"].isin(SEMANTIC_SCENARIOS))
        & (legacy_df["category"].isin({"activity", "physiology"}))
    ]
    for (method, cat), group in structural_cont.groupby(["method", "category"], observed=True):
        if cat is None:
            continue
        results.append(
            {
                "method": method,
                "scope": f"cat:{cat}",
                "skill_score": _skill(np.log(group["clipped_ratio"].values)),
                "n_tasks": len(group),
            }
        )

    # cat:<cat> for binary categories — log-space geomean of per-scenario
    # R values (3 structural scenarios × 1 collapsed task each). The
    # synthetic channel key (``cat_collapsed:<cat>``) is unchanged in the
    # parquet and pair_aggregator wire format; only the *scope label* is
    # ``cat:<cat>``.
    if not collapsed_df.empty:
        for (method, channel), group in collapsed_df.groupby(["method", "channel"], observed=True):
            cat_name = str(channel).split(":", 1)[1]
            results.append(
                {
                    "method": method,
                    "scope": f"cat:{cat_name}",
                    "skill_score": _skill(np.log(group["clipped_ratio"].values)),
                    "n_tasks": len(group),
                }
            )

    # --- semantic: L1 log-space mean over the 3 semantic scenarios ----
    sem_rows = _multi_scenario_skill(
        scen_log_R,
        SEMANTIC_SCENARIOS,
        scope_label="semantic",
    )
    results.extend(sem_rows)

    # --- overall: L1 log-space mean over all 6 scenarios --------------
    all_scenarios = set(scen_log_R["scenario"].unique()) if not scen_log_R.empty else set()
    overall_rows = _multi_scenario_skill(
        scen_log_R,
        all_scenarios,
        scope_label="overall",
    )
    results.extend(overall_rows)

    # --- task:<scenario>:<channel> (degenerate single-task scope) -----
    # Per-(method, scenario, channel) leaf scope. Skill is the single-task
    # case of 1 − exp(mean(log(R))) — i.e., 1 − clipped_ratio — so any
    # aggregated scope built from these same R values reproduces exactly.
    # Covers both real per-channel rows ("task:<scenario>:ch_K") and
    # collapsed-binary rows ("task:<scenario>:cat_collapsed:<cat>"). The
    # per-scenario channel inventory is inherited from the upstream pair
    # writer (e.g. workout_gap / intensity_failure naturally produce only
    # ch_5 and ch_6 tasks because no rows are written for unmasked
    # channels).
    for r in ratio_df.itertuples(index=False):
        results.append(
            {
                "method": r.method,
                "scope": f"task:{r.scenario}:{r.channel}",
                "skill_score": 1.0 - float(r.clipped_ratio),
                "n_tasks": 1,
            }
        )

    return pd.DataFrame(results)


def aggregate_task_ranks_to_scopes(per_task: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(method, scenario, channel) ``task_rank`` into per-(method, scope) ``avg_rank``.

    ``per_task`` must carry columns ``[method, scenario, channel, task_rank]``.
    Optionally, ``n_users`` is carried through as the scope-level
    ``max(n_users)`` over tasks-in-scope.

    Every task in scope contributes one ``task_rank`` value, equally
    weighted, to the scope arithmetic mean.

    Emits the standard scope set used by :func:`compute_skill_scores` and
    :func:`compute_average_rankings`:
      - ``<scenario>`` (one per scenario)
      - ``cat:<cat>`` (structural scenarios only, real channels only)
      - ``semantic``
      - ``overall`` (legacy per-channel; excludes ``cat_collapsed:*`` rows)
      - ``cat_collapsed:<cat>`` (one per binary category)
      - ``overall_binary_collapsed`` (continuous per-channel tasks +
        collapsed-binary tasks)

    Shared between the point-flow ``compute_average_rankings`` and the
    bootstrap Phase-2 consumer in ``bootstrap_skill_rank.py``.
    """
    df = per_task.copy()
    df["category"] = df["channel"].map(channel_category)
    df["is_collapsed"] = df["channel"].map(is_collapsed_channel)

    legacy = df[~df["is_collapsed"]]
    collapsed = df[df["is_collapsed"]]
    has_n_users = "n_users" in df.columns

    def _row(method: str, scope: str, group: pd.DataFrame) -> dict:
        out = {
            "method": method,
            "scope": scope,
            "avg_rank": float(group["task_rank"].mean()),
            "n_tasks": int(len(group)),
        }
        if has_n_users and len(group):
            out["n_users"] = int(group["n_users"].max())
        return out

    results: list[dict] = []

    # 3-level rank form — mirrors compute_skill_scores. The per-scenario
    # and overall_binary_collapsed scopes (plus semantic) collapse to one
    # ``scenario_rank`` value per (method, scenario), then L1 averages
    # over scenarios in scope. Bucket sourcing (activity/physiology from
    # continuous, sleep/workouts from cat_collapsed:*) is shared with the
    # skill kernel via ``_scenario_rank_per_method``.
    scen_rank = _scenario_rank_per_method(df, has_n_users)

    # Per-scenario scopes: one row per (method, scenario) present.
    for _, row in scen_rank.iterrows():
        rdict = {
            "method": row["method"],
            "scope": row["scenario"],
            "avg_rank": float(row["scenario_rank"]),
            "n_tasks": int(row["n_buckets"]),
        }
        if has_n_users:
            rdict["n_users"] = int(row["scenario_users"])
        results.append(rdict)

    # cat:<cat> (structural scenarios only, continuous-only)
    structural_cont = legacy[
        (~legacy["scenario"].isin(SEMANTIC_SCENARIOS))
        & (legacy["category"].isin({"activity", "physiology"}))
    ]
    for (method, cat), group in structural_cont.groupby(["method", "category"], observed=True):
        if cat is None:
            continue
        results.append(_row(method, f"cat:{cat}", group))

    # cat:<cat> for binary categories — arithmetic mean of per-scenario
    # task_ranks. The synthetic channel key (``cat_collapsed:<cat>``) is
    # unchanged; only the *scope label* is ``cat:<cat>``.
    if not collapsed.empty:
        for (method, channel), group in collapsed.groupby(["method", "channel"], observed=True):
            cat_name = str(channel).split(":", 1)[1]
            results.append(_row(method, f"cat:{cat_name}", group))

    # semantic: L1 mean over the 3 semantic scenarios.
    results.extend(
        _multi_scenario_rank(
            scen_rank,
            SEMANTIC_SCENARIOS,
            scope_label="semantic",
            has_n_users=has_n_users,
        )
    )

    # overall: L1 mean over all 6 scenarios.
    all_scenarios_rank = set(scen_rank["scenario"].unique()) if not scen_rank.empty else set()
    results.extend(
        _multi_scenario_rank(
            scen_rank,
            all_scenarios_rank,
            scope_label="overall",
            has_n_users=has_n_users,
        )
    )

    # --- task:<scenario>:<channel> (degenerate single-task rank scope) -
    # Pass each per-task row through unchanged: ``avg_rank`` is the
    # per-task ``task_rank`` (Stage-1 output). The aggregated scopes above
    # are arithmetic means over these same rank values, so any consumer
    # mean-aggregating ``task:*`` rows within a scope reproduces the
    # aggregated scope's ``avg_rank`` exactly.
    for r in df.itertuples(index=False):
        row = {
            "method": r.method,
            "scope": f"task:{r.scenario}:{r.channel}",
            "avg_rank": float(r.task_rank),
            "n_tasks": 1,
        }
        if has_n_users:
            row["n_users"] = int(r.n_users)
        results.append(row)

    cols = ["method", "scope", "avg_rank", "n_tasks"]
    if has_n_users:
        cols.append("n_users")
    if not results:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(results)[cols]


def _average_rankings_per_user(errors: pd.DataFrame) -> pd.DataFrame:
    """Per-user ranking — two-stage form.

    Stage 1: per ``(scenario, channel)`` task, pivot
    ``user_id × method`` on E, rank methods across each user
    (``method="average"``, ``ascending=True``), then take the
    ``nanmean`` over users → one ``task_rank`` per
    ``(method, scenario, channel)``.

    Stage 2: mean ``task_rank`` over tasks in scope.

    Per-user ranks are scale-free (each user's row is ranked across
    methods independently), so channel scale differences across tasks
    don't bias the result.
    """
    if "user_id" not in errors.columns:
        raise ValueError(
            "compute_average_rankings requires a 'user_id' column in "
            "``errors``. Build the per-user long frame from "
            "``pair_aggregator.aggregate_pairs(..., return_per_user=True)``."
        )
    df = errors[np.isfinite(errors["E"])].copy()
    if df.empty:
        return pd.DataFrame(columns=["method", "scope", "avg_rank", "n_tasks", "n_users"])

    task_rank_frames: list[pd.DataFrame] = []
    for (scenario, channel), grp in df.groupby(["scenario", "channel"], observed=True):
        pivot = grp.pivot(index="user_id", columns="method", values="E")
        if pivot.empty:
            continue
        ranks = pivot.rank(axis=1, method="average", ascending=True)
        long_rank = ranks.stack(future_stack=True).reset_index()
        long_rank.columns = ["user_id", "method", "rank"]
        long_rank["scenario"] = scenario
        long_rank["channel"] = channel
        task_rank_frames.append(long_rank)
    if not task_rank_frames:
        return pd.DataFrame(columns=["method", "scope", "avg_rank", "n_tasks", "n_users"])

    long_rank_all = pd.concat(task_rank_frames, ignore_index=True)
    per_task = (
        long_rank_all.groupby(["method", "scenario", "channel"], observed=True)
        .agg(task_rank=("rank", "mean"), n_users=("user_id", "nunique"))
        .reset_index()
    )
    return aggregate_task_ranks_to_scopes(per_task)


def compute_average_rankings(errors: pd.DataFrame) -> pd.DataFrame:
    """Average rank per method × scope. Lower E → better → rank 1.

    Two-stage form mirroring forecasting's full cross-task rank aggregation
    (per-user rank per task → mean over users per task → mean over tasks
    per scope). The bootstrap Phase-2 consumer in
    ``bootstrap_skill_rank.py`` shares the same Stage-2 helper so the
    point estimate matches the bootstrap identity-draw point estimate.

    Args:
        errors: long-format frame with columns ``[method, scenario,
            channel, user_id, E]`` — one row per (method, task, user).

    Emits the same scope set as :func:`compute_skill_scores`: per-channel
    legacy scopes (``<scenario>``, ``cat:*``, ``semantic``, ``overall``)
    plus the Part D collapsed-binary scopes (``cat_collapsed:sleep``,
    ``cat_collapsed:workouts``, ``overall_binary_collapsed``) when
    ``cat_collapsed:*`` rows are present.
    """
    return _average_rankings_per_user(errors)


def build_baseline_errors(
    errors: pd.DataFrame,
    *,
    baseline_continuous: str = BASELINE_CONTINUOUS,
    baseline_binary: str = BASELINE_BINARY,
) -> pd.DataFrame:
    """Extract global baseline errors (LOCF by default for both channel types).

    Returns one row per ``(scenario, channel)`` with the baseline E.
    """
    cont = errors[
        (errors["method"] == baseline_continuous) & (errors["channel_type"] == "continuous")
    ]
    binary = errors[(errors["method"] == baseline_binary) & (errors["channel_type"] == "binary")]
    return pd.concat([cont, binary], ignore_index=True)
