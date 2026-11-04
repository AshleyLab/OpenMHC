r"""Paired user-level bootstrap of the headline downstream benchmark stats (library).

Two-phase, mirroring the imputation paper-metrics pipeline:

* Phase 1 — :func:`compute_per_draw_errors` samples the declared test participants
  with replacement, sharing each draw across all tasks and methods. It recomputes
  every method's per-task primary metric on eligible participants and records the
  error ``E = 1 − metric`` as a long-format draws frame (:func:`write_draws_parquet`
  persists it).

* Phase 2 — :func:`aggregate_skill_rank` reconstructs each draw's per-task
  metric (``1 − E``) and reduces the draws to ``point / SE / 95 % CI`` for the macro
  (domain-balanced) **skill score** vs the baseline and the **average rank** (both
  Overall + per-domain).

The runnable CLIs are ``scripts/paper_results/downstream/bootstrap_downstream_draws.py`` (phase
1) and ``scripts/paper_results/downstream/aggregate_downstream_paper_metrics.py`` (phase 2).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from downstream_evaluation.evaluation.skill_score import (
    DEFAULT_CLIP_LOWER,
    DEFAULT_CLIP_UPPER,
    TASK_DOMAIN_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _auprc(y_true: np.ndarray, _pred: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray, _proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if np.isfinite(r) else float("nan")


def _pearson(y_true: np.ndarray, y_pred: np.ndarray, _proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r) if np.isfinite(r) else float("nan")


# Point metrics only — bypasses compute_*_metrics' internal 1000-iter SE
# bootstrap, which would otherwise turn each outer resample into 1000 nested
# resamples. Defaults are higher-is-better → error = 1 - metric.
PRIMARY_METRIC_FN = {
    "binary": _auprc,
    "ordinal": _spearman,
    "regression": _pearson,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_method_predictions(
    predictions_dir: Path,
    method: str,
    csvs_dir: Path,
) -> dict[str, dict]:
    """Return {task: {uids, y_true, y_pred, y_proba, task_type}} for one method."""
    # Prefer `eval_*<method>*.csv` (the production naming); fall back to any CSV
    # containing the method name (used by the smoke test CSV `smoke.csv` etc.).
    csv_candidates = sorted(csvs_dir.glob(f"eval_*{method}*.csv"))
    if not csv_candidates:
        csv_candidates = sorted(csvs_dir.glob(f"*{method}*.csv"))
    if not csv_candidates:
        # Last resort: any CSV in the dir whose `features` column contains
        # the method label (e.g. `smoke.csv` was written without the method
        # name in the file name).
        for path in sorted(csvs_dir.glob("*.csv")):
            try:
                head = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            if "features" in head.columns and (head["features"] == method).any():
                csv_candidates = [path]
                break
    if not csv_candidates:
        raise FileNotFoundError(f"No eval CSV matching '{method}' in {csvs_dir}")
    if len(csv_candidates) > 1:
        log.warning(
            "Multiple eval CSVs match '%s': %s — using %s",
            method,
            [p.name for p in csv_candidates],
            csv_candidates[0].name,
        )
    csv = pd.read_csv(csv_candidates[0])
    if "subgroup_variable" in csv.columns:
        csv = csv[csv["subgroup_variable"].fillna("") == ""]
    if "error" in csv.columns:
        csv = csv[csv["error"].fillna("") == ""]
    task_type_map = dict(zip(csv["task"], csv["task_type"]))

    method_dir = predictions_dir / method
    if not method_dir.exists():
        raise FileNotFoundError(f"No predictions dir for method '{method}' at {method_dir}")

    tasks: dict[str, dict] = {}
    for task_dir in sorted(method_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        parquet = task_dir / "test.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        candidates = [
            t for t in task_type_map if t.replace("/", "_").replace(" ", "_") == task_dir.name
        ]
        if not candidates:
            log.warning("No CSV task_type for parquet dir '%s' — skipping", task_dir.name)
            continue
        task_name = candidates[0]
        tasks[task_name] = {
            "uids": df["uid"].to_numpy(),
            "y_true": df["y_true"].to_numpy(),
            "y_pred": df["y_pred"].to_numpy(),
            "y_proba": df["y_proba"].to_numpy(),
            "task_type": task_type_map[task_name],
        }
    log.info("Loaded %d tasks for method '%s'", len(tasks), method)
    return tasks


def align_across_methods(
    method_tasks: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    """Per task, intersect uids across methods and reorder to a shared sequence."""
    methods = list(method_tasks.keys())
    common_tasks = set(method_tasks[methods[0]].keys())
    for m in methods[1:]:
        common_tasks &= set(method_tasks[m].keys())

    aligned: dict[str, dict[str, dict]] = {m: {} for m in methods}
    for task in sorted(common_tasks):
        for m in methods:
            uids = method_tasks[m][task]["uids"]
            if len(set(uids)) != len(uids):
                raise ValueError(f"Duplicate participant IDs for {m}/{task}")
        uid_sets = [set(method_tasks[m][task]["uids"]) for m in methods]
        common_uids = set.intersection(*uid_sets)
        if not common_uids:
            log.warning("Task '%s': empty uid intersection across methods — skipping", task)
            continue
        uid_order = np.array(sorted(common_uids))
        for m in methods:
            payload = method_tasks[m][task]
            uid_to_idx = {uid: i for i, uid in enumerate(payload["uids"])}
            idx = np.array([uid_to_idx[u] for u in uid_order])
            aligned[m][task] = {
                "uids": uid_order,
                "y_true": payload["y_true"][idx],
                "y_pred": payload["y_pred"][idx],
                "y_proba": payload["y_proba"][idx],
                "task_type": payload["task_type"],
            }
        ref_y = aligned[methods[0]][task]["y_true"]
        for m in methods[1:]:
            if not np.array_equal(aligned[m][task]["y_true"], ref_y):
                raise ValueError(f"Task {task!r}: y_true mismatch between {methods[0]!r} and {m!r}")
    return aligned


def _test_population(test_uids: Sequence[str]) -> np.ndarray:
    if isinstance(test_uids, (str, bytes)):
        raise ValueError("test_uids must contain participant IDs, not a filename or string")
    users = np.asarray(list(test_uids))
    if users.ndim != 1 or users.size == 0 or pd.isna(users).any():
        raise ValueError("The declared test population must be nonempty and one-dimensional")
    users = users.astype(str)
    if len(set(users)) != len(users):
        raise ValueError("The declared test population contains duplicate participant IDs")
    return np.sort(users)


def load_bootstrap_test_users(
    split_file: str | Path | None = None,
    *,
    data_dir: str | Path | None = None,
    version: str = "full",
) -> tuple[np.ndarray, dict]:
    """Load the entire declared test split and its provenance, independently of predictions."""
    if split_file is None:
        from openmhc._evaluate import _DatasetPaths

        split_file = _DatasetPaths.resolve(data_dir, version=version).splits_file
    path = Path(split_file).resolve()
    contents = path.read_bytes()
    split = json.loads(contents)
    if not isinstance(split, dict):
        raise ValueError(f"Split file must contain a train/validation/test mapping: {path}")
    groups = {}
    for name in ("train", "validation", "test"):
        if not isinstance(split.get(name), list):
            raise ValueError(f"Split file must contain a {name!r} participant list: {path}")
        values = split[name]
        if any(v is None for v in values):
            raise ValueError(f"Missing participant ID in {name!r}: {path}")
        groups[name] = set(map(str, values))
        if len(groups[name]) != len(values):
            raise ValueError(f"Duplicate participant IDs in {name!r}: {path}")
    if any(groups[a] & groups[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError(f"Train, validation and test participants must be disjoint: {path}")
    return _test_population(split["test"]), {
        "split_file": str(path), "split_sha256": hashlib.sha256(contents).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers (operate on per-resample dicts)
# ---------------------------------------------------------------------------


def _per_domain_skill_from_ratios(
    ratios: dict[str, float],
    domain_map: dict[str, str],
    clip_lower: float,
    clip_upper: float,
) -> dict[str, float]:
    """Skill score per domain ``+`` Overall.

    Per-domain S = 1 − geomean(clipped ratios for tasks in that domain).
    Overall S = mean of per-domain S (domain-balanced macro aggregate).
    """
    by_domain: dict[str, list[float]] = {}
    for task, r in ratios.items():
        if not np.isfinite(r):
            continue
        domain = domain_map.get(task)
        if domain is None:
            continue
        by_domain.setdefault(domain, []).append(np.clip(r, clip_lower, clip_upper))
    out: dict[str, float] = {}
    for domain, rs in by_domain.items():
        out[domain] = 1.0 - float(np.exp(np.mean(np.log(rs))))
    out["Overall"] = float(np.mean(list(out.values()))) if out else float("nan")
    return out


def _ratios_for_method(
    per_task_metric: dict[str, dict[str, float]],
    method: str,
    baseline: str,
) -> dict[str, float]:
    """E_method / E_baseline per task (errors = 1 − metric, both higher-is-better)."""
    out: dict[str, float] = {}
    for task, vals in per_task_metric.items():
        m_val = vals.get(method)
        b_val = vals.get(baseline)
        if m_val is None or b_val is None:
            continue
        m_err = 1.0 - m_val
        b_err = 1.0 - b_val
        if not (np.isfinite(m_err) and np.isfinite(b_err) and b_err > 0):
            continue
        out[task] = m_err / b_err
    return out


def _per_domain_avg_rank(
    per_task_metric: dict[str, dict[str, float]],
    methods: list[str],
    domain_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Per-domain + Overall average rank.

    Returns ``{domain_or_'Overall': {method: avg_rank}}``. Per-domain rank =
    mean of per-task ranks within that domain. Overall = mean of per-domain
    ranks (domain-balanced macro aggregate).
    """
    by_domain: dict[str, dict[str, list[float]]] = {}
    for task, method_vals in per_task_metric.items():
        domain = domain_map.get(task)
        if domain is None:
            continue
        # Series.rank handles ties via "average"; negate for descending
        # (higher metric = rank 1).
        ranks = pd.Series({m: -method_vals.get(m, np.nan) for m in methods}).rank(
            method="average",
            ascending=True,
        )
        for m, r in ranks.items():
            if pd.notna(r):
                by_domain.setdefault(domain, {}).setdefault(m, []).append(float(r))

    out: dict[str, dict[str, float]] = {}
    for domain, per_method in by_domain.items():
        out[domain] = {
            m: float(np.mean(per_method[m])) if m in per_method else float("nan") for m in methods
        }
    if out:
        out["Overall"] = {
            m: float(np.mean([out[d][m] for d in out if not np.isnan(out[d][m])]))
            if any(not np.isnan(out[d][m]) for d in out)
            else float("nan")
            for m in methods
        }
    else:
        out["Overall"] = {m: float("nan") for m in methods}
    return out


# ---------------------------------------------------------------------------
# Bootstrap loop
# ---------------------------------------------------------------------------


def _per_task_metric_on_indices(
    aligned: dict[str, dict[str, dict]],
    methods: list[str],
    tasks: list[str],
    task_indices_b: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Compute per-task per-method metric on the resampled rows."""
    out: dict[str, dict[str, float]] = {}
    for t in tasks:
        idx = task_indices_b[t]
        tt = aligned[methods[0]][t]["task_type"]
        fn = PRIMARY_METRIC_FN.get(tt)
        if fn is None:
            continue
        out[t] = {}
        for m in methods:
            p = aligned[m][t]
            y = p["y_true"][idx]
            pred = p["y_pred"][idx]
            proba = p["y_proba"][idx]
            try:
                v = float(fn(y, pred, proba))
            except Exception:
                v = float("nan")
            out[t][m] = v
    return out


# ---------------------------------------------------------------------------
# Two-phase pipeline: per-draw errors (phase 1) -> skill/rank (phase 2)
# ---------------------------------------------------------------------------

DRAW_COLS = [
    "method",
    "task",
    "task_type",
    "domain",
    "subgroup_attr",
    "subgroup_value",
    "draw",
    "E",
]

# Sentinel ``draw`` id for the point estimate — the full cohort, no resampling. The
# reported value is this point estimate; the bootstrap draws (``draw >= 0``) give only
# the standard error / CI around it (we never report the bootstrap mean as the value).
POINT_DRAW = -1


def compute_per_draw_errors(
    aligned: dict[str, dict[str, dict]],
    n_bootstrap: int,
    seed: int,
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
    *,
    test_uids: Sequence[str],
) -> pd.DataFrame:
    """Phase 1: per-(method, task, draw) error ``E = 1 - metric``.

    Each replicate samples the full ``test_uids`` population once. Its participant
    multiplicities are shared across tasks and methods; task eligibility is fixed
    and the resampled task sizes can vary. The original eligible cohorts supply the
    point estimates. ``subgroup_attr`` and ``subgroup_value`` hold the constant
    ``"all"``. Protocol metadata is attached to the returned frame and persisted by
    :func:`write_draws_parquet`.
    """
    methods = list(aligned.keys())
    if not methods or not aligned[methods[0]]:
        raise ValueError("At least one method and task are required")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    tasks = sorted(aligned[methods[0]].keys())
    task_type = {t: aligned[methods[0]][t]["task_type"] for t in tasks}
    users = _test_population(test_uids)
    user_index = {u: i for i, u in enumerate(users)}
    task_positions = {}
    for m in methods:
        if set(aligned[m]) != set(tasks):
            raise ValueError("All methods must have the same aligned tasks")
    for t in tasks:
        reference = aligned[methods[0]][t]
        uids = np.asarray(reference["uids"]).astype(str)
        if uids.ndim != 1 or len(set(uids)) != len(uids):
            raise ValueError(f"Task {t!r} must have unique one-dimensional participant IDs")
        if not set(uids) <= user_index.keys():
            raise ValueError(f"Task {t!r} contains participants outside the declared test split")
        if task_type[t] not in PRIMARY_METRIC_FN:
            raise ValueError(f"Unsupported task type for {t!r}: {task_type[t]!r}")
        for m in methods:
            payload = aligned[m][t]
            if payload["task_type"] != task_type[t] or not np.array_equal(
                np.asarray(payload["uids"]).astype(str), uids
            ) or not np.array_equal(payload["y_true"], reference["y_true"]):
                raise ValueError(f"Participant order, targets and task type must align for {t!r}")
            for key in ("y_true", "y_pred", "y_proba"):
                if np.asarray(payload[key]).shape != (len(uids),):
                    raise ValueError(f"Expected one-dimensional {key} for {m}/{t}")
            metric_key = "y_proba" if task_type[t] == "binary" else "y_pred"
            if not np.isfinite(payload["y_true"]).all() or not np.isfinite(payload[metric_key]).all():
                raise ValueError(f"Nonfinite targets or primary-metric predictions for {m}/{t}")
        if task_type[t] == "binary" and not set(np.unique(reference["y_true"])) <= {0, 1}:
            raise ValueError(f"Binary task {t!r} must use labels 0 and 1")
        task_positions[t] = np.array([user_index[u] for u in uids], dtype=int)
    rng = np.random.default_rng(seed)
    task_n = {t: len(aligned[methods[0]][t]["uids"]) for t in tasks}
    count_dtype = np.int16 if len(users) <= np.iinfo(np.int16).max else np.int64
    counts = np.empty((n_bootstrap, len(users)), dtype=count_dtype)
    for b in range(n_bootstrap):
        counts[b] = np.bincount(rng.integers(len(users), size=len(users)), minlength=len(users))

    rows: list[dict] = []
    omitted = {"all": 0}
    global_omissions = []

    def _emit(per_task: dict[str, dict[str, float]], b: int) -> None:
        for t, method_vals in per_task.items():
            dom = domain_map.get(t)
            for m, metric in method_vals.items():
                if metric is None or not np.isfinite(metric):
                    omitted["all"] += 1
                    continue
                rows.append(
                    {
                        "method": m,
                        "task": t,
                        "task_type": task_type[t],
                        "domain": dom,
                        "subgroup_attr": "all",
                        "subgroup_value": "all",
                        "draw": b,
                        "E": 1.0 - float(metric),
                    }
                )

    def _emit_for_indices(idx_b: dict[str, np.ndarray], b: int) -> None:
        """Emit errors for one set of row indices (draw ``b``)."""
        per_task_global = _per_task_metric_on_indices(aligned, methods, tasks, idx_b)
        for t, values in per_task_global.items():
            missing = [m for m, value in values.items() if not np.isfinite(value)]
            if missing:
                global_omissions.append({"task": t, "draw": b, "methods": missing,
                                         "sample_size": len(idx_b[t])})
        _emit(per_task_global, b)

    # Point estimate first — the full cohort in order, no resampling — then the B
    # paired bootstrap resamples that quantify its standard error.
    _emit_for_indices({t: np.arange(task_n[t]) for t in tasks}, POINT_DRAW)
    for b in range(n_bootstrap):
        indices = {t: np.repeat(np.arange(task_n[t]), counts[b, task_positions[t]]) for t in tasks}
        _emit_for_indices(indices, b)

    result = pd.DataFrame(rows, columns=DRAW_COLS)
    result.attrs["bootstrap"] = {
        "protocol": "shared_participant_v1", "n_boot": int(n_bootstrap), "seed": int(seed),
        "sampling_unit": "participant", "sampling_population": "entire declared test split",
        "sampling_population_n": len(users),
        "test_users_sha256": hashlib.sha256("\0".join(users).encode()).hexdigest(),
        "eligible_union_n": len(set().union(*(set(p) for p in task_positions.values()))),
        "shared_across": "all tasks and methods", "task_sample_sizes": "vary by replicate",
        "counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "counts_dtype": str(counts.dtype), "counts_shape": list(counts.shape),
        "point_draw": POINT_DRAW, "error_storage_dtype": "float32",
        "undefined_policy": "omit nonfinite metric cells; single-class targets omitted across all methods",
        "undefined_metric_cells": omitted, "global_omissions": global_omissions,
    }
    return result


def write_draws_parquet(df: pd.DataFrame, path: Path, meta: dict | None = None) -> None:
    """Write the phase-1 draws frame to parquet (zstd) + optional sidecar meta JSON."""
    protocol = df.attrs.get("bootstrap", {})
    metadata = dict(meta or {})
    for key, value in protocol.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"Metadata disagrees with computed bootstrap field {key!r}")
        metadata[key] = value
    df = df.copy()
    df.attrs.clear()
    df["E"] = df["E"].astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd")
    if metadata:
        path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def read_draws_parquet(path: Path) -> tuple[pd.DataFrame, dict | None]:
    """Read a phase-1 draws parquet and its sidecar meta JSON (if present)."""
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return df, meta


def _summarise(values: list[float], ci_level: float, point: float | None = None) -> dict[str, float]:
    """Deterministic point + bootstrap mean + SE / percentile-CI for one quantity.

    ``values`` are the bootstrap draws (the point draw excluded). ``point`` is the
    full-cohort deterministic estimate and is the reported centre for skill score and
    average rank alike; the draws supply only the SE and the percentile CI around it.
    ``mean``, the bootstrap mean of those draws, is returned alongside as a diagnostic
    (the gap between the two centres measures the resampling bias). When ``point`` is
    omitted the centre falls back to the bootstrap mean.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    boot_mean = float(np.mean(arr)) if len(arr) else float("nan")
    center = float(point) if point is not None and np.isfinite(point) else boot_mean
    if len(arr) == 0:
        return {
            "point": center,
            "mean": boot_mean,
            "se": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
    alpha = (1.0 - ci_level) / 2.0
    return {
        "point": center,
        "mean": boot_mean,
        "se": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "ci_lo": float(np.percentile(arr, 100 * alpha)),
        "ci_hi": float(np.percentile(arr, 100 * (1 - alpha))),
    }


def aggregate_skill_rank(
    draws: pd.DataFrame,
    baseline: str,
    *,
    clip_lower: float = DEFAULT_CLIP_LOWER,
    clip_upper: float = DEFAULT_CLIP_UPPER,
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
    ci_level: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Phase 2: summarise per-draw errors into skill / rank tables.

    Reconstructs each draw's per-task metric (``1 - E``) from the ``subgroup_attr ==
    "all"`` rows and reuses the point helpers, so per-draw skill/rank match the one-pass
    bootstrap exactly. Returns ``{skill_scores, avg_rankings}`` (mean / se / percentile
    ci at ``ci_level``).
    """
    methods = sorted(draws["method"].unique())
    glob = draws[draws["subgroup_attr"] == "all"]
    n_boot = int(draws.loc[draws["draw"] != POINT_DRAW, "draw"].nunique())

    task_dom = dict(zip(glob["task"], glob["domain"]))
    domains = sorted({d for d in task_dom.values() if d is not None})
    n_tasks = {"Overall": len({t for t, d in task_dom.items() if d is not None})}
    for dom in domains:
        n_tasks[dom] = len({t for t, d in task_dom.items() if d == dom})

    skill_draws: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
    rank_draws: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
    # Point estimates (full cohort, draw == POINT_DRAW) — the reported values.
    skill_point: dict[str, dict[str, float]] = {m: {} for m in methods}
    rank_point: dict[str, dict[str, float]] = {m: {} for m in methods}

    for b, g in glob.groupby("draw"):
        is_point = b == POINT_DRAW
        per_task_metric: dict[str, dict[str, float]] = {}
        for t, m_, e_ in zip(g["task"], g["method"], g["E"]):
            per_task_metric.setdefault(t, {})[m_] = 1.0 - e_
        ranks_b = _per_domain_avg_rank(per_task_metric, methods, domain_map)
        for m in methods:
            skill = _per_domain_skill_from_ratios(
                _ratios_for_method(per_task_metric, m, baseline),
                domain_map,
                clip_lower,
                clip_upper,
            )
            for scope, val in skill.items():
                if is_point:
                    skill_point[m][scope] = val
                else:
                    skill_draws[m].setdefault(scope, []).append(val)
            for scope, per_m in ranks_b.items():
                if is_point:
                    rank_point[m][scope] = per_m[m]
                else:
                    rank_draws[m].setdefault(scope, []).append(per_m[m])

    skill_rows, rank_rows = [], []
    for m in methods:
        for scope in ["Overall", *domains]:
            if scope in skill_draws[m]:
                pt = skill_point[m].get(scope)
                row = {"method": m, "scope": scope, **_summarise(skill_draws[m][scope], ci_level, pt)}
                row["n_boot"] = n_boot
                row["n_tasks"] = n_tasks.get(scope, 0)
                skill_rows.append(row)
            if scope in rank_draws[m]:
                pt = rank_point[m].get(scope)
                row = {"method": m, "scope": scope, **_summarise(rank_draws[m][scope], ci_level, pt)}
                row["n_boot"] = n_boot
                rank_rows.append(row)

    return {
        "skill_scores": pd.DataFrame(skill_rows),
        "avg_rankings": pd.DataFrame(rank_rows),
    }


# ---------------------------------------------------------------------------
# Live leaderboard reduce — point skill / rank straight from the
# per-user-pairs substrate, no bootstrap. The Track-1 leaderboard reduces this on
# each load (mirroring the imputation / forecasting tracks), so the board equals the
# paper's point estimate by construction.
# ---------------------------------------------------------------------------


def reduce_substrate_to_point(
    substrate: pd.DataFrame,
    baseline: str = "linear",
    *,
    clip_lower: float = DEFAULT_CLIP_LOWER,
    clip_upper: float = DEFAULT_CLIP_UPPER,
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
) -> dict[str, dict]:
    """Point skill / rank per method from the pooled per-user-pairs substrate.

    Reconstructs the aligned per-(method, task) predictions from the long substrate
    frame's ``subgroup_attr == "all"`` rows (the :mod:`per_user_pairs` layout), then
    runs the same point reducers the bootstrap uses — on the full cohort, with no
    resampling. Reusing those reducers makes the result identical to the paper's ``point``
    column by construction (the substrate stores float32 pairs, so values match the
    float64 paper point to ~1e-6, immaterial at leaderboard display precision).

    Returns ``{method: {"skill": {scope: value}, "rank": {scope: value}}}``: ``scope`` is
    ``"Overall"`` plus each health domain, ``skill`` the domain-balanced skill vs
    ``baseline``, and ``rank`` the cross-method average rank.
    """
    all_cell = substrate[substrate["subgroup_attr"] == "all"]
    methods = sorted(all_cell["method"].astype(str).unique())
    if baseline not in methods:
        raise ValueError(f"baseline {baseline!r} not among substrate methods {methods}")

    aligned_raw: dict[str, dict[str, dict]] = {m: {} for m in methods}
    for (m, task), g in all_cell.groupby(["method", "task"], observed=True):
        g = g.sort_values("user_id")
        aligned_raw[str(m)][str(task)] = {
            "uids": g["user_id"].astype(str).to_numpy(),
            "y_true": g["y_true"].to_numpy(np.float64),
            "y_pred": g["y_pred"].to_numpy(np.float64),
            "y_proba": g["y_proba"].to_numpy(np.float64),
            "task_type": str(g["task_type"].iloc[0]),
        }
    aligned = align_across_methods(aligned_raw)

    tasks = sorted(aligned[methods[0]].keys())
    full_idx = {t: np.arange(len(aligned[methods[0]][t]["uids"])) for t in tasks}
    per_task = _per_task_metric_on_indices(aligned, methods, tasks, full_idx)
    ranks = _per_domain_avg_rank(per_task, methods, domain_map)

    out: dict[str, dict] = {}
    for m in methods:
        skill = _per_domain_skill_from_ratios(
            _ratios_for_method(per_task, m, baseline), domain_map, clip_lower, clip_upper
        )
        out[m] = {
            "skill": skill,
            "rank": {scope: ranks[scope][m] for scope in ranks},
        }
    return out
