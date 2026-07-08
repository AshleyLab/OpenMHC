"""Persist per-(method, task) test predictions + a per-user subgroup map.

When the prediction engine runs with a predictions directory configured, it emits,
per (method, task), a ``test.parquet`` of ``uid, y_true, y_pred, y_proba`` plus a
single shared ``_subgroups.json`` (``{user_id: {age_group, sex}}``). Together these
are the input the paper-metrics bootstrap (``bootstrap_skill_rank``) paired-resamples
for skill / rank / fairness confidence intervals.

Layout::

    <predictions_dir>/<method>/<task>/test.parquet   # task = task name, "/"+" " -> "_"
    <predictions_dir>/<method>/fallback.json          # per-task {n_fallback, n_test}
    <predictions_dir>/_subgroups.json

The module is self-contained: demographics come from the labels lookup's ``age`` /
``BiologicalSex`` columns, so no Labels-API environment is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from downstream_evaluation.evaluation.metrics import get_task_type, prepare_predictions

# Five age bands (plus ``unknown``) — the subgroups the fairness analysis slices on.
_AGE_GROUP_BINS: tuple[tuple[float, float, str], ...] = (
    (-float("inf"), 30.0, "18-29"),
    (30.0, 40.0, "30-39"),
    (40.0, 50.0, "40-49"),
    (50.0, 60.0, "50-59"),
    (60.0, float("inf"), "60+"),
)

# Sentinels marking a missing label cell in the lookup (mirrors data.provider).
_MISSING_INT = -1
_MISSING_FLOAT = -1.0


def _safe_task(task: str) -> str:
    """Task name as a filesystem-safe directory (matches the bootstrap loader)."""
    return task.replace("/", "_").replace(" ", "_")


def write_task_predictions(
    predictions_dir: str | Path,
    method: str,
    task: str,
    uids,
    y_true,
    y_pred,
) -> None:
    """Write ``<predictions_dir>/<method>/<task>/test.parquet``.

    ``y_pred`` is the evaluator's raw test output: the class-1 probability for binary
    tasks, the point prediction otherwise. The ``y_pred`` / ``y_proba`` columns are derived
    from it so the bootstrap can read the probability (binary AUPRC) and the point
    prediction (ordinal Spearman / regression Pearson) it needs per task type. The
    per-task-type policy lives in ``prepare_predictions``, shared with the live metrics.
    """
    ttype = get_task_type(task)
    y_true = np.asarray(y_true)
    y_pred_col, y_proba = prepare_predictions(ttype, y_pred)

    df = (
        pd.DataFrame(
            {
                "uid": np.asarray(uids).astype(str),
                "y_true": y_true,
                "y_pred": y_pred_col,
                "y_proba": y_proba,
            }
        )
        .sort_values("uid")
        .reset_index(drop=True)
    )
    out_dir = Path(predictions_dir) / method / _safe_task(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "test.parquet", index=False)


def _age_group(value) -> str:
    """Bin a numeric age into one of the five bands, else ``unknown``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    for lo, hi, label in _AGE_GROUP_BINS:
        if lo <= v < hi:
            return label
    return "unknown"


def _sex(value) -> str:
    """Map ``BiologicalSex`` (1=male, 0=female) to a subgroup label, else ``unknown``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    return "male" if int(v) == 1 else ("female" if int(v) == 0 else "unknown")


def _first_valid_by_user(lookup: pd.DataFrame, col: str) -> dict[str, float]:
    """``{user_id: first non-sentinel value}`` for a constant-per-user label column."""
    arr = lookup[col].to_numpy()
    if np.issubdtype(arr.dtype, np.floating):
        valid = ~(np.isnan(arr) | (arr == _MISSING_FLOAT))
    else:
        valid = arr != _MISSING_INT
    sub = lookup.loc[valid, ["user_id", col]].drop_duplicates("user_id", keep="first")
    return dict(zip(sub["user_id"].astype(str), sub[col]))


def write_fallback_sidecar(
    predictions_dir: str | Path,
    method: str,
    per_task_counts: dict[str, dict],
) -> Path:
    """Persist a method's per-task fallback + cohort counts.

    Writes ``<predictions_dir>/<method>/fallback.json`` = ``{task: {n_fallback,
    n_test}}``, where ``n_fallback`` is how many test participants the harness
    substituted with the Linear baseline (a non-finite model prediction) and
    ``n_test`` the cohort size. The offline substrate producer reads it back to
    compute the method's ``overall_fallback_rate`` without re-running the model, so
    the rate is measured rather than hardcoded. Mirrors the imputation track's
    ``write_fallback_sidecar``.
    """
    out = {
        task: {"n_fallback": int(c["n_fallback"]), "n_test": int(c["n_test"])}
        for task, c in per_task_counts.items()
    }
    path = Path(predictions_dir) / method / "fallback.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return path


def read_fallback_sidecar(
    predictions_dir: str | Path,
    method: str,
) -> dict[str, dict] | None:
    """Read a method's fallback sidecar (``{task: {n_fallback, n_test}}``), or None if absent.

    Missing sidecar → None: a prediction dir written before the sidecar existed (the
    caller supplies an explicit rate for those legacy runs).
    """
    path = Path(predictions_dir) / method / "fallback.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def overall_fallback_rate(sidecar: dict[str, dict]) -> float:
    """Pool a fallback sidecar into one rate = ``Σ n_fallback / Σ n_test`` (0.0 if no cohort)."""
    total_fb = sum(int(c["n_fallback"]) for c in sidecar.values())
    total_n = sum(int(c["n_test"]) for c in sidecar.values())
    return total_fb / total_n if total_n else 0.0


def write_subgroup_map(
    predictions_dir: str | Path,
    lookup_path: str | Path,
    users,
) -> None:
    """Write ``<predictions_dir>/_subgroups.json`` = ``{uid: {age_group, sex}}``.

    Demographics are read from the labels lookup's ``age`` / ``BiologicalSex``
    columns (the first non-sentinel value per user). Users without a value fall into
    ``unknown`` so the union of subgroups covers the full evaluated set.
    """
    lookup = pd.read_parquet(lookup_path, columns=["user_id", "age", "BiologicalSex"])
    age_by = _first_valid_by_user(lookup, "age")
    sex_by = _first_valid_by_user(lookup, "BiologicalSex")
    subgroups = {
        u: {"age_group": _age_group(age_by.get(u)), "sex": _sex(sex_by.get(u))}
        for u in {str(x) for x in users}
    }
    out_dir = Path(predictions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "_subgroups.json").open("w") as f:
        json.dump(subgroups, f)
