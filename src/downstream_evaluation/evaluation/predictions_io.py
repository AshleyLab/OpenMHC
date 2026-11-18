"""Persist per-(method, task) test predictions.

When the prediction engine runs with a predictions directory configured, it emits,
per (method, task), a ``test.parquet`` of ``uid, y_true, y_pred, y_proba``. These are
the input the paper-metrics bootstrap (``bootstrap_skill_rank``) paired-resamples for
skill / rank confidence intervals.

Layout::

    <predictions_dir>/<method>/<task>/test.parquet   # task = task name, "/"+" " -> "_"
    <predictions_dir>/<method>/fallback.json          # per-task {n_fallback, n_test}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from downstream_evaluation.evaluation.metrics import get_task_type, prepare_predictions


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
    overall_fallback_rate(per_task_counts)
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
    caller must recover recorded counts or rerun evaluation for those legacy runs).
    """
    path = Path(predictions_dir) / method / "fallback.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def overall_fallback_rate(sidecar: dict[str, dict]) -> float | None:
    """Pool measured counts; an empty cohort has no measured rate."""
    total_fb = total_n = 0
    if not isinstance(sidecar, dict):
        raise ValueError("Missing fallback counts")
    for task, counts in sidecar.items():
        if not isinstance(counts, dict):
            raise ValueError(f"{task}: invalid fallback counts")
        n_test, n_fallback = counts.get("n_test"), counts.get("n_fallback")
        if (
            isinstance(n_test, bool) or isinstance(n_fallback, bool)
            or not isinstance(n_test, (int, np.integer))
            or not isinstance(n_fallback, (int, np.integer))
            or n_test < 0 or not 0 <= n_fallback <= n_test
        ):
            raise ValueError(f"{task}: missing or invalid fallback counts")
        total_fb += int(n_fallback)
        total_n += int(n_test)
    return total_fb / total_n if total_n else None
