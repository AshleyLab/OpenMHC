"""Regression tests for two Track 1 scoring-correctness fixes.

B1 — ordinal predictions must be scored (and persisted) on their RAW rank order, not
     rounded to int. The rank-merge fallback produces values in ``(0, 1]``; ``np.round``
     would collapse them to ``{0, 1}`` and turn Spearman's rho into a median-split binary.
B5 — ``compute_binary_metrics`` must not clip scores into ``[1e-10, 1 - 1e-10]`` before
     AUPRC. ``average_precision_score`` is rank-based, so clipping ties every out-of-``(0, 1)``
     score at a bound and distorts the metric for score-valued (logit) submissions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score

from downstream_evaluation.evaluation.evaluator import _metrics_for
from downstream_evaluation.evaluation.metrics import compute_binary_metrics, get_task_type
from downstream_evaluation.evaluation.predictions_io import write_task_predictions
from labels.api import LABEL_NAMES


def _an_ordinal_task() -> str:
    return next(t for t in sorted(LABEL_NAMES) if get_task_type(t) == "ordinal")


def test_ordinal_metric_scores_raw_ranks_not_rounded():
    """Ordinal Spearman is computed on the raw prediction, not ``np.round(pred)``."""
    task = _an_ordinal_task()
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 4, size=n)  # ordinal levels 0..3
    # Simulate a fallback-merged ordinal prediction: distinct percentile ranks in (0, 1].
    y_pred = rankdata(rng.normal(size=n)) / n

    out = _metrics_for(task, y_true, y_pred, seed=0)
    expected_raw, _ = spearmanr(y_true, y_pred)
    collapsed, _ = spearmanr(y_true, np.round(y_pred).astype(int))

    assert out["spearman_r"] == float(expected_raw)
    # Rounding collapses (0, 1] to {0, 1}; the fix must NOT reproduce that value.
    assert not np.isclose(out["spearman_r"], collapsed)
    # np.round(y_pred) would leave <=2 distinct values; the raw prediction keeps its ranks.
    assert len(np.unique(np.round(y_pred).astype(int))) <= 2
    assert len(np.unique(y_pred)) > 2


def test_ordinal_predictions_persisted_raw(tmp_path):
    """``write_task_predictions`` stores the raw ordinal ``y_pred`` (the bootstrap reads it)."""
    task = _an_ordinal_task()
    rng = np.random.default_rng(1)
    n = 50
    uids = [f"u{i}" for i in range(n)]
    y_true = rng.integers(0, 4, size=n)
    raw = rankdata(rng.normal(size=n)) / n  # in (0, 1]

    write_task_predictions(tmp_path, "m", task, uids, y_true, raw)
    safe = task.replace("/", "_").replace(" ", "_")
    df = pd.read_parquet(tmp_path / "m" / safe / "test.parquet")

    stored = df["y_pred"].to_numpy()
    raw_by_uid = dict(zip(uids, raw))
    expected = np.array([raw_by_uid[u] for u in df["uid"]])

    # y_pred keeps the continuous prediction (float), not a {0, 1} int collapse.
    assert np.issubdtype(stored.dtype, np.floating)
    assert np.allclose(stored, expected)
    assert len(np.unique(stored)) > 2


def test_binary_auprc_not_clipped_for_score_submissions():
    """AUPRC uses raw scores; out-of-``(0, 1)`` logits are not clamped to a bound."""
    rng = np.random.default_rng(2)
    n = 300
    y_true = rng.integers(0, 2, size=n)
    scores = rng.normal(size=n) * 5.0  # logit-scale, many values outside (0, 1)

    out = compute_binary_metrics(y_true, scores, seed=0)
    assert out["auprc"] == float(average_precision_score(y_true, scores))
    # The removed clip would have tied every out-of-range score at a bound and moved AUPRC.
    clipped = np.clip(scores, 1e-10, 1.0 - 1e-10)
    assert not np.isclose(out["auprc"], average_precision_score(y_true, clipped))
