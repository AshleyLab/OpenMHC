"""Canonical per-method ``per_user_errors.parquet`` producer.

Replaces the embedded per-user producer inside
:func:`bootstrap_skill_rank.compute_per_draw_errors` (the only previous
place this artifact was emitted) with a single, importable function the
public API, ``mhc-impute-eval`` runner, and paper scripts all share.

The producer takes **one method's** pairs directory and emits:

  * ``per_user_df`` — the long ``per_user_errors`` frame for that method,
    schema :data:`PER_USER_ERRORS_PARQUET_COLUMNS`. The per-user error
    ``E`` matches the bootstrap's :func:`_per_user_errors_for_cell`
    semantics: continuous = per-user MAE (``sae / n``); binary =
    ``1 − AUC`` (un-floored); collapsed binary = ``nanmean`` over the
    category's channels of ``1 − AUC[user, ch]``. NaN per-user values
    are dropped (one row per (user × channel) that has data).
    ``subgroup_attr`` and ``subgroup_value`` hold the constant ``"all"``.

  * ``display_metrics`` — ``{(scenario, split): metrics_dict}``. Carries
    user-macro per-channel display metrics (``mae``, ``rmse``, ``normalized_*``,
    ``balanced_accuracy``, ``roc_auc``, ``n_masked``) plus the
    ``continuous`` / ``binary`` headline aggregates. Used by
    :class:`openmhc._results.ImputationResults` to populate ``scenarios``.

Scenario-level binary filtering — binary and ``cat_collapsed:*`` rows
for scenarios in :data:`EXCLUDE_BINARY_SCENARIOS` are dropped, matching
the bootstrap's per-row filter at
:func:`compute_per_draw_errors`. ``include_auc=False`` drops every
binary row regardless of scenario.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES, N_CHANNELS
from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    PER_USER_ERRORS_PARQUET_COLUMNS,
    read_per_user_errors_parquet,
    write_per_user_errors_parquet,
)
from imputation_evaluation.evaluation.pair_aggregator import aggregate_pairs
from imputation_evaluation.evaluation.paper_metrics_core import EXCLUDE_BINARY_SCENARIOS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel-type resolution (same as the scripts' local helper, canonicalised)
# ---------------------------------------------------------------------------


def _channel_type_for(channel_label: str) -> str:
    """Resolve channel label → ``"continuous" | "binary" | "binary_collapsed"``.

    Mirrors the channel labels emitted by
    :func:`compute_per_draw_errors`: ``f"ch_{ch}"`` for raw channels (type
    determined by membership in :data:`CONTINUOUS_CHANNEL_INDICES`),
    ``f"cat_collapsed:{name}"`` for the two synthetic collapsed-binary
    tasks. Unknown labels raise — silent fallbacks hide drift.
    """
    if channel_label.startswith("cat_collapsed:"):
        return "binary_collapsed"
    if channel_label.startswith("ch_"):
        ch_idx = int(channel_label.split("_", 1)[1])
        return "continuous" if ch_idx in CONTINUOUS_CHANNEL_INDICES else "binary"
    raise ValueError(f"Unrecognised channel label: {channel_label!r}")


# ---------------------------------------------------------------------------
# Flattener: per_user map → 9-column tuple rows
# ---------------------------------------------------------------------------


def per_user_map_to_rows(
    per_user: Mapping[str, Mapping[str, float]],
    *,
    method: str,
    scenario: str,
    split: str,
    include_auc: bool = True,
) -> list[tuple]:
    """Flatten a ``{channel_label: {user_id: E}}`` map into 9-column tuples.

    Tuple order matches :data:`PER_USER_ERRORS_PARQUET_COLUMNS` so the
    rows plug straight into ``pd.DataFrame(rows, columns=...)``. Tuples
    (not dicts) keep in-flight memory ~10× smaller at production scale
    (~11M rows would otherwise OOM a 32 GB job — see
    :func:`compute_per_draw_errors` for the same allocation note).

    Scenario-level binary filtering is applied here:

      * binary and ``binary_collapsed`` rows are dropped when
        ``scenario ∈ EXCLUDE_BINARY_SCENARIOS``.
      * every binary / ``binary_collapsed`` row is dropped when
        ``include_auc=False``.

    These two filters mirror the bootstrap's row-level gates inside
    :func:`compute_per_draw_errors` so the producer's row set matches the
    bootstrap's per-user emission exactly.
    """
    rows: list[tuple] = []
    drop_binary = (not include_auc) or (scenario in EXCLUDE_BINARY_SCENARIOS)
    for ch_key, user_map in per_user.items():
        ch_type = _channel_type_for(ch_key)
        if drop_binary and ch_type in ("binary", "binary_collapsed"):
            continue
        for user_id, e in user_map.items():
            rows.append(
                (
                    method,
                    scenario,
                    split,
                    ch_key,
                    ch_type,
                    "all",
                    "all",
                    user_id,
                    float(e),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


def _load_channel_stds(
    method_pairs_dir: Path,
    channel_stds: np.ndarray | None,
) -> np.ndarray:
    if channel_stds is not None:
        arr = np.asarray(channel_stds, dtype=np.float64)
    else:
        cs_path = method_pairs_dir / "channel_stds.npy"
        if not cs_path.exists():
            raise FileNotFoundError(
                f"channel_stds.npy not found at {cs_path} — refusing to guess. "
                "Pass channel_stds= explicitly, or run mhc-impute-eval to "
                "generate the pairs dir."
            )
        arr = np.load(cs_path).astype(np.float64)
    if arr.shape[0] < N_CHANNELS:
        raise ValueError(
            f"channel_stds has {arr.shape[0]} entries, need at least {N_CHANNELS}"
        )
    return arr


def build_per_user_errors(
    method_pairs_dir: str | Path,
    method_name: str,
    *,
    scenarios: list[str],
    splits: list[str],
    channel_stds: np.ndarray | None = None,
    include_auc: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict]]:
    """Produce one method's per-user errors + display metrics in one pass.

    For each ``(scenario, split)`` under ``method_pairs_dir`` this calls
    :func:`aggregate_pairs` with ``return_per_user=True``, yielding both the
    per-channel display dict and the ``per_user`` map.

    The producer is the **single** importable entry point for
    ``per_user_errors`` generation. The public API, ``mhc-impute-eval``
    runner, and the paper Phase-0 driver all call this — guaranteeing
    that ``per_user_errors.parquet`` rows are byte-identical regardless
    of which surface produced them. The parity contract against
    :func:`compute_per_draw_errors`'s embedded per-user emission is
    pinned by the §1 LOCF parity test described in the design plan.

    Args:
        method_pairs_dir: Pairs root. Must contain
            ``manifest_<split>.parquet``, ``channel_stds.npy``, and
            per-scenario subdirs with
            ``<scenario>/<split>/pairs_ch{NN}.parquet``.
        method_name: Method label embedded in the ``method`` column.
        scenarios: Scenario names to process. Scenarios with no
            ``<scenario>/<split>/`` dir are skipped with a warning.
        splits: Split names to process (e.g. ``["test"]``).
        channel_stds: Per-channel stds; falls back to
            ``method_pairs_dir / "channel_stds.npy"``. Per-user
            ``E`` is std-independent (``sae/n`` and ``1 − AUC``); stds
            only affect the ``normalized_*`` display fields.
        include_auc: When ``False``, drop every binary /
            ``binary_collapsed`` row. Mirrors the bootstrap's
            ``include_auc`` semantics so the producer's per-user row set
            matches the bootstrap's per-user emission for both branches.

    Returns:
        ``(per_user_df, display_metrics)``:

          * ``per_user_df`` — DataFrame with
            :data:`PER_USER_ERRORS_PARQUET_COLUMNS`. One row per
            ``(method, scenario, split, channel, channel_type,
            subgroup_attr, subgroup_value, user_id)`` cell with a finite
            ``E_per_user``. The ``method`` column is constant
            (``method_name``); concatenate with other methods' frames
            before consuming.
          * ``display_metrics`` — ``{(scenario, split): metrics_dict}``.
            Keys: ``per_channel``,
            ``continuous``, ``binary``, ``n_samples`` — the exact subset
            of :func:`aggregate_pairs`'s return that
            :class:`ImputationResults.scenarios` exposes.

    Raises:
        FileNotFoundError: ``channel_stds.npy`` missing and no explicit
            ``channel_stds`` provided.
        ValueError: ``channel_stds`` has fewer entries than
            :data:`N_CHANNELS`.
    """
    method_pairs_dir = Path(method_pairs_dir)
    channel_stds = _load_channel_stds(method_pairs_dir, channel_stds)

    all_rows: list[tuple] = []
    display: dict[tuple[str, str], dict] = {}

    for split in splits:
        for scenario in scenarios:
            ssd = method_pairs_dir / scenario / split
            if not ssd.exists():
                logger.warning(
                    "method=%s scenario=%s split=%s: %s missing — skipping",
                    method_name,
                    scenario,
                    split,
                    ssd,
                )
                continue

            # ---- Display metrics + per_user map ----
            metrics_all = aggregate_pairs(ssd, channel_stds, return_per_user=True)
            display[(scenario, split)] = {
                "per_channel": metrics_all.get("per_channel", {}),
                "continuous": metrics_all.get("continuous", {}),
                "binary": metrics_all.get("binary", {}),
                "n_samples": metrics_all.get("n_samples", 0),
            }
            if "per_user" in metrics_all:
                all_rows.extend(
                    per_user_map_to_rows(
                        metrics_all["per_user"],
                        method=method_name,
                        scenario=scenario,
                        split=split,
                        include_auc=include_auc,
                    )
                )

    if all_rows:
        per_user_df = pd.DataFrame(all_rows, columns=PER_USER_ERRORS_PARQUET_COLUMNS)
    else:
        per_user_df = pd.DataFrame(columns=PER_USER_ERRORS_PARQUET_COLUMNS)
    return per_user_df, display


__all__ = [
    "PER_USER_ERRORS_PARQUET_COLUMNS",
    "build_per_user_errors",
    "per_user_map_to_rows",
    "read_per_user_errors_parquet",
    "write_per_user_errors_parquet",
]
