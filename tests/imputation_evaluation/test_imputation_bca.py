"""Tests for the per-user errors Parquet IO and the skill / rank BCa opt-in stub."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    PER_USER_ERRORS_PARQUET_COLUMNS,
    read_per_user_errors_parquet,
    write_per_user_errors_parquet,
)

# Load aggregate_imputation_paper_metrics as a module even though it lives under
# scripts/. It defines top-level functions we want to exercise directly without
# subprocess overhead.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(relpath: str) -> object:
    """Import a scripts/paper_results/<relpath>.py file as a module."""
    path = _REPO_ROOT / "scripts" / "paper_results" / f"{relpath}.py"
    name = Path(relpath).name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


paper_metrics_mod = _load_script_module("imputation/aggregate_imputation_paper_metrics")


# --------------------------------------------------------------------------
# Phase 1 per-user errors Parquet round-trip
# --------------------------------------------------------------------------


def _tiny_per_user_df() -> pd.DataFrame:
    """Smallest valid per-user errors frame — covers both channel + collapsed."""
    return pd.DataFrame(
        {
            "method": ["locf", "locf", "mymodel", "mymodel"],
            "scenario": ["s1", "s1", "s1", "s1"],
            "split": ["test", "test", "test", "test"],
            "channel": ["ch_0", "cat_collapsed:sleep", "ch_0", "cat_collapsed:sleep"],
            "channel_type": ["continuous", "binary_collapsed", "continuous", "binary_collapsed"],
            "subgroup_attr": ["all", "all", "all", "all"],
            "subgroup_value": ["all", "all", "all", "all"],
            "user_id": ["u1", "u1", "u1", "u1"],
            "E_per_user": [0.5, 0.3, 0.4, 0.2],
        }
    )


def test_phase1_per_user_errors_parquet_roundtrip(tmp_path):
    """Schema + dtype contract for the per-user Parquet IO."""
    src = _tiny_per_user_df()
    path = tmp_path / "per_user_errors.parquet"
    write_per_user_errors_parquet(src, path, meta={"n_boot": 1})
    out, meta = read_per_user_errors_parquet(path)
    assert list(out.columns) == PER_USER_ERRORS_PARQUET_COLUMNS
    assert meta == {"n_boot": 1}
    # Round-trip: same values modulo categorical / float32 conversion.
    out_cmp = out.assign(**{c: out[c].astype(str) for c in PER_USER_ERRORS_PARQUET_COLUMNS[:-1]})
    pd.testing.assert_series_equal(
        out_cmp["E_per_user"].astype(np.float64).reset_index(drop=True),
        src["E_per_user"].astype(np.float64).reset_index(drop=True),
        check_names=False,
        check_dtype=False,
    )


# --------------------------------------------------------------------------
# Skill / rank opt-in: default OFF, LOO not yet wired
# --------------------------------------------------------------------------


def test_skill_rank_bca_off_by_default_via_argparse():
    """``--bca-skill-rank`` defaults False — flag plumbing exists but is off.

    The CLI flag plumbing exists but is not enabled by default.
    """
    # Simulate parsing with no BCa flag.
    saved = sys.argv
    try:
        sys.argv = [
            "aggregate_imputation_paper_metrics",
            "--draws",
            "/tmp/x.parquet",
            "--output-dir",
            "/tmp/x",
        ]
        args = paper_metrics_mod._parse_args()
    finally:
        sys.argv = saved
    assert args.bca_skill_rank is False
    assert args.per_user_errors is None


def test_skill_rank_bca_opt_in_currently_stubbed(tmp_path):
    """Turning ``--bca-skill-rank`` on currently raises NotImplementedError.

    The LOO recompute for skill/rank is tracked as follow-up — see the
    error message and METRICS.md §S7.
    """
    # Build a tiny pair of parquets so the load-path validation passes.
    from imputation_evaluation.evaluation.bootstrap_skill_rank import (
        DRAWS_PARQUET_COLUMNS,
        write_draws_parquet,
    )

    draws_path = tmp_path / "draws.parquet"
    per_user_path = tmp_path / "per_user_errors.parquet"
    write_draws_parquet(
        pd.DataFrame(
            [
                {
                    "method": "locf",
                    "scenario": "s1",
                    "split": "test",
                    "channel": "ch_0",
                    "channel_type": "continuous",
                    "subgroup_attr": "all",
                    "subgroup_value": "all",
                    "draw": 0,
                    "E": 1.0,
                    "R": 1.0,
                    "rank": 1.0,
                }
            ]
        )[DRAWS_PARQUET_COLUMNS],
        draws_path,
    )
    write_per_user_errors_parquet(_tiny_per_user_df(), per_user_path)

    saved = sys.argv
    try:
        sys.argv = [
            "aggregate_imputation_paper_metrics",
            "--draws",
            str(draws_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--bca-skill-rank",
            "--per-user-errors",
            str(per_user_path),
        ]
        with pytest.raises(NotImplementedError, match="LOO jackknife"):
            paper_metrics_mod.main()
    finally:
        sys.argv = saved
