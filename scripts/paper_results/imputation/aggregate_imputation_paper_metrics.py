#!/usr/bin/env python
r"""Phase 2 of the imputation paper-metrics bootstrap.

Reads ``bootstrap_draws.parquet`` produced by phase 1
(``bootstrap_imputation_draws.py``) and emits the headline sidecar CSVs:

* ``skill_scores_bootstrap.csv``
* ``avg_rankings_bootstrap.csv``

Phase 2 is **fast** — rerun it with other clip bounds or a method subset
without re-resampling.

Example::

    python scripts/paper_results/aggregate_imputation_paper_metrics.py \
        --draws results/paper/bootstrap_draws.parquet \
        --output-dir results/paper/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    aggregate_skill_rank,
    read_draws_parquet,
    read_per_user_errors_parquet,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: summarise bootstrap_draws.parquet into paper sidecar CSVs",
    )
    p.add_argument(
        "--draws",
        type=Path,
        required=True,
        help="Path to bootstrap_draws.parquet from phase 1",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the two sidecar CSVs",
    )
    p.add_argument(
        "--baseline-method",
        default="locf",
        help="Method to treat as the skill-score baseline (default: locf)",
    )
    p.add_argument(
        "--clip-lower",
        type=float,
        default=1e-2,
        help="Lower clip bound for error ratios (default: 1e-2)",
    )
    p.add_argument(
        "--clip-upper",
        type=float,
        default=100.0,
        help="Upper clip bound for error ratios (default: 100.0)",
    )
    p.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Percentile CI level (default: 0.95)",
    )
    p.add_argument(
        "--method-filter",
        nargs="+",
        default=None,
        help=(
            "Restrict the rank comparison pool to these methods only. "
            "Skill values stay the same (pairwise vs. baseline); "
            "avg_rank values shift because the pool size changes. The "
            "baseline method (default: locf) MUST be in the filter or "
            "skill rows will be empty. See METRICS.md §8.1 for the full "
            "subset-recompute workflow."
        ),
    )
    bca_grp = p.add_mutually_exclusive_group()
    bca_grp.add_argument(
        "--bca-skill-rank",
        dest="bca_skill_rank",
        action="store_true",
        default=False,
        help=(
            "Opt-in BCa CI augmentation for skill / rank tables. Default OFF — "
            "skill and rank are near-unbiased, so the BCa interval is close to the "
            "percentile CI in practice. Requires --per-user-errors. NOTE: "
            "the LOO jackknife required by BCa is currently not wired up for "
            "skill / rank (it would require an N-iteration recompute of the "
            "point flow); passing this flag currently raises "
            "NotImplementedError. See METRICS.md §S7."
        ),
    )
    bca_grp.add_argument(
        "--no-bca-skill-rank",
        dest="bca_skill_rank",
        action="store_false",
        help="Explicitly disable BCa for skill / rank (the default).",
    )
    p.add_argument(
        "--per-user-errors",
        type=Path,
        default=None,
        help=(
            "Path to per_user_errors.parquet from Phase 1, required when "
            "--bca-skill-rank is on. Defaults to "
            "'<draws>.parent / per_user_errors.parquet'."
        ),
    )
    return p.parse_args()


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, meta = read_draws_parquet(args.draws)
    logger.info("Loaded %d rows from %s", len(df), args.draws)
    if meta is not None:
        logger.info(
            "Phase-1 meta: n_boot=%s, seed=%s, methods=%s, scenarios=%s",
            meta.get("n_boot"),
            meta.get("seed"),
            meta.get("methods"),
            meta.get("scenarios"),
        )
    if args.method_filter:
        df = df[df["method"].isin(args.method_filter)].copy()
        logger.info("After --method-filter: %d rows", len(df))

    if "rank" not in df.columns:
        logger.error(
            "The Phase-2 reducer requires a 'rank' column on the draws "
            "Parquet, but %s has none. Re-run Phase 1 "
            "(bootstrap_imputation_draws.py) to regenerate draws.",
            args.draws,
        )
        return 2

    if args.bca_skill_rank:
        # Validate the per-user-errors path exists even when stubbed so the
        # CLI fails early on configuration errors. Once the LOO recompute is
        # wired up the file is needed.
        per_user_path = (
            args.per_user_errors
            if args.per_user_errors is not None
            else args.draws.parent / "per_user_errors.parquet"
        )
        if not per_user_path.exists():
            logger.error(
                "--bca-skill-rank requires %s but the file does not exist; "
                "pass --per-user-errors PATH or rerun Phase 1 without "
                "--no-per-user-errors.",
                per_user_path,
            )
            return 2
        # Smoke-load to surface schema errors early (cheap; pyarrow lazy reads).
        _per_user_df, _ = read_per_user_errors_parquet(per_user_path)
        logger.info(
            "Loaded %d per-user rows from %s",
            len(_per_user_df),
            per_user_path,
        )
        raise NotImplementedError(
            "Skill / rank BCa is a planned parity sanity-check, but the LOO "
            "jackknife requires N (~5K user) recomputes of the point flow "
            "(compute_per_task_paired_R + compute_skill_scores + "
            "compute_average_rankings). Wiring + optimisation is tracked as "
            "follow-up work."
        )

    tables = aggregate_skill_rank(
        df,
        baseline_method=args.baseline_method,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        ci_level=args.ci_level,
    )

    out = args.output_dir
    paths = {
        "skill_scores": out / "skill_scores_bootstrap.csv",
        "avg_rankings": out / "avg_rankings_bootstrap.csv",
    }
    for key, path in paths.items():
        tbl = tables[key]
        tbl.to_csv(path, index=False, float_format="%.6f")
        logger.info("Wrote %s (%d rows)", path, len(tbl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
