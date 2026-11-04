#!/usr/bin/env python
r"""Phase 1 of the imputation paper-metrics bootstrap.

Reads each method's saved imputation pairs/, runs a paired participant-level
bootstrap **across methods** (shared resample matrix per split so inter-method
differences are paired), and writes a long-format Parquet of per-draw error
values plus a sidecar metadata JSON. Every row has ``subgroup_attr`` and
``subgroup_value`` equal to ``"all"``.

Phase 2 (``aggregate_imputation_paper_metrics.py``) reads that Parquet and
produces summary CSVs (mean / SE / 95 % CI for skill score and average rank).

Example::

    python scripts/paper_results/bootstrap_imputation_draws.py \
        --method-dirs configs/paper/bootstrap_method_dirs.json \
        --output results/paper/bootstrap_draws.parquet \
        --n-boot 1000 --seed 42 \
        --splits test

The JSON manifest maps method -> pairs_dir, e.g.::

    {
      "locf":            "results/imputation_eval/imputation_locf_X/pairs",
      "mae_daily_nodropout": "results/imputation_eval/mae_daily_nodropout_Y/pairs"
    }

Each pairs_dir must contain ``manifest_<split>.parquet`` and per-scenario
subdirs with ``<scenario>/<split>/pairs_ch{NN}.parquet``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    compute_per_draw_errors,
    compute_per_draw_errors_from_per_user_errors,
    write_draws_parquet,
    write_per_user_errors_parquet,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: build bootstrap_draws.parquet for the paper metrics pipeline",
    )
    p.add_argument(
        "--method-dirs",
        type=Path,
        required=True,
        help="JSON manifest mapping {method: pairs_dir}",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Parquet (.parquet). Sidecar .meta.json written alongside.",
    )
    p.add_argument(
        "--per-user-errors",
        type=Path,
        default=None,
        help=(
            "Output Parquet for per-user E values. Default: sibling of --output "
            "named 'per_user_errors.parquet'. Pass --no-per-user-errors to skip."
        ),
    )
    p.add_argument(
        "--no-per-user-errors",
        action="store_true",
        help="Skip per-user errors emission.",
    )
    p.add_argument(
        "--per-user-errors-dir",
        type=Path,
        default=None,
        help=(
            "Phase B fast path: directory of per-method per_user_errors.parquet "
            "files (file stem = method name). When set, skips the pair-reading + "
            "CellStats build and feeds compute_per_draw_errors_from_per_user_errors "
            "directly — much faster, and byte-equal to the from-pairs path given "
            "the same seed (verified by §3 of the imputation-metrics refactor)."
        ),
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of bootstrap draws (default 1000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed; per-cell seeds derived deterministically",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to process (default: test). Example: --splits test val",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenarios to process (default: auto-discover from first method's dir)",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Restrict to a subset of methods from --method-dirs (default: all)",
    )
    p.add_argument(
        "--no-auc",
        action="store_true",
        help="Skip AUC bootstrap for binary channels (faster)",
    )
    p.add_argument(
        "--channel-stds-path",
        type=Path,
        default=None,
        help="Override channel_stds.npy path (default: <first method dir>/channel_stds.npy)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail (non-zero exit) on any missing method dir or missing "
            "per-split manifest instead of warning-and-skipping. "
            "Required for runs whose numbers are published."
        ),
    )
    return p.parse_args()


def _discover_scenarios(method_dirs: dict[str, Path], split: str) -> list[str]:
    """Auto-discover scenarios = subdirs that contain a /<split>/ child."""
    seen: set[str] = set()
    for root in method_dirs.values():
        root = Path(root)
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and (child / split).is_dir():
                seen.add(child.name)
    return sorted(seen)


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()

    with args.method_dirs.open() as f:
        raw_dirs = json.load(f)
    method_dirs: dict[str, Path] = {m: Path(p) for m, p in raw_dirs.items()}
    if args.methods:
        method_dirs = {m: p for m, p in method_dirs.items() if m in args.methods}
    if not method_dirs:
        logger.error("No methods left after --methods filter")
        return 2

    # Validate paths
    for m, p in list(method_dirs.items()):
        if not p.exists():
            if args.strict:
                logger.error(
                    "[strict] method=%s: %s does not exist — aborting",
                    m,
                    p,
                )
                return 2
            logger.warning("method=%s: %s does not exist — skipping", m, p)
            method_dirs.pop(m)
    if not method_dirs:
        logger.error("All method dirs missing; aborting")
        return 2

    # Resolve scenarios
    if args.scenarios:
        scenarios = list(args.scenarios)
    else:
        # Use first split for discovery; downstream will skip cells absent for a split.
        scenarios = _discover_scenarios(method_dirs, args.splits[0])
    if not scenarios:
        logger.error("No scenarios discovered; aborting")
        return 2
    logger.info("Methods: %s", list(method_dirs.keys()))
    logger.info("Scenarios: %s", scenarios)
    logger.info("Splits: %s", args.splits)

    if args.strict and args.per_user_errors_dir is None:
        ref_dir = next(iter(method_dirs.values()))
        for split in args.splits:
            if not (ref_dir / f"manifest_{split}.parquet").exists():
                logger.error(
                    "[strict] No manifest_%s.parquet at %s — aborting",
                    split,
                    ref_dir,
                )
                return 2

    # Channel stds
    channel_stds = None
    if args.channel_stds_path:
        channel_stds = np.load(args.channel_stds_path)
        logger.info("Using channel_stds from %s", args.channel_stds_path)

    # Build per-draw errors (and, unless skipped, per-user errors).
    emit_per_user = not args.no_per_user_errors
    per_user_errors_path: Path | None = None
    if emit_per_user:
        per_user_errors_path = (
            args.per_user_errors
            if args.per_user_errors is not None
            else args.output.parent / "per_user_errors.parquet"
        )

    t0 = datetime.now()
    if args.per_user_errors_dir is not None:
        # ---- B1/B2 fast path: per_user_errors files in, draws out ----
        if not args.per_user_errors_dir.exists():
            logger.error(
                "--per-user-errors-dir does not exist: %s", args.per_user_errors_dir
            )
            return 2
        logger.info(
            "Phase B fast path — loading per-method per_user_errors from %s",
            args.per_user_errors_dir,
        )
        parts: list[pd.DataFrame] = []
        for method in method_dirs:
            path = args.per_user_errors_dir / f"{method}.parquet"
            if not path.exists():
                msg = f"missing per_user_errors file for method={method}: {path}"
                if args.strict:
                    logger.error("[strict] %s — aborting", msg)
                    return 2
                logger.warning("%s — skipping", msg)
                continue
            parts.append(pd.read_parquet(path))
        if not parts:
            logger.error("No per_user_errors files loaded — aborting")
            return 2
        per_user_input = pd.concat(parts, ignore_index=True)
        logger.info(
            "Loaded per_user_errors: %d rows across %d methods",
            len(per_user_input),
            per_user_input["method"].nunique(),
        )

        # Manifests (canonical user ordering — parity contract).
        # The legacy from-pairs path rebuilds canonical user ordering per split,
        # so the fast path must keep manifests split-scoped as well.
        method_manifests_by_split: dict[str, dict[str, pa.Table]] = {
            split: {} for split in args.splits
        }
        for split in args.splits:
            for method, root in method_dirs.items():
                manifest_path = Path(root) / f"manifest_{split}.parquet"
                if manifest_path.exists():
                    method_manifests_by_split[split][method] = pq.read_table(manifest_path)
                elif args.strict:
                    logger.error(
                        "[strict] missing manifest for method=%s split=%s: %s",
                        method,
                        split,
                        manifest_path,
                    )
                    return 2
                else:
                    logger.warning(
                        "missing manifest for method=%s split=%s: %s",
                        method,
                        split,
                        manifest_path,
                    )
        method_manifests_by_split = {
            split: manifests
            for split, manifests in method_manifests_by_split.items()
            if manifests
        }
        if not method_manifests_by_split:
            logger.error(
                "No manifests found under method_dirs for ordering — aborting"
            )
            return 2

        # Filter per_user_input to the requested splits/scenarios so the
        # cell iteration matches the legacy path's scope.
        per_user_input = per_user_input[per_user_input["split"].isin(args.splits)]
        if scenarios:
            per_user_input = per_user_input[per_user_input["scenario"].isin(scenarios)]
        # Published substrates may still hold age and sex rows.
        per_user_input = per_user_input[per_user_input["subgroup_attr"] == "all"]

        df = compute_per_draw_errors_from_per_user_errors(
            per_user_input,
            method_manifests_by_split,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        # The new path doesn't re-emit per_user_errors (input == output).
        per_user_df = per_user_input.reset_index(drop=True) if emit_per_user else None
    else:
        # ---- Legacy from-pairs path ----
        result = compute_per_draw_errors(
            method_dirs={m: Path(p) for m, p in method_dirs.items()},
            scenarios=scenarios,
            splits=args.splits,
            n_boot=args.n_boot,
            seed=args.seed,
            channel_stds=channel_stds,
            include_auc=not args.no_auc,
            emit_per_user_errors=emit_per_user,
        )
        if emit_per_user:
            df, per_user_df = result
        else:
            df = result
            per_user_df = None
    t_elapsed = (datetime.now() - t0).total_seconds()
    logger.info("Per-draw errors: %d rows in %.1fs", len(df), t_elapsed)
    if per_user_df is not None:
        logger.info("Per-user errors: %d rows", len(per_user_df))

    if df.empty:
        logger.error("No errors produced; aborting before write")
        return 1

    meta = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "splits": args.splits,
        "scenarios": scenarios,
        "methods": list(method_dirs.keys()),
        "method_dirs": {m: str(p) for m, p in method_dirs.items()},
        "include_auc": not args.no_auc,
        "elapsed_seconds": t_elapsed,
        "n_rows": len(df),
        "n_per_user_rows": int(len(per_user_df)) if per_user_df is not None else None,
        "per_user_errors_path": str(per_user_errors_path) if per_user_errors_path else None,
        "git_commit": _git_commit(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "argv": sys.argv,
    }
    write_draws_parquet(df, args.output, meta=meta)
    logger.info("Wrote %s and %s.meta.json", args.output, args.output)
    if per_user_df is not None and per_user_errors_path is not None:
        if per_user_df.empty:
            logger.warning(
                "Per-user errors frame is empty — writing %s anyway for schema stability",
                per_user_errors_path,
            )
        write_per_user_errors_parquet(per_user_df, per_user_errors_path, meta=meta)
        logger.info("Wrote %s", per_user_errors_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
