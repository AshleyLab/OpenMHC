#!/usr/bin/env python
r"""Phase 1 of the downstream paper-metrics bootstrap.

Reads per-(method, task) prediction parquets emitted by the eval pipeline when
``--output.save_predictions true`` is set, samples the declared test split once
per replicate with participant multiplicities shared across tasks and methods,
and writes ``bootstrap_draws.parquet``:
a long-format frame of per-(method, task, subgroup, draw) error ``E = 1 − metric``.

Phase 2 (``aggregate_downstream_paper_metrics.py``) turns those draws into the
skill / rank / fairness sidecar CSVs without re-resampling.

Usage::

    PYTHONPATH=src python scripts/paper_results/downstream/bootstrap_downstream_draws.py \
        --predictions_dir results/eval/final/predictions \
        --csvs_dir results/eval/final \
        --methods linear multirocket mae toto chronos2 xgboost wbm gru_d \
        --n_bootstrap 1000 --seed 42 \
        --output results/paper/bootstrap_draws.parquet

Fairness rows require ``predictions_dir/_subgroups.json`` (per-user
{age_group, sex}). When it is missing only the global (``subgroup_attr="all"``)
rows are written and fairness columns come out NaN in phase 2.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from downstream_evaluation.evaluation.bootstrap_skill_rank import (
    align_across_methods,
    compute_per_draw_errors,
    load_bootstrap_test_users,
    load_method_predictions,
    load_subgroup_map,
    write_draws_parquet,
)
from downstream_evaluation.evaluation.skill_score import TASK_DOMAIN_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    """Parse args, run the paired bootstrap, write the draws parquet."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--predictions_dir", type=Path, required=True)
    p.add_argument(
        "--csvs_dir",
        type=Path,
        required=True,
        help="Dir containing eval_*.csv files (used to look up task_type).",
    )
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline", default="linear", help="Skill / fairness baseline method (recorded in the meta sidecar).")
    p.add_argument("--output", type=Path, required=True, help="bootstrap_draws.parquet path")
    p.add_argument("--split-file", type=Path, help="Explicit train/validation/test split JSON; overrides dataset resolution.")
    p.add_argument("--data-dir", type=Path, help="Dataset root; defaults to MHC_DATA_DIR when --split-file is absent.")
    p.add_argument("--version", choices=["full", "xs"], default="full")
    p.add_argument("--fairness_attributes", nargs="*", default=["age_group", "sex"])
    args = p.parse_args()

    test_uids, split_meta = load_bootstrap_test_users(
        args.split_file, data_dir=args.data_dir, version=args.version
    )

    aligned = align_across_methods(
        {m: load_method_predictions(args.predictions_dir, m, args.csvs_dir) for m in args.methods}
    )
    log.info(
        "Aligned %d tasks across %d methods",
        len(aligned[args.methods[0]]),
        len(args.methods),
    )

    subgroup_map = load_subgroup_map(args.predictions_dir)
    attributes = args.fairness_attributes if subgroup_map is not None else None

    draws = compute_per_draw_errors(
        aligned,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        subgroup_map=subgroup_map,
        subgroup_attributes=attributes,
        domain_map=TASK_DOMAIN_MAP,
        test_uids=test_uids,
    )
    meta = {
        "n_boot": args.n_bootstrap,
        "seed": args.seed,
        "baseline": args.baseline,
        "methods": list(args.methods),
        "n_tasks": int(draws["task"].nunique()),
        "fairness_attributes": list(attributes) if attributes else [],
        **split_meta,
    }
    write_draws_parquet(draws, args.output, meta)
    log.info("Wrote %s (%d rows, %d draws)", args.output, len(draws), draws["draw"].nunique())


if __name__ == "__main__":
    main()
