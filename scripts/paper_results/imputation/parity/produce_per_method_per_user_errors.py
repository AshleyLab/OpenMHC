"""Produce one per-method ``per_user_errors.parquet`` via the canonical producer.

Single-method runner. Feeds a SLURM array (one task per method) that
populates ``paper-verification/per_user/<method>.parquet`` for every
method enumerated in ``paper/bootstrap_method_dirs.json``. The output
set then drives §2 (point parity) and §3 (bootstrap parity).

Parity against today's pooled ``paper/per_user_errors.parquet`` was
already established for ``locf`` in §1; that test pins the producer's
formula identity. This script just applies the same producer to every
other method — the row sets concat'd together must reproduce today's
pooled file slice-for-slice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from imputation_evaluation.evaluation.per_user_errors import (
    build_per_user_errors,
    write_per_user_errors_parquet,
)
from imputation_evaluation.sensitivity import sample_subgroups
from labels.api import AGE_GROUP_EDGES

logger = logging.getLogger(__name__)

DEFAULT_METHOD_DIRS = Path(
    "/scratch/users/schuetzn/openmhc-imputation-eval/paper/bootstrap_method_dirs.json"
)
DEFAULT_OUT_ROOT = Path(
    "/scratch/users/schuetzn/openmhc-imputation-eval/paper-verification/per_user"
)
DEFAULT_SPLITS = ["test"]
DEFAULT_SCENARIOS = [
    "intensity_failure",
    "random_noise",
    "signal_slice",
    "sleep_gap",
    "temporal_slice",
    "workout_gap",
]


def main() -> int:
    """Build one method's per-user errors artifact from saved pairs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, help="Method name (key in method_dirs JSON)")
    p.add_argument("--method-dirs", type=Path, default=DEFAULT_METHOD_DIRS)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    with args.method_dirs.open() as f:
        method_dirs = {m: Path(p) for m, p in json.load(f).items()}
    if args.method not in method_dirs:
        logger.error("Unknown method %s; known: %s", args.method, list(method_dirs.keys()))
        return 2

    pairs_dir = method_dirs[args.method]
    if not pairs_dir.exists():
        logger.error("Pairs dir does not exist: %s", pairs_dir)
        return 2

    out_path = args.out_root / f"{args.method}.parquet"
    if "paper-verification" not in out_path.parts and "paper" in out_path.parts:
        logger.error("REFUSING to write under paper/: %s", out_path)
        return 2

    # Subgroup mappings — use this method's own manifest. Manifests across
    # methods agree on (sample_idx → user_id, date) per the bootstrap's
    # _assert_manifests_agree check, so per-method demographic bucketing is
    # identical to using the reference method's manifest.
    subgroup_mappings: dict[str, dict[int, dict[str, str]]] = {}
    for split in args.splits:
        sg = sample_subgroups(pairs_dir, split)
        if sg is None:
            logger.error("Could not build subgroup mapping for split=%s", split)
            return 2
        subgroup_mappings[split] = sg

    per_user_df, _display = build_per_user_errors(
        method_pairs_dir=pairs_dir,
        method_name=args.method,
        scenarios=args.scenarios,
        splits=args.splits,
        subgroup_mappings=subgroup_mappings,
        include_auc=True,
        exclude_unknown=False,
    )
    logger.info("method=%s rows=%d", args.method, len(per_user_df))
    if per_user_df.empty:
        logger.error("Producer returned empty frame for %s", args.method)
        return 2

    meta = {
        "method": args.method,
        "scenarios": args.scenarios,
        "splits": args.splits,
        "age_bins": list(AGE_GROUP_EDGES),
        "source_pairs": str(pairs_dir),
        "purpose": "Phase B per-method per_user_errors fan-out",
    }
    write_per_user_errors_parquet(per_user_df, out_path, meta=meta)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
