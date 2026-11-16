"""Produce the Track-1 (downstream) leaderboard substrate — per-method prediction pairs.

Pools the per-(method, task) ``test.parquet`` files the eval pipeline writes
(``--output.save_predictions`` / ``PREDICTIONS_DIR``) into one per-user
prediction-pair parquet per method. This is the substrate the leaderboard ingests and
recomputes paired skill / rank from server-side against the ``linear`` baseline — see
``tools/leaderboard_docs/downstream/SCHEMA.md``.

Mirrors the other tracks' substrate producers
(``imputation/parity/produce_per_method_per_user_errors.py``,
``forecasting/parity/produce_seasonal_naive_substrate.py``): wraps the canonical
``build_per_user_pairs`` / ``write_per_user_pairs_parquet`` and writes
``<out-dir>/<method>.parquet`` plus a ``<method>.parquet.meta.json`` provenance
sidecar carrying ``overall_fallback_rate`` (the field
``tools/upload_leaderboard_substrate.py`` reads for issue #39).

Reads saved predictions and the selected dataset's schemas; it does not re-run
any model. Set ``MHC_DATA_DIR`` to the matching dataset root (all three schema
files are required) so task types use the correct release. The pooled rows are
the exact predictions the paper bootstrap (``bootstrap_downstream_draws.py``)
consumes, so the server-side recompute reproduces the committed ``results/paper``
leaderboard numbers (proven by ``parity_substrate.py``).

Usage::

    MHC_DATA_DIR=/path/to/dataset PYTHONPATH=src python scripts/paper_results/downstream/leaderboard/\
produce_per_method_per_user_pairs.py \
        --predictions-dir results/eval/final/predictions \
        --out-dir results/leaderboard_downstream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from downstream_evaluation.evaluation.per_user_pairs import (  # noqa: E402
    build_per_user_pairs,
    write_per_user_pairs_parquet,
)
from downstream_evaluation.evaluation.predictions_io import (  # noqa: E402
    overall_fallback_rate,
    read_fallback_sidecar,
)
from openmhc._constants import BENCHMARK_TASKS  # noqa: E402

# csv key -> (display_name, type), matching the registry in ``build_leaderboard_json.py``.
# ``linear`` is the skill baseline (scored against, not a competitor); it is
# built here for completeness and for the parity gate, but uploading it is the
# maintainer's call (SCHEMA: "do not submit it").
METHODS: dict[str, tuple[str, str]] = {
    "linear": ("Linear (baseline)", "Statistical"),
    "multirocket": ("MultiRocket", "Convolutional"),
    "xgboost": ("XGBoost", "Statistical"),
    "lsm2": ("LSM-2", "Self-Supervised"),
    "gru_d": ("GRU-D", "Deep Learning"),
    "wbm": ("WBM", "Self-Supervised"),
    "toto": ("Toto", "Foundation"),
    "chronos2": ("Chronos-2", "Foundation"),
}
SUBMITTER = "OpenMHC team"
SUBTRACK = "static"  # SCHEMA: static | longitudinal; all 8 score from one weekly embedding

def resolve_fallback_rate(
    predictions_dir: Path, method: str, task_sizes: dict[str, int]
) -> float:
    """Pool measured fallback counts after checking the exported task cohorts."""
    sidecar = read_fallback_sidecar(predictions_dir, method)
    if not isinstance(sidecar, dict) or not sidecar:
        raise ValueError(f"{method}: missing measured fallback.json; rerun the evaluation")
    if set(sidecar) != set(task_sizes):
        raise ValueError(f"{method}: fallback task set does not match the exported predictions")
    for task, counts in sidecar.items():
        if not isinstance(counts, dict):
            raise ValueError(f"{method}/{task}: invalid fallback counts")
        n_test, n_fallback = counts.get("n_test"), counts.get("n_fallback")
        if (
            type(n_test) is not int or type(n_fallback) is not int
            or n_test <= 0 or not 0 <= n_fallback <= n_test
        ):
            raise ValueError(f"{method}/{task}: invalid fallback counts")
        if n_test != task_sizes[task]:
            raise ValueError(f"{method}/{task}: fallback cohort size does not match prediction rows")
    return overall_fallback_rate(sidecar)


def main() -> int:
    """Build the per-method substrate parquet(s) + provenance sidecars."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=REPO_ROOT / "results/eval/final/predictions",
        help="Dir with <method>/<task>/test.parquet.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/leaderboard_downstream",
        help="Where to write <method>.parquet (+ .meta.json provenance sidecar).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        help="Methods to build (default: all 8 canonical methods).",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"predictions-dir={args.predictions_dir}")
    print(f"out-dir={args.out_dir}\n")
    print(f"{'method':12s} {'rows':>8s} {'tasks':>5s} {'users':>6s} {'fallback':>9s}")

    for method in args.methods:
        if method not in METHODS:
            raise SystemExit(f"unknown method {method!r}; known: {list(METHODS)}")
        df = build_per_user_pairs(
            args.predictions_dir,
            method_dir=method,
            tasks=BENCHMARK_TASKS,
            method_label=method,
        )
        if df.empty:
            raise SystemExit(f"empty substrate for {method!r} — no task predictions found")
        if not np.isfinite(df[["y_true", "y_pred", "y_proba"]].to_numpy()).all():
            raise SystemExit(f"{method}: non-finite pair values — fallback should already be applied")

        if df.duplicated(["task", "user_id"]).any():
            raise SystemExit(f"{method}: duplicate participant-task prediction pairs")
        task_sizes = df.groupby("task", observed=True).size().to_dict()
        rate = resolve_fallback_rate(args.predictions_dir, method, task_sizes)
        out = args.out_dir / f"{method}.parquet"
        write_per_user_pairs_parquet(
            df,
            out,
            meta={
                "method": method,
                "track": "downstream",
                "baseline": "linear",
                "overall_fallback_rate": rate,
                "n_pairs": int(len(df)),
                "n_tasks": int(df["task"].nunique()),
                "source_predictions": str(args.predictions_dir),
            },
        )
        n_users = df.loc[df.subgroup_attr == "all", "user_id"].nunique()
        print(
            f"{method:12s} {len(df):>8d} {df['task'].nunique():>5d} {n_users:>6d} {rate:>9.4f}"
        )

    print(f"\nWrote {len(args.methods)} substrate parquet(s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
