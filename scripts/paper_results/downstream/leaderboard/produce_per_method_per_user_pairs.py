"""Produce the Track-1 (downstream) leaderboard substrate — per-method prediction pairs.

Pools the per-(method, task) ``test.parquet`` files the eval pipeline writes
(``--output.save_predictions`` / ``PREDICTIONS_DIR``) into one per-user
prediction-pair parquet per method, subgroup-expanded over ``all`` / ``age_group``
/ ``sex`` (the fairness axes from ``_subgroups.json``). This is the substrate the
leaderboard ingests and recomputes paired skill / rank / fairness from server-side
against the ``linear`` baseline — see ``tools/leaderboard_docs/downstream/SCHEMA.md``.

Mirrors the other tracks' substrate producers
(``imputation/parity/produce_per_method_per_user_errors.py``,
``forecasting/parity/produce_seasonal_naive_substrate.py``): wraps the canonical
``build_per_user_pairs`` / ``write_per_user_pairs_parquet`` and writes
``<out-dir>/<method>.parquet`` plus a ``<method>.parquet.meta.json`` provenance
sidecar carrying ``overall_fallback_rate`` (the field
``tools/upload_leaderboard_substrate.py`` reads for issue #39).

Reads saved predictions only — it does not re-run any model. The pooled rows are
the exact predictions the paper bootstrap (``bootstrap_downstream_draws.py``)
consumes, so the server-side recompute reproduces the committed ``results/paper``
leaderboard numbers (proven by ``parity_substrate.py``).

Usage::

    PYTHONPATH=src python scripts/paper_results/downstream/parity/\
produce_per_method_per_user_pairs.py \
        --predictions-dir results/eval/final/predictions \
        --out-dir results/leaderboard_downstream
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
# ``linear`` is the skill / fairness baseline (scored against, not a competitor); it is
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
    predictions_dir: Path, method: str, overrides: dict[str, float]
) -> float:
    """The method's overall fallback rate (fraction substituted with the Linear baseline).

    Preference order: an explicit ``--fallback-rate`` override, else the measured
    ``fallback.json`` sidecar the eval writes next to the predictions
    (``Σ n_fallback / Σ n_test``), else ``0.0``. A prediction dir written before the
    sidecar existed has neither, so a legacy method with real fallback (e.g. ``wbm``)
    must be given an explicit override — otherwise it is reported as ``0.0`` with a
    warning rather than silently.
    """
    if method in overrides:
        return overrides[method]
    sidecar = read_fallback_sidecar(predictions_dir, method)
    if sidecar is not None:
        return overall_fallback_rate(sidecar)
    print(
        f"  [warn] {method}: no fallback.json sidecar and no --fallback-rate override; "
        "reporting fallback rate as 0.0",
        file=sys.stderr,
    )
    return 0.0


def main() -> int:
    """Build the per-method substrate parquet(s) + provenance sidecars."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--predictions-dir",
        type=Path,
        default=REPO_ROOT / "results/eval/final/predictions",
        help="Dir with <method>/<task>/test.parquet + _subgroups.json.",
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
    p.add_argument(
        "--fallback-rate",
        nargs="*",
        default=[],
        metavar="METHOD=RATE",
        help=(
            "Override a method's fallback rate for legacy prediction dirs that predate the "
            "fallback.json sidecar (e.g. wbm=0.6276). Fresh runs read the measured sidecar."
        ),
    )
    args = p.parse_args()

    overrides: dict[str, float] = {}
    for item in args.fallback_rate:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--fallback-rate expects METHOD=RATE, got {item!r}")
        overrides[name] = float(value)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subgroups = args.predictions_dir / "_subgroups.json"
    if not subgroups.exists():
        raise SystemExit(f"missing {subgroups} — fairness rows need the subgroup map")

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
        if df[["y_true", "y_pred", "y_proba"]].isna().any().any():
            raise SystemExit(f"{method}: NaN pair values — fallback should already be applied")

        rate = resolve_fallback_rate(args.predictions_dir, method, overrides)
        out = args.out_dir / f"{method}.parquet"
        write_per_user_pairs_parquet(
            df,
            out,
            meta={
                "method": method,
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
