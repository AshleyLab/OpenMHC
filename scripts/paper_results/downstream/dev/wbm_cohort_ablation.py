#!/usr/bin/env python
r"""WBM-cohort ablation: skill on the cohort WBM can actually score.

WBM emits a real prediction only for participants who have a weekly embedding; for
everyone else the harness substitutes the Linear baseline (≈66 % of test pairs).
That fallback dilutes WBM's headline skill and makes the cross-model comparison
uneven — most of WBM's "predictions" are literally the baseline, and the models it
is compared against are scored on a different (larger) cohort.

This ablation restricts every task's test cohort to the **WBM-real** participants —
the daily test cohort ∩ the weekly-embedding cohort, i.e. the exact per-task set WBM
itself scores (resolved from the same weekly lookup the WBM model consults) — and
recomputes the skill score there, with the Linear baseline **re-evaluated on the same
subset** so ``skill = 0`` still means "ties Linear" on identical participants. It
re-scores the *frozen* predictions; no model is re-run. The full cohort is recomputed
with the identical machinery for a like-for-like comparison.

The two cohorts share the production knobs (paired bootstrap, ``seed`` / ``clip`` /
``ci_level`` from ``configs/paper/downstream_paper.yaml``), so the ``full`` column
reproduces ``skill_scores_bootstrap.csv`` exactly and the two columns are directly
comparable.

Outputs (under ``--output-dir``, default ``results/paper/``):
  * ``wbm_cohort_ablation_skill.csv`` — skill per method × scope, both cohorts
    (``cohort`` column), schema matching ``skill_scores_bootstrap.csv``.
  * ``wbm_cohort_sizes.csv`` — per-task daily vs WBM-real cohort size + fallback rate.

Usage::

    MHC_DATA_DIR=/path/to/full/dataset PYTHONPATH=src python \
        scripts/paper_results/downstream/dev/wbm_cohort_ablation.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from downstream_evaluation.data.provider import LOOKUP_BY_GRANULARITY, TaskDataProvider
from downstream_evaluation.data.splits import load_split_file
from downstream_evaluation.evaluation.bootstrap_skill_rank import (
    aggregate_skill_rank_fairness,
    align_across_methods,
    compute_per_draw_errors,
    load_bootstrap_test_users,
    load_method_predictions,
)
from downstream_evaluation.evaluation.skill_score import (
    DEFAULT_CLIP_LOWER,
    DEFAULT_CLIP_UPPER,
    TASK_DOMAIN_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def wbm_real_masks(
    method_tasks: dict[str, dict[str, dict]],
    reference_method: str,
    data_dir: str | None,
) -> dict[str, set]:
    """``{task: set(wbm-real uids)}`` = daily test cohort ∩ weekly-embedding cohort.

    The weekly cohort per task comes from the same weekly lookup the WBM model
    consults (``_DatasetPaths.from_root`` → ``weekly_labels_lookup_*`` +
    ``TaskDataProvider``), so the mask is exactly the set of participants WBM
    scores rather than falls back on.
    """
    from openmhc._evaluate import _DatasetPaths

    paths = _DatasetPaths.from_root(data_dir)
    lookup = str(paths.root / "processed" / LOOKUP_BY_GRANULARITY["weekly"])
    prov = TaskDataProvider(lookup, load_split_file(paths.splits_file), granularity="weekly")

    masks: dict[str, set] = {}
    for task in method_tasks[reference_method]:
        daily = set(method_tasks[reference_method][task]["uids"])
        weekly = set(map(str, prov.task_data(task, "test").user_ids))
        masks[task] = daily & weekly
    return masks


def subset_predictions(
    method_tasks: dict[str, dict[str, dict]],
    masks: dict[str, set],
) -> dict[str, dict[str, dict]]:
    """Restrict every method's per-task arrays to the mask uids (alignment preserved)."""
    out: dict[str, dict[str, dict]] = {m: {} for m in method_tasks}
    for m, tasks in method_tasks.items():
        for task, p in tasks.items():
            keep = masks.get(task, set())
            idx = np.array([i for i, u in enumerate(p["uids"]) if u in keep], dtype=int)
            if idx.size == 0:
                continue
            out[m][task] = {
                "uids": p["uids"][idx],
                "y_true": p["y_true"][idx],
                "y_pred": p["y_pred"][idx],
                "y_proba": p["y_proba"][idx],
                "task_type": p["task_type"],
            }
    return out


def cohort_skill(
    method_tasks: dict[str, dict[str, dict]],
    label: str,
    *,
    baseline: str,
    n_bootstrap: int,
    seed: int,
    clip_lower: float,
    clip_upper: float,
    ci_level: float,
    test_uids,
) -> pd.DataFrame:
    """Run the paired bootstrap + skill aggregation for one cohort; tag with ``cohort``."""
    aligned = align_across_methods(method_tasks)
    draws = compute_per_draw_errors(
        aligned, n_bootstrap=n_bootstrap, seed=seed, subgroup_map=None, subgroup_attributes=None,
        test_uids=test_uids,
    )
    tables = aggregate_skill_rank_fairness(
        draws, baseline=baseline, clip_lower=clip_lower, clip_upper=clip_upper, ci_level=ci_level
    )
    skill = tables["skill_scores"]
    skill.insert(0, "cohort", label)
    skill.attrs["bootstrap"] = draws.attrs["bootstrap"]
    return skill


def cohort_sizes(
    method_tasks: dict[str, dict[str, dict]], reference_method: str, masks: dict[str, set]
) -> pd.DataFrame:
    """Per-task daily vs WBM-real cohort size + fallback rate."""
    rows = []
    for task in sorted(method_tasks[reference_method]):
        n_full = len(method_tasks[reference_method][task]["uids"])
        n_sub = len(masks.get(task, set()))
        rows.append(
            {
                "task": task,
                "domain": TASK_DOMAIN_MAP.get(task),
                "task_type": method_tasks[reference_method][task]["task_type"],
                "n_full": n_full,
                "n_wbm_real": n_sub,
                "fallback_rate": round(1 - n_sub / n_full, 4) if n_full else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Build the WBM-real masks, recompute skill on both cohorts, write the CSVs."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--predictions_dir", type=Path, default=Path("results/eval/final/predictions"))
    p.add_argument("--csvs_dir", type=Path, default=Path("results/eval/final"))
    p.add_argument("--output-dir", type=Path, default=Path("results/paper"))
    p.add_argument(
        "--data_dir",
        default=None,
        help="Dataset root for the weekly lookup. None → MHC_DATA_DIR (must be the 'full' release).",
    )
    p.add_argument("--methods", nargs="+", default=["linear", "lsm2", "xgboost", "wbm"])
    p.add_argument("--baseline", default="linear")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip-lower", type=float, default=DEFAULT_CLIP_LOWER)
    p.add_argument("--clip-upper", type=float, default=DEFAULT_CLIP_UPPER)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--split-file", type=Path, default=None)
    args = p.parse_args()
    test_uids, split_meta = load_bootstrap_test_users(args.split_file, data_dir=args.data_dir)

    raw = {m: load_method_predictions(args.predictions_dir, m, args.csvs_dir) for m in args.methods}
    masks = wbm_real_masks(raw, args.baseline, args.data_dir)

    sizes = cohort_sizes(raw, args.baseline, masks)
    n_full, n_sub = int(sizes.n_full.sum()), int(sizes.n_wbm_real.sum())
    log.info(
        "cohort: %d daily test pairs → %d WBM-real (overall fallback rate %.3f)",
        n_full,
        n_sub,
        1 - n_sub / n_full,
    )

    kw = dict(
        baseline=args.baseline,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        ci_level=args.ci_level,
        test_uids=test_uids,
    )
    full = cohort_skill(raw, "full", **kw)
    subset = cohort_skill(subset_predictions(raw, masks), "wbm_real_subset", **kw)
    skill = pd.concat([full, subset], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "wbm_cohort_bootstrap.meta.json").write_text(json.dumps({
        **split_meta, "cohorts": {"full": full.attrs["bootstrap"],
                                  "wbm_real_subset": subset.attrs["bootstrap"]},
    }, indent=2) + "\n")
    skill_path = args.output_dir / "wbm_cohort_ablation_skill.csv"
    sizes_path = args.output_dir / "wbm_cohort_sizes.csv"
    skill.to_csv(skill_path, index=False, float_format="%.6f")
    sizes.to_csv(sizes_path, index=False, float_format="%.6f")
    log.info("wrote %s", skill_path)
    log.info("wrote %s", sizes_path)

    overall = skill[skill.scope == "Overall"].pivot(
        index="method", columns="cohort", values="point"
    )
    log.info("Overall skill (point) — full vs WBM-real subset:")
    for m in args.methods:
        if m == args.baseline or m not in overall.index:
            continue
        f, s = overall.loc[m, "full"], overall.loc[m, "wbm_real_subset"]
        log.info("  %-9s full=%+.4f  wbm_real_subset=%+.4f  Δ=%+.4f", m, f, s, s - f)


if __name__ == "__main__":
    main()
