#!/usr/bin/env python
r"""Paper-style table (R, S, 5 domains) for the LSM-2 pretraining-cohort ablation.

``wbm_cohort_ablation.py`` answers "is WBM's weakness a *test*-cohort artifact?" by
re-scoring frozen predictions on the participants WBM can actually score. This script
answers the adjacent question — "is it a *pretraining*-cohort artifact?" — by putting
two LSM-2 arms pretrained on WBM's own cohort into the same field and rendering the
paper's headline table for both cohorts.

The field is Linear (baseline) + WBM + the two cohort-matched arms. The published
LSM-2 run is dropped from the **field**, not merely hidden from the table: average
rank is field-relative, so leaving it in the bootstrap while omitting it from the
output would silently contaminate every R. Skill is field-invariant and therefore
must not move — that is the validation anchor below.

Cohorts and bootstrap knobs are inherited from ``wbm_cohort_ablation`` so the two
scripts' numbers are directly comparable; this one additionally keeps ``avg_rankings``
(which ``cohort_skill()`` discards) and writes it out raw.

Validation anchor: re-running the 4-method field must reproduce the 5-method skill
CSV to the last decimal (max \|Δ\| = 0). A non-zero difference means the field change
leaked into skill, and the run aborts.

Outputs (under ``--output-dir``):
  * ``cohort_pretrain_rank.csv`` — average rank per method × scope, both cohorts.
  * ``cohort_pretrain_skill.csv`` — skill per method × scope, both cohorts.
  * ``cohort_pretrain_table.md`` — the rendered R/S/domain table.

Usage::

    MHC_DATA_DIR=/path/to/full/dataset PYTHONPATH=src python \
        scripts/paper_results/downstream/dev/cohort_pretrain_table.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
from wbm_cohort_ablation import subset_predictions, wbm_real_masks

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
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

LABELS = {
    "cohortdaily": "LSM-2 cohort-matched (minute)",
    "cohorthourly": "LSM-2 cohort-matched (hourly)",
    "lsm2": "LSM-2 published (full pretraining data)",
    "wbm": "WBM",
    "linear": "Linear (baseline)",
}
DOMAINS = [
    ("Demographics", "Demo."),
    ("Body metrics and biomarkers", "Body"),
    ("Medical conditions", "Medical"),
    ("Mental well-being", "Mental"),
    ("Sleep and lifestyle", "Sleep"),
]
COHORT_TITLES = {
    "wbm_real_subset": "WBM-real subset ({n:,} evals, the cohort WBM can score)",
    "full": "Full daily cohort ({n:,} evals)",
}


def n_evals(method_tasks: dict[str, dict[str, dict]], reference_method: str) -> int:
    """Total (participant, task) pairs scored in this cohort."""
    return sum(len(p["uids"]) for p in method_tasks[reference_method].values())


def cohort_tables(
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired bootstrap for one cohort; return ``(skill, rank)`` tagged with ``cohort``."""
    draws = compute_per_draw_errors(
        align_across_methods(method_tasks),
        n_bootstrap=n_bootstrap,
        seed=seed,
        subgroup_map=None,
        subgroup_attributes=None,
        test_uids=test_uids,
    )
    tables = aggregate_skill_rank_fairness(
        draws, baseline=baseline, clip_lower=clip_lower, clip_upper=clip_upper, ci_level=ci_level
    )
    out = []
    for key in ("skill_scores", "avg_rankings"):
        t = tables[key].copy()
        t.insert(0, "cohort", label)
        t.attrs["bootstrap"] = draws.attrs["bootstrap"]
        out.append(t)
    return out[0], out[1]


def check_anchor(skill: pd.DataFrame, anchor_csv: Path) -> None:
    """Assert skill is unchanged by the field edit, at the CSV's ``%.6f`` precision."""
    if not anchor_csv.exists():
        raise SystemExit(
            f"anchor CSV not found: {anchor_csv}\n"
            "Run wbm_cohort_ablation.py against the same input tree first — the "
            "field-invariance check has nothing to compare against without it."
        )
    ref = pd.read_csv(anchor_csv)
    keys = ["cohort", "method", "scope"]
    for name, frame in (("current skill", skill), ("anchor", ref)):
        missing = set(keys + ["point"]) - set(frame.columns)
        if missing:
            raise SystemExit(f"{name}: missing columns {sorted(missing)}")
        if frame.empty:
            raise SystemExit(f"{name}: empty skill table")
        if frame[keys].isna().any().any():
            raise SystemExit(f"{name}: null (cohort, method, scope) key")
        if frame.duplicated(keys).any():
            raise SystemExit(f"{name}: duplicate (cohort, method, scope) keys")
        if not pd.to_numeric(frame["point"], errors="coerce").map(math.isfinite).all():
            raise SystemExit(f"{name}: point estimates must be finite numbers")
    # The reference may contain dropped methods, but must cover every current row.
    merged = skill[keys + ["point"]].merge(
        ref[keys + ["point"]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_new", "_ref"),
    )
    missing_rows = merged.loc[merged["_merge"] != "both", keys]
    if not missing_rows.empty:
        raise SystemExit(f"anchor missing current skill rows: {missing_rows.to_dict('records')}")
    merged["point_new"] = pd.to_numeric(merged["point_new"])
    merged["point_ref"] = pd.to_numeric(merged["point_ref"])
    diff = (merged.point_new.round(6) - merged.point_ref.round(6)).abs()
    log.info(
        "ANCHOR skill vs %s: %d rows, max |Δ| = %.3e", anchor_csv.name, len(merged), diff.max()
    )
    if diff.max() != 0:
        worst = merged.loc[diff.idxmax()]
        raise SystemExit(
            f"ANCHOR FAILED: skill moved when the field changed (max |Δ| = {diff.max():.3e} at "
            f"{worst.cohort}/{worst.method}/{worst.scope}). Skill must be field-invariant; "
            "a non-zero difference means the dropped method leaked into the skill computation."
        )
    log.info("ANCHOR PASSED (skill is field-invariant)")


def render_table(skill: pd.DataFrame, rank: pd.DataFrame, order: list[str], sizes: dict) -> str:
    """Render the R / S / per-domain markdown table for every cohort."""
    lines = []
    for cohort, template in COHORT_TITLES.items():
        lines += [
            f"\n### {template.format(n=sizes[cohort])}\n",
            "| Method | R ↓ | S ↑ | " + " | ".join(f"{s} ↑" for _, s in DOMAINS) + " |",
            "|---" * (3 + len(DOMAINS)) + "|",
        ]
        for method in order:
            r = rank[(rank.cohort == cohort) & (rank.method == method) & (rank.scope == "Overall")]
            cells = []
            for scope, _ in [("Overall", "S")] + DOMAINS:
                s = skill[
                    (skill.cohort == cohort) & (skill.method == method) & (skill.scope == scope)
                ]
                cells.append(f"{float(s['point'].iloc[0]) * 100:+.1f}" if len(s) else "—")
            label = LABELS.get(method, method)
            lines.append(f"| {label} | {float(r['point'].iloc[0]):.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Score both cohorts on the 4-method field, verify the anchor, write the tables."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    tree = Path("results/eval/wbm_cohort_pretrain_input")
    p.add_argument("--predictions_dir", type=Path, default=tree / "predictions")
    p.add_argument("--csvs_dir", type=Path, default=tree)
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/paper/wbm_cohort_pretrain_ablation")
    )
    p.add_argument(
        "--anchor-csv",
        type=Path,
        default=None,
        help="5-method skill CSV to check field-invariance against. "
        "None → <output-dir>/wbm_cohort_ablation_skill.csv.",
    )
    p.add_argument(
        "--data_dir",
        default=None,
        help="Dataset root for the weekly lookup. None → MHC_DATA_DIR (must be the 'full' release).",
    )
    p.add_argument("--methods", nargs="+", default=["linear", "wbm", "cohortdaily", "cohorthourly"])
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
    cohorts = {"full": raw, "wbm_real_subset": subset_predictions(raw, masks)}
    sizes = {k: n_evals(v, args.baseline) for k, v in cohorts.items()}
    log.info("cohort sizes (evals): %s", sizes)

    kw = dict(
        baseline=args.baseline,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        ci_level=args.ci_level,
        test_uids=test_uids,
    )
    pairs = [cohort_tables(mt, label, **kw) for label, mt in cohorts.items()]
    skill = pd.concat([s for s, _ in pairs], ignore_index=True)
    rank = pd.concat([r for _, r in pairs], ignore_index=True)

    check_anchor(skill, args.anchor_csv or args.output_dir / "wbm_cohort_ablation_skill.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cohort_pretrain_bootstrap.meta.json").write_text(json.dumps({
        **split_meta, "cohorts": {label: pair[0].attrs["bootstrap"]
                                  for label, pair in zip(cohorts, pairs)},
    }, indent=2) + "\n")
    for name, df in (("cohort_pretrain_rank.csv", rank), ("cohort_pretrain_skill.csv", skill)):
        df.to_csv(args.output_dir / name, index=False, float_format="%.6f")
        log.info("wrote %s", args.output_dir / name)

    # Non-baseline methods ranked on the WBM-real subset, baseline pinned last; the same
    # order is reused for both tables so the two are read side by side.
    primary = rank[(rank.cohort == "wbm_real_subset") & (rank.scope == "Overall")]
    ranked = primary[primary.method != args.baseline].sort_values("point").method.tolist()
    order = ranked + [args.baseline]

    md = render_table(skill, rank, order, sizes)
    (args.output_dir / "cohort_pretrain_table.md").write_text(md + "\n")
    log.info("wrote %s", args.output_dir / "cohort_pretrain_table.md")
    print(md)

    print("\n=== Overall rank, with CIs ===")
    overall = rank[rank.scope == "Overall"][
        ["cohort", "method", "point", "mean", "se", "ci_lo", "ci_hi"]
    ]
    print(overall.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
