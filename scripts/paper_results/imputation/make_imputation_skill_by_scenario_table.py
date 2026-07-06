#!/usr/bin/env python3
"""Generate the arXiv imputation *skill-by-scenario* appendix table from HF.

Source of truth: the OpenMHC leaderboard HF dataset
(``MyHeartCounts/OpenMHC-leaderboard-data``). This table includes the dense
``LSM-2 (7-day)`` baseline, so it reduces the **dense-weekly** bootstrap
reference ``imputation/bootstrap_with_dense_weekly/draws.parquet`` (the main
16-method draws + the dense ``lsm2_weekly``) plus the matching per-method
substrate parquets — so all rows are ranked within one consistent 17-method pool.

Columns: Aggregate Skill Score $S$, Average Rank $R$, Fairness Skill Score
$S_{\\text{fair}}$ (the **disparity-ratio** score used by the main table — the
deprecated $S-\\lambda\\bar D$ score and its $\\bar D$ column are dropped), and
per-scenario Skill Scores for the six masking scenarios. Values are bootstrap
mean $\\pm$ SE ($B{=}1000$); $S_{\\text{fair}}$ is the deterministic point
estimate $\\pm$ bootstrap SE.

Reductions are the canonical ones (``aggregate_skill_rank_fairness`` for
skill/rank, ``compute_fairness_skill_scores`` for the disparity-ratio fairness),
identical to ``make_imputation_latex_tables.py`` but over the dense-weekly
superset and with ``bca=False`` (the table renders $\\pm$SE, so the BCa jackknife
is unnecessary).

Usage:
    python scripts/paper_results/imputation/make_imputation_skill_by_scenario_table.py \
        --out ~/MHC-benchmark/paper/sections_arxiv/appendix/imputation_skill_by_scenario_table.tex
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"
DRAWS_PATH = "imputation/bootstrap_with_dense_weekly/draws.parquet"
REFERENCE = "locf"

# key -> (latex_label, context, model_group). Labels mirror the existing appendix
# table (plain \cite). lsm2_weekly is the dense 7-day baseline.
METHODS: dict[str, tuple[str, str, str]] = {
    "linear": (r"Linear", "single", "stat"),
    "temporal_mean": (r"Temporal mean", "single", "stat"),
    "locf": (r"LOCF \textit{(reference)}", "single", "stat"),
    "temporal_mode": (r"Temporal mode", "single", "stat"),
    "mode": (r"Mode", "single", "stat"),
    "mean": (r"Mean", "single", "stat"),
    "lsm2": (r"LSM-2~\cite{xu2025lsm}", "single", "neural"),
    "dlinear": (r"DLinear~\cite{zeng2023dlinear}", "single", "neural"),
    "brits": (r"BRITS~\cite{cao2018brits}", "single", "neural"),
    "timesnet": (r"TimesNet~\cite{wu2023timesnet}", "single", "neural"),
    "fedformer": (r"FEDformer~\cite{zhou2022fedformer}", "single", "neural"),
    "personalized_temporal_mean": (r"Pers.\ temp.\ mean", "long", "stat"),
    "personalized_mean": (r"Pers.\ mean", "long", "stat"),
    "personalized_mode": (r"Pers.\ mode", "long", "stat"),
    "lsm2_weekly": (r"LSM-2 (7-day)", "long", "neural"),
    "lsm2_weekly_sparse": (r"LSM-2-Sparse (7-day)", "long", "neural"),
    "dlinear_weekly": (r"DLinear (7-day)~\cite{zeng2023dlinear}", "long", "neural"),
}

# (header, source, scope, center, scale100, lower_better, ref_zero)
#   source: "skill" | "rank" | "fair"; center is always the deterministic "point".
COLUMNS = [
    (r"$S\uparrow$", "skill", "overall", "point", True, False, True),
    (r"$R\downarrow$", "rank", "overall", "point", False, True, False),
    (r"$S_{\text{fair}}\uparrow$", "fair", "overall", "point", True, False, True),
    (r"Random noise\,$\uparrow$", "skill", "random_noise", "point", True, False, True),
    (r"Temporal slice\,$\uparrow$", "skill", "temporal_slice", "point", True, False, True),
    (r"Signal slice\,$\uparrow$", "skill", "signal_slice", "point", True, False, True),
    (r"Sleep gap\,$\uparrow$", "skill", "sleep_gap", "point", True, False, True),
    (r"Workout gap\,$\uparrow$", "skill", "workout_gap", "point", True, False, True),
    (r"Intensity failure\,$\uparrow$", "skill", "intensity_failure", "point", True, False, True),
]
NCOL = len(COLUMNS) + 1

SECTION_TITLE = {
    "single": r"\textbf{\emph{Single-day imputation}}",
    "long": r"\textbf{\emph{Long-context imputation ($\geq 7 \times 1440$ time steps)}}",
}
GROUP_TITLE = {
    "stat": r"\cellcolor[HTML]{EFEFEF}\textit{Statistical Models}",
    "neural": r"\cellcolor[HTML]{EFEFEF}\textit{Neural Models}",
}

HEADER_TMPL = r"""\begin{table}[t!]
    \renewcommand{\arraystretch}{1.05}
    \centering
    \captionsetup{width=\textwidth}
    \caption{\textbf{Imputation Results by Masking Scenario.} Aggregate Skill Score $S$ (in \%; $0=$LOCF reference), Average Rank $R$, Fairness Skill Score $S_{\text{fair}}$ (disparity-ratio; see Appendix~\ref{app:fairness_skillscore}), and per-scenario Skill Scores across all six masking scenarios (lower is better for $R$; higher otherwise). Single-day methods above; long-context methods ($\geq 7\times 1440$ time steps) below. Gradients computed within each track. Values are point estimates on the held-out test split; sub/superscripts give the $95\%$ bootstrap confidence interval ($B{=}1000$): the percentile interval for every column except $S_{\text{fair}}$, which uses the bias-corrected and accelerated (BCa) interval.}
    \label{tab:imputation_appendix_skill_by_scenario}
    \small
    \setlength{\tabcolsep}{1.5pt}
    \resizebox{\linewidth}{!}{%
    \begin{tabular}{l ccccccccc}
    \toprule[1.5pt]
    \textbf{Method} & $S\uparrow$ & $R\downarrow$ & $S_{\text{fair}}\uparrow$ & Random noise\,$\uparrow$ & Temporal slice\,$\uparrow$ & Signal slice\,$\uparrow$ & Sleep gap\,$\uparrow$ & Workout gap\,$\uparrow$ & Intensity failure\,$\uparrow$ \\
    \midrule
"""
FOOTER = r"""    \bottomrule[1.5pt]
    \end{tabular}%
    }
\end{table}
"""


def _require_finite_float(value, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing {label}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {label}: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return out


def reduce_from_hf(repo_id: str, revision: str | None) -> dict[str, dict[tuple[str, str], tuple[float, float]]]:
    """Return {method: {(source, scope): (center, se)}} from the dense-weekly HF substrate."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from imputation_evaluation.evaluation.bootstrap_skill_rank import (
        aggregate_skill_rank_fairness,
        compute_point_skill_rank,
        read_draws_parquet,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aggregate_fairness_skill_score import (  # noqa: E402
        SENSITIVE_ATTRS,
        compute_fairness_skill_scores,
    )
    from make_imputation_latex_tables import _attach_point  # noqa: E402

    draws_path = hf_hub_download(
        repo_id=repo_id, filename=DRAWS_PATH, repo_type="dataset", revision=revision
    )
    draws_df, _ = read_draws_parquet(Path(draws_path))
    all_rows = draws_df[draws_df["subgroup_attr"] == "all"]
    tables = aggregate_skill_rank_fairness(all_rows)

    per_method = [
        pd.read_parquet(
            hf_hub_download(
                repo_id=repo_id, filename=f"imputation/{m}.parquet",
                repo_type="dataset", revision=revision,
            )
        )
        for m in METHODS
    ]
    per_user_df = pd.concat(per_method, ignore_index=True)

    # Deterministic point (the reported center) for skill / rank, from the same
    # leaderboard reducers on the unresampled all/all cohort; percentile CI comes
    # from the bootstrap summary.
    # Average rank is cross-method: compute the point over the same method pool as
    # the bootstrap draws (restrict the substrate to those methods).
    draw_methods = set(tables["avg_rankings"]["method"].astype(str)) | set(
        tables["skill_scores"]["method"].astype(str)
    )
    pu_all = per_user_df[
        (per_user_df["subgroup_attr"] == "all")
        & (per_user_df["method"].astype(str).isin(draw_methods))
    ].rename(columns={"E_per_user": "E"})
    point = compute_point_skill_rank(pu_all, baseline_method=REFERENCE)
    skill_tbl = _attach_point(tables["skill_scores"], point["skill_scores"])
    rank_tbl = _attach_point(tables["avg_rankings"], point["avg_rankings"])

    # Fairness: deterministic point + BCa interval (matches the main table). The
    # BCa jackknife needs the per-user substrate.
    fairness = compute_fairness_skill_scores(
        draws_df, attrs=list(SENSITIVE_ATTRS), baseline_method=REFERENCE,
        bca=True, per_user_df=per_user_df,
    )
    fairness = fairness[(fairness["scope"] == "overall") & (fairness["split"] == "test")]

    out: dict[str, dict[tuple[str, str], tuple[float, float, float]]] = {m: {} for m in METHODS}

    def _ingest(df, source, center_col, lo_col, hi_col):
        for _, r in df[df["split"] == "test"].iterrows():
            m = r["method"]
            if m not in out:
                continue
            c = r.get(center_col)
            if c is None or not math.isfinite(float(c)):
                continue  # e.g. LOCF has no skill point (rendered as $0.0$)
            scope = str(r["scope"])
            lo = _require_finite_float(r.get(lo_col), f"{source} lo for method={m!r}, scope={scope!r}")
            hi = _require_finite_float(r.get(hi_col), f"{source} hi for method={m!r}, scope={scope!r}")
            out[m][(source, scope)] = (float(c), lo, hi)

    _ingest(skill_tbl, "skill", "point", "ci_lo", "ci_hi")
    _ingest(rank_tbl, "rank", "point", "ci_lo", "ci_hi")
    _ingest(fairness, "fair", "point", "bca_lo", "bca_hi")
    return out


def intensity(value: float, vmin: float, vmax: float, lower_better: bool) -> int:
    if vmax == vmin:
        return 0
    frac = (vmax - value) / (vmax - vmin) if lower_better else (value - vmin) / (vmax - vmin)
    return round(frac * 100)


def fmt_cell(method, center, lo, hi, scale100, ref_zero, n, is_best) -> str:
    """One LaTeX cell: optional color + ``$value^{+upper}_{-lower}$`` (percentile/BCa CI)."""
    if ref_zero and method == REFERENCE:
        return r"$0.0$"
    center = _require_finite_float(center, f"center for method={method!r}")
    s = 100.0 if scale100 else 1.0
    num = f"{center * s:+.1f}" if scale100 else f"{center * s:.1f}"
    up = f"{(hi - center) * s:.1f}"
    down = f"{(center - lo) * s:.1f}"
    body = rf"\mathbf{{{num}}}" if is_best else num
    color = rf"\cellcolor{{customblue!{n}}}" if n > 0 else ""
    return rf"{color}${body}^{{+{up}}}_{{-{down}}}$"


def build_body(data) -> str:
    lines: list[str] = []
    for ctx in ("single", "long"):
        if ctx == "long":
            lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{{NCOL}}}{{l}}{{{SECTION_TITLE[ctx]}}} \\")
        lines.append(r"    \hline")
        section = [m for m, (_, c, _) in METHODS.items() if c == ctx]
        bounds = []
        for _h, src, scope, _ctr, _s, _lb, _rz in COLUMNS:
            vals = [data[m][(src, scope)][0] for m in section if (src, scope) in data[m]]
            bounds.append((min(vals), max(vals)) if vals else (0.0, 0.0))
        for gi, grp in enumerate(("stat", "neural")):
            if gi > 0:
                lines.append(r"    \hline")
            lines.append(rf"    \multicolumn{{{NCOL}}}{{l}}{{{GROUP_TITLE[grp]}}} \\")
            members = [m for m in section if METHODS[m][2] == grp]
            members.sort(key=lambda m: -data[m].get(("skill", "overall"), (0.0, 0.0, 0.0))[0])
            for m in members:
                cells = []
                for ci, (_h, src, scope, _ctr, scale100, lower, ref_zero) in enumerate(COLUMNS):
                    if ref_zero and m == REFERENCE:
                        cells.append(fmt_cell(m, 0.0, 0.0, 0.0, scale100, ref_zero, 0, False))
                        continue
                    key = (src, scope)
                    if key not in data[m]:
                        raise ValueError(
                            f"Missing table cell for method={m!r}, source={src!r}, scope={scope!r}"
                        )
                    center, lo, hi = data[m][key]
                    vmin, vmax = bounds[ci]
                    n = intensity(center, vmin, vmax, lower)
                    best_val = vmin if lower else vmax
                    is_best = (center == best_val) and (m != REFERENCE)
                    cells.append(fmt_cell(m, center, lo, hi, scale100, ref_zero, n, is_best))
                lines.append(rf"    {METHODS[m][0]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--out", type=Path,
        default=Path.home()
        / "MHC-benchmark/paper/sections_arxiv/appendix/imputation_skill_by_scenario_table.tex",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    data = reduce_from_hf(args.repo_id, args.revision)
    missing = [m for m in METHODS if not data[m]]
    if missing:
        raise SystemExit(f"Methods missing from reduction: {missing}")

    table = HEADER_TMPL + build_body(data) + FOOTER
    if args.dry_run:
        print(table)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(f"Wrote {args.out} ({len(table)} bytes)")


if __name__ == "__main__":
    main()
