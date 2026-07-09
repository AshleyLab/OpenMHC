#!/usr/bin/env python3
"""Generate the arXiv *main* downstream (prediction-task) table from the bootstrap CSVs.

Reads the downstream bootstrap CSVs (``skill_scores_bootstrap.csv``,
``avg_rankings_bootstrap.csv``, ``fairness_skill_score_bootstrap.csv``) and emits
the full ``\\begin{table*}...\\end{table*}`` block for the main Track-1 results
table (booktabs rules, per-column ``customblue!N`` blue gradient, bold best cell),
mirroring the forecasting and imputation tables.

Uncertainty is a 95% bootstrap confidence interval rendered as an asymmetric
``value^{+upper}_{-lower}`` super/subscript (not the SE). The center is the
deterministic ``point`` estimate for every column; the interval is the percentile
CI (``ci_lo``/``ci_hi``) except ``S_fair`` (from
``fairness_skill_score_bootstrap.csv``, scope ``overall``), which uses the BCa
interval (``bca_lo``/``bca_hi``).

Linear is the baseline: in every skill / fairness / domain column it renders as a
plain ``$0.0$`` (no CI, no color); it still gets a real value+CI in the rank
column. Rows are sorted by overall skill (descending).

The paper preamble must define ``customblue`` (xcolor/colortbl) and provide
``tabularx`` + ``booktabs`` — same dependencies as the forecasting/imputation
tables; ``\\est`` is provided inline.

Usage:
    python scripts/paper_results/downstream/make_downstream_latex_tables.py \
        --results-dir results/paper/20260703_track1fix \
        --out results/paper/20260703_track1fix/downstream_main_results_table.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REFERENCE = "linear"

# key -> latex label. Order here is irrelevant (rows are skill-sorted at render).
METHODS: dict[str, str] = {
    "lsm2": r"\textsc{LSM-2}",
    "xgboost": r"\textsc{XGBoost}",
    "multirocket": r"\textsc{MultiRocket}",
    "wbm": r"\textsc{WBM}",
    "gru_d": r"\textsc{GRU-D}",
    "linear": r"\textsc{Linear} (baseline)",
    "chronos2": r"\textsc{Chronos-2}",
    "toto": r"\textsc{Toto}",
}

SKILL = "skill_scores_bootstrap.csv"
RANK = "avg_rankings_bootstrap.csv"
FAIR = "fairness_skill_score_bootstrap.csv"

# (header, csv, scope, center, lo, hi, scale100, lower_better, ref_zero)
COLUMNS: list[tuple[str, str, str, str, str, str, bool, bool, bool]] = [
    (r"$R\,\downarrow$",            RANK,  "Overall",                     "point", "ci_lo",  "ci_hi",  False, True,  False),
    (r"$S\,\uparrow$",              SKILL, "Overall",                     "point", "ci_lo",  "ci_hi",  True,  False, True),
    (r"$S_{\mathrm{fair}}\,\uparrow$", FAIR, "overall",                  "point", "bca_lo", "bca_hi", True,  False, True),
    (r"Demo.~$\uparrow$",           SKILL, "Demographics",                "point", "ci_lo",  "ci_hi",  True,  False, True),
    (r"Body~$\uparrow$",            SKILL, "Body metrics and biomarkers", "point", "ci_lo",  "ci_hi",  True,  False, True),
    (r"Medical~$\uparrow$",         SKILL, "Medical conditions",          "point", "ci_lo",  "ci_hi",  True,  False, True),
    (r"Mental~$\uparrow$",          SKILL, "Mental well-being",           "point", "ci_lo",  "ci_hi",  True,  False, True),
    (r"Sleep~$\uparrow$",           SKILL, "Sleep and lifestyle",         "point", "ci_lo",  "ci_hi",  True,  False, True),
]

NCOL = len(COLUMNS) + 1  # + method column


def _header() -> str:
    """Build the table preamble + header row from COLUMNS (keeps them in sync)."""
    col_spec = "*{%d}{>{\\centering\\arraybackslash}m{1.15cm}}" % len(COLUMNS)
    head_cells = "\n".join(rf"& \mbox{{{c[0]}}}" for c in COLUMNS)
    return rf"""\begin{{table*}}[t]
\centering
\captionsetup{{width=0.98\textwidth}}
\caption{{
\textbf{{Prediction-task Results.}}
We report Average Rank $R$, Aggregate Skill Score $S$
(in \%; $0=\textsc{{Linear}}$ baseline), Fairness-adjusted Skill Score
$S_{{\mathrm{{fair}}}}$, and per-domain Skill Scores for \textit{{Demographics}},
\textit{{Body}} metrics, \textit{{Medical}} conditions, \textit{{Mental}}
well-being, and \textit{{Sleep}}. Values are point estimates on the held-out test
split; subscripts and superscripts indicate the $95\%$ bootstrap confidence
interval ($1000$ resamples): the percentile interval for every column except
$S_{{\mathrm{{fair}}}}$, which uses the bias-corrected and accelerated (BCa) interval.
}}
\label{{tab:downstream_grouped_model_summary}}

\providecommand{{\est}}[3]{{%
  \ensuremath{{#1^{{\scriptscriptstyle +#2}}_{{\scriptscriptstyle -#3}}}}%
}}

\small
\renewcommand{{\arraystretch}}{{1.16}}
\setlength{{\tabcolsep}}{{2.2pt}}

\begin{{tabularx}}{{\textwidth}}{{
    >{{\raggedright\arraybackslash}}X
    {col_spec}
}}
\toprule[1.4pt]

\textbf{{Method}}
{head_cells} \\

\midrule
"""


FOOTER = r"""\bottomrule[1.4pt]
\end{tabularx}
\end{table*}
"""


def load_metric(
    path: Path, scope: str, center_col: str, lo_col: str, hi_col: str
) -> dict[str, tuple[float, float, float]]:
    """Return {method: (center, lo, hi)} for one scope."""
    out: dict[str, tuple[float, float, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in (center_col, lo_col, hi_col) if c not in fields]
        if missing:
            raise SystemExit(f"{path.name}: columns {missing} not in header {fields!r}")
        for r in reader:
            if r["scope"] != scope:
                continue
            c = r[center_col]
            if c in ("", None):
                continue
            center = float(c)
            lo = float(r[lo_col]) if r[lo_col] not in ("", None) else center
            hi = float(r[hi_col]) if r[hi_col] not in ("", None) else center
            out[r["method"]] = (center, lo, hi)
    return out


def load_columns(results_dir: Path) -> list[dict[str, tuple[float, float, float]]]:
    """Load each column's {method: (center, lo, hi)} map, in COLUMNS order."""
    return [
        load_metric(results_dir / fname, scope, center, lo, hi)
        for _h, fname, scope, center, lo, hi, _s100, _lower, _ref in COLUMNS
    ]


def intensity(value: float, vmin: float, vmax: float, lower_better: bool) -> int:
    """Per-column min-max intensity in [0, 100] (global over all methods)."""
    if vmax == vmin:
        return 0
    frac = (vmax - value) / (vmax - vmin) if lower_better else (value - vmin) / (vmax - vmin)
    return round(frac * 100)


def fmt_cell(
    method: str, center: float, lo: float, hi: float,
    scale100: bool, ref_zero: bool, n: int, is_best: bool,
) -> str:
    """One LaTeX cell: optional color + ``\\est{value}{upper}{lower}``."""
    if ref_zero and method == REFERENCE:
        return r"$0.0$"  # baseline reference: plain, no CI, no color
    if scale100:
        s, num = 100.0, f"{center * 100:+.1f}"
    else:
        s, num = 1.0, f"{center:.2f}"  # rank: 2 decimals, no sign
    up = f"{(hi - center) * s:.1f}" if scale100 else f"{(hi - center):.2f}"
    down = f"{(center - lo) * s:.1f}" if scale100 else f"{(center - lo):.2f}"
    body = rf"\mathbf{{{num}}}" if is_best else num
    color = rf"\cellcolor{{customblue!{n}}}" if n > 0 else ""
    return rf"{color}\est{{{body}}}{{{up}}}{{{down}}}"


def build_body(cols: list[dict[str, tuple[float, float, float]]]) -> str:
    """Render one data row per method, sorted by overall skill (descending)."""
    lines: list[str] = []
    bounds = []
    for ci in range(len(COLUMNS)):
        vals = [cols[ci][m][0] for m in METHODS if m in cols[ci]]
        bounds.append((min(vals), max(vals)) if vals else (0.0, 0.0))

    skill_idx = next(i for i, c in enumerate(COLUMNS) if c[1] == SKILL and c[2] == "Overall")
    order = sorted(METHODS, key=lambda m: -cols[skill_idx][m][0])

    for m in order:
        row = [METHODS[m]]
        for ci, col in enumerate(COLUMNS):
            _h, _f, _sc, _ctr, _lo, _hi, scale100, lower, ref_zero = col
            center, lo, hi = cols[ci][m]
            vmin, vmax = bounds[ci]
            n = intensity(center, vmin, vmax, lower)
            best_val = vmin if lower else vmax
            is_best = (center == best_val) and (m != REFERENCE)
            row.append("& " + fmt_cell(m, center, lo, hi, scale100, ref_zero, n, is_best))
        row[-1] = row[-1] + r" \\"
        lines.append("\n".join(row))
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=Path("results/paper/20260703_track1fix"))
    p.add_argument("--out", type=Path, default=None, help="Default: <results-dir>/downstream_main_results_table.tex")
    p.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing.")
    args = p.parse_args()

    cols = load_columns(args.results_dir)
    missing = [m for m in METHODS if m not in cols[0]]
    if missing:
        raise SystemExit(f"Methods missing from CSVs: {missing}")

    table = _header() + build_body(cols) + FOOTER
    if args.dry_run:
        print(table)
        return
    out = args.out or (args.results_dir / "downstream_main_results_table.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    print(f"Wrote {out} ({len(table)} bytes)")


if __name__ == "__main__":
    main()
