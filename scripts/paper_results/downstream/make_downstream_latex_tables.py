#!/usr/bin/env python3
"""Generate current Track 1 main and per-task tables from aggregate CSVs.

Select one results directory explicitly. Reference manifest hashes, when present,
are checked before reading values. Both tables are validated and rendered before
writing; no metrics are recomputed. Outputs default to <results-dir>/tables/.

The LaTeX fragments require booktabs, graphicx, and xcolor with its table option.
They define their own color and estimate macro and need no manuscript commands.

Usage::

    PYTHONPATH=src python scripts/paper_results/downstream/make_downstream_latex_tables.py \
        --results-dir results/paper
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from downstream_evaluation.evaluation.skill_score import DEFAULT_PRIMARY_METRICS, TASK_DOMAIN_MAP
from openmhc._constants import BENCHMARK_TASKS, TASK_DISPLAY_NAMES

REFERENCE = "linear"
METHODS = {
    "linear": r"\textsc{Linear} (reference)",
    "multirocket": r"\textsc{MultiRocket}",
    "xgboost": r"\textsc{XGBoost}",
    "gru_d": r"\textsc{GRU-D}",
    "lsm2": r"\textsc{LSM-2}",
    "wbm": r"\textsc{WBM}",
    "toto": r"\textsc{Toto} (FT)",
    "chronos2": r"\textsc{Chronos-2} (FT)",
}
GROUPS = {
    "Statistical Models": ("xgboost", "multirocket", "linear"),
    "Supervised Neural Models": ("gru_d",),
    "Time-Series Foundation Models": ("chronos2", "toto"),
    "Wearable Foundation Models": ("lsm2", "wbm"),
}
DOMAINS = {
    "Demographics": "Demographics",
    "Medical conditions": r"Medical Conditions \& Risk",
    "Body metrics and biomarkers": r"Vitals \& Blood Biomarkers",
    "Mental well-being": "Mental Well-Being",
    "Sleep and lifestyle": r"Sleep \& Lifestyle",
}
FILES = {
    "skill": "skill_scores_bootstrap.csv",
    "rank": "avg_rankings_bootstrap.csv",
    "task": "task_metrics_bootstrap.csv",
}
METRIC_LABELS = {"auprc": "AUPRC", "spearman_r": r"Spearman $\rho$", "pearson_r": "Pearson $r$"}
MAIN_COLUMNS = [("rank", "Overall"), ("skill", "Overall")] + [
    ("skill", domain) for domain in DOMAINS
]


def load_results(results_dir: Path) -> dict:
    """Read a complete eight-method result set, checking any reference manifest."""
    manifest_path = results_dir / "manifest.json"
    hashes = None
    if manifest_path.exists():
        hashes = json.loads(manifest_path.read_text())["files_sha256"]
    results = {}
    for kind, name in FILES.items():
        contents = (results_dir / name).read_bytes()
        if hashes is not None and hashlib.sha256(contents).hexdigest() != hashes.get(name):
            raise ValueError(f"Reference hash mismatch: {name}; use CSVs from one verified run")
        key_col = "task" if kind == "task" else "scope"
        center_col = "value" if kind == "task" else "point"
        reader = csv.DictReader(io.StringIO(contents.decode()))
        required = {"method", key_col, center_col, "ci_lo", "ci_hi"}
        if kind == "task":
            required.update(("task_type", "metric"))
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        rows = {}
        for row in reader:
            key = (row["method"], row[key_col])
            if key in rows:
                raise ValueError(f"{name}: duplicate row {key}")
            values = tuple(float(row[col]) for col in (center_col, "ci_lo", "ci_hi"))
            if not all(math.isfinite(value) for value in values) or values[1] > values[2]:
                raise ValueError(f"{name}: invalid estimate or interval for {key}")
            if kind == "skill" and key[0] == REFERENCE and values != (0.0, 0.0, 0.0):
                raise ValueError(f"{name}: Linear reference skill must be zero for {key}")
            if kind == "task" and DEFAULT_PRIMARY_METRICS.get(row["task_type"]) != row["metric"]:
                raise ValueError(f"{name}: inconsistent task type and metric for {key}")
            rows[key] = {"values": values, "metric": row.get("metric")}
        scopes = BENCHMARK_TASKS if kind == "task" else ["Overall", *DOMAINS]
        expected = {(method, scope) for method in METHODS for scope in scopes}
        if set(rows) != expected:
            raise ValueError(
                f"{name}: incomplete or unexpected rows; "
                f"missing={sorted(expected - rows.keys())}, extra={sorted(rows.keys() - expected)}"
            )
        results[kind] = rows
    for task in BENCHMARK_TASKS:
        metrics = {results["task"][method, task]["metric"] for method in METHODS}
        if len(metrics) != 1:
            raise ValueError(f"Inconsistent metrics across methods for {task}")
    return results


def intensity(value: float, vmin: float, vmax: float, lower_better: bool) -> int:
    """Return a per-column color intensity between zero and 100."""
    if vmax == vmin:
        return 0
    frac = (vmax - value) / (vmax - vmin) if lower_better else (value - vmin) / (vmax - vmin)
    return round(frac * 100)


def fmt_cell(method, center, lo, hi, scale100, ref_zero, n, is_best) -> str:
    """Format rank/skill with the rounding used by the verified paper exports."""
    if ref_zero and method == REFERENCE:
        return r"$0.0$"
    if scale100:
        s, num = 100.0, f"{center * 100:+.1f}"
    else:
        s, num = 1.0, f"{center:.2f}"
    up = f"{(hi - center) * s:.1f}" if scale100 else f"{(hi - center):.2f}"
    down = f"{(center - lo) * s:.1f}" if scale100 else f"{(center - lo):.2f}"
    body = rf"\mathbf{{{num}}}" if is_best else num
    color = rf"\cellcolor{{customblue!{n}}}" if n > 0 else ""
    return rf"{color}\est{{{body}}}{{{up}}}{{{down}}}"


def render_cells(results: dict, kind: str, scope: str) -> dict[str, str]:
    """Format one task or summary across all methods using a common color scale."""
    rows = results[kind]
    scale = 100 if kind == "skill" else 1
    centers = [rows[method, scope]["values"][0] * scale for method in METHODS]
    best = min(centers) if kind == "rank" else max(centers)
    cells = {}
    for method in METHODS:
        point, lo, hi = rows[method, scope]["values"]
        n = intensity(point * scale, min(centers), max(centers), kind == "rank")
        is_best = point * scale == best
        if kind == "task":
            number = f"{point:.3f}"
            if is_best:
                number = rf"\mathbf{{{number}}}"
            color = rf"\cellcolor{{customblue!{n}}}" if n else ""
            cells[method] = rf"{color}\est{{{number}}}{{{hi - point:.3f}}}{{{point - lo:.3f}}}"
        else:
            cells[method] = fmt_cell(
                method,
                point,
                lo,
                hi,
                kind == "skill",
                kind == "skill",
                n,
                is_best and method != REFERENCE,
            )
    return cells


def _table(caption: str, label: str, headers: list[str], rows: list[str], text_columns: int) -> str:
    """Wrap rows in a portable table fragment with local formatting definitions."""
    columns = "l" * text_columns + "c" * (len(headers) - text_columns)
    header = " & ".join(headers) + r" \\"
    body = "\n".join(rows)
    return rf"""% Requires booktabs, graphicx, and \usepackage[table]{{xcolor}}.
\begin{{table}}[htbp]
\centering
\definecolor{{customblue}}{{RGB}}{{205,227,245}}
\providecommand{{\est}}[3]{{\ensuremath{{#1^{{\scriptscriptstyle +#2}}_{{\scriptscriptstyle -#3}}}}}}
\caption{{{caption}}}
\label{{{label}}}
\small
\renewcommand{{\arraystretch}}{{1.15}}
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{{columns}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""


def render_main(results: dict) -> str:
    """Render seven summary columns, ordered by rank within each method class."""
    columns = [render_cells(results, kind, scope) for kind, scope in MAIN_COLUMNS]
    rows = []
    for title, methods in GROUPS.items():
        rows.append(r"\multicolumn{8}{l}{\textit{" + title + r"}} \\")
        ordered = sorted(methods, key=lambda m: results["rank"][m, "Overall"]["values"][0])
        for method in ordered:
            rows.append(
                " & ".join([METHODS[method], *(column[method] for column in columns)]) + r" \\"
            )
        rows.append(r"\midrule")
    rows.pop()
    return _table(
        "Prediction tasks: overall average rank $R$, overall skill $S$ and five domain "
        r"skills (\%; Linear reference). Values are held-out point estimates with "
        "percentile bootstrap intervals. Methods are ordered by rank within each class.",
        "tab:health_outcome_main_results",
        ["Method", r"$R\downarrow$", r"$S\uparrow$", *DOMAINS.values()],
        rows,
        1,
    )


def render_per_task(results: dict) -> str:
    """Render all task metrics, domain summaries and overall summaries."""
    rows = []

    def append_row(label, metric, kind, scope):
        cells = render_cells(results, kind, scope)
        rows.append(" & ".join([label, metric, *(cells[m] for m in METHODS)]) + r" \\")

    for domain, title in DOMAINS.items():
        rows.append(r"\multicolumn{10}{l}{\textit{" + title + r"}} \\")
        tasks = sorted(
            (task for task in BENCHMARK_TASKS if TASK_DOMAIN_MAP[task] == domain),
            key=lambda task: TASK_DISPLAY_NAMES[task],
        )
        for task in tasks:
            metric = results["task"][REFERENCE, task]["metric"]
            append_row(TASK_DISPLAY_NAMES[task], METRIC_LABELS[metric], "task", task)
        append_row("Domain Avg. Rank", r"$R\downarrow$", "rank", domain)
        append_row(r"Domain Skill (\%)", r"$S\uparrow$", "skill", domain)
        rows.append(r"\midrule")
    append_row("Macro Avg. Rank", r"$R\downarrow$", "rank", "Overall")
    append_row(r"Overall Skill (\%)", r"$S\uparrow$", "skill", "Overall")
    return _table(
        "Prediction per-task results: AUPRC (binary), Spearman correlation (ordinal), "
        "and Pearson correlation (regression), all higher is better. Values are point "
        "estimates with percentile bootstrap intervals. Overall rank and skill weight "
        "the five domains equally.",
        "tab:main_results",
        ["Task", "Metric", *METHODS.values()],
        rows,
        2,
    )


def main() -> None:
    """Validate one results directory and export both tables without editing a paper."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: <results-dir>/tables")
    parser.add_argument("--out", type=Path, help="Override the main table path only")
    parser.add_argument("--dry-run", action="store_true", help="Print both tables without writing")
    args = parser.parse_args()
    try:
        results = load_results(args.results_dir)
        main_table, per_task = render_main(results), render_per_task(results)
        input_names = {*FILES.values(), "manifest.json", "bootstrap_draws.meta.json"}
        manifest_path = args.results_dir / "manifest.json"
        if manifest_path.exists():
            input_names.update(json.loads(manifest_path.read_text())["files_sha256"])
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(main_table + "\n" + per_task)
        return
    output_dir = args.output_dir or args.results_dir / "tables"
    main_path = args.out or output_dir / "downstream_main_results_table.tex"
    task_path = output_dir / "downstream_per_task_results_table.tex"
    paths = [main_path.resolve(), task_path.resolve()]
    # Protect sources and provenance, while allowing generated tables to be replaced.
    protected = {(args.results_dir / name).resolve() for name in input_names}
    if len(set(paths)) != 2 or protected.intersection(paths):
        parser.error("Output paths must be distinct and must not overwrite input files")
    for path, table in ((main_path, main_table), (task_path, per_task)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
