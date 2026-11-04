"""The Track 3 paper scripts carry only skill and rank."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/paper_results/forecasting"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"forecasting_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draws_hold_only_skill_and_rank(tmp_path, monkeypatch):
    """The draws file and its metadata have no fairness rows or age bins."""
    script = _load("produce_forecasting_bootstrap_draws")
    (tmp_path / "skill_rank_models.json").write_text(json.dumps({"models": {"m": "/metrics"}}))
    row = {"model": "m", "scope": "overall", "draw": 0, "value": 0.5}
    monkeypatch.setattr(script, "build_per_user_metrics", lambda **_: pd.DataFrame())
    monkeypatch.setattr(
        script,
        "bootstrap_skill_rank",
        lambda **_: {
            "skill_draws": pd.DataFrame([row]),
            "rank_draws": pd.DataFrame([{**row, "metric": "overall"}]),
        },
    )
    monkeypatch.setattr(sys, "argv", ["draws", "--summary-dir", str(tmp_path)])
    assert script.main() == 0
    draws = pd.read_parquet(tmp_path / "bootstrap_draws.parquet")
    meta = json.loads((tmp_path / "bootstrap_draws.parquet.meta.json").read_text())
    assert set(draws["reduction"].astype(str)) == {"skill", "rank"}
    assert meta["reductions"] == ["skill", "rank"]
    assert "age_bins" not in meta


def test_pipeline_bootstrap_writes_only_skill_and_rank(tmp_path, monkeypatch):
    """Phase 3 needs no demographics and writes exactly the two bootstrap tables."""
    from forecasting_evaluation.metrics import bootstrap_skill_rank as module

    monkeypatch.setattr(
        module,
        "bootstrap_skill_rank",
        lambda **_: {"skill_scores": pd.DataFrame(), "avg_rankings": pd.DataFrame()},
    )
    cfg = {"output_root": str(tmp_path), "baseline": "seasonal_naive", "bootstrap": {}}
    _load("run_paper_pipeline")._phase_bootstrap(cfg, {}, False)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "forecasting_grouped_metric_rank_bootstrap.csv",
        "forecasting_skill_score_bootstrap.csv",
    ]


def test_latex_table_reads_only_skill_and_rank(tmp_path):
    """The table renders from the two CSVs, with one header and cell per column."""
    table = _load("make_forecasting_latex_tables")
    fields = ["model", "scope", "metric", "point", "ci_lo", "ci_hi"]
    for name in (table.SKILL, table.RANK):
        with (tmp_path / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, model in enumerate(table.MODELS):
                for column in table.COLUMNS:
                    writer.writerow(
                        dict(zip(fields, [model, column[2], column[9] or "", i, i - 0.5, i + 0.5]))
                    )
    body = table.build_body(table.load_columns(tmp_path))
    rendered = table.HEADER + body
    assert "fair" not in rendered.lower()
    assert f"*{{{len(table.COLUMNS)}}}" in table.HEADER
    assert table.HEADER.count(r"\mbox{") == len(table.COLUMNS)
    assert body.count("\n& ") == len(table.MODELS) * len(table.COLUMNS)
