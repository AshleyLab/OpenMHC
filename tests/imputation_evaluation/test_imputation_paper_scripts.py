"""The Track 2 paper scripts and frozen baseline carry only skill, rank and "all" rows."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.hf_config import N_CHANNELS
from imputation_evaluation.evaluation.bootstrap_skill_rank import write_draws_parquet
from imputation_evaluation.evaluation.per_user_errors import build_per_user_errors

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/paper_results"


def _load(relpath):
    """Import a scripts/paper_results script as a module."""
    path = SCRIPTS / f"{relpath}.py"
    spec = importlib.util.spec_from_file_location(Path(relpath).name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pairs_tree(root, offset):
    """One method's saved pairs: two users, one scenario, one continuous channel."""
    (root / "scenarioA/test").mkdir(parents=True)
    np.save(root / "channel_stds.npy", np.ones(N_CHANNELS))
    pq.write_table(
        pa.table(
            {
                "sample_idx": pa.array([0, 1], type=pa.int32()),
                "user_id": pa.array(["u0", "u1"], type=pa.utf8()),
                "date": pa.array(["2024-01-01", "2024-01-02"], type=pa.utf8()),
            }
        ),
        root / "manifest_test.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "sample_idx": pa.array([0, 1], type=pa.int32()),
                "gt": pa.array([1.0, 2.0], type=pa.float32()),
                "pred": pa.array([1.0 + offset, 2.5], type=pa.float32()),
            }
        ),
        root / "scenarioA/test/pairs_ch05.parquet",
    )


@pytest.fixture
def method_dirs(tmp_path):
    """Saved pairs for two methods plus their manifest JSON."""
    dirs = {"locf": tmp_path / "pairs/locf", "mean": tmp_path / "pairs/mean"}
    _pairs_tree(dirs["locf"], 1.0)
    _pairs_tree(dirs["mean"], 2.0)
    manifest = tmp_path / "method_dirs.json"
    manifest.write_text(json.dumps({m: str(p) for m, p in dirs.items()}))
    return dirs, manifest


def test_producer_emits_only_all_rows(method_dirs):
    """The canonical producer has no age or sex rows."""
    df, display = build_per_user_errors(
        method_dirs[0]["locf"], "locf", scenarios=["scenarioA"], splits=["test"]
    )
    assert set(df["subgroup_attr"]) == set(df["subgroup_value"]) == {"all"}
    assert len(df) == 2
    assert set(display) == {("scenarioA", "test")}


def test_bootstrap_cli_writes_only_all_rows(method_dirs, tmp_path, monkeypatch):
    """Draws, per-user errors and metadata carry no subgroup output."""
    output = tmp_path / "out/draws.parquet"
    argv = ["draws", "--method-dirs", str(method_dirs[1]), "--output", str(output)]
    argv += ["--n-boot", "2", "--no-auc"]
    monkeypatch.setattr(sys, "argv", argv)
    assert _load("imputation/bootstrap_imputation_draws").main() == 0
    for name in ("draws.parquet", "per_user_errors.parquet"):
        frame = pd.read_parquet(output.parent / name)
        assert set(frame["subgroup_attr"]) == set(frame["subgroup_value"]) == {"all"}
    meta = json.loads((output.parent / "draws.parquet.meta.json").read_text())
    assert not {"include_fairness", "age_bins", "exclude_unknown"} & set(meta)


def test_hydra_config_with_retired_sensitivity_fails():
    """The CLI drops unknown keys, so this one has to be rejected by name."""
    from omegaconf import OmegaConf

    from imputation_evaluation.hydra.cli import main

    with pytest.raises(ValueError, match="sensitivity"):
        main(OmegaConf.create({"sensitivity": {"enabled": True}}))


@pytest.mark.parametrize("flag", ["--no-fairness", "--exclude-unknown"])
def test_bootstrap_cli_rejects_retired_flags(flag, tmp_path, monkeypatch):
    """Retired flags fail instead of being ignored."""
    argv = ["draws", "--method-dirs", "x", "--output", str(tmp_path / "d"), flag]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        _load("imputation/bootstrap_imputation_draws")._parse_args()


def test_strict_bootstrap_requires_the_manifest(method_dirs, tmp_path, monkeypatch):
    """--strict still refuses a missing manifest on the from-pairs path."""
    (method_dirs[0]["locf"] / "manifest_test.parquet").unlink()
    argv = ["draws", "--method-dirs", str(method_dirs[1]), "--strict"]
    argv += ["--output", str(tmp_path / "d.parquet"), "--scenarios", "scenarioA"]
    monkeypatch.setattr(sys, "argv", argv)
    assert _load("imputation/bootstrap_imputation_draws").main() == 2


def test_point_cli_writes_only_skill_and_rank(method_dirs, tmp_path, monkeypatch):
    """The point flow writes exactly the two tables."""
    out = tmp_path / "point"
    argv = ["point", "--method-dirs", str(method_dirs[1]), "--output-dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert _load("compute_imputation_paper_metrics").main() == 0
    assert sorted(path.name for path in out.iterdir()) == ["avg_rankings.csv", "skill_scores.csv"]


def test_aggregate_cli_writes_only_skill_and_rank(tmp_path, monkeypatch):
    """Phase 2 writes exactly the two tables."""
    row = dict(
        scenario="scenarioA",
        split="test",
        channel_type="continuous",
        subgroup_attr="all",
        subgroup_value="all",
        draw=0,
        E=1.0,
        R=1.0,
        rank=1.0,
    )
    draws = tmp_path / "draws.parquet"
    rows = [
        {**row, "method": method, "channel": channel}
        for method in ("locf", "mean")
        for channel in ("ch_5", "ch_6")
    ]
    write_draws_parquet(pd.DataFrame(rows), draws)
    out = tmp_path / "agg"
    monkeypatch.setattr(sys, "argv", ["agg", "--draws", str(draws), "--output-dir", str(out)])
    assert _load("imputation/aggregate_imputation_paper_metrics").main() == 0
    assert sorted(path.name for path in out.iterdir()) == [
        "avg_rankings_bootstrap.csv",
        "skill_scores_bootstrap.csv",
    ]


def test_main_table_has_one_header_and_cell_per_column():
    """The main table renders from skill and rank only, with consistent columns."""
    table = _load("imputation/make_imputation_latex_tables")
    # (center, lo, hi) for every method in every column.
    cols = [
        {m: (float(i), i - 0.5, i + 0.5) for i, m in enumerate(table.METHODS)}
        for _ in table.COLUMNS
    ]
    rendered = table.HEADER + table.build_body(cols)
    assert "fair" not in rendered.lower()
    assert "bca" not in rendered.lower()
    assert re.search(r"\\begin\{tabular\}\{l (c+)\}", rendered).group(1) == "c" * len(table.COLUMNS)
    rows = [line for line in rendered.splitlines() if "$" in line and r"\\" in line and "&" in line]
    assert {line.count("&") for line in rows} == {len(table.COLUMNS)}
    assert "fairness_skill_score_bootstrap.csv" not in table.REDUCED_CSVS


def test_scenario_table_has_one_header_and_cell_per_column():
    """The by-scenario table renders from skill and rank only, with consistent columns."""
    table = _load("imputation/make_imputation_skill_by_scenario_table")
    data = {
        m: {(c[1], c[2]): (float(i), i - 0.5, i + 0.5) for c in table.COLUMNS}
        for i, m in enumerate(table.METHODS)
    }
    rendered = table.HEADER_TMPL + table.build_body(data)
    assert "fair" not in rendered.lower()
    assert "bca" not in rendered.lower()
    assert re.search(r"\\begin\{tabular\}\{l (c+)\}", rendered).group(1) == "c" * len(table.COLUMNS)
    rows = [line for line in rendered.splitlines() if "$" in line and r"\\" in line and "&" in line]
    assert {line.count("&") for line in rows} == {len(table.COLUMNS)}


def test_frozen_baseline_holds_only_all_rows():
    """The shipped LOCF baseline has no age or sex rows."""
    path = REPO / "src/openmhc/data/baselines/imputation_locf_per_user_errors.parquet"
    frame = pd.read_parquet(path)
    assert set(frame["subgroup_attr"].astype(str)) == {"all"}
    assert set(frame["subgroup_value"].astype(str)) == {"all"}
    assert set(frame["method"].astype(str)) == {"locf"}
    assert frame["E_per_user"].dtype == np.float32
