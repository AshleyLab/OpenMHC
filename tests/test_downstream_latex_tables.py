"""Synthetic aggregate tables, completeness and invalid-input safeguards."""

import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "downstream_tables", ROOT / "scripts/paper_results/downstream/make_downstream_latex_tables.py"
)
table = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(table)


@pytest.fixture
def inputs(tmp_path):
    """Generate small synthetic aggregates without using any saved paper results."""
    destination = tmp_path / "input"
    destination.mkdir()
    ranks = dict(linear=3, multirocket=5, xgboost=1, gru_d=6, lsm2=8, wbm=2, toto=4, chronos2=7)
    for kind, name in table.FILES.items():
        key, center = ("task", "value") if kind == "task" else ("scope", "point")
        fields = ["method", key, center, "ci_lo", "ci_hi"]
        scopes = table.BENCHMARK_TASKS if kind == "task" else ["Overall", *table.DOMAINS]
        if kind == "task":
            fields.extend(["task_type", "metric"])
        with (destination / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, method in enumerate(table.METHODS):
                for j, scope in enumerate(scopes):
                    if kind == "rank":
                        point, lo, hi = ranks[method], ranks[method] - 0.25, ranks[method] + 0.5
                    elif kind == "skill":
                        point = (i + 1) * 0.02
                        lo, hi = point - 0.02, point + 0.03
                        if method == "linear":
                            point = lo = hi = 0.0
                    else:
                        point = 0.2 + i * 0.01 + j * 0.001
                        lo, hi = point - 0.02, point + 0.03
                    row = dict(method=method, **{key: scope, center: point}, ci_lo=lo, ci_hi=hi)
                    if kind == "task":
                        task_type = ("binary", "ordinal", "regression")[j % 3]
                        row.update(
                            task_type=task_type, metric=table.DEFAULT_PRIMARY_METRICS[task_type]
                        )
                    writer.writerow(row)
    hashes = {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in table.FILES.values()
    }
    (destination / "manifest.json").write_text(json.dumps({"files_sha256": hashes}))
    return destination


def test_tables_have_all_rows_and_use_supplied_values(inputs):
    """Guard completeness, rounding and sorting using deliberately synthetic ranks."""
    results = table.load_results(inputs)
    main, tasks = table.render_main(results), table.render_per_task(results)
    main_rows = [line for line in main.splitlines() if line.startswith(r"\textsc{")]
    task_rows = [line for line in tasks.splitlines() if r"\est{" in line or "$0.0$" in line]
    assert len(main_rows) == 8
    assert len(task_rows) == 44  # 32 tasks + 10 domain summaries + 2 overall summaries.
    assert all(len(re.split(r"(?<!\\)&", row)) == 8 for row in main_rows)
    assert all(len(re.split(r"(?<!\\)&", row)) == 10 for row in task_rows)
    assert 8 * 7 + 44 * 8 == 408
    for name in table.TASK_DISPLAY_NAMES.values():
        assert sum(row.startswith(name + " & ") for row in task_rows) == 1
    assert [re.search(r"\\textsc\{([^}]+)\}", row)[1] for row in main_rows] == [
        "XGBoost",
        "Linear",
        "MultiRocket",
        "GRU-D",
        "Toto",
        "Chronos-2",
        "WBM",
        "LSM-2",
    ]
    wbm = next(row for row in main_rows if row.startswith(r"\textsc{WBM}"))
    assert r"\est{+12.0}{3.0}{2.0}" in wbm
    assert r"\est{2.00}{0.50}{0.25}" in wbm
    assert r"S_{\mathrm{fair}}" not in main
    assert r"\yuze" not in main + tasks


def test_manifest_rejects_changed_csv(inputs):
    """A modified reference file must not silently acquire verified status."""
    path = inputs / table.FILES["skill"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        table.load_results(inputs)


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "nan", "interval", "metric", "baseline"]
)
def test_invalid_new_run_rejected_before_writing(inputs, monkeypatch, problem):
    """Validate fresh-run CSVs too, even when they have no reference manifest."""
    (inputs / "manifest.json").unlink()
    kind = "task" if problem == "metric" else "skill"
    path = inputs / table.FILES[kind]
    with path.open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "nan":
        rows[0]["point"] = "nan"
    elif problem == "interval":
        rows[0]["ci_lo"], rows[0]["ci_hi"] = "1", "0"
    elif problem == "metric":
        rows[0]["metric"] = "rmse"
    else:
        next(row for row in rows if row["method"] == "linear")["point"] = "0.1"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(sys, "argv", ["tables", "--results-dir", str(inputs)])
    with pytest.raises(SystemExit) as exc:
        table.main()
    assert exc.value.code == 2
    assert not (inputs / "tables").exists()


def test_cli_dry_run_and_default_outputs(inputs, monkeypatch, capsys):
    """Generate both table fragments without touching any input CSV."""
    before = {path.name: path.read_bytes() for path in inputs.iterdir()}
    monkeypatch.setattr(sys, "argv", ["tables", "--results-dir", str(inputs), "--dry-run"])
    table.main()
    text = capsys.readouterr().out
    assert text.count(r"\begin{table}") == 2
    assert not (inputs / "tables").exists()
    monkeypatch.setattr(sys, "argv", ["tables", "--results-dir", str(inputs)])
    table.main()
    assert {path.name for path in (inputs / "tables").iterdir()} == {
        "downstream_main_results_table.tex",
        "downstream_per_task_results_table.tex",
    }
    for name, contents in before.items():
        assert (inputs / name).read_bytes() == contents


@pytest.mark.parametrize("target", ["skill_scores_bootstrap.csv", "manifest.json"])
def test_cli_cannot_overwrite_inputs(inputs, monkeypatch, target):
    """Protect the aggregate source even when an explicit output path is supplied."""
    path = inputs / target
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["tables", "--results-dir", str(inputs), "--out", str(path)])
    with pytest.raises(SystemExit) as exc:
        table.main()
    assert exc.value.code == 2
    assert path.read_bytes() == before
    assert not (inputs / "tables").exists()


def test_results_directory_is_required(monkeypatch):
    """Never select a potentially stale default results directory."""
    monkeypatch.setattr(sys, "argv", ["tables"])
    with pytest.raises(SystemExit) as exc:
        table.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("custom_main", [False, True])
def test_cli_can_regenerate_tables_in_results_directory(inputs, monkeypatch, custom_main):
    """Both root-level outputs and a custom main output must support repeat runs."""
    before = {path.name: path.read_bytes() for path in inputs.iterdir()}
    args = ["tables", "--results-dir", str(inputs)]
    main = inputs / "custom.tex" if custom_main else inputs / "downstream_main_results_table.tex"
    if custom_main:
        args += ["--out", str(main)]
        task = inputs / "tables/downstream_per_task_results_table.tex"
    else:
        args += ["--output-dir", str(inputs)]
        task = inputs / "downstream_per_task_results_table.tex"
    monkeypatch.setattr(sys, "argv", args)
    table.main()
    expected = [main.read_bytes(), task.read_bytes()]
    table.main()
    assert [main.read_bytes(), task.read_bytes()] == expected
    for name, contents in before.items():
        assert (inputs / name).read_bytes() == contents


@pytest.mark.parametrize("target", ["metadata", "manifest_input", "symlink", "same_output"])
def test_cli_rejects_output_collisions(inputs, monkeypatch, target):
    """Reject metadata, aliased inputs and overlapping outputs before writing."""
    if target == "metadata":
        path = inputs / "bootstrap_draws.meta.json"
        path.write_text('{"seed": 42}')
    elif target == "manifest_input":
        path = inputs / "provenance.txt"
        path.write_text("Synthetic run provenance")
        manifest_path = inputs / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    elif target == "symlink":
        path = inputs / "alias.tex"
        path.symlink_to(inputs / table.FILES["skill"])
    else:
        path = inputs / "tables/downstream_per_task_results_table.tex"
    before = {p: p.read_bytes() for p in inputs.iterdir() if p.is_file()}
    monkeypatch.setattr(sys, "argv", ["tables", "--results-dir", str(inputs), "--out", str(path)])
    with pytest.raises(SystemExit) as exc:
        table.main()
    assert exc.value.code == 2
    assert not (inputs / "tables").exists()
    for p, contents in before.items():
        assert p.read_bytes() == contents
