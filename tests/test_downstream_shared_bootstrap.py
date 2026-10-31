"""Track-1 resampling contracts and public CLI integration."""

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from downstream_evaluation.evaluation import bootstrap_skill_rank as engine

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = {"binary": "D1", "ordinal": "D2", "regression": "D2"}


def predictions():
    """Overlapping, unequal task cohorts, ties, and a test user with no eligible task."""
    specs = {
        "binary": (["u0", "u1", "u2", "u3"], [0, 1, 0, 1], [0.2, 0.7, 0.7, 0.6]),
        "ordinal": (["u1", "u2", "u3", "u4"], [0, 1, 1, 2], [1, 0, 1, 2]),
        "regression": (["u0", "u2", "u4"], [0.0, 1.0, 3.0], [1.0, 0.5, 2.0]),
    }
    result = {m: {} for m in ["linear", "model"]}
    for task, (uids, y, p) in specs.items():
        for method in result:
            pp = np.array(p, dtype=float)
            if method == "model":
                pp = pp[::-1].copy()
            result[method][task] = dict(
                uids=np.array(uids),
                y_true=np.array(y, dtype=float),
                y_pred=pp,
                y_proba=pp,
                task_type=task,
            )
    return result, ["u0", "u1", "u2", "u3", "u4", "unused"]


def calculate(raw, population, n=20, **kwargs):
    return engine.compute_per_draw_errors(
        raw, n_bootstrap=n, seed=42, test_uids=population, domain_map=DOMAIN, **kwargs
    )


def canonical(frame):
    frame = frame.copy()
    frame.attrs.clear()
    return frame.sort_values(engine.DRAW_COLS[:-1]).reset_index(drop=True)


def test_shared_participant_counts_and_full_sampling_population(monkeypatch):
    raw, population = predictions()
    observed = []
    original = engine._per_task_metric_on_indices

    def record(aligned, methods, tasks, indices, mask_per_task=None):
        if mask_per_task is None:
            observed.append(
                {t: Counter(aligned[methods[0]][t]["uids"][indices[t]]) for t in tasks}
            )
        return original(aligned, methods, tasks, indices, mask_per_task)

    monkeypatch.setattr(engine, "_per_task_metric_on_indices", record)
    draws = calculate(raw, population)
    rng = np.random.default_rng(42)
    saw_varying_size = False
    for actual in observed[1:]:
        sampled = Counter(
            rng.choice(sorted(population), size=len(population), replace=True)
        )
        for task, payload in raw["linear"].items():
            expected = Counter({u: sampled[u] for u in payload["uids"] if sampled[u]})
            assert actual[task] == expected
            saw_varying_size |= sum(actual[task].values()) != len(payload["uids"])
        assert (
            actual["binary"]["u2"]
            == actual["ordinal"]["u2"]
            == actual["regression"]["u2"]
        )
    assert saw_varying_size
    assert draws.attrs["bootstrap"]["sampling_population_n"] == 6
    assert draws.attrs["bootstrap"]["eligible_union_n"] == 5


def test_metrics_and_subgroups_match_independent_expanded_samples():
    raw, population = predictions()
    groups = {u: {"sex": "A" if u in ["u0", "u1", "u4"] else "B"} for u in population}
    draws = calculate(raw, population, subgroup_map=groups, subgroup_attributes=["sex"])
    values = draws.set_index(
        ["method", "task", "subgroup_attr", "subgroup_value", "draw"]
    ).E
    rng = np.random.default_rng(42)
    resamples = [None] + [
        rng.choice(sorted(population), size=len(population), replace=True)
        for _ in range(20)
    ]
    expected_keys = set()
    for draw, sampled in enumerate(resamples, start=-1):
        for method, tasks in raw.items():
            for task, payload in tasks.items():
                lookup = {u: i for i, u in enumerate(payload["uids"])}
                for attr, value in [("all", "all"), ("sex", "A"), ("sex", "B")]:
                    source = payload["uids"] if sampled is None else sampled
                    indices = [
                        lookup[u]
                        for u in source
                        if u in lookup and (attr == "all" or groups[u][attr] == value)
                    ]
                    yy, pp = payload["y_true"][indices], payload["y_pred"][indices]
                    key = (method, task, attr, value, draw)
                    if len(np.unique(yy)) < 2 or (
                        task != "binary" and len(np.unique(pp)) < 2
                    ):
                        assert key not in values.index
                        continue
                    metric = (
                        average_precision_score(yy, pp)
                        if task == "binary"
                        else (
                            spearmanr(yy, pp).statistic
                            if task == "ordinal"
                            else pearsonr(yy, pp).statistic
                        )
                    )
                    assert values.loc[key] == pytest.approx(1 - metric, abs=1e-14)
                    expected_keys.add(key)
    assert set(values.index) == expected_keys


def test_order_independence_and_subset_population():
    raw, population = predictions()
    original = calculate(raw, population)
    reordered = {
        m: dict(reversed(list(tasks.items())))
        for m, tasks in reversed(list(raw.items()))
    }
    changed = calculate(reordered, population[::-1])
    pd.testing.assert_frame_equal(
        canonical(original), canonical(changed), check_exact=True
    )
    subset = {m: {"binary": tasks["binary"]} for m, tasks in raw.items()}
    restricted = calculate(subset, population)
    assert (
        restricted.attrs["bootstrap"]["counts_sha256"]
        == original.attrs["bootstrap"]["counts_sha256"]
    )
    pd.testing.assert_frame_equal(
        canonical(original[original.task == "binary"]),
        canonical(restricted),
        check_exact=True,
    )


def test_points_are_original_cohort_metrics():
    raw, population = predictions()
    draws = calculate(raw, population)
    for row in draws[draws.draw == -1].itertuples():
        p = raw[row.method][row.task]
        metric = engine.PRIMARY_METRIC_FN[row.task](
            p["y_true"], p["y_pred"], p["y_proba"]
        )
        assert row.E == 1 - metric


def test_empty_and_single_class_samples_are_omitted_without_imputation():
    raw, population = predictions()
    for m in raw:
        raw[m] = {
            "binary": {
                key: (value[:2] if isinstance(value, np.ndarray) else value)
                for key, value in raw[m]["binary"].items()
            }
        }
    draws = calculate(raw, population, n=100)
    assert set(draws[draws.draw == -1].method) == set(raw)
    coverage = draws.groupby("draw").method.nunique()
    assert coverage.eq(2).all() and len(coverage) < 101
    omissions = draws.attrs["bootstrap"]["global_omissions"]
    assert any(row["sample_size"] == 0 for row in omissions)
    assert all(set(row["methods"]) == set(raw) for row in omissions)
    assert len(omissions) + len(coverage) == 101


@pytest.mark.parametrize("population", [[], ["u0", "u0"], ["u0"], "split.json"])
def test_invalid_populations_fail(population):
    raw, _ = predictions()
    with pytest.raises(ValueError):
        calculate(raw, population)


def test_missing_population_is_not_inferred():
    raw, _ = predictions()
    with pytest.raises(TypeError, match="test_uids"):
        engine.compute_per_draw_errors(raw, n_bootstrap=2, seed=42)


@pytest.mark.parametrize("change", ["targets", "order", "duplicates", "nonfinite"])
def test_invalid_prediction_alignment_fails(change):
    raw, population = predictions()
    if change == "targets":
        raw["model"]["binary"]["y_true"][0] = 1
    if change == "order":
        raw["model"]["binary"]["uids"] = raw["model"]["binary"]["uids"][::-1]
    if change == "duplicates":
        raw["linear"]["binary"]["uids"][1] = "u0"
    if change == "nonfinite":
        raw["model"]["binary"]["y_proba"][0] = np.nan
    with pytest.raises(ValueError):
        calculate(raw, population)


def test_alignment_does_not_hide_duplicates_or_target_disagreement():
    raw, _ = predictions()
    raw["model"]["binary"]["uids"][1] = "u0"
    with pytest.raises(ValueError, match="Duplicate"):
        engine.align_across_methods(raw)
    raw, _ = predictions()
    raw["model"]["binary"]["y_true"][0] = 1
    with pytest.raises(ValueError, match="y_true mismatch"):
        engine.align_across_methods(raw)


def test_metadata_persisted_and_conflicts_rejected(tmp_path):
    raw, population = predictions()
    draws = calculate(raw, population)
    destination = tmp_path / "draws.parquet"
    engine.write_draws_parquet(draws, destination, {"baseline": "linear"})
    saved, meta = engine.read_draws_parquet(destination)
    assert saved.E.dtype == np.float32
    assert (
        meta["sampling_population_n"] == 6
        and meta["protocol"] == "shared_participant_v1"
    )
    assert meta["counts_sha256"] == draws.attrs["bootstrap"]["counts_sha256"]
    with pytest.raises(ValueError, match="Metadata disagrees"):
        engine.write_draws_parquet(draws, tmp_path / "bad.parquet", {"seed": 123})
    assert not (tmp_path / "bad.parquet").exists()


@pytest.mark.parametrize("change", ["duplicate", "overlap", "missing", "empty"])
def test_invalid_split_files_fail(tmp_path, change):
    split = {"train": ["train"], "validation": ["val"], "test": ["u0", "u1"]}
    if change == "duplicate":
        split["test"].append("u0")
    if change == "overlap":
        split["test"].append("train")
    if change == "missing":
        del split["validation"]
    if change == "empty":
        split["test"] = []
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError):
        engine.load_bootstrap_test_users(path)


@pytest.mark.parametrize("version", ["full", "xs"])
def test_dataset_split_resolution(version, tmp_path, monkeypatch):
    from openmhc._dataset import write_dataset_marker

    n = {"full": 11894, "xs": 593}[version]
    root = tmp_path / version
    (root / "splits").mkdir(parents=True)
    filename = (
        "sharable_users_seed42_2026" + ("_xs" if version == "xs" else "") + ".json"
    )
    split = {
        "train": [f"t{i}" for i in range(n - 3)],
        "validation": ["v"],
        "test": ["u1", "u0"],
    }
    path = root / "splits" / filename
    path.write_text(json.dumps(split))
    write_dataset_marker(root, version=version, n_users=n)
    monkeypatch.setenv("MHC_DATA_DIR", str(root))
    users, meta = engine.load_bootstrap_test_users(version=version)
    assert users.tolist() == ["u0", "u1"]
    assert meta["split_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        engine.load_bootstrap_test_users(
            data_dir=root, version="xs" if version == "full" else "full"
        )


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_pipeline_forwards_split_and_bootstrap_options():
    pipeline = module(
        "downstream_pipeline_test",
        "scripts/paper_results/downstream/run_paper_pipeline.py",
    )
    calls = []
    pipeline._run = lambda cmd, dry_run: calls.append(cmd)
    cfg = dict(
        predictions_dir="predictions",
        csvs_dir="csvs",
        n_bootstrap=1000,
        seed=42,
        data_dir="dataset-xs",
        version="xs",
        split_file="custom.json",
        baseline_method="reference",
        fairness_attributes=[],
    )
    pipeline._phase1_bootstrap(cfg, Path("draws.parquet"), ["reference", "model"], True)
    cmd = calls[0]
    for key, value in [
        ("--data-dir", "dataset-xs"),
        ("--version", "xs"),
        ("--split-file", "custom.json"),
        ("--baseline", "reference"),
    ]:
        assert cmd[cmd.index(key) + 1] == value
    assert cmd[cmd.index("--fairness_attributes") + 1] == "--output"


def test_cli_round_trip_uses_explicit_split_and_disables_subgroups(
    tmp_path, monkeypatch
):
    raw, population = predictions()
    for method, tasks in raw.items():
        csv = []
        for task, payload in tasks.items():
            directory = tmp_path / "predictions" / method / task
            directory.mkdir(parents=True)
            pd.DataFrame(
                {k: payload[k] for k in ["y_true", "y_pred", "y_proba"]}
                | {"uid": payload["uids"]}
            ).to_parquet(directory / "test.parquet")
            csv.append({"task": task, "task_type": task, "features": method})
        pd.DataFrame(csv).to_csv(tmp_path / f"eval_{method}.csv", index=False)
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": ["train"], "validation": ["val"], "test": population})
    )
    (tmp_path / "predictions/_subgroups.json").write_text(
        json.dumps({u: {"sex": "A"} for u in population})
    )
    cli = module(
        "downstream_cli_test",
        "scripts/paper_results/downstream/bootstrap_downstream_draws.py",
    )
    monkeypatch.setattr(cli, "TASK_DOMAIN_MAP", DOMAIN)
    destination = tmp_path / "output/draws.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bootstrap",
            "--predictions_dir",
            str(tmp_path / "predictions"),
            "--csvs_dir",
            str(tmp_path),
            "--methods",
            "linear",
            "model",
            "--split-file",
            str(split),
            "--n_bootstrap",
            "4",
            "--seed",
            "42",
            "--fairness_attributes",
            "--output",
            str(destination),
        ],
    )
    cli.main()
    saved, meta = engine.read_draws_parquet(destination)
    assert meta["sampling_population_n"] == 6 and meta["fairness_attributes"] == []
    assert meta["split_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert set(saved.subgroup_attr) == {"all"}
    expected = calculate(raw, population, n=4)
    expected.E = expected.E.astype("float32")
    pd.testing.assert_frame_equal(
        canonical(saved), canonical(expected), check_exact=True
    )


def test_cohort_helpers_preserve_full_population_after_restriction(monkeypatch):
    dev = ROOT / "scripts/paper_results/downstream/dev"
    monkeypatch.syspath_prepend(str(dev))
    wbm = module("wbm_cohort_test", str(dev / "wbm_cohort_ablation.py"))
    pretrain = module("pretrain_cohort_test", str(dev / "cohort_pretrain_table.py"))
    raw, population = predictions()
    names = {"binary": "PH", "ordinal": "sleep_time_categories", "regression": "age"}
    raw = {m: {names[t]: p for t, p in tasks.items()} for m, tasks in raw.items()}
    subset = wbm.subset_predictions(
        raw, {t: {"u0", "u1", "u2", "u3"} for t in names.values()}
    )
    kwargs = dict(
        baseline="linear",
        n_bootstrap=5,
        seed=42,
        clip_lower=0.01,
        clip_upper=100.0,
        ci_level=0.95,
        test_uids=population,
    )
    full = wbm.cohort_skill(raw, "full", **kwargs)
    restricted = wbm.cohort_skill(subset, "subset", **kwargs)
    skill, rank = pretrain.cohort_tables(subset, "subset", **kwargs)
    assert not full.empty and not restricted.empty and not rank.empty
    for table in (restricted, skill, rank):
        assert table.attrs["bootstrap"]["sampling_population_n"] == 6
        assert table.attrs["bootstrap"]["eligible_union_n"] == 4
        assert (
            table.attrs["bootstrap"]["counts_sha256"]
            == full.attrs["bootstrap"]["counts_sha256"]
        )
    pd.testing.assert_frame_equal(restricted, skill)


@pytest.fixture
def anchor_case(tmp_path, monkeypatch):
    """Use a baseline and a nonzero method to expose incomplete comparisons."""
    dev = ROOT / "scripts/paper_results/downstream/dev"
    monkeypatch.syspath_prepend(str(dev))
    helper = module("pretrain_anchor_test", str(dev / "cohort_pretrain_table.py"))
    skill = pd.DataFrame(
        [("full", "linear", "Overall", 0.0), ("full", "wbm", "Overall", 0.1234561)],
        columns=["cohort", "method", "scope", "point"],
    )
    return helper, skill, tmp_path / "anchor.csv"


def test_anchor_accepts_complete_reference_superset_at_csv_precision(anchor_case):
    """Dropped reference methods are allowed, with comparison at CSV precision."""
    helper, skill, path = anchor_case
    ref = pd.concat([skill, skill.iloc[[1]].assign(method="lsm2")], ignore_index=True)
    ref.to_csv(path, index=False, float_format="%.6f")
    helper.check_anchor(skill, path)


@pytest.mark.parametrize(
    "problem, message",
    [
        ("partial", "missing"),
        ("duplicate_current", "duplicate"),
        ("duplicate_reference", "duplicate"),
        ("nan", "finite"),
        ("infinity", "finite"),
        ("nonnumeric", "finite"),
        ("null_key", "null"),
        ("missing_column", "columns"),
        ("empty", "empty"),
        ("changed", "skill moved"),
    ],
)
def test_anchor_rejects_invalid_or_incomplete_comparisons(anchor_case, problem, message):
    """Partial matches, ambiguous keys and invalid values must never pass the anchor."""
    helper, skill, path = anchor_case
    ref = skill.copy()
    if problem == "partial":
        ref = ref.iloc[:1]
    elif problem == "duplicate_current":
        skill = pd.concat([skill, skill.iloc[:1]], ignore_index=True)
    elif problem == "duplicate_reference":
        ref = pd.concat([ref, ref.iloc[:1]], ignore_index=True)
    elif problem in {"nan", "infinity", "nonnumeric"}:
        invalid = {"nan": float("nan"), "infinity": float("inf"), "nonnumeric": "bad"}
        skill["point"] = [0.0, invalid[problem]]
    elif problem == "null_key":
        ref.loc[1, "scope"] = None
    elif problem == "missing_column":
        ref = ref.drop(columns="point")
    elif problem == "empty":
        skill = skill.iloc[:0]
    else:
        skill.loc[1, "point"] = 0.99
    ref.to_csv(path, index=False)
    with pytest.raises(SystemExit, match=message):
        helper.check_anchor(skill, path)
