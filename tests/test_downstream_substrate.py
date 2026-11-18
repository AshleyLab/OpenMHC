"""Track-1 leaderboard substrate — per-user prediction pairs (issue #39).

Builds the substrate from synthetic per-(method, task) prediction files and label
schemas, so it needs no real dataset. Verifies the schema, the constant "all" cell, and
parquet round-trip.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream_evaluation.evaluation.per_user_pairs import (
    PER_USER_PAIRS_PARQUET_COLUMNS,
    build_per_user_pairs,
    read_per_user_pairs_parquet,
    write_per_user_pairs_parquet,
)
from downstream_evaluation.evaluation.predictions_io import _safe_task

pytestmark = pytest.mark.usefixtures("label_schema_environment")


def _write_task(pred_dir, method, task, df):
    d = pred_dir / method / _safe_task(task)
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "test.parquet", index=False)


def test_build_per_user_pairs_schema(tmp_path):
    """Build one all-cohort row per participant and task."""
    method = "linear"
    _write_task(
        tmp_path,
        method,
        "Diabetes",
        pd.DataFrame(
            {"uid": ["u0", "u1"], "y_true": [0, 1], "y_pred": [0, 1], "y_proba": [0.2, 0.8]}
        ),
    )
    _write_task(
        tmp_path,
        method,
        "age",
        pd.DataFrame(
            {
                "uid": ["u0", "u1"],
                "y_true": [30.0, 55.0],
                "y_pred": [31.0, 50.0],
                "y_proba": [31.0, 50.0],
            }
        ),
    )

    df = build_per_user_pairs(tmp_path, method, ["Diabetes", "age"])

    assert list(df.columns) == PER_USER_PAIRS_PARQUET_COLUMNS
    assert set(df["method"]) == {"linear"}
    # 2 tasks × 2 users, every row labelled all/all.
    assert len(df) == 4
    assert set(df["subgroup_attr"]) == set(df["subgroup_value"]) == {"all"}
    # task_type is derived from the task name.
    assert set(df[df.task == "Diabetes"]["task_type"]) == {"binary"}
    assert set(df[df.task == "age"]["task_type"]) == {"regression"}


def test_missing_task_files_are_skipped(tmp_path):
    """Pool only tasks with saved predictions."""
    _write_task(
        tmp_path,
        "m",
        "Diabetes",
        pd.DataFrame({"uid": ["u0"], "y_true": [1], "y_pred": [1], "y_proba": [0.9]}),
    )

    # "age" has no file → silently skipped, only Diabetes rows emitted.
    df = build_per_user_pairs(tmp_path, "m", ["Diabetes", "age"])
    assert set(df["task"]) == {"Diabetes"}


def test_write_read_roundtrip_and_meta(tmp_path):
    """Preserve pairs and bind metadata to the serialized file."""
    _write_task(
        tmp_path,
        "m",
        "Diabetes",
        pd.DataFrame({"uid": ["u0"], "y_true": [1], "y_pred": [1], "y_proba": [0.9]}),
    )
    df = build_per_user_pairs(tmp_path, "m", ["Diabetes"])

    p = tmp_path / "m.parquet"
    write_per_user_pairs_parquet(df, p, meta={"method": "m", "overall_fallback_rate": 0.0})

    back, meta = read_per_user_pairs_parquet(p)
    assert list(back.columns) == PER_USER_PAIRS_PARQUET_COLUMNS
    assert meta["method"] == "m"
    assert meta["overall_fallback_rate"] == 0.0
    assert meta["substrate_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    write_per_user_pairs_parquet(df, p)
    assert not Path(f"{p}.meta.json").exists()
    assert str(back["y_proba"].dtype) == "float32"
    assert str(back["method"].dtype) == "category"


def test_fallback_sidecar_roundtrip_and_pool(tmp_path):
    """Sidecar write -> read -> pooled overall_fallback_rate; absent method -> None."""
    from downstream_evaluation.evaluation.predictions_io import (
        overall_fallback_rate,
        read_fallback_sidecar,
        write_fallback_sidecar,
    )

    counts = {"Diabetes": {"n_fallback": 3, "n_test": 10}, "age": {"n_fallback": 0, "n_test": 5}}
    path = write_fallback_sidecar(tmp_path, "wbm", counts)
    assert path == tmp_path / "wbm" / "fallback.json"
    assert read_fallback_sidecar(tmp_path, "wbm") == counts
    # pooled rate = (3 + 0) / (10 + 5); empty cohort -> unknown; absent method -> None.
    assert overall_fallback_rate(read_fallback_sidecar(tmp_path, "wbm")) == 3 / 15
    assert overall_fallback_rate({}) is None
    assert read_fallback_sidecar(tmp_path, "linear") is None


@pytest.fixture
def producer():
    """Load the public offline exporter."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / (
        "scripts/paper_results/downstream/leaderboard/produce_per_method_per_user_pairs.py"
    )
    spec = importlib.util.spec_from_file_location("produce_pairs", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_fallback_rate_requires_measured_counts(tmp_path, producer):
    """Pool counts, including zero, without interpreting missing data as zero."""
    from downstream_evaluation.evaluation.predictions_io import write_fallback_sidecar

    write_fallback_sidecar(tmp_path, "wbm", {
        "t": {"n_fallback": 2, "n_test": 8}, "u": {"n_fallback": 1, "n_test": 2},
    })
    assert producer.resolve_fallback_rate(tmp_path, "wbm", {"t": 8, "u": 2}) == 3 / 10
    with pytest.raises(ValueError, match="missing measured"):
        producer.resolve_fallback_rate(tmp_path, "linear", {"t": 8})
    write_fallback_sidecar(tmp_path, "linear", {"t": {"n_fallback": 0, "n_test": 8}})
    assert producer.resolve_fallback_rate(tmp_path, "linear", {"t": 8}) == 0.0


@pytest.mark.parametrize("counts,sizes,message", [
    ({}, {"t": 8}, "missing measured"),
    ({"t": {"n_fallback": 1, "n_test": 8}}, {"u": 8}, "task set"),
    ({"t": {"n_fallback": 1, "n_test": 8}}, {"t": 9}, "cohort size"),
    *[({"t": value}, {"t": 8}, "invalid fallback") for value in (
        None, {}, {"n_fallback": -1, "n_test": 8}, {"n_fallback": 9, "n_test": 8},
        {"n_fallback": True, "n_test": 8}, {"n_fallback": 1.5, "n_test": 8},
        {"n_fallback": 0, "n_test": 0}, {"n_fallback": 1, "n_test": "8"},
    )],
])
def test_producer_rejects_inconsistent_counts(tmp_path, producer, counts, sizes, message):
    """Reject missing, malformed and mismatched measurement cohorts."""
    folder = tmp_path / "wbm"
    folder.mkdir()
    (folder / "fallback.json").write_text(json.dumps(counts))
    with pytest.raises(ValueError, match=message):
        producer.resolve_fallback_rate(tmp_path, "wbm", sizes)


def test_producer_runs_from_saved_predictions_alone(tmp_path, monkeypatch):
    """The substrate producer needs no group map and emits only "all" rows."""
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    leaderboard = repo / "scripts/paper_results/downstream/leaderboard"
    script = leaderboard / "produce_per_method_per_user_pairs.py"
    spec = importlib.util.spec_from_file_location("produce_pairs", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    predictions, out = tmp_path / "predictions", tmp_path / "out"
    _write_task(
        predictions,
        "wbm",
        "Diabetes",
        pd.DataFrame({"uid": ["u0"], "y_true": [1], "y_pred": [1], "y_proba": [0.9]}),
    )
    (predictions / "wbm/fallback.json").write_text(json.dumps({
        "Diabetes": {"n_fallback": 1, "n_test": 1},
    }))
    argv = ["produce", "--predictions-dir", str(predictions), "--out-dir", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--methods", "wbm"])
    assert mod.main() == 0
    df, meta = read_per_user_pairs_parquet(out / "wbm.parquet")
    assert meta["overall_fallback_rate"] == 1.0
    assert set(df["subgroup_attr"]) == set(df["subgroup_value"]) == {"all"}


@pytest.mark.parametrize("missing_prediction", [False, True])
def test_evaluate_prediction_output_needs_no_enrollment_file(
    label_schema_environment, tmp_path, monkeypatch, missing_prediction
):
    """A non-Linear method on non-age tasks writes its substrate from the lookup alone."""
    import openmhc
    from downstream_evaluation.data.provider import lookup_filename
    from labels import api
    from openmhc._dataset import write_dataset_marker

    root = label_schema_environment.parent
    assert not (label_schema_environment / "enrollment_info.json").exists()
    users = {"train": ["t0", "t1"], "test": ["u0", "u1", "u2", "u3"]}
    split = {
        "train": users["train"] + [f"pad{i}" for i in range(586)],
        "validation": ["v"],
        "test": users["test"],
    }
    (root / "splits").mkdir()
    (root / "splits/sharable_users_seed42_2026_xs.json").write_text(json.dumps(split))
    write_dataset_marker(root, version="xs", n_users=593)
    (root / "processed").mkdir()
    pd.DataFrame(
        {
            "user_id": users["train"] + users["test"],
            "date": pd.Timestamp("2025-06-01"),
            "Diabetes": [0, 1, 0, 1, 0, 1],
        }
    ).to_parquet(root / "processed" / lookup_filename("daily", full_history=True))

    class Constant:
        name = "constant"
        needs_segments = False

        def set_context(self, context):
            self.n = len(context.user_ids)

        def predict(self, data):
            predictions = np.linspace(0.1, 0.9, self.n)
            if missing_prediction:
                predictions[0] = np.nan
            return predictions

    from downstream_evaluation.evaluation.evaluator import (
        DownstreamEvaluator,
        _combine_with_fallback,
    )

    def synthetic_baseline(self, task, ttype, predictions, train_td, test_td):
        return _combine_with_fallback(predictions, np.full(len(predictions), 0.6), ttype)

    monkeypatch.setattr(DownstreamEvaluator, "_apply_fallback", synthetic_baseline)
    out = tmp_path / "out"
    results = openmhc.evaluate_prediction(
        Constant(), version="xs", tasks=["Diabetes"], data_dir=root, output_dir=out
    )

    assert sorted(results.per_user_pairs["user_id"]) == users["test"]
    assert set(results.per_user_pairs["subgroup_attr"]) == {"all"}
    saved, meta = read_per_user_pairs_parquet(out / "constant.parquet")
    assert set(saved["subgroup_value"]) == {"all"}
    expected_rate = int(missing_prediction) / 4
    assert results.overall_fallback_rate == meta["overall_fallback_rate"] == expected_rate
    assert json.loads((out / "constant/fallback.json").read_text()) == {
        "Diabetes": {"n_fallback": int(missing_prediction), "n_test": 4}
    }
    tools = str(Path(__file__).resolve().parents[1] / "tools")
    monkeypatch.syspath_prepend(tools)
    from upload_leaderboard_substrate import resolve_fallback_rate

    assert resolve_fallback_rate(
        out / "constant.parquet", "constant", "downstream", None, None
    ) == expected_rate
    expected_predictions = np.linspace(0.1, 0.9, 4)
    if missing_prediction:
        expected_predictions[0] = 0.6
    np.testing.assert_allclose(saved["y_proba"], expected_predictions, rtol=1e-6)
    assert saved["y_true"].tolist() == [0, 1, 0, 1]
    assert not (out / "_subgroups.json").exists()
    assert api.STORE._enrollment is None


def test_reduce_substrate_to_point_baseline_invariants(tmp_path):
    """reduce_substrate_to_point runs from the pooled substrate; the baseline scores 0."""
    from downstream_evaluation.evaluation.bootstrap_skill_rank import reduce_substrate_to_point
    from downstream_evaluation.evaluation.per_user_pairs import build_per_user_pairs

    uids = [f"u{i}" for i in range(6)]
    data = {
        "linear": {
            "Diabetes": ([0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0], [0.4, 0.6, 0.45, 0.55, 0.5, 0.5]),
            "age": ([25.0, 35, 45, 55, 65, 30], [30.0, 38, 42, 50, 60, 35], [30.0, 38, 42, 50, 60, 35]),
        },
        "good": {
            "Diabetes": ([0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0], [0.1, 0.9, 0.2, 0.8, 0.85, 0.15]),
            "age": ([25.0, 35, 45, 55, 65, 30], [26.0, 36, 44, 54, 64, 31], [26.0, 36, 44, 54, 64, 31]),
        },
    }
    tasks = ["Diabetes", "age"]
    for method, per_task in data.items():
        for task, (yt, yp, pr) in per_task.items():
            _write_task(
                tmp_path, method, task,
                pd.DataFrame({"uid": uids, "y_true": yt, "y_pred": yp, "y_proba": pr}),
            )
    df = pd.concat(
        [build_per_user_pairs(tmp_path, m, tasks, method_label=m) for m in data], ignore_index=True
    )

    res = reduce_substrate_to_point(df, baseline="linear")

    assert set(res) == {"linear", "good"}
    assert set(res["good"]) == {"skill", "rank"}
    # Baseline vs self -> exactly 0 skill.
    assert res["linear"]["skill"]["Overall"] == pytest.approx(0.0, abs=1e-12)
    # Both methods carry a finite Overall rank + skill.
    for m in ("linear", "good"):
        assert "Overall" in res[m]["rank"]
        assert res[m]["skill"]["Overall"] == res[m]["skill"]["Overall"]  # not NaN
    # An unknown baseline is a loud error.
    with pytest.raises(ValueError, match="baseline"):
        reduce_substrate_to_point(df, baseline="nonexistent")

    # Reordering valid rows must preserve every headline result.
    assert reduce_substrate_to_point(df.iloc[::-1]) == res

    # Substrates built before S_fair was retired also hold age and sex rows. They are
    # ignored, even when one user's rows disagree.
    baked = pd.concat(
        [
            df.assign(subgroup_attr="sex", subgroup_value="male"),
            df[df["user_id"] == "u0"].assign(subgroup_attr="sex", subgroup_value="female"),
        ]
    )
    assert reduce_substrate_to_point(pd.concat([df, baked], ignore_index=True)) == res

    # A cohort view is a membership filter of the "all" rows.
    cohort = reduce_substrate_to_point(df[df["user_id"].isin(uids[:4])])
    assert cohort["linear"]["skill"]["Overall"] == pytest.approx(0.0, abs=1e-12)
    assert cohort["good"] != res["good"]


def test_producer_rejects_duplicate_participant_task_rows(tmp_path, monkeypatch, producer):
    """Matching counts must not hide a duplicate participant in a task's predictions."""
    predictions, out = tmp_path / 'predictions', tmp_path / 'out'
    _write_task(predictions, 'wbm', 'Diabetes', pd.DataFrame({
        'uid': ['u0', 'u0'], 'y_true': [1, 1], 'y_pred': [1, 1], 'y_proba': [0.9, 0.9],
    }))
    (predictions / 'wbm/fallback.json').write_text(json.dumps({
        'Diabetes': {'n_fallback': 0, 'n_test': 2},
    }))
    monkeypatch.setattr(sys, 'argv', [
        'produce', '--predictions-dir', str(predictions), '--out-dir', str(out), '--methods', 'wbm',
    ])
    with pytest.raises(SystemExit, match='duplicate participant-task'):
        producer.main()
    assert not (out / 'wbm.parquet').exists()


@pytest.mark.parametrize('column', ['y_true', 'y_pred', 'y_proba'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_producer_rejects_nonfinite_pairs(tmp_path, monkeypatch, producer, column, value):
    """Neither missing nor infinite pairs may be published after fallback scoring."""
    predictions, out = tmp_path / 'predictions', tmp_path / 'out'
    frame = pd.DataFrame({'uid': ['u0'], 'y_true': [1.0], 'y_pred': [1.0], 'y_proba': [0.9]})
    frame[column] = value
    _write_task(predictions, 'wbm', 'Diabetes', frame)
    (predictions / 'wbm/fallback.json').write_text(json.dumps({
        'Diabetes': {'n_fallback': 0, 'n_test': 1},
    }))
    monkeypatch.setattr(sys, 'argv', [
        'produce', '--predictions-dir', str(predictions), '--out-dir', str(out), '--methods', 'wbm',
    ])
    with pytest.raises(SystemExit, match='non-finite pair'):
        producer.main()
    assert not (out / 'wbm.parquet').exists()
