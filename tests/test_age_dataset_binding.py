"""Keep lookup ages and prediction-export metadata on the selected dataset."""

import json
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from downstream_evaluation.config import EvalConfig
from downstream_evaluation.data.provider import TaskDataProvider, lookup_filename
from labels import api


@pytest.fixture
def bound_datasets(label_schema_environment, tmp_path):
    """Two complete roots share a user but have different birth years."""
    folder = label_schema_environment
    root = folder.parent
    (folder / "last_labels.json").write_text(json.dumps({
        "age": {"u": {"timestamps": ["2025-06-01"], "values": [99]}},
    }))
    (folder / "enrollment_info.json").write_text(json.dumps({
        "u": {"birth_year": 1980, "createdOn": "2015-01-01"},
    }))
    (root / "processed").mkdir()
    lookup = root / "processed" / lookup_filename("daily", full_history=True)
    pd.DataFrame({
        "user_id": ["u"], "date": [pd.Timestamp("2025-06-01")],
        "age": [99.0], "BiologicalSex": [1],
    }).to_parquet(lookup)
    other = tmp_path / "other_dataset"
    shutil.copytree(root, other)
    (other / "labels/enrollment_info.json").write_text(json.dumps({
        "u": {"birth_year": 1990, "createdOn": "2015-01-01"},
    }))
    return root, other, lookup


@pytest.mark.parametrize("cached", [False, True])
def test_provider_rejects_other_dataset_before_age_reads(bound_datasets, cached):
    """A global configure switch cannot supply or silently reuse another root's ages."""
    root, other, lookup = bound_datasets
    provider = TaskDataProvider(lookup, {"test": ["u"]})
    if cached:
        assert provider.task_data("age", "test").labels.tolist() == [45]
    api.configure(other)
    with pytest.raises(ValueError, match="does not match the provider lookup"):
        provider.task_data("age", "test")
    # Lookup-only tasks do not read the Labels API.
    assert provider.task_data("BiologicalSex", "test").labels.tolist() == [1]
    api.configure(root)
    assert provider.task_data("age", "test").labels.tolist() == [45]


def test_provider_accepts_canonical_dataset_alias(bound_datasets, tmp_path):
    """A symlink to the same dataset is not a configuration conflict."""
    root, _, lookup = bound_datasets
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    provider = TaskDataProvider(alias / "processed" / lookup.name, {"test": ["u"]}, data_dir=root)
    assert provider.task_data("age", "test").labels.tolist() == [45]


def test_copied_lookup_requires_explicit_dataset_for_ages(bound_datasets, tmp_path):
    """An exported lookup's location alone cannot identify its age payloads."""
    root, _, lookup = bound_datasets
    copied = tmp_path / "copied.parquet"
    shutil.copyfile(lookup, copied)
    with pytest.raises(ValueError, match="Pass data_dir="):
        TaskDataProvider(copied, {"test": ["u"]}).task_data("age", "test")
    provider = TaskDataProvider(copied, {"test": ["u"]}, data_dir=root)
    assert provider.task_data("age", "test").labels.tolist() == [45]


def test_explicit_dataset_cannot_override_another_lookup_root(bound_datasets):
    """An explicit root must agree with a lookup in the standard dataset layout."""
    _, other, lookup = bound_datasets
    with pytest.raises(ValueError, match="data_dir does not match"):
        TaskDataProvider(lookup, {"test": ["u"]}, data_dir=other)


@pytest.mark.parametrize("field", ["labels", "enrollment"])
def test_provider_detects_payload_reconfiguration(bound_datasets, field):
    """Changing only an age payload path must also invalidate a provider's binding."""
    from dataclasses import replace

    _, other, lookup = bound_datasets
    provider = TaskDataProvider(lookup, {"test": ["u"]})
    assert provider.task_data("age", "test").labels.tolist() == [45]
    name = "last_labels.json" if field == "labels" else "enrollment_info.json"
    api.configure(paths=replace(api.STORE.paths, **{field: other / "labels" / name}))
    with pytest.raises(ValueError, match=f"provider lookup \\({field}\\)"):
        provider.task_data("age", "test")


@pytest.mark.parametrize("fault", ["missing", "truncated", "non_object", "wrong_dataset"])
def test_export_preflight_fails_before_loading_or_evaluating(bound_datasets, tmp_path, monkeypatch, fault):
    """Broken enrollment is reported before model work or prediction output starts."""
    from downstream_evaluation import runner

    root, other, _ = bound_datasets
    enrollment = root / "labels/enrollment_info.json"
    error = ValueError
    if fault == "missing":
        enrollment.unlink()
        error = FileNotFoundError
    elif fault == "truncated":
        enrollment.write_text("{")
    elif fault == "non_object":
        enrollment.write_text("[]")
    else:
        api.configure(other)
    output = tmp_path / "predictions"
    config = EvalConfig(str(root), {"test": ["u"]}, tasks=["BiologicalSex"], predictions_dir=str(output))
    monkeypatch.setattr(runner, "DataLoader", lambda *a, **k: pytest.fail("Preflight must precede sensor loading"))
    monkeypatch.setattr(runner, "DownstreamEvaluator", lambda *a, **k: pytest.fail("Preflight must precede evaluation"))
    with pytest.raises(error):
        runner.run_eval(config, SimpleNamespace())
    assert not output.exists()


@pytest.mark.parametrize("export", [False, True])
def test_runner_preflight_only_requires_enrollment_for_export(bound_datasets, tmp_path, monkeypatch, export):
    """Valid export writes the group map; lookup-only evaluation needs no enrollment."""
    from downstream_evaluation import runner

    root, _, _ = bound_datasets
    if not export:
        (root / "labels/enrollment_info.json").unlink()
    output = tmp_path / "predictions"
    config = EvalConfig(
        str(root), {"test": ["u"]}, tasks=["BiologicalSex"],
        predictions_dir=str(output) if export else None,
    )
    calls = []

    def run(provider, loader, model, tasks, **kwargs):
        calls.append(tasks)
        if export:
            assert api.STORE._enrollment is not None
        assert provider.task_data("BiologicalSex", "test").labels.tolist() == [1]
        return {"BiologicalSex": {"n_test": 1}}

    monkeypatch.setattr(runner, "DownstreamEvaluator", lambda **k: SimpleNamespace(run=run))
    results = runner.run_eval(config, SimpleNamespace(needs_segments=False, name="dummy"))
    assert calls == [["BiologicalSex"]]
    assert results["BiologicalSex"]["n_test"] == 1
    if export:
        assert json.loads((output / "_subgroups.json").read_text()) == {
            "u": {"age_group": "30-39", "sex": "male"},
        }
    else:
        assert not output.exists()
