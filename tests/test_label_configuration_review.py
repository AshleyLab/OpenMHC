"""Regression cases from the resolver PR review, using distinct A/B datasets."""

import json
import multiprocessing
import os
import runpy
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_label_configuration import clean_configuration, write_dataset  # noqa: F401


@pytest.mark.parametrize("schema", ["label_types", "ordinal_dictionary", "validity_config"])
@pytest.mark.parametrize("damage", ["missing", "json", "shape"])
def test_invalid_schema_does_not_switch_live_stores(tmp_path, schema, damage):
    """A schema failure must leave metadata, label caches and devices on A."""
    from devices import api as devices
    from labels import api
    from openmhc._evaluate import _ensure_labels_env

    a = write_dataset(tmp_path / "a", 20, "context_a")
    b = write_dataset(tmp_path / "b", 40, "context_b")
    _ensure_labels_env(a)
    timestamp = pd.Timestamp("2020-01-01")
    assert api.get_labels("u", timestamp, "age").value == 21
    assert devices.get_devices("u", timestamp).phone.name == "20"
    cached_labels, cached_devices = api.STORE._labels_index, devices.STORE._timelines
    damaged = getattr(b, schema)
    if damage == "missing":
        damaged.unlink()
    else:
        damaged.write_text("{broken" if damage == "json" else "[]")
    with pytest.raises((ValueError, FileNotFoundError)):
        _ensure_labels_env(b)
    assert api._PATHS == a
    assert api.STORE._labels_index is cached_labels
    assert devices.STORE._timelines is cached_devices
    assert api.LABEL_NAMES == ["age", "context_a"]
    assert api.get_labels("u", timestamp, "age").value == 21
    assert devices.get_devices("u", timestamp).phone.name == "20"


def test_public_label_configuration_switches_devices_too(tmp_path):
    """The documented dataset switch keeps both public accessors consistent."""
    import devices
    import labels

    a = write_dataset(tmp_path / "a", 20, "context_a")
    b = write_dataset(tmp_path / "b", 40, "context_b")
    for paths, value in [(a, 20), (b, 40), (a, 20)]:
        labels.configure(paths=paths)
        assert labels.get_labels("u", pd.Timestamp("2020-01-01"), "age").value == value + 1
        assert devices.get_devices("u", pd.Timestamp("2020-01-01")).phone.name == str(value)


def test_repeated_evaluation_keeps_caches_but_explicit_configure_refreshes(tmp_path):
    """Keep caches for unchanged evaluation paths, but honor forced refreshes."""
    from devices import api as devices
    from labels import api
    from openmhc._evaluate import _ensure_labels_env

    paths = write_dataset(tmp_path / "a", 20, "context_a")
    _ensure_labels_env(paths)
    api.get_labels("u", pd.Timestamp("2020-01-01"), "age")
    devices.get_device_timeline("u")
    index, timelines = api.STORE._labels_index, devices.STORE._timelines
    _ensure_labels_env(paths)
    assert api.STORE._labels_index is index
    assert devices.STORE._timelines is timelines
    api.configure(paths=paths)
    assert api.STORE._labels_index is None
    assert devices.STORE._timelines is None


class LabelReadingImputer:
    """A custom imputer that queries both stores inside the actual pool worker."""

    def impute(self, data, observed_mask, target_mask):
        """Fail if either child store points at the ambient B dataset."""
        import devices
        import labels

        timestamp = pd.Timestamp("2020-01-01")
        assert labels.get_labels("u", timestamp, "age").value == 21
        assert devices.get_devices("u", timestamp).phone.name == "20"
        return np.zeros_like(data)


class TinyMaskCache:
    """Mask two timesteps on nineteen channels without dataset dependencies."""

    def get_batch_masks(self, split, scenario, indices):
        """Return deterministic masks for the tiny worker regression."""
        return list(range(len(indices))), np.ones((len(indices), 19, 2), dtype=np.float32)


class LabelFreeImputer:
    """An imputer that does not require dataset label metadata."""

    def impute(self, data, observed_mask, target_mask):
        """Allow workers to evaluate tensors without configuring label schemas."""
        from labels import api

        assert not api._SCHEMAS_CONFIGURED
        return np.zeros_like(data)


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_imputation_workers_without_label_configuration(tmp_path, monkeypatch, start_method):
    """Standalone tensor evaluation must not require unused label schemas."""
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} unavailable on this platform")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from imputation_evaluation.evaluation import evaluator
    from labels import api
    from openmhc._label_paths import LabelPaths

    monkeypatch.setattr(api, "_SCHEMAS_CONFIGURED", False)
    monkeypatch.setattr(api, "_PATHS", LabelPaths.resolve())
    monkeypatch.setattr(evaluator, "ProcessPoolExecutor", partial(
        ProcessPoolExecutor, mp_context=multiprocessing.get_context(start_method)))
    loader = DataLoader(TensorDataset(torch.ones(1, 19, 2), torch.ones(1, 19, 2)), batch_size=1)
    engine = evaluator.ImputationEvaluator(
        ["random_noise"], num_eval_workers=2, pairs_dir=tmp_path / "pairs")
    counts = engine._evaluate_split(loader, TinyMaskCache(), LabelFreeImputer(), np.ones(19), "test")
    assert counts["random_noise"]["n_applicable"] == 1


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
@pytest.mark.parametrize("ambient_root", [False, True])
def test_actual_imputation_worker_uses_selected_paths(tmp_path, monkeypatch, start_method, ambient_root):
    """Exercise the actual pool initializer and custom imputer for both contexts."""
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} unavailable on this platform")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from imputation_evaluation.evaluation import evaluator
    from openmhc._evaluate import _ensure_labels_env

    a = write_dataset(tmp_path / "a", 20, "context_a")
    write_dataset(tmp_path / "b", 40, "context_b")
    if ambient_root:
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "b"))
    _ensure_labels_env(a)
    monkeypatch.setattr(evaluator, "ProcessPoolExecutor", partial(
        ProcessPoolExecutor, mp_context=multiprocessing.get_context(start_method)))
    loader = DataLoader(TensorDataset(torch.ones(1, 19, 2), torch.ones(1, 19, 2)), batch_size=1)
    engine = evaluator.ImputationEvaluator(["random_noise"], num_eval_workers=2, pairs_dir=tmp_path / "pairs")
    counts = engine._evaluate_split(loader, TinyMaskCache(), LabelReadingImputer(), np.ones(19), "test")
    assert counts["random_noise"]["n_applicable"] == 1


@pytest.mark.parametrize("constructor_root", [False, True])
def test_xgboost_cache_miss_receives_evaluation_root(tmp_path, monkeypatch, constructor_root):
    """The evaluator's A wins over ambient B and a conflicting constructor B."""
    from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator
    from downstream_evaluation.models.xgboost import model
    from openmhc._evaluate import _ensure_labels_env

    a = write_dataset(tmp_path / "a", 20, "context_a")
    _ensure_labels_env(a)
    monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "b"))
    method = model.XGBoost(data_dir=str(tmp_path / "b") if constructor_root else None,
                           features_dir=str(tmp_path / "features"))

    class BuildReached(Exception):
        pass

    def build(**kwargs):
        assert kwargs["data_dir"] == str(tmp_path / "a")
        raise BuildReached

    monkeypatch.setattr(model, "extract_xgboost_features", build)
    td = SimpleNamespace(task="age", split="train", user_ids=np.array(["u"]),
                         dates=None, inputs=None, labels=np.array([21]))
    with pytest.raises(BuildReached):
        DownstreamEvaluator(data_dir=str(tmp_path / "a"))._eval_task(method, "age", td, td)


def test_missing_schema_environment_override_is_rejected(tmp_path, monkeypatch):
    """A missing explicitly selected schema cannot erase live metadata."""
    from labels import api

    a = write_dataset(tmp_path / "a", 20, "context_a")
    api.configure(paths=a)
    monkeypatch.setenv("LABEL_TYPES_PATH", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="Unset LABEL_TYPES_PATH"):
        api.configure(data_dir=tmp_path / "b")
    assert api._PATHS == a


def test_bad_ordinal_mapping_is_rejected_before_switch(tmp_path):
    """Valid JSON with invalid mapping structure is also a configuration error."""
    from labels import api

    a = write_dataset(tmp_path / "a", 20, "context_a")
    api.configure(paths=a)
    b = write_dataset(tmp_path / "b", 40, "context_b")
    broken = b.ordinal_dictionary
    broken.write_text('{"age": []}')
    with pytest.raises(ValueError):
        api.configure(paths=b)
    assert api._PATHS == a


@pytest.mark.parametrize("schema", ["ordinal_dictionary", "validity_config"])
def test_first_evaluation_validates_import_time_schema_paths(tmp_path, schema):
    """Matching initial paths must not bypass validation of the other schemas."""
    paths = write_dataset(tmp_path / "a", 20, "context_a")
    getattr(paths, schema).write_text("{broken")
    env = dict(os.environ, MHC_DATA_DIR=str(tmp_path / "a"),
               LABEL_TYPES_PATH=str(paths.label_types),
               ORDINAL_DICTIONARY_PATH=str(paths.ordinal_dictionary),
               VALIDITY_CONFIG_PATH=str(paths.validity_config))
    code = """from labels import api
from openmhc._evaluate import _ensure_labels_env
from openmhc._label_paths import LabelPaths
paths = LabelPaths.resolve()
assert api._PATHS == paths
_ensure_labels_env(paths)
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "JSONDecodeError" in result.stderr


def test_full_history_feature_build_does_not_reset_labels(tmp_path, monkeypatch):
    """The default feature path never reads labels and must not reconfigure them."""
    import polars as pl

    from downstream_evaluation.models.xgboost import model
    from labels import api
    from openmhc._dataset import write_dataset_marker

    a = write_dataset(tmp_path / "a", 20, "context_a")
    api.configure(paths=a)
    api.get_labels("u", pd.Timestamp("2020-01-01"), "age")
    index = api.STORE._labels_index
    write_dataset(tmp_path / "b", 40, "context_b")
    write_dataset_marker(tmp_path / "b", version="xs", n_users=593)
    monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "b"))
    for suffix, function in (
        ("pipeline_curve_analysis", "build_curve_analysis_features"),
        ("pipeline_day_dynamics", "build_signal_processing_features"),
        ("pipeline_timeseries", "build_user_features_chunked"),
    ):
        name = "downstream_evaluation.models.xgboost." + suffix
        module = ModuleType(name)
        setattr(module, function, lambda **kwargs: None)
        monkeypatch.setitem(sys.modules, name, module)

    class LookupReached(Exception):
        pass

    def read_lookup(*args, **kwargs):
        assert api._PATHS == a
        assert api.STORE._labels_index is index
        raise LookupReached

    monkeypatch.setattr(pl, "read_parquet", read_lookup)
    with pytest.raises(LookupReached):
        model.extract_xgboost_features(str(tmp_path / "features"))


def test_fingerprint_uses_actual_paths_and_covariate_helper(tmp_path, monkeypatch, capsys):
    """Changing the real exclusion helper must change the fingerprint too."""
    from downstream_evaluation import demo_covariates
    from downstream_evaluation.models import linear
    from labels import api

    paths = write_dataset(tmp_path / "a", 20, "context_a")
    api.configure(paths=paths)
    monkeypatch.setattr(linear, "_DEMO_COVARIATES", ["age", "BiologicalSex", "extra"])
    monkeypatch.setattr(demo_covariates, "_kept_cov_indices", lambda task, covariates: [1])
    script = runpy.run_path(str(Path(__file__).parents[1] / "scripts/fingerprint_labels.py"))
    script["main"]()
    result = json.loads(capsys.readouterr().out)
    assert result["files"]["USER_DEVICE_INFO_PATH"]["path"] == str(paths.devices)
    assert len(result["files"]) == 9
    assert all(value == ["age", "extra"] for value in result["covariate_exclusions"].values())


def test_configuration_fixture_restores_explicit_state_and_environment(tmp_path, monkeypatch, write_label_schemas):
    """Teardown restores A and warm caches even when the saved environment says B."""
    from devices import api as devices
    from labels import api

    paths = write_dataset(tmp_path / "a", 20, "context_a")
    api.configure(paths=paths)
    api.get_labels("u", pd.Timestamp("2020-01-01"), "age")
    devices.get_device_timeline("u")
    index, timelines = api.STORE._labels_index, devices.STORE._timelines
    monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "b"))
    with pytest.MonkeyPatch.context() as nested:
        fixture = clean_configuration.__wrapped__(nested, tmp_path, write_label_schemas)
        next(fixture)
        assert "MHC_DATA_DIR" not in os.environ
        with pytest.raises(StopIteration):
            next(fixture)
    assert os.environ["MHC_DATA_DIR"] == str(tmp_path / "b")
    assert api._PATHS == paths
    assert api.STORE._labels_index is index
    assert devices.STORE._timelines is timelines


def test_empty_label_schema_preserves_active_state(tmp_path):
    """Reject an empty schema without replacing configured metadata or caches."""
    from labels import api

    a = write_dataset(tmp_path / "a", 20, "context_a")
    b = write_dataset(tmp_path / "b", 40, "context_b")
    api.configure(paths=a)
    api.get_labels("u", pd.Timestamp("2020-01-01"), "age")
    cached = api.STORE._labels_index
    b.label_types.write_text("{}")
    with pytest.raises(ValueError, match="nonempty.*label_types.json"):
        api.configure(paths=b)
    assert api._PATHS == a
    assert api._SCHEMAS_CONFIGURED
    assert api.LABEL_NAMES == ["age", "context_a"]
    assert api.STORE._labels_index is cached


def test_direct_cutoffs_initialize_metadata(tmp_path, monkeypatch):
    """Direct feature-helper calls must not silently return zero cutoff dates."""
    from downstream_evaluation.models.xgboost.preprocessing import build_cutoff_dates
    from labels import api

    paths = write_dataset(tmp_path / "dataset", 20, "context_a")
    monkeypatch.setenv("MHC_DATA_DIR", str(paths.schema_dir.parent))
    monkeypatch.setattr(api, "_SCHEMAS_CONFIGURED", False)
    api.TARGET_NAMES.clear()
    assert build_cutoff_dates() == {"u": "2021-01-01"}
    assert api._SCHEMAS_CONFIGURED
    assert api._PATHS == paths


@pytest.mark.parametrize("configured", [False, True])
def test_schema_environment_fixture_restores_state(tmp_path, monkeypatch, write_label_schemas, configured):
    """The shared synthetic fixture restores API state and direct env writes."""
    label_schema_environment = runpy.run_path(
        str(Path(__file__).resolve().with_name("conftest.py"))
    )["label_schema_environment"]

    from devices import api as devices
    from labels import api

    a = write_dataset(tmp_path / "a", 20, "context_a")
    api.configure(paths=a)
    api.get_labels("u", pd.Timestamp("2020-01-01"), "age")
    devices.get_devices("u", pd.Timestamp("2020-01-01"))
    monkeypatch.setattr(api, "_SCHEMAS_CONFIGURED", configured)
    store, device_store = api.STORE, devices.STORE
    labels_index, timelines = store._labels_index, device_store._timelines
    names = api.LABEL_NAMES
    before_env = dict(os.environ)
    with pytest.MonkeyPatch.context() as nested:
        fixture = label_schema_environment.__wrapped__(nested, tmp_path, write_label_schemas)
        next(fixture)
        os.environ["LABELS_DATA_PATH"] = "legacy-direct-write"
        with pytest.raises(StopIteration):
            next(fixture)
    assert dict(os.environ) == before_env
    assert api._PATHS == a
    assert api._SCHEMAS_CONFIGURED == configured
    assert api.LABEL_NAMES is names
    assert names == ["age", "context_a"]
    assert api.STORE is store and devices.STORE is device_store
    assert store._labels_index is labels_index and device_store._timelines is timelines
