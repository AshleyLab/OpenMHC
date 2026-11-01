"""Regression coverage for switching datasets after consumers are imported."""

import importlib
import json
import os
from dataclasses import replace

import pandas as pd
import pytest

from openmhc._evaluate import _DatasetPaths, _ensure_labels_env
from openmhc._label_paths import LabelPaths


@pytest.fixture(autouse=True)
def clean_configuration(monkeypatch):
    """Keep configuration changes independent of test execution order."""
    for key in (
        "MHC_DATA_DIR",
        "LABELS_DATA_PATH",
        "CONTEXT_LABELS_PATH",
        "ENROLLMENT_DATA_PATH",
        "LABEL_VALIDITY_PATH",
        "HEALTHKIT_DAILY_PATH",
        "LABEL_TYPES_PATH",
        "ORDINAL_DICTIONARY_PATH",
        "VALIDITY_CONFIG_PATH",
        "USER_DEVICE_INFO_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    from devices import api as devices
    from labels import api

    api.configure()
    devices.configure()
    yield
    api.configure(paths=LabelPaths.resolve())
    devices.configure(paths=LabelPaths.resolve())


def write_dataset(root, value, context_name):
    """Use distinct labels, validity, enrollment and device data in each root."""
    folder = root / "labels"
    folder.mkdir(parents=True)
    series = {"u": {"timestamps": ["2020-01-01", "2020-01-02"], "values": [value, value + 1]}}
    files = {
        "last_labels.json": {"age": series},
        "context_labels.json": {context_name: series},
        "enrollment_info.json": {"u": {"birth_year": 2000 - value}},
        "label_validity.json": {"age": {"u": [False, True]}},
        "healthkit_daily.json": {"happiness": series},
        "user_device_info.json": {"u": {"phone": [["2020-01-01", "2020-01-02", str(value)]]}},
        "label_types.json": {
            "age": {"type": "continuous", "target": True},
            context_name: {"type": "continuous", "target": False},
        },
        "ordinal_dictionary.json": {},
        "validity_config.json": {},
    }
    for name, content in files.items():
        (folder / name).write_text(json.dumps(content))
    # Resolver PR intentionally retains repo schema defaults. Select synthetic
    # schemas explicitly so this also tests updates to imported metadata lists.
    return replace(
        LabelPaths.resolve(root),
        label_types=folder / "label_types.json",
        ordinal_dictionary=folder / "ordinal_dictionary.json",
        validity_config=folder / "validity_config.json",
    )


def test_switch_refreshes_imported_consumers_and_all_caches(tmp_path):
    """Switch A to B and back without leaving imported consumers or caches stale."""
    import context.api as context
    import downstream_evaluation.evaluation.metrics as metrics
    import imputation_evaluation.sensitivity as sensitivity
    from devices import api as devices
    from labels import api

    # Legacy tests elsewhere reload labels.api. Bind consumers to the current
    # module once, before the A -> B -> A lifecycle under test (no reloads there).
    for module in (context, metrics, sensitivity):
        importlib.reload(module)
    store, device_store = api.STORE, devices.STORE
    names, targets, contexts, types = (
        api.LABEL_NAMES,
        api.TARGET_NAMES,
        api.CONTEXT_NAMES,
        api.LABEL_TYPES,
    )
    result_class = api.LabelResult
    before = dict(os.environ)
    a = write_dataset(tmp_path / "a", 20, "context_a")
    b = write_dataset(tmp_path / "b", 40, "context_b")
    t = pd.Timestamp("2020-01-01")

    for paths, value, context_name in (
        (a, 20, "context_a"),
        (b, 40, "context_b"),
        (a, 20, "context_a"),
    ):
        _ensure_labels_env(paths)
        assert api.STORE is store is sensitivity.STORE
        assert devices.STORE is device_store
        assert api.LabelResult is result_class
        assert api.LABEL_TYPES is types is metrics.LABEL_TYPES
        assert api.LABEL_NAMES is names is metrics.LABEL_NAMES is context.LABEL_NAMES
        assert api.CONTEXT_NAMES is contexts is context.CONTEXT_NAMES
        assert api.TARGET_NAMES is targets
        assert names == ["age", context_name]
        assert targets == ["age"] and contexts == [context_name]
        assert api.get_labels("u", t, "age").value == value + 1  # validity mask applied
        assert context.get_context("u", t, context_name).value == value
        assert store.enrollment.get_birth_year("u") == 2000 - value
        assert api.get_labels_windowed("u", t, "happiness", window_days=0).value == value
        assert devices.get_devices("u", t).phone.name == str(value)
        assert dict(os.environ) == before


def test_failed_schema_configuration_leaves_previous_store_usable(tmp_path):
    """Reject malformed new metadata before changing the live configuration."""
    from labels import api

    a = write_dataset(tmp_path / "a", 20, "context_a")
    b = write_dataset(tmp_path / "b", 40, "context_b")
    api.configure(paths=a)
    b.label_types.write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        api.configure(paths=b)
    assert api.LABEL_TYPES_PATH == a.label_types
    assert api.get_labels("u", pd.Timestamp("2020-01-01"), "age").value == 21


def test_explicit_root_beats_mhc_env_and_resolves_devices(monkeypatch, tmp_path):
    """Honor explicit roots and deliberate per-device file overrides."""
    monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "old"))
    paths = _DatasetPaths._build(tmp_path / "new", "xs")
    assert paths.label_paths.labels == tmp_path / "new/labels/last_labels.json"
    assert paths.label_paths.devices == tmp_path / "new/labels/user_device_info.json"
    monkeypatch.setenv("USER_DEVICE_INFO_PATH", str(tmp_path / "device_override.json"))
    assert LabelPaths.resolve(tmp_path / "new").devices == tmp_path / "device_override.json"


def test_device_file_stays_optional(tmp_path):
    """Permit configuration without the device file but fail on actual lookup."""
    from devices import api

    _ensure_labels_env(LabelPaths.resolve(tmp_path))
    with pytest.raises(FileNotFoundError, match="user_device_info.json"):
        api.get_device_timeline("u")


@pytest.mark.parametrize("track", ["prediction", "imputation", "forecasting"])
def test_public_evaluation_switches_dataset_before_runner(monkeypatch, tmp_path, track):
    """Each public entry point configures the selected root before running models."""
    import openmhc
    from labels import api
    from openmhc._dataset import write_dataset_marker

    runner_name = (
        "downstream_evaluation.runner" if track == "prediction" else f"{track}_evaluation.runner"
    )
    runner = importlib.import_module(runner_name)
    roots = [tmp_path / "a", tmp_path / "b"]
    for root, value in zip(roots, (20, 40)):
        write_dataset(root, value, "context")
        for folder in (
            "splits",
            "processed/daily_hf",
            "hourly_trajectory",
            "forecasting_sample_index",
        ):
            (root / folder).mkdir(parents=True, exist_ok=True)
        split = {"train": ["u"] + [f"u{i}" for i in range(592)], "validation": [], "test": []}
        (root / "splits/sharable_users_seed42_2026_xs.json").write_text(json.dumps(split))
        write_dataset_marker(root, version="xs", n_users=593)

    class ReachedRunner(Exception):
        """Stop at the model boundary after observing real API configuration."""

    observed = []

    def capture(*args, **kwargs):
        observed.append(
            (api.STORE.labels_path, api.get_labels("u", pd.Timestamp("2020-01-01"), "age").value)
        )
        raise ReachedRunner

    monkeypatch.setattr(runner, "run_eval", capture)

    class Model:
        """Satisfy the public model protocols without running inference."""

        def predict(self, *args, **kwargs):
            raise AssertionError("The configuration check must stop before inference")

        def impute(self, *args, **kwargs):
            raise AssertionError("The configuration check must stop before inference")

    model = Model()
    store = api.STORE
    environment = dict(os.environ)
    kwargs = {"tasks": ["age"]} if track == "prediction" else {"max_samples": 1}
    for root in roots:
        with pytest.raises(ReachedRunner):
            getattr(openmhc, f"evaluate_{track}")(model, version="xs", data_dir=root, **kwargs)
        assert api.STORE is store
        assert dict(os.environ) == environment
    assert observed == [
        (roots[0] / "labels/last_labels.json", 21),
        (roots[1] / "labels/last_labels.json", 41),
    ]


def test_package_schema_paths_follow_configuration(tmp_path):
    """Expose current schema paths through the package after configuration."""
    import labels
    from labels import api

    paths = write_dataset(tmp_path / "a", 20, "context_a")
    api.configure(paths=paths)
    assert labels.LABEL_TYPES_PATH == paths.label_types
    assert labels.ORDINAL_DICTIONARY_PATH == paths.ordinal_dictionary


@pytest.mark.parametrize("explicit_files_only", [False, True])
def test_fairness_command_uses_resolver(monkeypatch, tmp_path, explicit_files_only):
    """Preserve root selection and the CLI explicit-file compatibility path."""
    from forecasting_evaluation.metrics import fairness_skill_score_summary as summary

    monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "environment"))
    monkeypatch.setenv("LABELS_DATA_PATH", str(tmp_path / "env_labels.json"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "fairness",
            "--baseline",
            "base",
            "--data-dir",
            str(tmp_path / "explicit"),
            "--labels-path",
            str(tmp_path / "cli_labels.json"),
        ],
    )
    if explicit_files_only:
        monkeypatch.delenv("MHC_DATA_DIR")
        monkeypatch.delenv("LABELS_DATA_PATH")
        monkeypatch.setattr(
            "sys.argv",
            [
                "fairness",
                "--baseline",
                "base",
                "--labels-path",
                str(tmp_path / "cli_labels.json"),
                "--enrollment-path",
                str(tmp_path / "cli_enrollment.json"),
            ],
        )
    monkeypatch.setattr(summary, "_load_models_dict", lambda args: {})
    seen = {}

    class ReachedEvaluation(Exception):
        pass

    def capture(**kwargs):
        seen.update(kwargs)
        raise ReachedEvaluation

    monkeypatch.setattr(summary, "compute_fairness_skill_score_tables", capture)
    with pytest.raises(ReachedEvaluation):
        summary.main()
    assert seen["labels_path"] == tmp_path / "cli_labels.json"
    expected_enrollment = (
        tmp_path / "cli_enrollment.json"
        if explicit_files_only
        else tmp_path / "explicit/labels/enrollment_info.json"
    )
    assert seen["enrollment_path"] == expected_enrollment


def test_xgboost_feature_builder_configures_labels_before_cutoffs(monkeypatch, tmp_path):
    """The feature-building entry point passes resolved paths to label configuration."""
    import sys
    from types import ModuleType

    from downstream_evaluation.models.xgboost import model, preprocessing
    from labels import api
    from openmhc._dataset import write_dataset_marker

    # Optional feature-extraction packages are irrelevant before cutoff selection.
    for suffix, function in (
        ("pipeline_curve_analysis", "build_curve_analysis_features"),
        ("pipeline_day_dynamics", "build_signal_processing_features"),
        ("pipeline_timeseries", "build_user_features_chunked"),
    ):
        name = "downstream_evaluation.models.xgboost." + suffix
        module = ModuleType(name)
        setattr(module, function, lambda **kwargs: None)
        monkeypatch.setitem(sys.modules, name, module)

    root = tmp_path / "features_dataset"
    paths = write_dataset(root, 20, "context_a")
    write_dataset_marker(root, version="xs", n_users=593)

    class CutoffReached(Exception):
        pass

    def check_cutoffs(**kwargs):
        assert api.STORE.labels_path == paths.labels
        assert api.STORE.enrollment_path == paths.enrollment
        raise CutoffReached

    monkeypatch.setattr(preprocessing, "build_cutoff_dates", check_cutoffs)
    with pytest.raises(CutoffReached):
        model.extract_xgboost_features(
            str(tmp_path / "features"), data_dir=str(root), max_future_days=1
        )
