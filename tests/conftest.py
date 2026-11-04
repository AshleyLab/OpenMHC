"""Synthetic dataset schemas shared by label-dependent tests."""

import json
import os

import pytest


@pytest.fixture
def write_label_schemas():
    """Create minimal schemas without depending on a particular release."""

    def write(root):
        folder = root / "labels"
        folder.mkdir(parents=True, exist_ok=True)
        types = {
            "age": "continuous",
            "BiologicalSex": "binary",
            "Diabetes": "binary",
            "WeightKilograms": "continuous",
            "TotalCholesterol": "continuous",
            "BMI_categories": "ordinal",
            "happiness": "continuous",
            "cardiovascular_disease": "binary",
            "sleep_diagnosis1": "binary",
            "Watch_RestingHeartRate": "continuous",
            "framingham_risk": "continuous",
        }
        types = {key: {"type": value, "target": True} for key, value in types.items()}
        for name in ("field_family_history", "field_race"):
            types[name] = {"type": "multi_categorical", "target": False}
        schemas = {
            "label_types.json": types,
            "ordinal_dictionary.json": {
                "BMI_categories": {
                    "Underweight": 0,
                    "Normal weight": 1,
                    "Overweight": 2,
                    "Obesity": 3,
                    "Morbid Obesity": 4,
                }
            },
            "validity_config.json": {},
        }
        for name, value in schemas.items():
            (folder / name).write_text(json.dumps(value))
        return folder

    return write


@pytest.fixture
def label_schema_environment(monkeypatch, tmp_path, write_label_schemas):
    """Isolate legacy per-file fixtures and configure a synthetic dataset."""
    from devices import api as devices
    from labels import api

    paths, configured = api._PATHS, api._SCHEMAS_CONFIGURED
    metadata = {name: (getattr(api, name), getattr(api, name).copy()) for name in (
        "LABEL_TYPES", "_LABEL_META", "LABEL_NAMES", "TARGET_NAMES", "CONTEXT_NAMES",
        "MULTI_CATEGORICAL_NAMES",
    )}
    store, device_store = api.STORE, devices.STORE
    store_state, device_state = vars(store).copy(), vars(device_store).copy()
    env_keys = (
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
    )
    previous_env = {key: os.environ.get(key) for key in env_keys}
    for key in env_keys:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "synthetic_dataset"
    folder = write_label_schemas(root)
    monkeypatch.setenv("MHC_DATA_DIR", str(root))
    try:
        api.configure(root)
        yield folder
    finally:
        monkeypatch.undo()
        # Some legacy helpers write os.environ directly instead of monkeypatch.
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        api._PATHS, api._SCHEMAS_CONFIGURED = paths, configured
        for name, (container, value) in metadata.items():
            if isinstance(container, dict):
                container.clear()
                container.update(value)
            else:
                container[:] = value
            setattr(api, name, container)
        api.STORE, devices.STORE = store, device_store
        vars(store).clear()
        vars(store).update(store_state)
        vars(device_store).clear()
        vars(device_store).update(device_state)


@pytest.fixture
def write_age_dataset(write_label_schemas):
    """Create a complete synthetic schema root for age-specific payloads."""
    return write_label_schemas


@pytest.fixture
def age_label_environment(label_schema_environment):
    """Reuse schema setup and restore both configured and unconfigured states."""
    return label_schema_environment
