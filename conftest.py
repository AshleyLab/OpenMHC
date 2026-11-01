"""Synthetic dataset schemas shared by label-dependent tests."""

import json

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
    root = tmp_path / "synthetic_dataset"
    folder = write_label_schemas(root)
    monkeypatch.setenv("MHC_DATA_DIR", str(root))
    from labels import api

    api.configure(root)
    return folder
