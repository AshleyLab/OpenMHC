"""Isolated dataset payloads for age API and evaluation tests."""

import pytest


@pytest.fixture
def write_age_dataset():
    """Create a labels directory for synthetic participant payloads."""

    def write(root):
        folder = root / "labels"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    return write


@pytest.fixture
def age_label_environment(monkeypatch, tmp_path, write_age_dataset):
    """Configure test payloads while keeping the repository's schema defaults."""
    from labels import api

    previous = api._PATHS
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
    root = tmp_path / "age_dataset"
    folder = write_age_dataset(root)
    monkeypatch.setenv("MHC_DATA_DIR", str(root))
    api.configure(root)
    yield folder
    api.configure(paths=previous)
