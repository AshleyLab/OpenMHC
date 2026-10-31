"""Regression checks for canonical hourly masks and incompatible embedding caches."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from downstream_evaluation.models import tsfm
from downstream_evaluation.models.chronos2 import Chronos2
from downstream_evaluation.models.toto import Toto


@pytest.fixture(autouse=True)
def synthetic_dataset(tmp_path, monkeypatch):
    """Cache tests use a synthetic dataset marker, never the user's dataset."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "dataset_version.json").write_text(json.dumps({"version": "xs"}))
    monkeypatch.setenv("MHC_DATA_DIR", str(root))


def hourly_days(sleep_hours=8):
    values = np.ones((1, 24, 19), dtype=np.float32)
    values[:, :, 9:] = np.nan
    values[:, :, 7:9] = 0
    values[:, :sleep_hours, 7:9] = 1
    values[:, 3:7, 0] = np.nan
    values[:, 10:12, 2] = np.nan
    values[:, 13:15, 6] = np.nan
    return values


@pytest.mark.parametrize("sleep_hours", [0, 2, 8])
def test_timeline_preserves_stored_cleaning_decisions(sleep_hours):
    """Even observed all-zero sleep comes from the input mask, not a new heuristic."""
    values = hourly_days(sleep_hours)
    original = values.copy()
    timeline = tsfm.build_user_timeline(values, ["2020-01-01"], 19)
    np.testing.assert_equal(timeline.values, original[0])
    np.testing.assert_equal(values, original)
    assert np.isnan(timeline.values[:, 9:]).sum() == 240
    assert (timeline.values[:, 7:9] == 0).sum() == 2 * (24 - sleep_hours)


@pytest.mark.parametrize("window_hours", [30, 100])
def test_window_preserves_missing_days_padding_and_observation_mask(window_hours):
    values = np.concatenate([hourly_days(), hourly_days()])
    values[1, 10:] = np.nan
    timeline = tsfm.build_user_timeline(values, ["2020-01-01", "2020-01-03"], 19)
    expected = np.concatenate([
        values[0], np.full((24, 19), np.nan, dtype=np.float32), values[1, :10],
    ])
    history = expected[-window_hours:]
    expected_window = np.full((window_hours, 19), np.nan, dtype=np.float32)
    expected_window[-len(history):] = history
    example = tsfm.build_window(timeline, "synthetic", "age", "2020-01-03", window_hours, 19, None)
    np.testing.assert_equal(example.padding_mask.T, np.isfinite(expected_window))
    np.testing.assert_equal(
        np.where(example.padding_mask, example.window, np.nan).T, expected_window
    )
    assert (example.window[~example.padding_mask] == 0).all()


def test_empty_and_all_missing_histories():
    assert tsfm.build_user_timeline(np.empty((0, 24, 19)), [], 19) is None
    timeline = tsfm.build_user_timeline(
        np.full((1, 24, 19), np.nan), ["2020-01-01"], 19
    )
    assert not timeline.observed_hours.any()
    assert tsfm.build_window(timeline, "synthetic", "age", "2020-01-01", 48, 19, None) is None


@pytest.mark.parametrize("cls", [Chronos2, Toto])
@pytest.mark.parametrize("full_history", [True, False])
def test_default_cache_is_separate_from_legacy_cache(cls, full_history):
    encoder = cls()
    encoder.set_temporal_window(SimpleNamespace(is_full_history=full_history))
    variant = "from_raw_full_history" if full_history else "from_raw"
    old = Path("results") / f"{encoder.name}_embeddings" / variant
    assert encoder._resolve_cache_dir() == old / tsfm.INPUT_PREPROCESSING_VERSION
    assert encoder._resolve_cache_dir() != old


def write_cache(encoder, version):
    path = encoder._task_file("test", "age")
    attrs = {} if version is None else {"input_preprocessing": version}
    if version == tsfm.INPUT_PREPROCESSING_VERSION:
        attrs["cache_identity"] = json.dumps(encoder._task_identity("test", "age"))
    writer = tsfm._TaskWriter(
        path, 19, attrs
    )
    data = np.arange(19 * 4, dtype=np.float32).reshape(1, 19, 4)
    writer.append(data, ["synthetic"], ["2020-01-01"])
    writer.close()
    return path, data.mean(axis=1)[0]


@pytest.mark.parametrize("cls", [Chronos2, Toto])
@pytest.mark.parametrize("version", [None, "old_preprocessing"])
def test_legacy_cache_rejected_before_loading_or_writing(cls, version, tmp_path, monkeypatch):
    encoder = cls(cache_dir=str(tmp_path))
    path, _ = write_cache(encoder, version)
    before = path.read_bytes()
    monkeypatch.setattr(tsfm, "FINAL_TASKS", ["new_task", "age"])
    monkeypatch.setattr(
        encoder, "_ensure_loaded",
        lambda: pytest.fail("Legacy cache must be rejected before loading an encoder"),
    )
    with pytest.raises(ValueError, match="fresh cache_dir"):
        encoder._ensure_split("test")
    with pytest.raises(ValueError, match="fresh cache_dir"):
        encoder._load_task("test", "age")
    assert path.read_bytes() == before
    assert not encoder._task_file("test", "new_task").exists()


@pytest.mark.parametrize("cls", [Chronos2, Toto])
def test_versioned_cache_can_be_reused_without_encoder_loading(cls, tmp_path, monkeypatch):
    encoder = cls(cache_dir=str(tmp_path))
    _, expected = write_cache(encoder, tsfm.INPUT_PREPROCESSING_VERSION)
    monkeypatch.setattr(tsfm, "FINAL_TASKS", ["age"])
    monkeypatch.setattr(
        encoder, "_ensure_loaded",
        lambda: pytest.fail("A complete compatible cache should not load the encoder"),
    )
    encoder._ensure_split("test")
    np.testing.assert_equal(encoder._load_task("test", "age")["synthetic"], expected)


@pytest.mark.parametrize("cls", [Chronos2, Toto])
def test_extraction_passes_canonical_masks_and_records_cache_version(cls, tmp_path, monkeypatch):
    values = hourly_days()
    encoder = cls(cache_dir=str(tmp_path))
    encoder._grouped = {"test": ["synthetic"]}
    encoder._loader = SimpleNamespace(user_days=lambda uid: (values, ["2020-01-01"]))
    encoder._handle = object()
    encoder._window_hours = 48
    encoder.set_temporal_window(SimpleNamespace(weeks_after=lambda task: None))
    monkeypatch.setattr(tsfm, "_label_timestamp", lambda *args: __import__("pandas").Timestamp("2020-01-01"))
    monkeypatch.setattr(tsfm, "_DatasetPaths_root", lambda data_dir: Path("synthetic_hourly"))
    calls = []

    def run_batch(handle, examples, window_hours):
        calls.append(examples)
        assert window_hours == 48
        actual = np.where(examples[0].padding_mask, examples[0].window, np.nan).T
        expected = np.concatenate([np.full((24, 19), np.nan), values[0]])
        np.testing.assert_equal(actual, expected)
        return np.ones((1, 19, 4), dtype=np.float32)

    monkeypatch.setattr(encoder, "_run_batch", run_batch)
    encoder._build_split("test", ["age"])
    assert len(calls) == 1
    with h5py.File(encoder._task_file("test", "age"), "r") as f:
        assert f.attrs["input_preprocessing"] == tsfm.INPUT_PREPROCESSING_VERSION
    np.testing.assert_equal(encoder._load_task("test", "age")["synthetic"], np.ones(4))
