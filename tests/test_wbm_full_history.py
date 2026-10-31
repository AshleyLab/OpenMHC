"""Weekly input selection and pooling across the participant's full history."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from downstream_evaluation.config import TemporalWindowConfig
from downstream_evaluation.data.provider import lookup_filename
from downstream_evaluation.models.wbm import WBMProbe
from openmhc._evaluate import _DatasetPaths


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "validation": [], "test": ["u1", "u2"]}))
    monkeypatch.setattr(_DatasetPaths, "from_root", lambda root: SimpleNamespace(root=tmp_path, splits_file=splits))
    frame = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "week_start": ["2020-01-01", "2024-01-01", "2024-01-01"],
        "n_valid_days": [5, 5, 5], "n_valid_hours": [-1, -1, -1],
        "Diabetes": [1, 1, 0], "age": [50., 50., 60.],
    })
    frame.to_parquet(processed / lookup_filename("weekly", full_history=True))
    old = frame.copy()
    old.loc[[1, 2], ["Diabetes", "age"]] = -1
    old.to_parquet(processed / "weekly_labels_lookup_stride7_windowed.parquet")
    return tmp_path


@pytest.mark.parametrize("task", ["Diabetes", "age"])
def test_late_weeks_contribute_to_pooling_and_coverage(dataset, task):
    model = WBMProbe(str(dataset))
    model.set_temporal_window(TemporalWindowConfig())
    td = model._weekly_td(task, "test")
    assert td.user_ids.tolist() == ["u1", "u2"]
    assert list(td.dates[0]) == ["2020-01-01", "2024-01-01"]
    model._wbm._by_key = {
        ("u1", "2020-01-01"): np.array([2.]),
        ("u1", "2024-01-01"): np.array([6.]),
        ("u2", "2024-01-01"): np.array([9.]),
    }
    model._wbm._dim = 1
    pooled, available = model._wbm.encode_cohort(task, td)
    np.testing.assert_array_equal(pooled, [[4.], [9.]])
    assert available.tolist() == [True, True]


def test_missing_full_history_lookup_does_not_use_windowed_file(dataset):
    (dataset / "processed" / lookup_filename("weekly", full_history=True)).unlink()
    with pytest.raises(FileNotFoundError, match="requires the full-history weekly lookup"):
        WBMProbe(str(dataset))._weekly_td("Diabetes", "test")


def test_capped_policy_is_rejected():
    with pytest.raises(ValueError, match="requires full-history"):
        WBMProbe().set_temporal_window(TemporalWindowConfig.windowed())
    with pytest.raises(ValueError, match="requires full history"):
        lookup_filename("weekly", full_history=False)


def test_release_wrapper_validates_temporal_policy():
    from openmhc.encoders import WBM
    model = WBM(model_path="unused.ckpt")
    model.set_temporal_window(TemporalWindowConfig())
    with pytest.raises(ValueError, match="requires full-history"):
        model.set_temporal_window(TemporalWindowConfig.windowed())


@pytest.mark.parametrize("grain", ["daily", "series"])
def test_daily_policy_selection_is_preserved(grain):
    assert lookup_filename(grain, full_history=True) == "daily_labels_lookup_full_history.parquet"
    assert lookup_filename(grain, full_history=False) == "daily_labels_lookup.parquet"
