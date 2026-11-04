"""Verify release preparation without generating replacement mask patterns."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from imputation_evaluation.masking.generator import ScenarioMasks
from openmhc._constants import MASKING_SCENARIOS
from openmhc._evaluate import _dataset_masks_dir

SCRIPTS = Path(__file__).parents[1] / "scripts"


@pytest.fixture
def prepared_source(tmp_path):
    """Provide all twelve files with distinct mask patterns and sample keys."""
    root = tmp_path / "full"
    masks = _dataset_masks_dir(root, "full")
    keys = {}
    for split in ("val", "test"):
        keys[split] = [(f"{split}_{user}", "2020-01-01") for user in ("a", "b", "c")]
        (masks / split).mkdir(parents=True)
        for number, scenario in enumerate(MASKING_SCENARIOS):
            ScenarioMasks(
                indices=np.array([0, 2]),
                masks_packed=np.array([number + 1, number + 10], dtype=np.uint8),
                shape=(2, 1, 8),
                user_ids=np.array([keys[split][i][0] for i in (0, 2)]),
                dates=np.array(["2020-01-01", "2020-01-01"]),
            ).save(masks / split / f"{scenario}.npz")
    return root, masks, keys


@pytest.mark.parametrize("problem", [None, "existing_output", "missing_file", "wrong_dates"])
def test_full_preparation_validates_before_writing(tmp_path, monkeypatch, prepared_source, problem):
    """The CLI preserves bits while shifting dates, and rejects incomplete inputs."""
    root, masks, keys = prepared_source
    module = runpy.run_path(str(SCRIPTS / "migrate_imputation_masks.py"))
    target = tmp_path / "target"
    output = tmp_path / "output"
    # Target sample order differs from the source order.
    target_keys = {
        split: [(user, "2020-01-08") for user, _ in reversed(rows)] for split, rows in keys.items()
    }
    if problem == "existing_output":
        output.mkdir()
        (output / "sentinel").write_text("unchanged")
    elif problem == "missing_file":
        (masks / "test" / f"{MASKING_SCENARIOS[-1]}.npz").unlink()
    elif problem == "wrong_dates":
        target_keys["test"][0] = ("test_c", "2020-01-09")
    monkeypatch.setitem(
        module["main"].__globals__,
        "sample_keys",
        lambda path, version: keys if path == root else target_keys,
    )
    offsets = tmp_path / "offsets.parquet"
    users = [user for rows in keys.values() for user, _ in rows]
    pd.DataFrame({"user_id": users, "offset_weeks": [1] * len(users)}).to_parquet(offsets)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_imputation_masks.py",
            "--source-data-dir",
            str(root),
            "--target-data-dir",
            str(target),
            "--source-masks",
            str(masks),
            "--output",
            str(output),
            "--version",
            "full",
            "--offsets",
            str(offsets),
        ],
    )
    if problem:
        with pytest.raises((FileExistsError, FileNotFoundError, ValueError)):
            module["main"]()
        if problem == "existing_output":
            assert (output / "sentinel").read_text() == "unchanged"
            assert len(list(output.iterdir())) == 1
        else:
            assert not output.exists()
        return
    module["main"]()
    assert len(list(output.rglob("*.*"))) == 12  # No private offset table is copied.
    for file in masks.glob("*/*.npz"):
        before = ScenarioMasks.load(file)
        after = ScenarioMasks.load(output / file.relative_to(masks))
        np.testing.assert_array_equal(after.masks_packed, before.masks_packed)
        np.testing.assert_array_equal(after.indices, [2, 0])
        np.testing.assert_array_equal(after.user_ids, before.user_ids)
        np.testing.assert_array_equal(after.dates, ["2020-01-08", "2020-01-08"])
        assert after.shape == before.shape


@pytest.mark.parametrize("problem", [None, "existing_output", "missing_file", "wrong_dates"])
def test_xs_preparation_validates_before_writing(tmp_path, monkeypatch, prepared_source, problem):
    """The XS CLI selects Full rows and rejects mismatched releases before output."""
    root, masks, keys = prepared_source
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = runpy.run_path(str(SCRIPTS / "precompute_xs_masks.py"))
    xs = tmp_path / "xs"
    output = _dataset_masks_dir(xs, "xs")
    xs_keys = {split: list(reversed(rows)) for split, rows in keys.items()}
    if problem == "existing_output":
        output.mkdir(parents=True)
        (output / "sentinel").write_text("unchanged")
    elif problem == "missing_file":
        (masks / "test" / f"{MASKING_SCENARIOS[-1]}.npz").unlink()
    elif problem == "wrong_dates":
        xs_keys["test"][0] = ("test_c", "2020-01-08")
    monkeypatch.setattr(
        module["_DatasetPaths"], "resolve", lambda path, version: SimpleNamespace(root=path)
    )
    monkeypatch.setitem(
        module["main"].__globals__,
        "sample_keys",
        lambda path, version: keys if path == root else xs_keys,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precompute_xs_masks.py",
            "--full-data-dir",
            str(root),
            "--data-dir",
            str(xs),
        ],
    )
    if problem:
        with pytest.raises((FileExistsError, FileNotFoundError, ValueError)):
            module["main"]()
        if problem == "existing_output":
            assert (output / "sentinel").read_text() == "unchanged"
            assert len(list(output.iterdir())) == 1
        else:
            assert not output.exists()
        return
    module["main"]()
    assert len(list(output.rglob("*.npz"))) == 12
    for file in masks.glob("*/*.npz"):
        before = ScenarioMasks.load(file)
        after = ScenarioMasks.load(output / file.relative_to(masks))
        np.testing.assert_array_equal(after.indices, [0, 2])
        np.testing.assert_array_equal(after.masks_packed, before.masks_packed[::-1])
        np.testing.assert_array_equal(after.user_ids, before.user_ids[::-1])
        np.testing.assert_array_equal(after.dates, before.dates[::-1])


def test_preparation_keys_match_evaluator_filters_and_order(tmp_path, monkeypatch):
    """Preparation and evaluation must agree after wear-time and variance filtering."""
    from datasets import Dataset

    import data.filters.daily_filters as filters
    from imputation_evaluation.config import DataConfig
    from imputation_evaluation.data.data_loader import ImputationDataLoader

    module = runpy.run_path(str(SCRIPTS / "migrate_imputation_masks.py"))
    hf = tmp_path / "processed/daily_hf"
    Dataset.from_dict(
        {
            "user_id": ["a", "b", "a", "b", "c", "a"],
            "date": [
                "2020-01-03",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-01",
                "2020-01-02",
            ],
            "total_nonwear_minutes": [0, 0, 721, 0, 0, 720],
            "channel_variance": [
                [10000.0] * 19,
                [0.0] * 19,
                [10000.0] * 19,
                [float("nan")] * 19,
                [10000.0] * 19,
                [10000.0] * 19,
            ],
        }
    ).save_to_disk(hf)
    split_file = tmp_path / "splits/sharable_users_seed42_2026_xs.json"
    split_file.parent.mkdir()
    split_file.write_text(json.dumps({"train": ["c"], "validation": ["a"], "test": ["b"]}))
    monkeypatch.setattr(
        filters, "hf_cache_path", lambda label, ds: str(tmp_path / f"{label}.arrow")
    )
    loader = ImputationDataLoader(DataConfig(daily_hf_dir=str(hf), split_file=str(split_file)))
    indices, users, dates = loader.load_split_indices()
    expected = {split: [(users[i], dates[i]) for i in indices[split]] for split in ("val", "test")}
    assert expected == {
        "val": [("a", "2020-01-03"), ("a", "2020-01-02")],
        "test": [("b", "2020-01-02")],
    }
    assert module["sample_keys"](tmp_path, "xs") == expected
