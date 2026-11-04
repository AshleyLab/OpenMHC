"""Dataset masks must preserve sample identity across release date variants."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from imputation_evaluation.masking.generator import (
    MaskCache,
    ScenarioMasks,
    pack_masks,
)
from openmhc import _evaluate


@pytest.fixture
def cache():
    """Build byte-aligned masks with explicit participant/date identities."""
    packed, shape = pack_masks(np.array([[[1, 0, 1, 0, 1, 0, 1, 0]], [[0, 1, 0, 1, 0, 1, 0, 1]]]))
    result = MaskCache()
    result.add(
        "test",
        "random_noise",
        ScenarioMasks(
            indices=np.array([0, 2]),
            masks_packed=packed,
            shape=shape,
            user_ids=np.array(["a", "c"]),
            dates=np.array(["2020-01-01", "2020-01-03"]),
        ),
    )
    return result


def test_roundtrip_preserves_masks_and_sample_keys(cache, tmp_path):
    """Saving and loading preserves masks and their sample identities."""
    cache.save(tmp_path)
    restored = MaskCache.load(tmp_path, ["random_noise"], ["test"])
    data = {"user_id": ["a", "b", "c"], "date": ["2020-01-01", "2020-01-02", "2020-01-03"]}
    restored.check_sample_keys(data, {"test": [0, 1, 2]})
    before, after = cache.get("test", "random_noise"), restored.get("test", "random_noise")
    for field in ("indices", "masks_packed", "user_ids", "dates"):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    np.testing.assert_array_equal(before.get_mask(2), after.get_mask(2))
    _evaluate._raise_if_lfs_pointer_masks(tmp_path)  # A small real NPZ is valid.


@pytest.mark.parametrize(
    "change", ["date", "order", "missing", "index", "negative", "duplicate", "float", "length"]
)
def test_wrong_release_masks_are_rejected(cache, change):
    """Reject mismatched dates, ordering, metadata and sample bounds."""
    data = {"user_id": ["a", "b", "c"], "date": ["2020-01-01", "2020-01-02", "2020-01-03"]}
    masks = cache.get("test", "random_noise")
    if change == "date":
        data["date"][0] = "2020-01-08"
    elif change == "order":
        data["user_id"] = ["c", "b", "a"]
    elif change == "missing":
        masks.dates = None
    elif change == "index":
        masks.indices[1] = 3
    elif change == "negative":
        masks.indices[0] = -1
    elif change == "duplicate":
        masks.indices[1] = 0
    elif change == "float":
        masks.indices = np.array([0.0, 2.0])
    else:
        masks.indices = np.array([0])
    with pytest.raises(ValueError, match="Mask cache"):
        cache.check_sample_keys(data, {"test": [0, 1, 2]})


@pytest.mark.parametrize("version", ["full", "xs"])
def test_public_api_selects_masks_from_requested_root(tmp_path, monkeypatch, version):
    """Switching dataset roots also switches the selected mask directory."""
    import imputation_evaluation.runner as runner

    class ReachedRunner(Exception):
        pass

    monkeypatch.setattr(_evaluate, "_ensure_labels_env", lambda _: None)
    for name in ("unshifted", "shifted"):
        root = tmp_path / name
        mask_dir = _evaluate._dataset_masks_dir(root, version)
        mask_dir.mkdir(parents=True)
        (root / "processed/daily_hf").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "splits").mkdir()
        count = _evaluate.EXPECTED_N_USERS[version]
        (root / "dataset_version.json").write_text(
            json.dumps({"version": version, "n_users": count})
        )
        (root / "splits" / _evaluate._SPLIT_FILENAMES[version]).write_text(
            json.dumps({"train": [f"user-{i}" for i in range(count)], "val": [], "test": []})
        )

        def check_config(config, **kwargs):
            assert config.masking.masks_file == str(mask_dir)
            raise ReachedRunner

        monkeypatch.setattr(runner, "run_eval", check_config)
        # An explicit dataset root takes precedence over a different environment root.
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "other_dataset"))
        with pytest.raises(ReachedRunner):
            _evaluate.evaluate_imputation(
                SimpleNamespace(impute=lambda x: x),
                version=version,
                data_dir=root,
                masking_scenarios=["random_noise"],
            )
        # The same resolution works when the root comes only from the environment.
        monkeypatch.setenv("MHC_DATA_DIR", str(root))
        with pytest.raises(ReachedRunner):
            _evaluate.evaluate_imputation(SimpleNamespace(impute=lambda x: x), version=version)


@pytest.mark.parametrize("version", ["full", "xs"])
def test_canonical_run_requires_dataset_masks(tmp_path, monkeypatch, version):
    """Missing release masks fail before evaluation starts."""
    monkeypatch.setattr(_evaluate, "_ensure_labels_env", lambda _: None)
    paths = SimpleNamespace(
        root=tmp_path, version=version, label_paths=None, require=lambda *args: None
    )
    monkeypatch.setattr(_evaluate._DatasetPaths, "resolve", lambda *args, **kwargs: paths)
    with pytest.raises(FileNotFoundError, match="accompanying this dataset"):
        _evaluate.evaluate_imputation(
            SimpleNamespace(impute=lambda x: x), version=version, data_dir=tmp_path
        )


@pytest.mark.parametrize("options", [{"seed": 7}, {"n_days": 7}, {"max_samples": 2}])
def test_noncanonical_xs_generates_masks_for_its_selected_samples(tmp_path, monkeypatch, options):
    """Custom XS runs must not accidentally reuse the canonical release cache."""
    import imputation_evaluation.runner as runner

    class ReachedRunner(Exception):
        pass

    monkeypatch.setattr(_evaluate, "_ensure_labels_env", lambda _: None)
    paths = SimpleNamespace(
        root=tmp_path,
        version="xs",
        daily_hf=tmp_path / "processed/daily_hf",
        splits_file=tmp_path / "splits.json",
        label_paths=None,
        require=lambda *args: None,
    )
    monkeypatch.setattr(_evaluate._DatasetPaths, "resolve", lambda *args, **kwargs: paths)
    # Even if the canonical cache exists, these runs must generate their own masks.
    _evaluate._dataset_masks_dir(tmp_path, "xs").mkdir(parents=True)

    def check_config(config, **kwargs):
        assert config.masking.masks_file is None
        raise ReachedRunner

    monkeypatch.setattr(runner, "run_eval", check_config)
    with pytest.raises(ReachedRunner):
        _evaluate.evaluate_imputation(
            SimpleNamespace(impute=lambda x: x), version="xs", data_dir=tmp_path, **options
        )


def test_missing_scenario_file_is_not_silently_regenerated(cache, tmp_path):
    """Require every requested released mask file to exist."""
    cache.save(tmp_path)
    with pytest.raises(FileNotFoundError, match="temporal_slice.npz"):
        MaskCache.load(tmp_path, ["random_noise", "temporal_slice"], ["test"])


def test_sample_identity_uses_split_local_indices(cache):
    """The split may select non-contiguous rows from the underlying HF store."""
    data = {
        "user_id": ["unused", "c", "a", "unused", "b"],
        "date": ["2020-02-01", "2020-01-03", "2020-01-01", "2020-02-02", "2020-01-02"],
    }
    cache.check_sample_keys(data, {"test": [2, 4, 1]})


@pytest.mark.parametrize("wrong_date", [False, True])
def test_runner_checks_loaded_masks_before_fitting(cache, tmp_path, monkeypatch, wrong_date):
    """Validate loaded sample keys before a method can fit."""
    import imputation_evaluation.runner as runner

    class ReachedFit(Exception):
        pass

    cache.save(tmp_path)
    data = {"user_id": ["a", "b", "c"], "date": ["2020-01-01", "2020-01-02", "2020-01-03"]}
    if wrong_date:
        data["date"][0] = "2020-01-08"
    loaded = SimpleNamespace(hf_dataset=data, split_indices={"test": [0, 1, 2]}, train_loader=[])
    monkeypatch.setattr(
        runner, "ImputationDataLoader", lambda _: SimpleNamespace(load_splits=lambda **kw: loaded)
    )
    monkeypatch.setattr(
        runner, "create_mask_generators", lambda _: [SimpleNamespace(name="random_noise")]
    )

    def fit(_):
        assert not wrong_date, "The method must not fit with masks from the wrong date variant"
        raise ReachedFit

    config = SimpleNamespace(
        data=SimpleNamespace(batch_size=1, num_workers=0, pin_memory=False),
        masking=SimpleNamespace(masks_file=str(tmp_path)),
        evaluation=SimpleNamespace(eval_splits=["test"]),
    )
    with pytest.raises(ValueError if wrong_date else ReachedFit):
        runner.run_eval(config, SimpleNamespace(name="test", fit=fit))


def test_migration_shifts_dates_and_remaps_without_changing_bits(cache):
    """Move dates and indices while preserving the exact masking pattern."""
    import runpy
    from pathlib import Path

    remap = runpy.run_path(str(Path(__file__).parents[1] / "scripts/migrate_imputation_masks.py"))[
        "remap_masks"
    ]
    original = cache.get("test", "random_noise")
    source = [("a", "2020-01-01"), ("b", "2020-01-02"), ("c", "2020-01-03")]
    target = [("c", "2019-12-27"), ("b", "2020-01-02"), ("a", "2020-01-15")]
    result = remap(original, source, target, {"a": 2, "c": -1})
    np.testing.assert_array_equal(result.indices, [2, 0])
    np.testing.assert_array_equal(result.dates, ["2020-01-15", "2019-12-27"])
    np.testing.assert_array_equal(result.masks_packed, original.masks_packed)
    np.testing.assert_array_equal(result.get_mask(0), original.get_mask(2))
    np.testing.assert_array_equal(original.dates, ["2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError, match="offset table"):
        remap(original, source, target, {"a": 2})
    with pytest.raises(ValueError, match="absent"):
        remap(original, source, target, {"a": 1, "c": -1})
    with pytest.raises(ValueError, match="source dataset"):
        remap(original, list(reversed(source)), target, {"a": 2, "c": -1})


def test_xs_extraction_preserves_full_mask_rows(cache):
    """XS selects Full rows and changes only their split-local indices."""
    import runpy
    from pathlib import Path

    subset = runpy.run_path(str(Path(__file__).parents[1] / "scripts/migrate_imputation_masks.py"))[
        "subset_masks"
    ]
    full = cache.get("test", "random_noise")
    xs = subset(full, [("c", "2020-01-03"), ("unmasked", "2020-02-01")])
    np.testing.assert_array_equal(xs.indices, [0])
    np.testing.assert_array_equal(xs.user_ids, ["c"])
    np.testing.assert_array_equal(xs.get_mask(0), full.get_mask(2))
    assert xs.get_mask(1) is None
    assert xs.shape == (1, *full.shape[1:])
    empty = subset(full, [("unmasked", "2020-02-01")])
    assert empty.n_applicable == 0
    assert empty.masks_packed.size == 0
