"""Embedding provenance guards, exercised with synthetic files and no model downloads."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from downstream_evaluation.config import TemporalWindowConfig
from downstream_evaluation.models import _cache_identity as identity
from downstream_evaluation.models import tsfm
from downstream_evaluation.models.chronos2 import Chronos2
from downstream_evaluation.models.lsm2 import model as lsm2
from downstream_evaluation.models.toto import Toto
from downstream_evaluation.models.wbm import model as wbm


@pytest.fixture
def inputs(tmp_path):
    """Tiny source files with enough layout to exercise real provenance resolution."""
    root = tmp_path / "data"
    for directory in ("processed/daily_hourly_hf", "processed/daily_hf", "labels", "splits"):
        (root / directory).mkdir(parents=True)
    (root / "dataset_version.json").write_text('{"version": "xs"}')
    for name in ("daily_hourly_hf", "daily_hf"):
        (root / "processed" / name / "data.arrow").write_bytes(b"synthetic source")
    (root / "splits/split.json").write_text('{"test": ["u1"]}')
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"synthetic checkpoint A")
    return root, checkpoint, tmp_path / "cache"


def write_task(model, complete=True):
    """Write the same metadata and completion marker as production extraction."""
    path = model._task_file("test", "age")
    attrs = {
        "input_preprocessing": tsfm.INPUT_PREPROCESSING_VERSION,
        "cache_identity": json.dumps(model._task_identity("test", "age")),
    }
    writer = tsfm._TaskWriter(path, 19, attrs)
    writer.append(np.ones((1, 19, 4), dtype=np.float32), ["u1"], ["2020-01-01"])
    writer.close(complete=complete)
    return path


@pytest.mark.parametrize("cls", [Toto, Chronos2])
@pytest.mark.parametrize(
    "change",
    [
        "checkpoint",
        "checkpoint_bytes",
        "dataset",
        "source",
        "policy",
        "missing_identity",
        "incomplete",
    ],
)
def test_tsfm_rejects_mismatch_before_model_load_or_writes(inputs, cls, change, monkeypatch):
    """The same cache path cannot silently cross checkpoints, datasets or policies."""
    root, checkpoint, cache = inputs
    kwargs = dict(data_dir=str(root), checkpoint=str(checkpoint), cache_dir=str(cache))
    path = write_task(cls(**kwargs), complete=change != "incomplete")
    if change == "checkpoint":
        other = checkpoint.with_name("other.ckpt")
        other.write_bytes(b"checkpoint B")
        kwargs["checkpoint"] = str(other)
    elif change == "checkpoint_bytes":
        checkpoint.write_bytes(b"replaced checkpoint at the same path")
    elif change == "dataset":
        other = root.with_name("other_data")
        shutil.copytree(root, other)
        kwargs["data_dir"] = str(other)
    elif change == "source":
        (root / "processed/daily_hourly_hf/data.arrow").write_bytes(b"updated input")
    elif change == "missing_identity":
        with h5py.File(path, "a") as handle:
            del handle.attrs["cache_identity"]
    before = path.read_bytes()
    model = cls(**kwargs)
    if change == "policy":
        model.set_temporal_window(TemporalWindowConfig.windowed())
    monkeypatch.setattr(tsfm, "FINAL_TASKS", ["new_task", "age"])
    monkeypatch.setattr(model, "_ensure_loaded", lambda: pytest.fail("must reject before GPU load"))
    with pytest.raises(ValueError, match="fresh cache_dir"):
        model._ensure_split("test")
    with pytest.raises(ValueError, match="fresh cache_dir"):
        model._load_task("test", "age")
    assert path.read_bytes() == before
    assert not model._task_file("test", "new_task").exists()


@pytest.mark.parametrize(
    "cls, settings",
    [
        (Toto, {"lora_alpha": 16}),
        (Toto, {"base_checkpoint": "other/base"}),
        (Chronos2, {"prediction_length": 48}),
        (Chronos2, {"window_hours": 48}),
    ],
)
def test_tsfm_model_settings_are_part_of_identity(inputs, cls, settings):
    """Feature-affecting subclass parameters must invalidate a shared task cache."""
    root, checkpoint, cache = inputs
    kwargs = dict(data_dir=str(root), checkpoint=str(checkpoint), cache_dir=str(cache))
    write_task(cls(**kwargs))
    with pytest.raises(ValueError, match="settings"):
        cls(**kwargs, **settings)._load_task("test", "age")


@pytest.mark.parametrize("cls", [Toto, Chronos2])
def test_tsfm_reuses_canonical_dataset_alias_without_loading_model(inputs, cls, monkeypatch):
    """A symlink to the same dataset is a valid cache hit, even without model loading."""
    root, checkpoint, cache = inputs
    write_task(cls(data_dir=str(root), checkpoint=str(checkpoint), cache_dir=str(cache)))
    alias = root.with_name("alias")
    alias.symlink_to(root, target_is_directory=True)
    model = cls(data_dir=str(alias), checkpoint=str(checkpoint), cache_dir=str(cache))
    monkeypatch.setattr(model, "_ensure_loaded", lambda: pytest.fail("cache hit loaded model"))
    monkeypatch.setattr(tsfm, "FINAL_TASKS", ["age"])
    model._ensure_split("test")
    np.testing.assert_array_equal(model._load_task("test", "age")["u1"], np.ones(4))


@pytest.mark.parametrize("module, cls", [(lsm2, lsm2.LSM2), (wbm, wbm.WBM)])
@pytest.mark.parametrize(
    "change", ["none", "checkpoint", "dataset", "legacy", "partial", "metadata", "shape"]
)
def test_numpy_cache_reuse_and_rejection(inputs, module, cls, change, monkeypatch):
    """Daily/weekly caches are validated before either extraction or NumPy loading."""
    root, checkpoint, cache = inputs
    expected = module._cache_identity(str(checkpoint), str(root), 42)
    arrays = {
        "embeddings.npy": np.ones((1, 4)),
        "user_ids.npy": np.array(["u1"]),
        module.CACHE_FILES[-1]: np.array(["2020-01-01"]),
    }
    if change == "shape":
        arrays["embeddings.npy"] = np.ones((2, 4))
    identity.write_numpy_cache(cache, expected, arrays)
    if change == "checkpoint":
        checkpoint.write_bytes(b"different checkpoint")
    elif change == "dataset":
        other = root.with_name("other")
        shutil.copytree(root, other)
        root = other
    elif change == "legacy":
        (cache / identity.METADATA_FILE).unlink()
    elif change == "partial":
        (cache / "user_ids.npy").unlink()
    elif change == "metadata":
        (cache / identity.METADATA_FILE).write_text("{")
    before = {p.name: p.read_bytes() for p in cache.iterdir()}
    monkeypatch.setattr(
        module, f"extract_{cls.name}_embeddings", lambda **kw: pytest.fail("unexpected extraction")
    )
    model = cls(data_dir=str(root), checkpoint=str(checkpoint), cache_dir=str(cache))
    if change == "none":
        model._ensure_embeddings()
        np.testing.assert_array_equal(model._by_key["u1", "2020-01-01"], np.ones(4))
    else:
        if change != "shape":
            monkeypatch.setattr(
                np, "load", lambda *a, **kw: pytest.fail("loaded incompatible cache")
            )
        with pytest.raises(ValueError, match="fresh cache_dir"):
            model._ensure_embeddings()
    assert {p.name: p.read_bytes() for p in cache.iterdir()} == before


def test_wbm_normalization_override_invalidates_cache(inputs):
    """A release-bundled normalization file is also an extraction input."""
    root, checkpoint, cache = inputs
    stats = root / "stats.json"
    stats.write_text('{"mean": 0}')
    old = wbm._cache_identity(str(checkpoint), str(root), 42, str(stats))
    stats.write_text('{"mean": 1}')
    new = wbm._cache_identity(str(checkpoint), str(root), 42, str(stats))
    with pytest.raises(ValueError, match="settings"):
        identity.validate_identity(old, new, cache)


def test_local_checkpoint_directory_and_remote_references(inputs):
    """Hash local model directories; compare remote revisions without network access."""
    root, _, _ = inputs
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights A")
    old = identity.checkpoint_identity(str(checkpoint))
    weights.write_bytes(b"weights B")
    assert old != identity.checkpoint_identity(str(checkpoint))
    assert identity.checkpoint_identity("hf://org/repo@rev1") != identity.checkpoint_identity(
        "hf://org/repo@rev2"
    )
    assert identity.checkpoint_identity("Datadog/Toto-Open-Base-1.0") == {
        "reference": "Datadog/Toto-Open-Base-1.0"
    }


@pytest.mark.parametrize("module, cls", [(lsm2, lsm2.LSM2), (wbm, wbm.WBM)])
def test_numpy_extraction_records_identity_and_supports_a_warm_run(
    inputs, module, cls, monkeypatch
):
    """Exercise real extraction/writing/reading around a tiny deterministic CPU encoder."""
    import pandas as pd
    import torch

    from downstream_evaluation.data.provider import lookup_filename
    from openmhc._evaluate import _DatasetPaths

    root, checkpoint, cache = inputs
    paths = _DatasetPaths.from_root(str(root))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if module is lsm2:
        paths.splits_file.write_text(json.dumps({"train": [], "validation": [], "test": ["u1"]}))
        pd.DataFrame({"user_id": ["u1"], "date": ["2020-01-01"]}).to_parquet(
            root / "processed" / lookup_filename("daily", full_history=True)
        )
        fake = SimpleNamespace(
            patch_size=1,
            forward_encoder_dense=lambda x, mask: (
                torch.ones((len(x), 2, 4)),
                torch.zeros((len(x), 2)),
            ),
        )
        monkeypatch.setattr(module, "_load_lsm2_model", lambda *args: fake)
        monkeypatch.setattr(module, "_build_transforms", lambda *args: [])
        loader = SimpleNamespace(
            participant_minute=lambda *args: (np.ones((1, 19, 2)), ["2020-01-01"])
        )
    else:
        import data.datasets.indexed_week_dataset as indexed
        import data.processing.build_window_index as windows

        class TinyWeeks:
            def __len__(self):
                return 1

            def __getitem__(self, key):
                if isinstance(key, str):
                    return {"user_id": ["u1"], "week_start": ["2020-01-01"]}[key]
                return {}

        monkeypatch.setattr(indexed, "IndexedWeekDataset", lambda **kw: TinyWeeks())
        monkeypatch.setattr(windows, "load_window_index", lambda *args: None)
        monkeypatch.setattr(
            module, "_resolve_norm_stats", lambda *args: (np.zeros(19), np.ones(19))
        )
        monkeypatch.setattr(
            module, "_process_example", lambda *args: np.ones((168, 38), dtype=np.float32)
        )
        monkeypatch.setattr(
            module, "_load_wbm_encoder", lambda *args: lambda x: (None, torch.ones((len(x), 4)))
        )
        loader = SimpleNamespace(as_daily_rows=lambda: [])
    model = cls(data_dir=str(root), checkpoint=str(checkpoint), cache_dir=str(cache))
    model.set_loader(loader)
    model._ensure_embeddings()
    np.testing.assert_array_equal(model._by_key["u1", "2020-01-01"], np.ones(4))
    assert json.loads((cache / identity.METADATA_FILE).read_text())["model"] == cls.name
    monkeypatch.setattr(
        module,
        f"extract_{cls.name}_embeddings",
        lambda **kw: pytest.fail("warm run extracted again"),
    )
    fresh = cls(data_dir=str(root), checkpoint=str(checkpoint), cache_dir=str(cache))
    fresh._ensure_embeddings()
    np.testing.assert_array_equal(fresh._by_key["u1", "2020-01-01"], np.ones(4))


def test_injected_loader_must_match_recorded_dataset(inputs):
    """Reject an explicitly injected loader from a different dataset or resolution."""
    from downstream_evaluation.data.loader import DataLoader

    root, checkpoint, _ = inputs
    other = root.with_name("other")
    shutil.copytree(root, other)
    for model in (
        Toto(data_dir=str(root)),
        lsm2.LSM2(data_dir=str(root)),
        wbm.WBM(data_dir=str(root)),
    ):
        with pytest.raises(ValueError, match="Injected loader"):
            model.set_loader(DataLoader(str(other)))
    with pytest.raises(ValueError, match="resolution"):
        lsm2.LSM2(data_dir=str(root)).set_loader(DataLoader(str(root), resolution="hourly"))


def test_failed_tsfm_extraction_cannot_be_reused(inputs, monkeypatch):
    """A task file written before a later batch failure must remain incomplete."""
    import pandas as pd

    root, checkpoint, cache = inputs
    model = Toto(data_dir=str(root), checkpoint=str(checkpoint), cache_dir=str(cache), batch_size=1)
    model._grouped = {"test": ["u1", "u2"]}
    model._loader = SimpleNamespace(user_days=lambda uid: (np.ones((1, 24, 19)), ["2020-01-01"]))
    model._handle, model._window_hours = object(), 24
    monkeypatch.setattr(tsfm, "_label_timestamp", lambda *args: pd.Timestamp("2020-01-01"))

    def run_batch(handle, examples, window):
        if examples[0].user_id == "u2":
            raise RuntimeError("synthetic extraction failure")
        return np.ones((1, 19, 4))

    monkeypatch.setattr(model, "_run_batch", run_batch)
    with pytest.raises(RuntimeError, match="synthetic extraction failure"):
        model._build_split("test", ["age"])
    with pytest.raises(ValueError, match="Incomplete"):
        Toto(data_dir=str(root), checkpoint=str(checkpoint), cache_dir=str(cache))._load_task(
            "test", "age"
        )


@pytest.mark.parametrize("method", ["lsm2", "wbm", "toto", "chronos2"])
@pytest.mark.parametrize("custom_checkpoint", [False, True])
def test_public_config_forwards_cache_and_checkpoint(inputs, method, custom_checkpoint):
    """The documented CLI override must reach each actual encoder constructor."""
    from hydra import compose, initialize_config_dir

    from downstream_evaluation.config import DataConfig, MethodConfig
    from downstream_evaluation.hydra import cli, registry

    root, checkpoint, cache = inputs
    cli.register_configs()
    config_dir = Path(__file__).resolve().parents[1] / "configs/downstream"
    overrides = [f"method={method}", f"method.cache_dir={cache}"]
    if custom_checkpoint:
        overrides.append(f"method.checkpoint={checkpoint}")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(config_name="eval", overrides=overrides)
    model, _ = getattr(registry, f"_{method}")(
        MethodConfig(**config.method), DataConfig(data_dir=str(root))
    )
    encoder = model._wbm if method == "wbm" else model
    assert encoder._resolve_cache_dir() == cache
    if custom_checkpoint:
        actual = encoder.checkpoint if method in {"toto", "chronos2"} else encoder._checkpoint
        assert actual == str(checkpoint)


def test_public_wbm_wrapper_forwards_cache(inputs):
    """Released WBM checkpoints can also select a fresh cache through the public API."""
    from openmhc.encoders import WBM

    root, checkpoint, cache = inputs
    model = WBM(model_path=str(checkpoint), data_dir=str(root), cache_dir=str(cache))
    assert model._model._wbm._resolve_cache_dir() == cache
