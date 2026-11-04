"""Prevent stale dataset metadata from surviving a supposedly successful download."""

import io
import json
import zipfile

import numpy as np
import pytest

from openmhc import _dataset


def test_download_rejects_nonempty_root_before_network(tmp_path, monkeypatch):
    """A re-download must leave an existing root untouched and explain recovery."""
    labels = tmp_path / "labels"
    labels.mkdir()
    schema = labels / "label_types.json"
    schema.write_text('{"old": "continuous"}')

    def unexpected_request(*args, **kwargs):
        raise AssertionError("An existing dataset must be rejected before downloading")

    monkeypatch.setattr(_dataset.urllib.request, "urlopen", unexpected_request)
    with pytest.raises(FileExistsError, match="new directory"):
        _dataset.download_dataset("xs", dest=tmp_path)
    assert schema.read_text() == '{"old": "continuous"}'
    assert not (tmp_path / "dataset_version.json").exists()


@pytest.mark.parametrize("subdir", ["labels", "splits", "forecasting_sample_index", "imputation"])
def test_extraction_collision_fails_before_any_moves(tmp_path, subdir):
    """An extracted incoming release must not be silently mixed with old metadata."""
    existing = tmp_path / subdir
    incoming = tmp_path / "data" / subdir
    existing.mkdir()
    incoming.mkdir(parents=True)
    (existing / "value.json").write_text("old")
    (incoming / "value.json").write_text("new")
    archive = tmp_path / "archives/daily_hf"
    archive.mkdir(parents=True)
    (archive / "sentinel").write_text("incoming")
    with pytest.raises(FileExistsError, match="fresh directory"):
        _dataset._post_process_xs(tmp_path)
    assert (existing / "value.json").read_text() == "old"
    assert (incoming / "value.json").read_text() == "new"
    assert (archive / "sentinel").exists()
    assert not (tmp_path / "processed").exists()


@pytest.mark.parametrize("precreate", [False, True])
def test_download_installs_complete_schemas_in_fresh_root(tmp_path, monkeypatch, precreate):
    """Both a new destination and an empty directory receive the incoming schemas."""
    root = tmp_path / "download"
    if precreate:
        root.mkdir()
    schemas = {
        "label_types.json": {"age": {"type": "continuous", "target": True}},
        "ordinal_dictionary.json": {},
        "validity_config.json": {},
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for filename, value in schemas.items():
            archive.writestr("data/labels/" + filename, json.dumps(value))
    payload = stream.getvalue()
    monkeypatch.setattr(_dataset.urllib.request, "urlopen", lambda request: io.BytesIO(payload))
    assert _dataset.download_dataset("xs", dest=root) == root
    for filename, value in schemas.items():
        assert json.loads((root / "labels" / filename).read_text()) == value
    assert not (root / "data/labels").exists()
    assert _dataset.read_dataset_marker(root)["version"] == "xs"


@pytest.mark.parametrize("version", ["xs", "full"])
@pytest.mark.parametrize("layout", ["data", "canonical", "nested_zip"])
def test_download_installs_masks_at_loader_path(tmp_path, monkeypatch, version, layout):
    """Install every released mask unchanged where the evaluation loader looks."""
    from openmhc._constants import MASKING_SCENARIOS
    from openmhc._evaluate import _dataset_masks_dir

    root = tmp_path / "download"
    relative = _dataset_masks_dir(root, version).relative_to(root)
    mask = io.BytesIO()
    np.savez_compressed(
        mask,
        indices=np.array([0]),
        masks_packed=np.array([170], dtype=np.uint8),
        shape=np.array([1, 1, 8]),
        user_ids=np.array(["fixture_user"]),
        dates=np.array(["2026-01-01"]),
    )
    mask_bytes = mask.getvalue()
    bundle = io.BytesIO()
    prefix = "" if layout == "canonical" else "data/"
    with zipfile.ZipFile(bundle, "w") as archive:
        for split in ["val", "test"]:
            for scenario in MASKING_SCENARIOS:
                archive.writestr(f"{prefix}{relative}/{split}/{scenario}.npz", mask_bytes)
    payload = bundle.getvalue()
    if layout == "nested_zip":
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w") as archive:
            archive.writestr("release.zip", payload)
        payload = outer.getvalue()
    # Exercise the Full installer without claiming that its public DOI is enabled.
    monkeypatch.setitem(_dataset._VERSION_DOIS, version, "doi:test/fixture")
    monkeypatch.setattr(_dataset.urllib.request, "urlopen", lambda request: io.BytesIO(payload))

    assert _dataset.download_dataset(version, dest=root) == root
    installed = _dataset_masks_dir(root, version)
    for split in ["val", "test"]:
        for scenario in MASKING_SCENARIOS:
            assert (installed / split / f"{scenario}.npz").read_bytes() == mask_bytes
    assert not (root / "data/imputation").exists()
    assert not (root / "release.zip").exists()
    assert _dataset.read_dataset_marker(root)["version"] == version


@pytest.mark.parametrize("with_archives", [False, True])
def test_full_mask_collision_preserves_both_copies(tmp_path, with_archives):
    """Full installation must not overwrite or mix masks from two releases."""
    existing = tmp_path / "imputation"
    incoming = tmp_path / "data/imputation"
    existing.mkdir()
    incoming.mkdir(parents=True)
    (existing / "sentinel").write_bytes(b"old")
    (incoming / "sentinel").write_bytes(b"new")
    if with_archives:
        parts = tmp_path / "archives/daily_hf_full.tar.gz.part-00"
        parts.parent.mkdir()
        parts.write_bytes(b"untouched archive")
    with pytest.raises(FileExistsError, match="fresh directory"):
        _dataset._post_process_full(tmp_path)
    assert (existing / "sentinel").read_bytes() == b"old"
    assert (incoming / "sentinel").read_bytes() == b"new"
    if with_archives:
        assert parts.read_bytes() == b"untouched archive"
