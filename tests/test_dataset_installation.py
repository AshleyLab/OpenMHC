"""Prevent stale dataset metadata from surviving a supposedly successful download."""

import io
import json
import zipfile

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


@pytest.mark.parametrize("subdir", ["labels", "splits", "forecasting_sample_index"])
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
