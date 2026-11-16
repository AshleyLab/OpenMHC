"""Exercise release ZIPs containing loose metadata and split sensor archives."""

import io
import json
import tarfile
import zipfile

import pytest

from openmhc import _dataset as dataset


def _tar_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("marker_prefix", ["", "data/"])
def test_full_download_layout(tmp_path, monkeypatch, dataverse_response, multipart, marker_prefix):
    """Actual public downloader assembles matching bytes and preserves revision."""
    marker = {
        "version": "full",
        "n_users": 11894,
        "dataset_revision": "2026-10-31T12:01:59.312287Z",
    }
    expected = {
        "processed/daily_hf/data.arrow": b"minute sensor test payload",
        "processed/daily_hourly_hf/data.arrow": b"hourly sensor test payload",
        "labels/last_labels.json": b"{}",
        "splits/sharable_users_seed42_2026.json": b"{}",
        "forecasting_sample_index/index.json": b"{}",
        "processed/daily_labels_lookup.parquet": b"lookup test payload",
        "processed/normalization_stats.json": b"{}",
    }
    minute = _tar_bytes({"daily_hf/data.arrow": expected["processed/daily_hf/data.arrow"]})
    hourly = _tar_bytes(
        {"daily_hourly_hf/data.arrow": expected["processed/daily_hourly_hf/data.arrow"]}
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.writestr(marker_prefix + "dataset_version.json", json.dumps(marker))
        for path, content in expected.items():
            if "/data.arrow" not in path:
                bundle.writestr("data/" + path, content)
        bundle.writestr("archives/daily_hourly_hf_full.tar.gz", hourly)
        if multipart:
            # Cross both gzip and tar boundaries; no temporary joined archive.
            pieces = [minute[:9], minute[9:41], minute[41:]]
            for index, piece in enumerate(pieces):
                bundle.writestr(f"archives/daily_hf_full.tar.gz.part-{index:02d}", piece)
        else:
            bundle.writestr("archives/daily_hf_full.tar.gz", minute)

    monkeypatch.setitem(dataset._VERSION_DOIS, "full", "doi:test/synthetic")
    dataverse_response(output.getvalue(), "full")
    root = dataset.download_dataset("full", dest=tmp_path / "download")
    assert dataset.read_dataset_marker(root) == marker
    assert {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} == {
        *expected,
        "dataset_version.json",
    }
    for path, content in expected.items():
        assert (root / path).read_bytes() == content
    assert not (root / "data").exists()
    assert not (root / "archives").exists()


def test_full_metadata_without_archives(tmp_path):
    """Metadata normalization also runs when no multipart directory exists."""
    (tmp_path / "data/labels").mkdir(parents=True)
    (tmp_path / "data/labels/last_labels.json").write_text("{}")
    dataset._post_process_full(tmp_path)
    assert (tmp_path / "labels/last_labels.json").read_text() == "{}"
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("suffixes", [["00", "02"], ["00", "0"], ["bad"]])
def test_invalid_multipart_preserves_inputs(tmp_path, suffixes):
    """Reject incomplete/ambiguous numbering before consuming any parts."""
    archives = tmp_path / "archives"
    archives.mkdir()
    for suffix in suffixes:
        (archives / f"daily_hf_full.tar.gz.part-{suffix}").write_bytes(b"test")
    with pytest.raises(ValueError, match="multipart"):
        dataset._post_process_full(tmp_path)
    assert len(list(archives.iterdir())) == len(suffixes)


def test_corrupt_multipart_preserves_inputs(tmp_path):
    """A failed extraction must retain the original parts for diagnosis/retry."""
    archives = tmp_path / "archives"
    archives.mkdir()
    part = archives / "daily_hf_full.tar.gz.part-00"
    part.write_bytes(b"not a gzip archive")
    with pytest.raises((tarfile.ReadError, OSError)):
        dataset._post_process_full(tmp_path)
    assert part.read_bytes() == b"not a gzip archive"
