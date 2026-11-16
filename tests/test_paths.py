"""Path-resolution tests for the openmhc API.

These verify that the API and the dataset downloader agree on where data
lives. Without this, ``download_dataset`` would write to one location and
``evaluate_*`` would look in another.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import openmhc
from openmhc._dataset import (
    DATASET_VERSION_FILENAME,
    data_dir,
    write_dataset_marker,
)
from openmhc._evaluate import _DatasetPaths
from openmhc._label_paths import LabelPaths


def _write_minimal_split(root: Path, version: str = "full") -> None:
    """Build a minimal version-tagged dataset root.

    Writes both the split file (with the canonical user count for the
    requested version) and the ``dataset_version.json`` marker, so
    ``_DatasetPaths.resolve(root, version=...)`` is happy without needing
    the full payload on disk.
    """
    expected = {"full": 11894, "xs": 593}[version]
    name = {
        "full": "sharable_users_seed42_2026.json",
        "xs": "sharable_users_seed42_2026_xs.json",
    }[version]
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    users = [f"u{i}" for i in range(expected)]
    payload = {"train": users, "validation": [], "test": []}
    (split_dir / name).write_text(json.dumps(payload))
    write_dataset_marker(root, version=version, n_users=expected)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Clear dataset / labels env vars so each test starts from a known state.

    Without this, tests that set ``LABELS_DATA_PATH`` / ``CONTEXT_LABELS_PATH``
    would leak into sibling test modules (e.g. ``tests/test_labels_api.py``)
    and make ordering matter.
    """
    for var in (
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
        monkeypatch.delenv(var, raising=False)
    yield


class TestDataDir:
    """Resolution order and normalization of ``data_dir``."""

    def test_raises_when_not_configured(self):
        """With no argument and no env var, ``data_dir`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            data_dir()

    def test_explicit_override_wins(self, tmp_path):
        """An explicit path argument is resolved and returned."""
        result = data_dir(tmp_path / "explicit")
        assert result == (tmp_path / "explicit").resolve()

    def test_env_var_override(self, monkeypatch, tmp_path):
        """When unset explicitly, ``MHC_DATA_DIR`` supplies the root."""
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "from-env"))
        result = data_dir()
        assert result == (tmp_path / "from-env").resolve()

    def test_explicit_wins_over_env(self, monkeypatch, tmp_path):
        """An explicit argument takes precedence over ``MHC_DATA_DIR``."""
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "from-env"))
        result = data_dir(tmp_path / "explicit")
        assert result == (tmp_path / "explicit").resolve()

    def test_tilde_expansion(self):
        """A leading ``~`` is expanded to the user's home directory."""
        result = data_dir("~/some-mhc-data")
        assert result == (Path.home() / "some-mhc-data").resolve()


class TestDatasetPaths:
    """Confirm the eval pipeline routes through ``data_dir`` consistently."""

    def test_all_subpaths_under_root(self, tmp_path):
        """Every resolved dataset subpath sits under the configured root."""
        _write_minimal_split(tmp_path, version="full")
        paths = _DatasetPaths.resolve(tmp_path, version="full")
        assert paths.daily_hourly_hf == tmp_path / "processed" / "daily_hourly_hf"
        assert paths.daily_hf == tmp_path / "processed" / "daily_hf"
        assert paths.window_index == tmp_path / "processed" / "window_index_w7_s7_d5.parquet"
        assert (
            paths.weekly_labels_lookup
            == tmp_path / "processed" / "weekly_labels_lookup_stride7_full_history.parquet"
        )
        assert paths.splits_file == tmp_path / "splits" / "sharable_users_seed42_2026.json"
        assert paths.norm_stats == tmp_path / "processed" / "normalization_stats_hourly.json"
        assert paths.clip_dates == tmp_path / "labels" / "clip_dates.json"
        assert paths.labels_dir == tmp_path / "labels"

    def test_env_override_propagates(self, monkeypatch, tmp_path):
        """``MHC_DATA_DIR`` becomes the resolved root for the eval paths."""
        _write_minimal_split(tmp_path, version="full")
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path))
        paths = _DatasetPaths.resolve(version="full")
        assert paths.root == tmp_path.resolve()
        assert paths.daily_hf.parent.parent == tmp_path.resolve()

    def test_explicit_override_propagates(self, tmp_path):
        """An explicit root argument becomes the resolved ``paths.root``."""
        root = tmp_path / "explicit"
        _write_minimal_split(root, version="full")
        paths = _DatasetPaths.resolve(root, version="full")
        assert paths.root == (tmp_path / "explicit").resolve()

    def test_version_required(self, tmp_path):
        """Resolving without a ``version`` argument raises ``ValueError``."""
        _write_minimal_split(tmp_path, version="full")
        with pytest.raises(ValueError, match="version='xs' or version='full'"):
            _DatasetPaths.resolve(tmp_path)

    def test_marker_required(self, tmp_path):
        """A missing ``dataset_version.json`` marker raises ``FileNotFoundError``."""
        # split file present but no marker
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        (split_dir / "sharable_users_seed42_2026.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match=DATASET_VERSION_FILENAME):
            _DatasetPaths.resolve(tmp_path, version="full")

    def test_marker_version_mismatch_rejected(self, tmp_path):
        """A marker version differing from the requested version is rejected."""
        _write_minimal_split(tmp_path, version="xs")
        with pytest.raises(ValueError, match="is version 'xs'"):
            _DatasetPaths.resolve(tmp_path, version="full")

    def test_split_user_count_mismatch_rejected(self, tmp_path):
        """A split-file user count that disagrees with the marker is rejected."""
        # Marker claims full (11894 users), but split file holds only 5.
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        (split_dir / "sharable_users_seed42_2026.json").write_text(
            '{"train": ["u0","u1","u2"], "validation": ["u3"], "test": ["u4"]}'
        )
        write_dataset_marker(tmp_path, version="full")  # n_users defaults to 11894
        with pytest.raises(ValueError, match="contains 5 users"):
            _DatasetPaths.resolve(tmp_path, version="full")

    def test_from_root_trusts_marker_without_version_arg(self, tmp_path):
        """Internal callers use ``from_root``: no version arg, version from the marker.

        Regression guard — the bundled models and the loader resolve paths this way
        (they get an already-validated root). A version-required ``resolve`` broke the
        whole downstream track end-to-end. ``from_root`` needs only the marker, not the
        split file.
        """
        write_dataset_marker(tmp_path, version="xs", n_users=593)  # marker only, no split
        paths = _DatasetPaths.from_root(tmp_path)
        assert paths.version == "xs"  # trusted from the marker, not asserted by the caller
        # daily_labels_lookup is the field the version merge dropped; the linear baseline
        # (and others) read it.
        assert (
            paths.daily_labels_lookup == tmp_path / "processed" / "daily_labels_lookup.parquet"
        )

    def test_from_root_matches_resolve(self, tmp_path):
        """``from_root`` and ``resolve(version=...)`` derive identical sub-paths."""
        _write_minimal_split(tmp_path, version="full")
        assert _DatasetPaths.from_root(tmp_path) == _DatasetPaths.resolve(tmp_path, version="full")


class TestLabelsEnvWiring:
    """Evaluation configures paths without persisting them in the environment."""

    def test_configures_payloads_without_writing_environment(self, tmp_path, write_label_schemas):
        """Configure all participant payloads without persisting environment defaults."""
        from labels import api
        from openmhc._evaluate import _ensure_labels_env

        write_label_schemas(tmp_path)
        before = dict(os.environ)
        paths = LabelPaths.resolve(tmp_path)
        _ensure_labels_env(paths)
        assert dict(os.environ) == before
        assert api.STORE.paths.labels == paths.labels
        assert api.STORE.paths.context == paths.context
        assert api.STORE.paths.enrollment == paths.enrollment
        assert api.STORE.paths.validity == paths.validity
        assert api.STORE.paths.daily == paths.daily

    def test_respects_user_overrides(self, monkeypatch, tmp_path, write_label_schemas):
        """Retain explicit overrides while resolving remaining files from the root."""
        monkeypatch.setenv("LABELS_DATA_PATH", "/somewhere/else.json")
        monkeypatch.setenv("CONTEXT_LABELS_PATH", "/somewhere/ctx.json")
        monkeypatch.setenv("ENROLLMENT_DATA_PATH", "/somewhere/enrollment.json")
        from labels import api
        from openmhc._evaluate import _ensure_labels_env

        write_label_schemas(tmp_path)
        before = dict(os.environ)
        _ensure_labels_env(LabelPaths.resolve(tmp_path))
        assert dict(os.environ) == before
        assert api.STORE.paths.labels == Path("/somewhere/else.json")
        assert api.STORE.paths.context == Path("/somewhere/ctx.json")
        assert api.STORE.paths.enrollment == Path("/somewhere/enrollment.json")
        assert api.STORE.paths.validity == tmp_path / "labels" / "label_validity.json"


class TestLabelsApiResolution:
    """Resolve schemas and payloads from the selected dataset."""

    def test_all_label_files_resolve_from_dataset_root(
        self, monkeypatch, tmp_path, write_label_schemas
    ):
        """Configure every schema and payload from MHC_DATA_DIR."""
        folder = write_label_schemas(tmp_path)
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path))
        from labels import api

        api.configure()
        assert api._PATHS.labels == folder / "last_labels.json"
        assert api._PATHS.context == folder / "context_labels.json"
        assert api._PATHS.enrollment == folder / "enrollment_info.json"
        assert api._PATHS.validity == folder / "label_validity.json"
        assert api._PATHS.daily == folder / "healthkit_daily.json"
        assert api._PATHS.label_types == folder / "label_types.json"
        assert api._PATHS.ordinal_dictionary == folder / "ordinal_dictionary.json"
        assert api._PATHS.validity_config == folder / "validity_config.json"

    def test_per_file_env_overrides_dataset_root(self, monkeypatch, tmp_path, write_label_schemas):
        """Explicit per-file overrides remain supported."""
        root = tmp_path / "dataset"
        write_label_schemas(root)
        custom = tmp_path / "custom_labels.json"
        monkeypatch.setenv("LABELS_DATA_PATH", str(custom))
        from labels import api

        api.configure(root)
        assert api._PATHS.labels == custom
        assert api._PATHS.enrollment == root / "labels/enrollment_info.json"

    def test_configuration_requires_explicit_root(self):
        """No selected dataset means no implicit repository fallback."""
        from labels import api

        with pytest.raises(ValueError, match="data_dir=|MHC_DATA_DIR"):
            api.configure()


class TestDownloadDatasetSurface:
    """Verify the download helper signature and error paths."""

    def test_requires_explicit_destination(self):
        """Downloading with no destination or ``MHC_DATA_DIR`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.download_dataset(version="xs")

    def test_unknown_version_raises(self):
        """An unrecognized ``version`` string raises ``ValueError``."""
        from openmhc import download_dataset

        with pytest.raises(ValueError, match="version must be one of"):
            download_dataset(version="bogus")

    def test_unpublished_version_raises(self, monkeypatch):
        """A deliberately disabled release fails before resolving a destination."""
        from openmhc import download_dataset
        from openmhc._dataset import _VERSION_DOIS

        monkeypatch.setitem(_VERSION_DOIS, "full", None)
        with pytest.raises(ValueError, match="not yet published"):
            download_dataset(version="full")


class TestPublicApisRequireDatasetRoot:
    """Each public ``evaluate_*`` entry point fails fast without a dataset root."""

    def test_evaluate_prediction_fails_fast(self):
        """``evaluate_prediction`` raises ``ValueError`` when no root is configured."""

        class DummyMethod:
            def predict(self, data):
                return data

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_prediction(DummyMethod(), version="xs")

    def test_evaluate_imputation_fails_fast(self):
        """``evaluate_imputation`` raises ``ValueError`` when no root is configured."""

        class DummyImputer:
            def impute(self, data, observed_mask, target_mask):
                return data

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_imputation(DummyImputer(), version="xs")

    def test_evaluate_forecasting_fails_fast(self):
        """``evaluate_forecasting`` raises ``ValueError`` when no root is configured."""

        class DummyForecaster:
            def predict(self, history, horizon):
                return history

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_forecasting(DummyForecaster(), version="xs")


class TestDatasetRevision:
    """Canonical revision timestamps survive writes, downloads and derivatives."""

    @pytest.mark.parametrize("revision", ["2026-10-31T14:32:07Z", "2026-10-31T14:32:07.123456Z"])
    def test_roundtrip_and_preservation(self, tmp_path, revision):
        """Omitting a revision preserves it and unrelated marker metadata."""
        from openmhc._dataset import read_dataset_marker

        marker = write_dataset_marker(tmp_path, "full", dataset_revision=revision, daily_hf_rows=10)
        payload = json.loads(marker.read_text())
        payload["custom_metadata"] = "retain me"
        marker.write_text(json.dumps(payload))
        write_dataset_marker(tmp_path, "full")
        assert read_dataset_marker(tmp_path) == payload

    @pytest.mark.parametrize(
        "revision",
        [
            "2026-10-31",
            "2026-10-31T14:32:07",
            "2026-10-31T14:32:07+01:00",
            "2026-10-31T14:32:07+00:00",
            "2026-02-30T14:32:07Z",
            "2026-10-31T25:00:00Z",
            "2026-10-31T14:32:07Z ",
            "",
            123,
            None,
        ],
    )
    def test_invalid_stored_revision_rejected(self, tmp_path, revision):
        """A present revision must be a real UTC timestamp with explicit Z suffix."""
        from openmhc._dataset import read_dataset_marker

        marker = tmp_path / DATASET_VERSION_FILENAME
        marker.write_text(
            json.dumps({"version": "full", "n_users": 11894, "dataset_revision": revision})
        )
        with pytest.raises(ValueError, match="dataset_revision"):
            read_dataset_marker(tmp_path)

    def test_invalid_write_preserves_existing_marker(self, tmp_path):
        """Validate before replacing the user's existing marker."""
        marker = write_dataset_marker(tmp_path, "full", dataset_revision="2026-10-31T14:32:07Z")
        before = marker.read_bytes()
        with pytest.raises(ValueError, match="dataset_revision"):
            write_dataset_marker(tmp_path, "full", dataset_revision="not a timestamp")
        assert marker.read_bytes() == before

    def test_derivatives_keep_revision_while_size_changes(self, tmp_path):
        """Full and XS can carry the same data revision with different sizes."""
        from openmhc._dataset import read_dataset_marker

        revision = "2026-10-31T14:32:07Z"
        for size in ("full", "xs"):
            root = tmp_path / size
            _write_minimal_split(root, size)
            write_dataset_marker(root, size, dataset_revision=revision)
            assert _DatasetPaths.resolve(root, version=size).version == size
        a, b = (read_dataset_marker(tmp_path / size) for size in ("full", "xs"))
        assert a["dataset_revision"] == b["dataset_revision"] == revision
        assert a["n_users"] == 11894 and b["n_users"] == 593

    def test_second_revision_same_day_is_distinct(self, tmp_path):
        """Explicit updates distinguish changes within a day without a numbered version."""
        from openmhc._dataset import read_dataset_marker

        first, second = "2026-10-31T14:32:07Z", "2026-10-31T14:32:07.000001Z"
        write_dataset_marker(tmp_path, "full", dataset_revision=first)
        write_dataset_marker(tmp_path, "full", dataset_revision=second)
        assert read_dataset_marker(tmp_path)["dataset_revision"] == second

    @pytest.mark.parametrize("prefix", ["", "data/"])
    def test_download_preserves_packaged_revision(self, tmp_path, monkeypatch, dataverse_response, prefix):
        """Exercise real archive extraction with a synthetic network response."""
        import io
        import zipfile

        import openmhc._dataset as dataset

        payload = {"version": "xs", "n_users": 593, "dataset_revision": "2026-10-31T14:32:07Z"}
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(prefix + DATASET_VERSION_FILENAME, json.dumps(payload))
        dataverse_response(archive.getvalue(), "xs")
        root = dataset.download_dataset("xs", dest=tmp_path)
        assert dataset.read_dataset_marker(root) == payload

    def test_switching_size_drops_previous_row_count(self, tmp_path):
        """Deriving XS keeps the revision but must not retain Full's sensor-row count."""
        from openmhc._dataset import read_dataset_marker

        revision = "2026-10-31T14:32:07Z"
        write_dataset_marker(tmp_path, "full", dataset_revision=revision, daily_hf_rows=1000)
        write_dataset_marker(tmp_path, "xs")
        payload = read_dataset_marker(tmp_path)
        assert payload["dataset_revision"] == revision
        assert payload["version"] == "xs" and payload["n_users"] == 593
        assert "daily_hf_rows" not in payload
