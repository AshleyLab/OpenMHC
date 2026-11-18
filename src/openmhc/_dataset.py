"""Dataset download and discovery for OpenMHC.

The dataset is hosted on Harvard Dataverse. This module provides a thin
wrapper around the Dataverse access API so users don't have to remember
the DOI or directory layout.

Two versions are available:

- ``"xs"`` — 593-user subset suitable for quickstarts and smoke tests (~1.9 GB)
- ``"full"`` — full 11,894-user release (~38 GB)

Large benchmark payloads are always resolved from one explicit dataset root:

1. Explicit ``data_dir=`` / ``dest=`` argument
2. ``MHC_DATA_DIR`` environment variable

If neither is provided, OpenMHC raises instead of silently falling back to a
cache directory.

Every dataset root carries a ``dataset_version.json`` marker file preserved by
:func:`download_dataset`. The marker pins the version and expected user count
so the resolver can fail loudly if a directory ever ends up holding mismatched
contents (e.g. an XS bundle whose split file got renamed to the canonical
"full" name). For existing roots that predate the marker, use
:func:`write_dataset_marker` to write one explicitly.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

_DATAVERSE_BASE = "https://dataverse.harvard.edu"

# Identify the client explicitly for Dataverse access requests.
_USER_AGENT = "openmhc/0.1.0 (+https://github.com/AshleyLab/myheartcounts-dataset)"

# DOIs per version. Set to None for versions that haven't been published yet.
_VERSION_DOIS: dict[str, str | None] = {
    "xs": "doi:10.7910/DVN/RIKSK4",
    "full": "doi:10.7910/DVN/XNBITM",
}

# Expected user count per version (sum across train/validation/test in the
# canonical split file). The resolver cross-checks the loaded split file
# against these numbers, so a renamed/swapped split fails loudly.
EXPECTED_N_USERS: dict[str, int] = {
    "xs": 593,
    "full": 11894,
}

DATASET_VERSION_FILENAME = "dataset_version.json"

Version = Literal["xs", "full"]


class DatasetGuestbookRequiredError(PermissionError):
    """Dataverse requires a guestbook response; download and install manually."""


def _missing_dataset_root_error() -> ValueError:
    return ValueError(
        "OpenMHC dataset root is required for large benchmark payloads. "
        "Provide it with `data_dir=` / `dest=` or set `MHC_DATA_DIR`. "
        "For example, use `openmhc.download_dataset(dest='~/.cache/openmhc/data')` "
        "or `export MHC_DATA_DIR=~/.cache/openmhc/data`."
    )


def data_dir(override: str | Path | None = None) -> Path:
    """Resolve the explicit dataset directory for large payloads.

    Args:
        override: Explicit path. If provided, returned as-is.

    Returns:
        Absolute path to the dataset directory. May not exist yet.

    Raises:
        ValueError: If neither ``override`` nor ``MHC_DATA_DIR`` is provided.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    env = os.getenv("MHC_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    raise _missing_dataset_root_error()


def write_dataset_marker(
    root: str | Path,
    version: Version,
    *,
    n_users: int | None = None,
    daily_hf_rows: int | None = None,
    dataset_revision: str | None = None,
) -> Path:
    """Write a ``dataset_version.json`` marker to an existing dataset root.

    Use this when you have a dataset root that predates the marker file
    (e.g. an old XS cache) and need to make it usable with the strict
    resolver.

    Args:
        root: Path to the dataset root that contains ``processed/``,
            ``splits/``, ``labels/``, etc.
        version: ``"xs"`` or ``"full"``. Stored verbatim and cross-checked
            by :class:`_DatasetPaths`.
        n_users: Expected user count across all splits. Defaults to the
            canonical count for ``version``.
        daily_hf_rows: Expected row count of ``processed/daily_hf/``.
            Preserved if omitted for the same dataset size; cleared when
            switching between Full and XS unless explicitly supplied.
        dataset_revision: Canonical data revision as a UTC timestamp ending in Z,
            for example ``2026-10-31T14:32:07Z``. Omission preserves an existing
            revision. Legacy markers without a revision remain readable.

    Returns:
        Path to the written marker file.
    """
    if not isinstance(version, str) or version not in EXPECTED_N_USERS:
        raise ValueError(f"version must be one of {sorted(EXPECTED_N_USERS)}, got {version!r}")
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root_path}")
    marker = root_path / DATASET_VERSION_FILENAME
    payload = read_dataset_marker(root_path) if marker.exists() else {}
    if payload.get("version") != version:
        payload.pop("daily_hf_rows", None)
    payload.update(
        version=version,
        n_users=n_users if n_users is not None else EXPECTED_N_USERS[version],
    )
    if dataset_revision is not None:
        payload["dataset_revision"] = dataset_revision
    if daily_hf_rows is not None:
        payload["daily_hf_rows"] = daily_hf_rows
    _validate_marker_counts(payload)
    _validate_dataset_revision(payload)
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    return marker


def _validate_dataset_revision(payload: dict) -> None:
    """Accept legacy markers and validate any supplied UTC revision timestamp."""
    if "dataset_revision" not in payload:
        return
    revision = payload["dataset_revision"]
    if not isinstance(revision, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z",
        revision,
    ):
        raise ValueError("dataset_revision must be a UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(revision[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("dataset_revision must be a valid UTC timestamp") from exc


def read_dataset_marker(root: str | Path) -> dict:
    """Read ``dataset_version.json`` from a dataset root.

    Args:
        root: Path to the dataset root.

    Returns:
        Parsed marker payload.

    Raises:
        FileNotFoundError: If the marker is missing. The error message
            includes instructions for backfilling it with
            :func:`write_dataset_marker`.
        ValueError: If the marker is malformed.
    """
    root_path = Path(root).expanduser().resolve()
    marker = root_path / DATASET_VERSION_FILENAME
    if not marker.exists():
        raise FileNotFoundError(
            f"Missing dataset version marker: {marker}\n\n"
            "Every OpenMHC dataset root must contain a `dataset_version.json` "
            "file declaring which release ('xs' or 'full') lives there. "
            "`download_dataset` preserves the packaged marker; for a pre-existing "
            "root, run:\n"
            "    openmhc.write_dataset_marker(root, version='xs')  # or 'full'\n"
        )
    try:
        payload = json.loads(marker.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed dataset marker {marker}: {exc}") from exc
    if not isinstance(payload, dict) or "version" not in payload:
        raise ValueError(f"dataset marker {marker} is missing required 'version' field")
    if not isinstance(payload["version"], str) or payload["version"] not in EXPECTED_N_USERS:
        raise ValueError(
            f"dataset marker {marker} has unknown version "
            f"{payload['version']!r}; expected one of {sorted(EXPECTED_N_USERS)}"
        )
    _validate_marker_counts(payload)
    _validate_dataset_revision(payload)
    return payload


def _validate_marker_counts(payload: dict) -> None:
    """Reject invalid counts consistently on marker reads and writes."""
    for field, minimum in (("n_users", 1), ("daily_hf_rows", 0)):
        if field in payload and (type(payload[field]) is not int or payload[field] < minimum):
            raise ValueError(f"dataset marker {field} must be an integer >= {minimum}")


def download_dataset(
    version: str = "xs",
    dest: str | Path | None = None,
    api_token: str | None = None,
) -> Path:
    """Download the OpenMHC dataset from Harvard Dataverse.

    Fetches the latest published file inventory, downloads each original file
    with size/checksum verification, and extracts into a staging directory on
    the destination filesystem. Only a successful installation is moved into
    place. This avoids the dataset-ZIP service, which can omit restricted files
    or exceed the server's bundle size limit. If the dataset is
    restricted, supply your Dataverse API token via ``api_token=`` or the
    ``DATAVERSE_API_TOKEN`` environment variable. Releases requiring a guestbook
    response must be downloaded through Dataverse's website and installed
    manually; the exception includes the release and setup links.

    Args:
        version: ``"xs"`` (593-user quickstart subset, ~1.9 GB) or
            ``"full"`` (11,894-user release, ~38 GB).
        dest: Where to put the data. Must be provided explicitly here or via
            ``MHC_DATA_DIR``. Must be a new or empty directory; existing dataset
            roots are not updated in place.
        api_token: Optional Dataverse API token for restricted datasets.

    Returns:
        Path to the downloaded dataset directory.

    Raises:
        ValueError: If the version, release metadata, checksum or layout is invalid.
        DatasetGuestbookRequiredError: If Dataverse requires a guestbook response.
        PermissionError: If restricted files require a token or access is denied.
        FileExistsError: If the destination already contains files. Download to
            a fresh directory, validate it, then select that root instead.
    """
    if not isinstance(version, str) or version not in _VERSION_DOIS:
        raise ValueError(f"version must be one of {sorted(_VERSION_DOIS)}, got {version!r}")
    doi = _VERSION_DOIS[version]
    if doi is None:
        available = [v for v, d in _VERSION_DOIS.items() if d]
        raise ValueError(
            f"the {version!r} dataset is not yet published. Available versions: {available}"
        )

    target = data_dir(dest)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(
            f"Dataset destination is not an empty directory: {target}. "
            "Download the complete release into a new directory, then select it "
            "with data_dir= or MHC_DATA_DIR. Existing roots are not repaired in place."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    token = api_token or os.getenv("DATAVERSE_API_TOKEN")
    print(f"Downloading {version!r} dataset ({doi}) → {target}")
    # Staging beside the destination keeps the final move on one filesystem.
    with tempfile.TemporaryDirectory(prefix=".openmhc-download-", dir=target.parent) as scratch:
        stage = Path(scratch) / "dataset"
        stage.mkdir()
        _download_release(doi, stage, token)
        _validate_requested_marker(stage, version)
        _unpack_nested_archives(stage)
        if version == "xs":
            _post_process_xs(stage)
        else:
            _post_process_full(stage)
        _validate_requested_marker(stage, version)
        # Replace an empty destination atomically; a populated one is refused.
        stage.replace(target)
    print(f"Done. Reuse this root via MHC_DATA_DIR={target} or data_dir={target!s}.")
    return target


def _validate_requested_marker(root: Path, version: str) -> None:
    """Require the packaged identity to agree with the requested release."""
    markers = [read_dataset_marker(p) for p in (root, root / "data")
               if (p / DATASET_VERSION_FILENAME).exists()]
    if not markers:
        raise ValueError("Downloaded release is missing its dataset marker")
    for marker in markers:
        if marker["version"] != version or marker.get("n_users") != EXPECTED_N_USERS[version]:
            raise ValueError(f"Dataset marker does not match requested {version!r} release")
    if len(markers) == 2 and markers[0] != markers[1]:
        raise ValueError("Conflicting dataset markers at bundle root and data/")


def _open_dataverse(url: str, token: str | None, *, doi: str):
    """Open a bounded-time request without forwarding credentials on redirects."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    if token:
        request.add_unredirected_header("X-Dataverse-key", token)
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            with exc:
                body = exc.read()
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                payload = None
            message = payload.get("message") if isinstance(payload, dict) else None
            if (isinstance(message, str)
                    and "required guestbook response" in message.lower()):
                raise DatasetGuestbookRequiredError(
                    f"This release ({doi}) requires a Dataverse guestbook response. "
                    f"Open https://doi.org/{doi.removeprefix('doi:')}, complete the "
                    "required form, and download the original release files through "
                    "the browser. Follow the manual setup instructions to install them: "
                    "https://github.com/AshleyLab/OpenMHC/blob/main/docs/manual-dataset-setup.md. "
                    "Completing the browser form does not authorize this installer's "
                    "API download requests."
                ) from exc
            # Preserve the status, headers and readable body of unrelated errors.
            raise urllib.error.HTTPError(
                exc.url, exc.code, exc.msg, exc.headers, io.BytesIO(body)
            ) from exc
        if exc.code in (401, 403):
            raise PermissionError(
                "Dataverse denied access. Check the release's access requirements "
                "and supply an authorized api_token or DATAVERSE_API_TOKEN."
            ) from exc
        raise


def _relative_path(name: str) -> Path:
    """Accept portable relative paths, never traversal or absolute paths."""
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or name.startswith("/") or ".." in name.split("/")):
        raise ValueError(f"Unsafe archive/release path: {name!r}")
    path = PurePosixPath(name)
    if not path.parts:
        raise ValueError(f"Empty archive/release path: {name!r}")
    return Path(*path.parts)


def _release_files(payload: dict, doi: str) -> list[tuple[Path, int, int, str, str, bool]]:
    """Validate and pin file IDs, paths, sizes and checksums from one release."""
    release = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(release, dict):
        raise ValueError("Invalid Dataverse release metadata")
    if (release.get("versionState") != "RELEASED"
            or release.get("datasetPersistentId") != doi):
        raise ValueError("Dataverse metadata is not the requested published release")
    files = release.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Published release has no file inventory")
    entries, paths, ids = [], set(), set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("dataFile"), dict):
            raise ValueError("Invalid Dataverse file metadata")
        data = item["dataFile"]
        label = item.get("label")
        if not isinstance(label, str) or "/" in label:
            raise ValueError("Invalid release filename")
        directory = item.get("directoryLabel", "")
        if not isinstance(directory, str) or type(item.get("restricted", False)) is not bool:
            raise ValueError("Invalid release directory or restriction flag")
        path = _relative_path(f"{directory}/{label}" if directory else label)
        size, file_id = data.get("filesize"), data.get("id")
        if type(size) is not int or size < 0 or type(file_id) is not int or file_id <= 0:
            raise ValueError(f"Invalid release file size/id: {path}")
        if path in paths or file_id in ids or any(p in paths for p in path.parents):
            raise ValueError(f"Duplicate or conflicting release file: {path}")
        if any(path in p.parents for p in paths):
            raise ValueError(f"Conflicting release directory: {path}")
        checksum = data.get("checksum", {})
        if not isinstance(checksum, dict):
            raise ValueError(f"Missing release checksum for {path}")
        algorithm = str(checksum.get("type", "")).lower().replace("-", "")
        digest = str(checksum.get("value", "")).lower()
        if algorithm not in {"md5", "sha1", "sha256", "sha512"}:
            raise ValueError(f"Unsupported release checksum for {path}")
        digest_length = hashlib.new(algorithm).digest_size * 2
        if not re.fullmatch(rf"[0-9a-f]{{{digest_length}}}", digest):
            raise ValueError(f"Invalid release checksum for {path}")
        paths.add(path)
        ids.add(file_id)
        entries.append((path, file_id, size, algorithm, digest, bool(item.get("restricted"))))
    if not {Path(DATASET_VERSION_FILENAME), Path("data") / DATASET_VERSION_FILENAME} & paths:
        raise ValueError("Published release inventory is missing its dataset marker")
    return entries


def _download_release(doi: str, dest: Path, token: str | None) -> None:
    """Download every inventoried original file, failing on omissions or corruption."""
    query = urllib.parse.urlencode({"persistentId": doi})
    url = f"{_DATAVERSE_BASE}/api/datasets/:persistentId/versions/:latest-published?{query}"
    with _open_dataverse(url, token, doi=doi) as response:
        files = _release_files(json.load(response), doi)
    if not token and any(entry[-1] for entry in files):
        raise PermissionError("This release contains restricted files; supply an authorized Dataverse API token")
    for path, file_id, size, algorithm, expected, _ in files:
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, received = hashlib.new(algorithm), 0
        url = f"{_DATAVERSE_BASE}/api/access/datafile/{file_id}?format=original"
        with _open_dataverse(url, token, doi=doi) as response, target.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > size:
                    raise ValueError(f"Download size mismatch for {path}")
                digest.update(chunk)
                output.write(chunk)
        if received != size or digest.hexdigest() != expected:
            raise ValueError(f"Download size/checksum mismatch for {path}")


def _extract_archive_recursive(archive: Path, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, recursively unpacking nested archives.

    Supports existing ZIP bundles containing uploaded tarballs. After
    extracting the outer archive, any inner ``.tar``, ``.tar.gz``, ``.tgz``, or ``.zip``
    files are unpacked in place and removed.
    """
    _extract_one(archive, dest)
    _unpack_nested_archives(dest)


def _unpack_nested_archives(dest: Path) -> None:
    """Unpack each newly revealed archive, with a bound on nesting depth."""
    for _ in range(32):
        archives = sorted(p for p in dest.rglob("*") if p.is_file() and _looks_like_archive(p))
        if not archives:
            return
        for path in archives:
            target = path.parent
            # Full HDF5 archives contain bare files; XS contains hdf5/.
            if path.name == "hdf5_sharable_2026_full.tar.gz":
                target = path.parent / "hdf5"
            _extract_one(path, target)
            path.unlink()
    raise ValueError("Archive nesting exceeds 32 levels")


def _looks_like_archive(path: Path) -> bool:
    """Detect by suffix; falls through ``.bin`` / ``.zip`` / ``.tar*``."""
    name = path.name.lower()
    return (
        name.endswith(".tar.gz")
        or name.endswith(".tgz")
        or name.endswith(".tar")
        or name.endswith(".zip")
    )


def _extract_one(archive: Path, dest: Path) -> None:
    """Extract a single archive (zip / tar / tar.gz) into ``dest``.

    Detects format via the magic-byte sniffers in :mod:`zipfile` and
    :mod:`tarfile` so the file extension does not need to identify the format.
    """
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = _extraction_target(dest, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive links are not supported")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
        return
    if tarfile.is_tarfile(archive):
        with archive.open("rb") as stream:
            magic = stream.read(2)
            stream.seek(0)
            if magic == b"\x1f\x8b":
                _extract_gzip_tar(stream, dest)
            else:
                with tarfile.open(fileobj=stream, mode="r|*") as tf:
                    _extract_tar_members(tf, dest)
        return
    raise ValueError(f"could not detect archive format for {archive!r}; expected zip or tar/tar.gz")


def _extraction_target(dest: Path, name: str) -> Path:
    """Keep members inside the destination even if an existing parent is a link."""
    target = dest / _relative_path(name)
    if not target.resolve().is_relative_to(dest.resolve()):
        raise ValueError(f"Archive path escapes destination: {name!r}")
    if any(p.is_symlink() for p in (target, *target.parents) if p != dest and p.is_relative_to(dest)):
        raise ValueError(f"Archive path follows a link: {name!r}")
    return target


def _extract_tar_members(archive: tarfile.TarFile, dest: Path) -> None:
    """Extract regular files/directories without overwrites, links or special files."""
    for member in archive:
        if member.isdir() and member.name in (".", "./"):
            continue
        target = _extraction_target(dest, member.name)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
        else:
            raise ValueError(f"Unsupported archive member type: {member.name}")


def _extract_gzip_tar(stream, dest: Path) -> None:
    """Read through the gzip trailer so truncation and checksum errors surface."""
    with gzip.GzipFile(fileobj=stream, mode="rb") as uncompressed:
        with tarfile.open(fileobj=uncompressed, mode="r|") as archive:
            _extract_tar_members(archive, dest)
        while uncompressed.read(1024 * 1024):
            pass


_METADATA_SUBDIRS = ("splits", "labels", "forecasting_sample_index", "imputation")


def _validate_install_layout(dest: Path) -> None:
    """Reject metadata collisions and conflicting markers before moving files."""
    targets = set()
    for source, target in _layout_moves(dest):
        if target.exists() or target.is_symlink() or target in targets:
            raise FileExistsError(
                f"Cannot install {source}: conflicting destination {target}. "
                "Extract or download the complete release into a fresh directory; "
                "do not mix files from different releases."
            )
        targets.add(target)
    markers = [read_dataset_marker(root) for root in (dest, dest / "data")
               if (root / DATASET_VERSION_FILENAME).exists()]
    if len(markers) == 2 and markers[0] != markers[1]:
        raise ValueError("Conflicting dataset markers at bundle root and data/")


def _post_process_xs(dest: Path) -> None:
    """Reorganize the extracted XS bundle into the canonical dataset layout.

    The Dataverse XS deposit lays files out with directory labels: large
    HuggingFace Arrow / HDF5 archives under ``archives/`` and metadata under
    ``data/`` (``data/processed``, ``data/labels``, ``data/splits``,
    ``data/forecasting_sample_index``, ``data/imputation``). After the
    tarballs are extracted, this moves everything to the canonical paths the
    evaluation API expects (``processed/daily_hf``, ``splits/``, ``labels/``,
    ``forecasting_sample_index/``, ``hourly_trajectory/``, …) and removes the
    now-empty staging directories.
    """
    _validate_install_layout(dest)

    data = dest / "data"
    source_marker = data / DATASET_VERSION_FILENAME
    target_marker = dest / DATASET_VERSION_FILENAME
    if source_marker.exists():
        if target_marker.exists():
            source_marker.unlink()  # Equality was checked before any moves.
        else:
            source_marker.rename(target_marker)
    for source, target in _layout_moves(dest):
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    for staging in (dest / "archives", data / "processed", data):
        if staging.is_dir() and not any(staging.iterdir()):
            staging.rmdir()
    if (dest / "archives").exists():
        raise ValueError("Unrecognized or conflicting files remain in archives/")


def _layout_moves(dest: Path) -> list[tuple[Path, Path]]:
    """Plan layout normalization once for both collision checks and moves."""
    moves = [(dest / "data" / name, dest / name) for name in _METADATA_SUBDIRS]
    for name in ("daily_hf", "daily_hourly_hf", "hourly_trajectory", "minute_trajectory", "hdf5"):
        parent = dest / "processed" if name in ("daily_hf", "daily_hourly_hf") else dest
        moves.append((dest / "archives" / name, parent / name))
    processed = dest / "data" / "processed"
    if processed.is_dir():
        moves.extend((p, dest / "processed" / p.name) for p in processed.iterdir())
    moves.append((dest / "normalization_stats.json", dest / "processed" / "normalization_stats.json"))
    return [(source, target) for source, target in moves if source.exists()]


def _post_process_full(dest: Path) -> None:
    """Handle multi-part archives left in place after full bundle extraction.

    Dataverse serves each ``*.tar.gz.part-NN`` as a separate file. Downloaded
    parts sit in ``archives/`` and must be
    concatenated and streamed into ``tarfile`` before they can be unpacked.
    The part files are removed on success.
    """
    _validate_install_layout(dest)
    archives_dir = dest / "archives"

    # Group part files by their base tar.gz name.
    groups: dict[str, list[Path]] = {}
    for part in archives_dir.glob("*.tar.gz.part-*"):
        base = part.name.split(".part-")[0]
        groups.setdefault(base, []).append(part)

    # Resolve each group's extraction target from the archive name.
    _dest_map = {
        "daily_hf_full.tar.gz": dest / "processed",
        "daily_hourly_hf_full.tar.gz": dest / "processed",
        "hdf5_sharable_2026_full.tar.gz": dest / "hdf5",
        "minute_trajectory_full.tar.gz": dest,
        "hourly_trajectory_full.tar.gz": dest,
    }

    # Check every group before consuming any parts, including groups encountered
    # after an otherwise valid archive. Numbering starts at zero without gaps.
    for base_name, parts in groups.items():
        if base_name not in _dest_map:
            raise ValueError(f"Unsupported Full multipart archive: {base_name}")
        suffixes = [path.name.rsplit(".part-", 1)[1] for path in parts]
        if any(not re.fullmatch(r"[0-9]+", suffix) for suffix in suffixes):
            raise ValueError(f"Invalid multipart suffix for {base_name}")
        parts.sort(key=lambda path: int(path.name.rsplit(".part-", 1)[1]))
        numbers = [int(path.name.rsplit(".part-", 1)[1]) for path in parts]
        if numbers != list(range(len(parts))):
            raise ValueError(f"Missing or duplicate multipart segments for {base_name}")

    for base_name, parts in groups.items():
        target_dir = _dest_map[base_name]
        target_dir.mkdir(parents=True, exist_ok=True)

        # Stream concatenated parts through tarfile without writing a temp file.
        class _CatStream(io.RawIOBase):
            def __init__(self, paths: list[Path]) -> None:
                super().__init__()
                self._paths = paths
                self._current = None
                self._idx = 0

            def readable(self) -> bool:
                return True

            def close(self) -> None:
                if self._current is not None:
                    self._current.close()
                super().close()

            def readinto(self, b: bytearray) -> int:
                while self._idx < len(self._paths):
                    if self._current is None:
                        self._current = self._paths[self._idx].open("rb")
                    n = self._current.readinto(b)
                    if n:
                        return n
                    self._current.close()
                    self._current = None
                    self._idx += 1
                return 0

        with io.BufferedReader(_CatStream(parts)) as stream:
            _extract_gzip_tar(stream, target_dir)

        for part in parts:
            part.unlink(missing_ok=True)

    # Full shares XS's metadata and single-archive layout, including masks.
    # Normalize it even when the bundle contains no multipart archives.
    _post_process_xs(dest)
