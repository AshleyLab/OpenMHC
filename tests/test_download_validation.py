"""Validate release markers and multipart boundaries during installation."""

import io
import json
import tarfile
import urllib.error
import urllib.parse
import zipfile

import pytest

from openmhc import DatasetGuestbookRequiredError
from openmhc import _dataset as dataset


@pytest.mark.parametrize("version,doi", [
    ("xs", "doi:10.7910/DVN/RIKSK4"),
    ("full", "doi:10.7910/DVN/XNBITM"),
])
def test_download_requests_current_deposit(tmp_path, monkeypatch, dataverse_response, version, doi):
    """Use the configured release DOI in the actual request without network I/O."""
    payload = io.BytesIO()
    marker = {"version": version, "n_users": dataset.EXPECTED_N_USERS[version]}
    with zipfile.ZipFile(payload, "w") as bundle:
        bundle.writestr("dataset_version.json", json.dumps(marker))
    response = dataverse_response(payload.getvalue(), version)
    root = dataset.download_dataset(version, dest=tmp_path / "installed")
    requests = response["requests"]
    assert len(requests) == 2
    url = urllib.parse.urlparse(requests[0].full_url)
    assert url.scheme == "https" and url.netloc == "dataverse.harvard.edu"
    assert url.path == "/api/datasets/:persistentId/versions/:latest-published"
    assert urllib.parse.parse_qs(url.query) == {"persistentId": [doi]}
    assert dataset.read_dataset_marker(root) == marker


def tar_bytes(member):
    """Create a gzip fixture that can be split at arbitrary byte boundaries."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(b"sensor fixture")
        archive.addfile(info, io.BytesIO(b"sensor fixture"))
    return stream.getvalue()


def download(tmp_path, monkeypatch, dataverse_response, files, version="full"):
    """Exercise the public installer using an in-memory Dataverse response."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    monkeypatch.setitem(dataset._VERSION_DOIS, version, "doi:test/synthetic")
    dataverse_response(stream.getvalue(), version)
    return dataset.download_dataset(version, dest=tmp_path / "installed")


@pytest.mark.parametrize("family,member,target", [
    ("daily_hf", "daily_hf/value.arrow", "processed/daily_hf/value.arrow"),
    ("daily_hourly_hf", "daily_hourly_hf/value.arrow", "processed/daily_hourly_hf/value.arrow"),
    ("minute_trajectory", "minute_trajectory/value.arrow", "minute_trajectory/value.arrow"),
    ("hourly_trajectory", "hourly_trajectory/value.arrow", "hourly_trajectory/value.arrow"),
    ("hdf5_sharable_2026", "participant.h5", "hdf5/participant.h5"),
])
def test_multipart_family_destination(tmp_path, monkeypatch, dataverse_response, family, member, target):
    """Every supported multipart family reaches its loader's directory."""
    payload = tar_bytes(member)
    files = {f"archives/{family}_full.tar.gz.part-{i:02d}": part
             for i, part in enumerate([payload[:9], payload[9:41], payload[41:]])}
    root = download(tmp_path, monkeypatch, dataverse_response, files)
    assert (root / target).read_bytes() == b"sensor fixture"
    assert not (root / "archives").exists()


@pytest.mark.parametrize("suffixes", [["00", "02"], ["00", "0"], ["bad"]])
def test_valid_archive_with_invalid_part_numbering_is_rejected(tmp_path, suffixes):
    """Reject invalid numbering even when concatenation produces valid gzip."""
    # A complete valid gzip stream prevents gzip corruption from masking a
    # missing numbering check. Empty extra parts do not change archive bytes.
    archives = tmp_path / "archives"
    archives.mkdir()
    before = {}
    for i, suffix in enumerate(suffixes):
        path = archives / f"daily_hf_full.tar.gz.part-{suffix}"
        before[path] = tar_bytes("daily_hf/value.arrow") if i == 0 else b""
        path.write_bytes(before[path])
    with pytest.raises(ValueError, match="multipart"):
        dataset._post_process_full(tmp_path)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_markers_preserve_identity(tmp_path, monkeypatch, dataverse_response, conflict):
    """Deduplicate equal markers and reject different release identities."""
    marker = {"version": "full", "n_users": 11894, "dataset_revision": "2026-10-31T14:32:07Z"}
    other = {**marker, "dataset_revision": "2026-11-01T14:32:07Z"} if conflict else marker
    files = {"dataset_version.json": json.dumps(marker), "data/dataset_version.json": json.dumps(other)}
    if conflict:
        with pytest.raises(ValueError, match="[Cc]onflicting.*marker"):
            download(tmp_path, monkeypatch, dataverse_response, files)
    else:
        root = download(tmp_path, monkeypatch, dataverse_response, files)
        assert dataset.read_dataset_marker(root) == marker
        assert not (root / "data/dataset_version.json").exists()


def test_marker_rewrite_keeps_existing_revision(tmp_path):
    """Rewriting a marker retains its revision and unrelated metadata."""
    marker = {"version": "full", "n_users": 11894, "dataset_revision": "2026-10-31T14:32:07Z",
              "daily_hf_rows": 10, "custom_metadata": "retain me"}
    (tmp_path / "dataset_version.json").write_text(json.dumps(marker))
    dataset.write_dataset_marker(tmp_path, "full")
    assert dataset.read_dataset_marker(tmp_path) == marker


def test_multipart_numbers_are_sorted_numerically(tmp_path, monkeypatch, dataverse_response):
    """Unpadded part 10 must follow part 9, even with reversed ZIP ordering."""
    payload = tar_bytes("daily_hf/value.arrow")
    pieces = [payload[i * len(payload) // 12:(i + 1) * len(payload) // 12] for i in range(12)]
    files = {f"archives/daily_hf_full.tar.gz.part-{i}": pieces[i] for i in reversed(range(12))}
    root = download(tmp_path, monkeypatch, dataverse_response, files)
    assert (root / "processed/daily_hf/value.arrow").read_bytes() == b"sensor fixture"
    assert not (root / "archives").exists()


@pytest.mark.parametrize("bad_name", ["unknown_full.tar.gz.part-00", "minute_trajectory_full.tar.gz.part-01"])
def test_all_multipart_groups_are_validated_before_extraction(tmp_path, monkeypatch, bad_name):
    """An invalid later group must leave earlier valid archive parts intact."""
    archives = tmp_path / "archives"
    archives.mkdir()
    good = archives / "daily_hf_full.tar.gz.part-00"
    bad = archives / bad_name
    good.write_bytes(tar_bytes("daily_hf/value.arrow"))
    bad.write_bytes(tar_bytes("minute_trajectory/value.arrow"))
    before = {p: p.read_bytes() for p in [good, bad]}
    path_type = type(archives)
    original_glob = path_type.glob

    def ordered_glob(path, pattern):
        if path == archives and pattern == "*.tar.gz.part-*":
            return iter([good, bad])
        return original_glob(path, pattern)

    monkeypatch.setattr(path_type, "glob", ordered_glob)
    with pytest.raises(ValueError, match="multipart"):
        dataset._post_process_full(tmp_path)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (tmp_path / "processed").exists()


@pytest.mark.parametrize("process", [dataset._post_process_xs, dataset._post_process_full])
def test_conflicting_markers_fail_before_layout_changes(tmp_path, process):
    """Marker conflicts retain both markers and all extracted payloads in place."""
    data = tmp_path / "data"
    data.mkdir()
    marker = {"version": "full", "n_users": 11894, "dataset_revision": "2026-10-31T14:32:07Z"}
    paths = [tmp_path / "dataset_version.json", data / "dataset_version.json"]
    paths[0].write_text(json.dumps(marker))
    paths[1].write_text(json.dumps({**marker, "dataset_revision": "2026-11-01T14:32:07Z"}))
    incoming = data / "imputation"
    incoming.mkdir()
    payload = incoming / "sentinel"
    payload.write_bytes(b"unchanged masks")
    before = {p: p.read_bytes() for p in [*paths, payload]}
    with pytest.raises(ValueError, match="Conflicting.*markers"):
        process(tmp_path)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (tmp_path / "processed").exists()
    assert not (tmp_path / "imputation").exists()


@pytest.mark.parametrize("version", ["xs", "full"])
@pytest.mark.parametrize("prefix", ["", "data/"])
def test_download_rejects_invalid_revision(tmp_path, monkeypatch, dataverse_response, version, prefix):
    """Malformed packaged revisions cannot become successfully installed roots."""
    marker = {"version": version, "n_users": dataset.EXPECTED_N_USERS[version],
              "dataset_revision": "2026-02-30T00:00:00Z"}
    with pytest.raises(ValueError, match="dataset_revision"):
        download(tmp_path, monkeypatch, dataverse_response, {prefix + "dataset_version.json": json.dumps(marker)}, version)
    assert not (tmp_path / "installed").exists()


def release_response(dataverse_response, *, version="xs", files=None, add_marker=True):
    """Create a small inventory with independently checksummed original files."""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, value in (files or {"README.md": b"synthetic release"}).items():
            archive.writestr(name, value)
    return dataverse_response(payload.getvalue(), version, add_marker=add_marker)


@pytest.mark.parametrize("failure", ["short", "long", "checksum", "http", "interrupted"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_transfer_never_installs_partial_root(tmp_path, monkeypatch, dataverse_response, failure, existing):
    """Bad/truncated/denied transfers leave the requested destination untouched."""
    import urllib.error

    response = release_response(dataverse_response)
    root = tmp_path / "installed"
    if existing:
        root.mkdir()
    original = dataset.urllib.request.urlopen
    if failure == "short":
        response["bodies"][1] = b""
    elif failure == "long":
        response["bodies"][1] += b"extra"
    elif failure == "checksum":
        response["bodies"][1] = b"x" * len(response["bodies"][1])
    else:
        def fail(request, **kwargs):
            if "/datafile/" in request.full_url:
                if failure == "http":
                    raise urllib.error.HTTPError(request.full_url, 403, "denied", {}, None)
                raise ConnectionError("interrupted transfer")
            return original(request, **kwargs)
        monkeypatch.setattr(dataset.urllib.request, "urlopen", fail)
    with pytest.raises((ValueError, PermissionError, ConnectionError)):
        dataset.download_dataset("xs", root)
    assert root.exists() == existing
    assert not root.exists() or not list(root.iterdir())
    assert not list(tmp_path.glob(".openmhc-download-*"))


@pytest.mark.parametrize("version", ["xs", "full"])
@pytest.mark.parametrize("token", [None, "fixture-token"])
@pytest.mark.parametrize("existing", [False, True])
def test_guestbook_rejection_gives_manual_setup_guidance(
    tmp_path, monkeypatch, dataverse_response, version, token, existing,
):
    """Translate the real Dataverse rejection, even after a file was staged."""
    release_response(dataverse_response, version=version)
    root = tmp_path / "installed"
    if existing:
        root.mkdir()
    guestbook_id = 729 if version == "xs" else 728
    body = json.dumps({
        "status": "ERROR",
        "message": "You may not download this file without the required "
                   f"Guestbook response for guestbookID {guestbook_id}.",
    }).encode()
    original = dataset.urllib.request.urlopen
    rejection = None

    def fail(request, **kwargs):
        nonlocal rejection
        if "/datafile/2?" in request.full_url:
            rejection = urllib.error.HTTPError(
                request.full_url, 400, "Bad Request", {}, io.BytesIO(body),
            )
            raise rejection
        return original(request, **kwargs)

    monkeypatch.setattr(dataset.urllib.request, "urlopen", fail)
    with pytest.raises(DatasetGuestbookRequiredError) as caught:
        dataset.download_dataset(version, root, api_token=token)
    assert isinstance(caught.value, PermissionError)
    assert caught.value.__cause__ is rejection
    message = str(caught.value)
    doi = dataset._VERSION_DOIS[version]
    assert doi in message
    assert f"https://doi.org/{doi.removeprefix('doi:')}" in message
    assert "https://github.com/AshleyLab/OpenMHC/blob/main/docs/manual-dataset-setup.md" in message
    assert "download the original release files through the browser" in message
    assert "does not authorize" in message
    assert root.exists() == existing
    assert not root.exists() or not list(root.iterdir())
    assert not list(tmp_path.glob(".openmhc-download-*"))


@pytest.mark.parametrize("code,body", [
    (400, b'{"status":"ERROR","message":"Invalid file format"}'),
    (400, b"<html>Bad request</html>"),
    (400, b"\xff"),
    (400, b"[]"),
    (400, b"null"),
    (400, b'{"message": null}'),
    (404, b'{"message":"required Guestbook response"}'),
    (500, b'{"message":"required Guestbook response"}'),
])
def test_other_http_errors_keep_status_headers_and_body(
    tmp_path, monkeypatch, dataverse_response, code, body,
):
    """Unrelated failures remain inspectable HTTP errors, including bad JSON."""
    release_response(dataverse_response)
    original = dataset.urllib.request.urlopen
    failed_url = None

    def fail(request, **kwargs):
        nonlocal failed_url
        if "/datafile/" in request.full_url:
            failed_url = request.full_url
            raise urllib.error.HTTPError(
                failed_url, code, "original reason", {"X-Error-ID": "fixture"}, io.BytesIO(body),
            )
        return original(request, **kwargs)

    monkeypatch.setattr(dataset.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.HTTPError) as caught:
        dataset.download_dataset("xs", tmp_path / "installed")
    with caught.value as error:
        assert error.code == code
        assert error.url == failed_url
        assert error.reason == "original reason"
        assert error.headers["X-Error-ID"] == "fixture"
        assert error.read() == body
    assert not (tmp_path / "installed").exists()
    assert not list(tmp_path.glob(".openmhc-download-*"))


@pytest.mark.parametrize("code", [401, 403])
def test_access_denied_still_raises_permission_error(tmp_path, monkeypatch, code):
    """Authentication and file permissions keep their existing token guidance."""
    def fail(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, code, "denied", {}, None)

    monkeypatch.setattr(dataset.urllib.request, "urlopen", fail)
    with pytest.raises(PermissionError, match="api_token") as caught:
        dataset.download_dataset("xs", tmp_path / "installed")
    assert not isinstance(caught.value, DatasetGuestbookRequiredError)


@pytest.mark.parametrize("problem", ["draft", "wrong_doi", "empty", "no_marker", "duplicate", "path", "checksum", "size", "id", "file_parent"])
def test_invalid_inventory_fails_before_payload_requests(tmp_path, dataverse_response, problem):
    """Validate the entire published inventory before starting file downloads."""
    response = release_response(dataverse_response, add_marker=problem != "no_marker")
    release = response["metadata"]["data"]
    file = release["files"][0]
    if problem == "draft":
        release["versionState"] = "DRAFT"
    elif problem == "wrong_doi":
        release["datasetPersistentId"] = "doi:wrong"
    elif problem == "empty":
        release["files"] = []
    elif problem == "duplicate":
        release["files"].append(file.copy())
    elif problem == "path":
        file["directoryLabel"] = "../outside"
    elif problem == "checksum":
        file["dataFile"]["checksum"]["value"] = "invalid"
    elif problem == "size":
        file["dataFile"]["filesize"] = -1
    elif problem == "id":
        file["dataFile"]["id"] = True
    elif problem == "file_parent":
        release["files"][1]["directoryLabel"] = "README.md"
    with pytest.raises(ValueError):
        dataset.download_dataset("xs", tmp_path / "installed")
    assert len(response["requests"]) == 1
    assert not (tmp_path / "installed").exists()


def test_restricted_inventory_needs_token_before_file_requests(tmp_path, dataverse_response):
    """A public subset must not be mistaken for the complete restricted release."""
    response = release_response(dataverse_response, version="full")
    response["metadata"]["data"]["files"][0]["restricted"] = True
    with pytest.raises(PermissionError, match="token"):
        dataset.download_dataset("full", tmp_path / "installed")
    assert len(response["requests"]) == 1


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_token_is_used_but_not_forwarded_on_redirect(tmp_path, monkeypatch, dataverse_response, source):
    """Authenticate Dataverse requests without exposing tokens to storage redirects."""
    response = release_response(dataverse_response, version="full")
    response["metadata"]["data"]["files"][0]["restricted"] = True
    if source == "environment":
        monkeypatch.setenv("DATAVERSE_API_TOKEN", "fixture-token")
    dataset.download_dataset("full", tmp_path / "installed",
                             api_token="fixture-token" if source == "argument" else None)
    for request in response["requests"]:
        assert request.get_header("X-dataverse-key") == "fixture-token"
        redirected = dataset.urllib.request.HTTPRedirectHandler().redirect_request(
            request, None, 302, "redirect", {}, "https://storage.example/signed-file")
        assert redirected.get_header("X-dataverse-key") is None
        assert request.get_header("User-agent") == dataset._USER_AGENT


def test_pinned_inventory_downloads_every_file_id(tmp_path, dataverse_response):
    """Use individual original-file URLs, not the potentially partial ZIP endpoint."""
    response = release_response(dataverse_response, files={"one.txt": b"one", "two.txt": b"two"})
    root = dataset.download_dataset("xs", tmp_path / "installed")
    assert (root / "one.txt").read_bytes() == b"one"
    assert (root / "two.txt").read_bytes() == b"two"
    assert [urllib.parse.urlparse(r.full_url).path for r in response["requests"][1:]] == [
        "/api/access/datafile/1", "/api/access/datafile/2", "/api/access/datafile/3"]


@pytest.mark.parametrize("name", ["../escape", "/absolute", "dir/../../escape", "C:/escape", "dir\\escape"])
@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_paths_cannot_escape(tmp_path, kind, name):
    """Reject unsafe paths rather than normalizing them into surprising destinations."""
    archive = tmp_path / ("input." + kind)
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(name, b"fixture")
    else:
        archive.write_bytes(tar_bytes(name))
    with pytest.raises(ValueError, match="[Uu]nsafe"):
        dataset._extract_one(archive, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_tar_rejects_links_and_special_files(tmp_path, member_type):
    """Release extraction creates only regular files and directories."""
    archive = tmp_path / "input.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("link")
        member.type = member_type
        member.linkname = "../outside"
        output.addfile(member)
    with pytest.raises(ValueError, match="member type"):
        dataset._extract_one(archive, tmp_path / "output")


def test_archive_rejects_existing_symlink_parent(tmp_path):
    """Even a pre-existing directory link cannot redirect an extraction write."""
    output, outside = tmp_path / "output", tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "linked").symlink_to(outside, target_is_directory=True)
    archive = tmp_path / "input.tar.gz"
    archive.write_bytes(tar_bytes("linked/escape"))
    with pytest.raises(ValueError):
        dataset._extract_one(archive, output)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("damage", ["truncated", "checksum"])
def test_gzip_trailer_is_validated(tmp_path, multipart, damage):
    """Valid tar contents must not hide a missing or corrupted gzip trailer."""
    payload = tar_bytes("daily_hf/value.arrow")
    if damage == "truncated":
        payload = payload[:-8]
    else:
        payload = payload[:-8] + bytes([payload[-8] ^ 255]) + payload[-7:]
    if multipart:
        path = tmp_path / "archives/daily_hf_full.tar.gz.part-00"
        path.parent.mkdir()
    else:
        path = tmp_path / "input.tar.gz"
    path.write_bytes(payload)
    with pytest.raises((EOFError, OSError, tarfile.ReadError)):
        if multipart:
            dataset._post_process_full(tmp_path)
        else:
            dataset._extract_one(path, tmp_path / "output")
    assert path.read_bytes() == payload


@pytest.mark.parametrize("source,target", [
    ("data/processed/value.json", "processed/value.json"),
    ("archives/daily_hf/value.arrow", "processed/daily_hf/value.arrow"),
    ("normalization_stats.json", "processed/normalization_stats.json"),
])
def test_payload_layout_collisions_preserve_both_inputs(tmp_path, source, target):
    """Reject metadata and sensor payload collisions before any layout moves."""
    incoming, existing = tmp_path / source, tmp_path / target
    incoming.parent.mkdir(parents=True, exist_ok=True)
    existing.parent.mkdir(parents=True, exist_ok=True)
    incoming.write_bytes(b"incoming")
    existing.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        dataset._post_process_xs(tmp_path)
    assert incoming.read_bytes() == b"incoming"
    assert existing.read_bytes() == b"existing"


def test_deep_nested_archives_are_fully_unpacked(tmp_path):
    """Discover archives revealed by earlier extraction passes."""
    payload = tar_bytes("payload.txt")
    for name in ["inner.tar.gz", "middle.zip", "outer.zip"]:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr(name, payload)
        payload = stream.getvalue()
    archive = tmp_path / "download.bin"
    archive.write_bytes(payload)
    dest = tmp_path / "installed"
    dataset._extract_archive_recursive(archive, dest)
    assert [p.name for p in dest.iterdir()] == ["payload.txt"]
    assert (dest / "payload.txt").read_bytes() == b"sensor fixture"


def test_single_full_hdf5_archive_reaches_loader_path(tmp_path, monkeypatch, dataverse_response):
    """Full HDF5 bare members have the same destination with or without splitting."""
    root = download(tmp_path, monkeypatch, dataverse_response,
                    {"archives/hdf5_sharable_2026_full.tar.gz": tar_bytes("participant.h5")})
    assert (root / "hdf5/participant.h5").read_bytes() == b"sensor fixture"


@pytest.mark.parametrize("field,value", [("n_users", 0), ("n_users", -1), ("n_users", True),
                                         ("n_users", 3.5), ("daily_hf_rows", -1),
                                         ("daily_hf_rows", False), ("daily_hf_rows", 1.2)])
def test_invalid_marker_write_preserves_identity(tmp_path, field, value):
    """A writer must not emit a marker that its own reader would reject."""
    path = dataset.write_dataset_marker(tmp_path, "xs")
    before = path.read_bytes()
    with pytest.raises(ValueError, match=field):
        dataset.write_dataset_marker(tmp_path, "xs", **{field: value})
    assert path.read_bytes() == before


@pytest.mark.parametrize("problem", [None, [], {"data": []}, {"data": None}])
def test_malformed_release_response_fails_cleanly(tmp_path, dataverse_response, problem):
    """Invalid JSON shapes fail without requesting payloads or installing a root."""
    response = release_response(dataverse_response)
    response["metadata"] = problem
    with pytest.raises(ValueError, match="metadata"):
        dataset.download_dataset("xs", tmp_path / "installed")
    assert len(response["requests"]) == 1
    assert not (tmp_path / "installed").exists()


@pytest.mark.parametrize("kind", ["corrupt", "traversal", "collision"])
def test_extraction_failure_discards_entire_staging(tmp_path, dataverse_response, kind):
    """Successful transfers must not publish a root when extraction fails."""
    files = {"archives/broken.tar.gz": b"not an archive"}
    if kind == "traversal":
        files = {"archives/broken.tar.gz": tar_bytes("../../../escape")}
    elif kind == "collision":
        files = {"data/processed/value.json": b"incoming", "processed/value.json": b"existing"}
    release_response(dataverse_response, files=files)
    root = tmp_path / "installed"
    root.mkdir()
    before = root.stat().st_ino
    with pytest.raises((ValueError, OSError)):
        dataset.download_dataset("xs", root)
    assert root.stat().st_ino == before
    assert not list(root.iterdir())
    assert not list(tmp_path.glob(".openmhc-download-*"))
    assert not (tmp_path / "escape").exists()


def test_destination_populated_during_download_is_preserved(tmp_path, monkeypatch, dataverse_response):
    """A competing writer cannot lose files at the final installation step."""
    release_response(dataverse_response)
    root = tmp_path / "installed"
    root.mkdir()
    original = dataset._download_release

    def competing_write(*args):
        original(*args)
        (root / "keep.txt").write_text("other process")

    monkeypatch.setattr(dataset, "_download_release", competing_write)
    with pytest.raises(OSError):
        dataset.download_dataset("xs", root)
    assert (root / "keep.txt").read_text() == "other process"
    assert list(root.iterdir()) == [root / "keep.txt"]
    assert not list(tmp_path.glob(".openmhc-download-*"))


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_duplicate_archive_members_do_not_overwrite(tmp_path, kind):
    """Repeated paths cannot silently choose which release bytes survive."""
    archive = tmp_path / ("input." + kind)
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("same.txt", b"first")
            with pytest.warns(UserWarning, match="Duplicate"):
                output.writestr("same.txt", b"second")
    else:
        with tarfile.open(archive, "w") as output:
            for content in (b"first", b"second"):
                member = tarfile.TarInfo("same.txt")
                member.size = len(content)
                output.addfile(member, io.BytesIO(content))
    dest = tmp_path / "output"
    with pytest.raises(FileExistsError):
        dataset._extract_one(archive, dest)
    assert (dest / "same.txt").read_bytes() == b"first"


def test_zip_symlinks_are_rejected(tmp_path):
    """ZIP link attributes must not be interpreted as ordinary release members."""
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as output:
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = 0o120777 << 16
        output.writestr(member, "../outside")
    with pytest.raises(ValueError, match="links"):
        dataset._extract_one(path, tmp_path / "output")
