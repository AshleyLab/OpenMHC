"""The public demographics file: evaluated users only, every age known, with provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import build_leaderboard_demographics as builder  # noqa: E402
import upload_leaderboard_substrate as uploader  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError  # noqa: E402


def _substrate(path, users):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user_id": users, "E": [0.0] * len(users)}).to_parquet(path)


@pytest.fixture
def published(tmp_path, monkeypatch):
    """Substrates as the Hub would serve them, including a draws file that must be ignored."""
    root = tmp_path / "hub"
    _substrate(root / "downstream/linear.parquet", ["a", "b"])
    _substrate(root / "imputation/locf.parquet", ["a", "b", "c"])
    _substrate(root / "imputation/bootstrap/draws.parquet", ["not_evaluated"])
    _substrate(root / "forecasting/seasonal_naive.parquet", ["c"])
    requested = {}

    def snapshot(repo_id, **kwargs):
        requested.update(repo_id=repo_id, **kwargs)
        return str(root)

    monkeypatch.setattr(builder, "snapshot_download", snapshot)
    monkeypatch.setattr(
        builder,
        "HfApi",
        lambda: SimpleNamespace(
            dataset_info=lambda repo_id, revision=None: SimpleNamespace(sha="abc123")
        ),
    )
    return requested


@pytest.fixture
def dataset(label_schema_environment):
    folder = label_schema_environment
    (folder / "enrollment_info.json").write_text(
        json.dumps(
            {
                "a": {"birth_year": 1980, "createdOn": "2020-01-01"},
                "b": {"birth_year": 2000, "createdOn": "2020-01-01"},
                "c": {"birth_year": 1950, "createdOn": "2020-01-01"},
                "not_evaluated": {"birth_year": 1990, "createdOn": "2020-01-01"},
            }
        )
    )
    (folder / "last_labels.json").write_text(
        json.dumps({"BiologicalSex": {"a": {"timestamps": ["2020-01-01"], "values": ["Male"]}}})
    )
    (folder.parent / "dataset_version.json").write_text(json.dumps({"version": "full"}))
    return folder.parent


def test_writes_evaluated_users_with_provenance(published, dataset, tmp_path):
    """Only users already in published substrates are written, sorted and grouped."""
    out = tmp_path / "out"
    builder.build(dataset, out)

    frame = pd.read_parquet(out / "demographics.parquet")
    assert frame.to_dict("records") == [
        {"user_id": "a", "age_group": "40-49", "sex": "male"},
        {"user_id": "b", "age_group": "18-29", "sex": "unknown"},
        {"user_id": "c", "age_group": "60+", "sex": "unknown"},
    ]
    meta = json.loads((out / "demographics.meta.json").read_text())
    assert meta["n_users"] == 3
    assert meta["age_group_edges"] == [18, 30, 40, 50, 60]
    assert meta["dataset_version"] == {"version": "full"}
    assert meta["substrate_revision"] == "abc123"
    assert "createdOn" in meta["age_rule"]
    assert "code_commit" in meta
    assert published["revision"] == "abc123"


def test_unknown_age_is_rejected_and_nothing_is_written(published, dataset, tmp_path):
    """The release must cover every evaluated user; a gap must not reach the public file."""
    enrollment = dataset / "labels/enrollment_info.json"
    payload = json.loads(enrollment.read_text())
    del payload["c"]
    enrollment.write_text(json.dumps(payload))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="1/3"):
        builder.build(dataset, out)
    assert not out.exists()


def test_no_published_substrates_is_an_error(monkeypatch, dataset, tmp_path):
    monkeypatch.setattr(builder, "snapshot_download", lambda repo_id, **kwargs: str(tmp_path))
    monkeypatch.setattr(
        builder,
        "HfApi",
        lambda: SimpleNamespace(
            dataset_info=lambda repo_id, revision=None: SimpleNamespace(sha="abc")
        ),
    )
    with pytest.raises(SystemExit, match="No published substrates"):
        builder.build(dataset, tmp_path / "out")


def _demographics(tmp_path, users):
    path = tmp_path / "demographics.parquet"
    pd.DataFrame({"user_id": users, "age_group": "30-39", "sex": "male"}).to_parquet(path)
    return path


def test_upload_accepts_users_in_the_demographics_file(tmp_path, monkeypatch):
    file = _demographics(tmp_path, ["a", "b"])
    monkeypatch.setattr(uploader, "hf_hub_download", lambda *a, **k: str(file))
    parquet = tmp_path / "method.parquet"
    _substrate(parquet, ["a"])
    uploader.validate_users_have_demographics(parquet, "repo")


def test_upload_rejects_users_left_out_of_every_cohort(tmp_path, monkeypatch):
    file = _demographics(tmp_path, ["a"])
    monkeypatch.setattr(uploader, "hf_hub_download", lambda *a, **k: str(file))
    parquet = tmp_path / "method.parquet"
    _substrate(parquet, ["a", "new_1", "new_2"])
    with pytest.raises(SystemExit, match="2 user") as error:
        uploader.validate_users_have_demographics(parquet, "repo")
    assert "every cohort" in str(error.value)
    assert "new_1" not in str(error.value)


@pytest.mark.parametrize("track", ["downstream", "imputation", "forecasting"])
def test_upload_checks_users_before_writing_to_the_hub(track, tmp_path, monkeypatch):
    """Every track is checked, and a rejected file leaves the Hub untouched."""
    file = _demographics(tmp_path, ["a"])
    monkeypatch.setattr(uploader, "hf_hub_download", lambda *a, **k: str(file))
    monkeypatch.setattr(
        uploader, "HfApi", lambda: pytest.fail("the Hub was touched before the check")
    )
    monkeypatch.setattr(uploader, "validate_method_column", lambda *a: None)
    source = tmp_path / "source"
    _substrate(source / "m.parquet", ["a", "new"])
    monkeypatch.setattr(
        sys, "argv", ["upload", "--dir", str(source), "--method", "m", "--track", track]
    )
    with pytest.raises(SystemExit, match="1 user"):
        uploader.main()


def test_upload_fails_when_the_demographics_file_is_not_published(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(uploader, "hf_hub_download", missing)
    parquet = tmp_path / "method.parquet"
    _substrate(parquet, ["a"])
    with pytest.raises(SystemExit, match="demographics.parquet"):
        uploader.validate_users_have_demographics(parquet, "repo")
