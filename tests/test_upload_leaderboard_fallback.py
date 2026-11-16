"""Uploader downstream method-column guard + fallback-rate extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import upload_leaderboard_substrate as uploader  # noqa: E402


def test_validate_method_column_downstream_pass(tmp_path):
    """Accept a substrate whose method matches the upload name."""
    pq = tmp_path / "linear.parquet"
    pd.DataFrame({"method": ["linear", "linear"], "y_true": [0, 1]}).to_parquet(pq)
    # Mirrors the uploader gate for --track downstream: must not raise.
    uploader.validate_method_column(pq, "linear")


def test_validate_method_column_mismatch_raises(tmp_path):
    """Reject predictions labeled as a different method."""
    pq = tmp_path / "linear.parquet"
    pd.DataFrame({"method": ["custom"], "y_true": [0]}).to_parquet(pq)
    with pytest.raises(SystemExit):
        uploader.validate_method_column(pq, "linear")


def _write_results(tmp_path, payload):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(payload))
    return p


def test_worst_case_fallback_rate_forecasting_top_level(tmp_path):
    """Read the measured pooled forecasting rate."""
    # Forecasting / downstream shape: a single top-level scalar.
    p = _write_results(
        tmp_path,
        {"per_channel": {"ch_0": {"mae": 1.0}}, "overall_fallback_rate": 0.0123},
    )
    assert uploader._worst_case_fallback_rate(p) == pytest.approx(0.0123)


def test_worst_case_fallback_rate_forecasting_zero(tmp_path):
    """Preserve an explicitly measured zero."""
    p = _write_results(tmp_path, {"per_channel": {}, "overall_fallback_rate": 0.0})
    assert uploader._worst_case_fallback_rate(p) == 0.0


def test_worst_case_fallback_rate_imputation_nested_max(tmp_path):
    """Use the largest measured scenario/split rate."""
    # Imputation shape: max across all scenarios[*][*].overall_fallback_rate cells.
    p = _write_results(
        tmp_path,
        {
            "random_noise": {"test": {"overall_fallback_rate": 0.01}},
            "sleep_gap": {
                "test": {"overall_fallback_rate": 0.05},
                "val": {"overall_fallback_rate": 0.20},
            },
        },
    )
    assert uploader._worst_case_fallback_rate(p) == pytest.approx(0.20)


def test_worst_case_fallback_rate_legacy_absent_returns_none(tmp_path):
    """Distinguish a missing legacy measurement from zero."""
    # Legacy run with no fallback field anywhere -> None (not 0.0).
    p = _write_results(tmp_path, {"per_channel": {"ch_0": {"mae": 1.0}}})
    assert uploader._worst_case_fallback_rate(p) is None


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf"), True, "0.2", None])
@pytest.mark.parametrize("nested", [False, True])
def test_invalid_rates_rejected(tmp_path, rate, nested):
    """Both result shapes reject values that cannot be measured fractions."""
    payload = {"overall_fallback_rate": rate}
    if nested:
        payload = {"scenarios": {"gap": {"test": payload}}}
    with pytest.raises(ValueError, match="fallback rate"):
        uploader._worst_case_fallback_rate(_write_results(tmp_path, payload))


def test_partially_measured_imputation_rejected(tmp_path):
    """A missing scenario must not silently lower the maximum."""
    payload = {"scenarios": {"gap": {
        "val": {"overall_fallback_rate": 0.1}, "test": {"mae": 1.0},
    }}}
    with pytest.raises(ValueError, match="fallback rate"):
        uploader._worst_case_fallback_rate(_write_results(tmp_path, payload))


@pytest.fixture
def upload_case(tmp_path, monkeypatch):
    """Intercept all Hub access and provide synthetic local and remote files."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    src = tmp_path / "wbm.parquet"
    pd.DataFrame({
        "method": ["wbm"], "model": ["wbm"], "user_id": ["synthetic-user"],
    }).to_parquet(src)
    provenance = Path(f"{src}.meta.json")
    provenance.write_text(json.dumps({"method": "wbm", "overall_fallback_rate": 0.25}))
    demographics = tmp_path / "demographics.parquet"
    pd.DataFrame({"user_id": ["synthetic-user"]}).to_parquet(demographics)
    remote_meta = tmp_path / "remote.meta.json"
    remote_meta.write_text(json.dumps({
        "display_name": "Existing WBM", "type": "Self-Supervised", "submitter": "Team",
        "subtrack": "static", "fallback_rate": 0.9, "overall_fallback_rate": 0.8,
    }))
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha="saved-revision")
    uploaded_predictions = []

    def commit(**kwargs):
        uploaded_predictions.append(Path(kwargs["operations"][0].path_or_fileobj).read_bytes())
        return SimpleNamespace(commit_url="mock-commit")

    api.create_commit.side_effect = commit
    factory = Mock(return_value=api)
    monkeypatch.setattr(uploader, "HfApi", factory)

    def download(repo_id, filename, **kwargs):
        assert kwargs["revision"] == "saved-revision"
        if filename == "demographics.parquet":
            return str(demographics)
        assert filename.endswith("/wbm.meta.json")
        return str(remote_meta)

    reader = Mock(side_effect=download)
    monkeypatch.setattr(uploader, "hf_hub_download", reader)
    argv = ["upload", "--dir", str(tmp_path), "--method", "wbm", "--track", "downstream"]
    monkeypatch.setattr(sys, "argv", argv)
    return SimpleNamespace(
        src=src, provenance=provenance, demographics=demographics, remote_meta=remote_meta,
        api=api, factory=factory, reader=reader, argv=argv,
        uploaded_predictions=uploaded_predictions,
    )


@pytest.mark.parametrize("rate", [0.0, 0.25, 1.0])
def test_upload_commits_predictions_and_measured_rate_together(upload_case, rate):
    """Overwrite stale rates while preserving display fields in one pinned commit."""
    case = upload_case
    case.provenance.write_text(json.dumps({"method": "wbm", "overall_fallback_rate": rate}))
    uploader.main()
    case.api.create_repo.assert_not_called()
    case.api.upload_file.assert_not_called()
    case.api.create_commit.assert_called_once()
    commit = case.api.create_commit.call_args.kwargs
    assert commit["parent_commit"] == "saved-revision"
    prediction, metadata = commit["operations"]
    assert prediction.path_in_repo == "downstream/wbm.parquet"
    assert case.uploaded_predictions == [case.src.read_bytes()]
    assert not Path(prediction.path_or_fileobj).exists()
    assert metadata.path_in_repo == "downstream/wbm.meta.json"
    meta = json.loads(metadata.path_or_fileobj)
    assert meta == {
        "display_name": "Existing WBM", "type": "Self-Supervised", "submitter": "Team",
        "subtrack": "static", "fallback_rate": rate,
    }


@pytest.mark.parametrize("payload", [
    {}, {"overall_fallback_rate": None}, {"overall_fallback_rate": -1},
    {"overall_fallback_rate": float("nan")}, {"overall_fallback_rate": True},
    {"overall_fallback_rate": 0.2, "method": "different"},
    {"overall_fallback_rate": 0.2, "track": "forecasting"},
    {"overall_fallback_rate": 0.2, "substrate_sha256": "wrong"}, [],
])
def test_bad_provenance_stops_before_hub_access(upload_case, payload):
    """Invalid or conflicting provenance cannot reach an upload call."""
    case = upload_case
    case.provenance.write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        uploader.main()
    case.factory.assert_not_called()
    case.reader.assert_not_called()


@pytest.mark.parametrize("extra", [[], ["--fallback-rate", "0.0"], ["--fallback-rate", "nan"]])
def test_missing_measurement_cannot_be_replaced_by_manual_rate(upload_case, extra):
    """Absence is not zero, even when an old command supplies a numeric override."""
    case = upload_case
    case.provenance.unlink()
    case.argv.extend(extra)
    with pytest.raises(SystemExit):
        uploader.main()
    case.factory.assert_not_called()


@pytest.mark.parametrize("value", [None, [], {"overall_fallback_rate": 0.5}, {}])
def test_explicit_bad_results_stop_before_upload(upload_case, value, tmp_path):
    """A missing file, legacy metrics list or conflicting rate stops publication."""
    path = tmp_path / "results.json"
    if value is not None:
        path.write_text(json.dumps(value))
    upload_case.argv.extend(["--results-json", str(path)])
    with pytest.raises(SystemExit):
        uploader.main()
    upload_case.factory.assert_not_called()


@pytest.mark.parametrize("track,payload", [
    ("downstream", {"overall_fallback_rate": 0.25}),
    ("forecasting", {"overall_fallback_rate": 0.25}),
    ("imputation", {"gap": {"test": {"overall_fallback_rate": 0.25}}}),
])
def test_results_json_and_display_updates_remain_supported(upload_case, tmp_path, track, payload):
    """Existing result-file workflows use the same commit path for all tracks."""
    case = upload_case
    case.provenance.unlink()
    _write_results(tmp_path, payload)
    case.argv[-1] = track
    case.argv.extend([
        "--results-json", str(tmp_path), "--name", "Updated name", "--fallback-rate", "0.25",
    ])
    uploader.main()
    meta = json.loads(case.api.create_commit.call_args.kwargs["operations"][1].path_or_fileobj)
    assert meta["fallback_rate"] == 0.25
    assert meta["display_name"] == "Updated name"


def test_manual_rate_cannot_override_measurement(upload_case):
    """The retained flag only checks agreement with measured metadata."""
    upload_case.argv.extend(["--fallback-rate", "0.1"])
    with pytest.raises(SystemExit):
        uploader.main()
    upload_case.factory.assert_not_called()


@pytest.mark.parametrize("failure", ["method", "cohort", "missing-demographics", "bad-remote-meta"])
def test_existing_guards_prevent_remote_writes(upload_case, failure):
    """Keep main's method/cohort checks before any mutation."""
    case = upload_case
    if failure == "method":
        pd.DataFrame({"method": ["other"], "user_id": ["synthetic-user"]}).to_parquet(case.src)
    elif failure == "cohort":
        pd.DataFrame({"user_id": ["someone-else"]}).to_parquet(case.demographics)
    elif failure == "missing-demographics":
        case.reader.side_effect = uploader.EntryNotFoundError("missing demographics")
    else:
        case.remote_meta.write_text("[]")
    with pytest.raises(SystemExit):
        uploader.main()
    case.api.create_commit.assert_not_called()
    case.api.create_repo.assert_not_called()
    case.api.upload_file.assert_not_called()


def test_new_method_uses_default_display_metadata(upload_case):
    """Absence of remote display metadata is allowed; absence of measurements is not."""
    case = upload_case
    case.reader.side_effect = [str(case.demographics), uploader.EntryNotFoundError("new method")]
    uploader.main()
    meta = json.loads(case.api.create_commit.call_args.kwargs["operations"][1].path_or_fileobj)
    assert meta["display_name"] == "wbm"
    assert meta["fallback_rate"] == 0.25


def test_new_repository_created_only_after_validation(upload_case):
    """A new repository receives both files in its first payload commit."""
    from types import SimpleNamespace

    case = upload_case
    case.api.repo_info.side_effect = [
        uploader.RepositoryNotFoundError("new repo"), SimpleNamespace(sha="initial-revision"),
    ]
    uploader.main()
    case.reader.assert_not_called()
    case.api.create_repo.assert_called_once()
    assert case.api.create_repo.call_args.kwargs["exist_ok"] is False
    assert case.api.create_commit.call_args.kwargs["parent_commit"] == "initial-revision"


def test_public_writer_to_uploader_and_stale_sidecar_detection(upload_case):
    """Carry a measured rate from the public exporter, then detect replaced predictions."""
    from downstream_evaluation.evaluation.per_user_pairs import write_per_user_pairs_parquet

    case = upload_case
    frame = pd.DataFrame({
        "method": ["wbm"], "task": ["synthetic-task"], "task_type": ["binary"],
        "subgroup_attr": ["all"], "subgroup_value": ["all"], "user_id": ["synthetic-user"],
        "y_true": [1], "y_pred": [1], "y_proba": [0.8],
    })
    write_per_user_pairs_parquet(frame, case.src, meta={
        "method": "wbm", "track": "downstream", "overall_fallback_rate": 1.0,
    })
    uploader.main()
    meta = json.loads(case.api.create_commit.call_args.kwargs["operations"][1].path_or_fileobj)
    assert meta["fallback_rate"] == 1.0
    # A new valid parquet at the same path must not inherit the old measured rate.
    frame.assign(y_proba=0.9).to_parquet(case.src)
    case.factory.reset_mock()
    with pytest.raises(SystemExit):
        uploader.main()
    case.factory.assert_not_called()


@pytest.mark.parametrize("bad_split", [None, [], 0])
def test_malformed_imputation_cell_is_not_skipped(tmp_path, bad_split):
    """Malformed cells must not disappear from the maximum-rate calculation."""
    payload = {"scenarios": {"gap": {
        "val": {"overall_fallback_rate": 0.1}, "test": bad_split,
    }}}
    with pytest.raises(ValueError, match="incomplete"):
        uploader._worst_case_fallback_rate(_write_results(tmp_path, payload))


def test_failed_commit_does_not_report_success_or_upload_separately(upload_case, capsys):
    """An atomic commit failure must propagate without a separate parquet upload."""
    case = upload_case
    case.api.create_commit.side_effect = RuntimeError("remote revision changed")
    with pytest.raises(RuntimeError, match="revision changed"):
        uploader.main()
    assert "Uploaded" not in capsys.readouterr().out
    case.api.upload_file.assert_not_called()
    case.api.create_commit.assert_called_once()


def test_lost_repository_access_does_not_skip_cohort_validation(upload_case):
    """An existing repository becoming inaccessible must stop the upload."""
    case = upload_case
    case.reader.side_effect = uploader.RepositoryNotFoundError("inaccessible")
    with pytest.raises(uploader.RepositoryNotFoundError):
        uploader.main()
    case.api.create_commit.assert_not_called()
    case.api.create_repo.assert_not_called()


def _submission_metadata(packet):
    block = packet.split('.meta.json ---\n', 1)[1]
    return json.JSONDecoder().raw_decode(block)[0]


@pytest.mark.parametrize('track', ['downstream', 'imputation', 'forecasting'])
@pytest.mark.parametrize('rate', [0.0, 0.25, 1.0])
def test_results_submission_packet_preserves_measured_rate(track, rate, tmp_path):
    """All result classes carry the same rate used by the uploader into the packet."""
    from openmhc import ForecastingResults, ImputationResults, PredictionResults

    if track == 'downstream':
        result = PredictionResults(records=[], overall_fallback_rate=rate)
        payload = {'overall_fallback_rate': rate}
    elif track == 'forecasting':
        result = ForecastingResults(per_channel={}, overall_fallback_rate=rate)
        payload = {'overall_fallback_rate': rate}
    else:
        payload = {
            'gap': {'test': {'overall_fallback_rate': rate}},
            'noise': {'test': {'overall_fallback_rate': rate / 2}},
        }
        result = ImputationResults(scenarios=payload)
    packet = result.to_submission_yaml(
        method_name='Example Model', submitter_team='Example Team', code_url='https://example.com',
    )
    meta = _submission_metadata(packet)
    assert meta['fallback_rate'] == rate
    assert meta['display_name'] == 'Example Model'
    assert meta['submitter'] == 'Example Team'
    assert f'{track}/example_model.parquet' in packet
    assert meta['fallback_rate'] == uploader._worst_case_fallback_rate(_write_results(tmp_path, payload))


@pytest.mark.parametrize('track', ['downstream', 'imputation', 'forecasting'])
@pytest.mark.parametrize('rate', [None, True, '0.1', -0.1, 1.1, float('nan'), float('inf')])
def test_submission_packet_rejects_invalid_rate(track, rate):
    """Invalid rates cannot be rendered into submission metadata on any track."""
    from openmhc import ForecastingResults, ImputationResults, PredictionResults

    if track == 'downstream':
        result = PredictionResults(records=[], overall_fallback_rate=rate)
    elif track == 'forecasting':
        result = ForecastingResults(per_channel={}, overall_fallback_rate=rate)
    else:
        result = ImputationResults(scenarios={'gap': {'test': {'overall_fallback_rate': rate}}})
    with pytest.raises(ValueError, match='measured fallback rate'):
        result.to_submission_yaml('Example', 'Team', 'https://example.com')


@pytest.mark.parametrize('scenarios', [
    {}, {'gap': {}}, {'gap': None}, {'gap': {'test': None}},
    {'gap': {'test': {'mae': 1.0}}},
    {'gap': {'val': {'overall_fallback_rate': 0.2}, 'test': {'mae': 1.0}}},
])
def test_imputation_submission_requires_complete_measurements(scenarios):
    """Legacy missing measurements must not become a zero in submission metadata."""
    from openmhc import ImputationResults

    with pytest.raises(ValueError, match='measured'):
        ImputationResults(scenarios=scenarios).to_submission_yaml(
            'Example', 'Team', 'https://example.com',
        )


@pytest.mark.parametrize('renderer_name', [
    'prediction_to_submission_yaml', 'imputation_to_submission_yaml', 'forecasting_to_submission_yaml',
])
def test_direct_submission_renderer_requires_rate(renderer_name):
    """Lower-level callers must supply a rate; there is no implicit zero."""
    from openmhc import _submission

    renderer = getattr(_submission, renderer_name)
    with pytest.raises(TypeError, match='fallback_rate'):
        renderer('Example', 'Team', 'https://example.com')
    assert _submission_metadata(renderer(
        'Example', 'Team', 'https://example.com', fallback_rate=0.25,
    ))['fallback_rate'] == 0.25


@pytest.mark.parametrize('identity', [
    {'method': 'other'}, {'track': 'imputation'},
    {'config': {'method': 'other'}}, {'config': {'track': 'forecasting'}},
])
def test_results_from_another_method_or_track_are_rejected(upload_case, tmp_path, identity):
    """An explicit identity mismatch must not be hidden by a matching numeric rate."""
    path = _write_results(tmp_path, {'overall_fallback_rate': 0.25, **identity})
    upload_case.argv.extend(['--results-json', str(path)])
    with pytest.raises(SystemExit):
        uploader.main()
    upload_case.factory.assert_not_called()


def test_matching_imputation_config_identity_is_supported(upload_case, tmp_path):
    """Read the actual wrapped runner shape, including its method configuration."""
    case = upload_case
    path = _write_results(tmp_path, {
        'scenarios': {'gap': {'test': {'overall_fallback_rate': 0.25}}},
        'config': {'method': 'wbm', 'seed': 42, 'mask_seed': 42},
    })
    case.argv[-1] = 'imputation'
    case.argv.extend(['--results-json', str(path)])
    uploader.main()
    case.api.create_commit.assert_called_once()


def test_forecasting_model_identity_checked_before_upload(upload_case):
    """Forecasting uses the model column, which must match the destination name."""
    case = upload_case
    case.argv[-1] = 'forecasting'
    pd.DataFrame({'model': ['wrong'], 'user_id': ['synthetic-user']}).to_parquet(case.src)
    with pytest.raises(SystemExit, match='model column'):
        uploader.main()
    case.factory.assert_not_called()


def test_original_file_changes_do_not_change_validated_upload(upload_case):
    """Publish the validated snapshot even if another process replaces the source."""
    from types import SimpleNamespace

    case = upload_case
    original = case.src.read_bytes()

    def remote_info(*args, **kwargs):
        pd.DataFrame({'method': ['other'], 'user_id': ['new-user']}).to_parquet(case.src)
        case.provenance.write_text(json.dumps({'overall_fallback_rate': 0.9}))
        return SimpleNamespace(sha='saved-revision')

    case.api.repo_info.side_effect = remote_info
    uploader.main()
    assert case.src.read_bytes() != original
    assert case.uploaded_predictions == [original]
    operations = case.api.create_commit.call_args.kwargs['operations']
    assert json.loads(operations[1].path_or_fileobj)['fallback_rate'] == 0.25
    assert not Path(operations[0].path_or_fileobj).exists()


def test_failed_commit_cleans_staged_copy(upload_case):
    """A failed upload leaves the user's original files and removes temporary copies."""
    case = upload_case
    original = case.src.read_bytes()
    case.api.create_commit.side_effect = RuntimeError('failed commit')
    with pytest.raises(RuntimeError, match='failed commit'):
        uploader.main()
    operations = case.api.create_commit.call_args.kwargs['operations']
    assert not Path(operations[0].path_or_fileobj).exists()
    assert case.src.read_bytes() == original
