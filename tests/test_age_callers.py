"""Keep target dates and eligibility while using enrollment ages for subgroups."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from labels import api


@pytest.fixture
def caller_dataset(age_label_environment):
    """Separate survey age, enrollment age, lookup values and missing metadata."""
    folder = age_label_environment
    enrollment = {
        "known": {"birth_year": 1980, "createdOn": "2019-12-31", "age": 99},
        "no_target": {"birth_year": 1980, "createdOn": "2020-01-01"},
        "under18": {"birth_year": 2000, "createdOn": "2014-01-01"},
        "missing_year": {"createdOn": "2019-12-31", "age": 99},
        "no_date": {"birth_year": 1980, "age": 99},
        "bad_date": {"birth_year": 1980, "createdOn": "invalid"},
        "bad_year": {"birth_year": 1980.5, "createdOn": "2019-01-01"},
    }
    (folder / "enrollment_info.json").write_text(json.dumps(enrollment))
    (folder / "last_labels.json").write_text(
        json.dumps(
            {
                "age": {"known": {"timestamps": ["2025-06-01"], "values": [99]}},
                "BiologicalSex": {
                    u: {"timestamps": ["2025-06-01"], "values": ["Male"]} for u in enrollment
                },
            }
        )
    )
    # Eligibility is already in the lookup; callers must not rebuild it here.
    (folder / "label_validity.json").write_text(json.dumps({"age": {"known": [False]}}))
    lookup = pd.DataFrame(
        {
            "user_id": ["known", "known", "known", "no_target", "missing_year"],
            "date": pd.to_datetime(
                ["2018-01-01", "2019-01-01", "2026-01-01", "2020-01-01", "2020-01-01"]
            ),
            "age": [88.0, -1.0, 88.0, -1.0, np.nan],
            "BiologicalSex": [1, 1, 1, 1, 1],
            "BMI_values": [25.0, 25.0, 25.0, np.nan, -1.0],
        }
    )
    return folder, lookup


@pytest.mark.parametrize("grain", ["daily", "weekly", "series"])
def test_target_changes_values_only(caller_dataset, tmp_path, grain):
    """Daily/weekly masks and missing-target cohorts survive dynamic age lookup."""
    from downstream_evaluation.data.provider import TaskDataProvider

    _, lookup = caller_dataset
    date_col = "week_start" if grain == "weekly" else "date"
    lookup = lookup.rename(columns={"date": date_col})
    path = tmp_path / "lookup.parquet"
    lookup.to_parquet(path)
    splits = {"train": ["known", "no_target", "missing_year"], "validation": ["known"]}
    provider = TaskDataProvider(path, splits, grain)
    td = provider.task_data("age", "train")
    assert td.user_ids.tolist() == ["known"]
    assert td.labels.tolist() == [45]
    assert provider.cohort("age")["train"] == ["known"]
    np.testing.assert_array_equal(td.dates[0], lookup.iloc[[0, 2]][date_col].to_numpy())
    assert provider.task_data("age", "val").labels.tolist() == [45]
    assert provider.task_data("BiologicalSex", "train").user_ids.tolist() == [
        "known",
        "missing_year",
        "no_target",
    ]
    assert (
        api.get_labels("known", pd.Timestamp("2020-01-01"), "age", return_valid_only=False).value
        == 99
    )


@pytest.mark.parametrize("fault", ["missing_year", "missing_series", "multiple", "missing_value"])
def test_eligible_target_metadata_errors_do_not_shrink_cohort(caller_dataset, tmp_path, fault):
    """Broken eligible targets fail explicitly instead of falling back or dropping users."""
    from downstream_evaluation.data.provider import TaskDataProvider

    folder, lookup = caller_dataset
    if fault == "missing_year":
        payload = json.loads((folder / "enrollment_info.json").read_text())
        del payload["known"]["birth_year"]
        (folder / "enrollment_info.json").write_text(json.dumps(payload))
    else:
        payload = json.loads((folder / "last_labels.json").read_text())
        if fault == "missing_series":
            payload["age"] = {}
        elif fault == "multiple":
            payload["age"]["known"] = {
                "timestamps": ["2019-01-01", "2025-06-01"],
                "values": [39, 45],
            }
        else:
            payload["age"]["known"]["values"] = [None]
        (folder / "last_labels.json").write_text(json.dumps(payload))
    path = tmp_path / "lookup.parquet"
    lookup.to_parquet(path)
    with pytest.raises(ValueError, match="Cannot calculate age target"):
        TaskDataProvider(path, {"test": ["known"]}).task_data("age", "test")


def test_tsfm_uses_same_age_anchor(caller_dataset):
    """The age window date stays at the survey, independent of query and validity mask."""
    from downstream_evaluation.models.tsfm import _label_timestamp

    for date in ("1900-01-01", "2100-01-01"):
        assert _label_timestamp("known", "age", pd.Timestamp(date)) == pd.Timestamp("2025-06-01")
    assert _label_timestamp("no_target", "age", pd.Timestamp("2020-01-01")) is None
    assert _label_timestamp("known", "BiologicalSex", pd.Timestamp("2020-01-01")) == pd.Timestamp(
        "2025-06-01"
    )


def test_linear_age_covariate_keeps_reference_missingness_and_exclusions(caller_dataset):
    """Use survey age without filling absent age labels from enrollment metadata."""
    from downstream_evaluation.demo_covariates import (
        apply_demographics,
        build_demo_user_lookup_from_labels_df,
    )

    _, lookup = caller_dataset
    covs = ["age", "BiologicalSex", "BMI_values"]
    demo = build_demo_user_lookup_from_labels_df(lookup, covs)
    np.testing.assert_array_equal(demo["known"], [45, 1, 25])
    np.testing.assert_array_equal(demo["no_target"], [0, 1, 0])
    np.testing.assert_array_equal(demo["missing_year"], [0, 1, 0])
    x = np.zeros((1, 2))
    np.testing.assert_array_equal(apply_demographics(x, ["known"], "age", demo, covs), [[0, 0]])
    np.testing.assert_array_equal(
        apply_demographics(x, ["known"], "Diabetes", demo, covs), [[0, 0, 45, 1, 25]]
    )


def test_all_tracks_use_enrollment_and_keep_unknown_users(caller_dataset, tmp_path):
    """Survey/sample/fixed dates cannot change enrollment subgroups in any track."""
    from downstream_evaluation.evaluation.predictions_io import write_subgroup_map
    from forecasting_evaluation.metrics.fairness_skill_score_summary import load_user_demographics
    from imputation_evaluation.sensitivity import build_subgroup_mapping

    folder, lookup = caller_dataset
    users = ["known", "no_target", "missing_year", "no_date", "bad_date", "bad_year", "absent"]
    path = tmp_path / "lookup.parquet"
    lookup.drop(columns="age").to_parquet(path)
    out = tmp_path / "predictions"
    write_subgroup_map(out, path, users)
    prediction = json.loads((out / "_subgroups.json").read_text())
    forecast = load_user_demographics(
        user_ids=set(users),
        labels_path=folder / "last_labels.json",
        enrollment_path=folder / "enrollment_info.json",
    )
    hf = {"user_id": users + ["known"], "date": ["1900-01-01"] * len(users) + ["2099-01-01"]}
    impute = build_subgroup_mapping(
        hf, {"test": list(range(len(users) + 1))}, [18, 30, 40, 50, 60], store=api.STORE
    )["test"]
    expected = {u: "unknown" for u in users}
    expected.update(known="30-39", no_target="40-49")
    assert {u: prediction[u]["age_group"] for u in users} == expected
    assert {u: forecast[u]["age_group"] for u in users} == expected
    assert {u: impute[i]["age_group"] for i, u in enumerate(users)} == expected
    assert impute[len(users)]["age_group"] == "30-39"
    assert (
        prediction["known"]["sex"]
        == forecast["known"]["sex"]
        == impute[users.index("known")]["sex"]
        == "male"
    )
    assert api.age_at_enrollment("known") == 39
    assert api.age_at("known", pd.Timestamp("2025-06-01")) == 45


@pytest.mark.parametrize("created_on", [None, "", "NaT", "bad", 123, True])
def test_invalid_enrollment_date_never_uses_stored_age(created_on):
    """A missing enrollment date must not turn into today's date or a fixed date."""
    enrollment = api.EnrollmentIndex(
        {"u": {"birth_year": 1980, "createdOn": created_on, "age": 99}}
    )
    with pytest.raises((KeyError, ValueError)):
        enrollment.age_at_enrollment("u")


@pytest.mark.parametrize(
    "script",
    [
        "compute_imputation_paper_metrics.py",
        "imputation/bootstrap_imputation_draws.py",
        "imputation/parity/produce_per_method_per_user_errors.py",
        "imputation/parity/parity_locf.py",
    ],
)
def test_offline_imputation_matches_enrollment(caller_dataset, tmp_path, monkeypatch, script):
    """Saved-pair analysis shares live age semantics, including repeated samples."""
    path = Path(__file__).resolve().parents[1] / "scripts/paper_results" / script
    spec = importlib.util.spec_from_file_location("age_offline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = pa.table(
        {
            "sample_idx": [7, 2, 9, 4],
            "user_id": ["known", "known", "no_target", "missing_year"],
            "date": ["1900-01-01", "2099-01-01", "invalid", "2025-01-01"],
        }
    )
    monkeypatch.setattr(module, "load_sample_manifest", lambda *args: manifest)
    groups = module._build_subgroup_mapping(tmp_path, "test", [18, 30, 40, 50, 60])
    assert {i: row["age_group"] for i, row in groups.items()} == {
        7: "30-39",
        2: "30-39",
        9: "40-49",
        4: "unknown",
    }


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (14, "unknown"),
        (17, "unknown"),
        (18, "18-29"),
        (29, "18-29"),
        (30, "30-39"),
        (39, "30-39"),
        (40, "40-49"),
        (49, "40-49"),
        (50, "50-59"),
        (59, "50-59"),
        (60, "60+"),
        (90, "60+"),
    ],
)
def test_age_band_boundaries_agree_across_tracks(age, expected):
    """Under-18 ages stay unknown; each decade starts at the same age in all tracks."""
    from downstream_evaluation.evaluation.predictions_io import _age_group
    from forecasting_evaluation.metrics.fairness_skill_score_summary import bin_age as forecast_bin
    from imputation_evaluation.sensitivity import bin_age as imputation_bin

    assert (
        _age_group(age)
        == forecast_bin(age)
        == imputation_bin(age, [18, 30, 40, 50, 60])
        == expected
    )


def test_callers_follow_dataset_switches(caller_dataset, tmp_path):
    """Imported callers use the newly configured birth years and enrollment dates."""
    import shutil

    import labels
    from downstream_evaluation.data.provider import TaskDataProvider
    from downstream_evaluation.demo_covariates import build_demo_user_lookup_from_labels_df
    from downstream_evaluation.evaluation.predictions_io import write_subgroup_map
    from imputation_evaluation.sensitivity import get_user_demographics

    folder, lookup = caller_dataset
    b = tmp_path / "dataset_b"
    shutil.copytree(folder.parent, b)
    payload = json.loads((b / "labels/enrollment_info.json").read_text())
    payload["known"].update(birth_year=1990, createdOn="2018-01-01")
    (b / "labels/enrollment_info.json").write_text(json.dumps(payload))
    path = tmp_path / "lookup.parquet"
    lookup.to_parquet(path)
    for root, target_age, enrollment_age, group in (
        (folder.parent, 45, 39, "30-39"),
        (b, 35, 28, "18-29"),
        (folder.parent, 45, 39, "30-39"),
    ):
        labels.configure(root)
        td = TaskDataProvider(path, {"test": ["known"]}).task_data("age", "test")
        assert td.labels.tolist() == [target_age]
        demo = build_demo_user_lookup_from_labels_df(lookup, ["age"])
        assert demo["known"][0] == target_age
        assert (
            get_user_demographics(api.STORE, ["known"])["known"]["age_at_enrollment"]
            == enrollment_age
        )
        write_subgroup_map(tmp_path / "predictions", path, ["known"])
        groups = json.loads((tmp_path / "predictions/_subgroups.json").read_text())
        assert groups["known"]["age_group"] == group
