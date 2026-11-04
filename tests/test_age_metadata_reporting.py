"""Expose incomplete age coverage without swallowing broken enrollment files."""

import json
import logging

import pandas as pd
import pytest


@pytest.fixture
def demographic_inputs(age_label_environment, tmp_path):
    """Write a valid cohort with one missing birth year and usable sex labels."""
    folder = age_label_environment
    (folder / "enrollment_info.json").write_text(
        json.dumps(
            {
                "good": {"birth_year": 1980, "createdOn": "2020-01-01"},
                "bad": {"age": 40, "createdOn": "2020-01-01"},
            }
        )
    )
    (folder / "last_labels.json").write_text(
        json.dumps(
            {
                "BiologicalSex": {
                    uid: {"timestamps": ["2020-01-01"], "values": [True]} for uid in ("good", "bad")
                },
                "age": {
                    uid: {"timestamps": ["2020-01-01"], "values": [40]} for uid in ("good", "bad")
                },
            }
        )
    )
    return folder, tmp_path / "predictions"


def _groups(track, inputs, users):
    folder, output = inputs
    if track == "prediction":
        from downstream_evaluation.evaluation.predictions_io import write_subgroup_map

        write_subgroup_map(output, users)
        return json.loads((output / "_subgroups.json").read_text())
    if track == "imputation":
        from imputation_evaluation.sensitivity import build_subgroup_mapping
        groups = build_subgroup_mapping(
            {"user_id": users},
            {"test": list(range(len(users)))},
        )["test"]
        return {uid: groups[i] for i, uid in enumerate(users)}
    from forecasting_evaluation.metrics.fairness_skill_score_summary import load_user_demographics

    return load_user_demographics(
        user_ids=set(users),
        labels_path=folder / "last_labels.json",
        enrollment_path=folder / "enrollment_info.json",
    )


def _coverage_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "unknown age groups" in r.getMessage()]


@pytest.mark.parametrize("track", ["prediction", "imputation", "forecasting"])
def test_partial_age_coverage_warns_and_keeps_users(track, demographic_inputs, caplog):
    """Count unique users, retain sex and unknown ages, and warn once per cohort."""
    caplog.set_level(logging.WARNING)
    groups = _groups(track, demographic_inputs, ["good", "bad", "good"])
    assert set(groups) == {"good", "bad"}
    assert groups["good"]["age_group"] == "40-49"
    assert groups["bad"] == {"age_group": "unknown", "sex": "male"}
    warnings = _coverage_warnings(caplog)
    assert len(warnings) == 1
    assert "1/2 users have unknown age groups" in warnings[0]
    assert "birth_year" in warnings[0] and "createdOn" in warnings[0]
    assert "no usable age groups" not in warnings[0]


@pytest.mark.parametrize("track", ["prediction", "imputation", "forecasting"])
@pytest.mark.parametrize(
    "record",
    [
        {"age": 40, "createdOn": "2020-01-01"},
        {"birth_year": 1980},
        {"birth_year": "invalid", "createdOn": "2020-01-01"},
        {"birth_year": 1980, "createdOn": "invalid"},
        {"birth_year": 2005, "createdOn": "2020-01-01"},
        None,
        [],
    ],
)
def test_all_unknown_ages_warn_that_fairness_is_unavailable(
    track, record, demographic_inputs, caplog
):
    """Invalid individual records remain unknown, with explicit all-unknown reporting."""
    folder, _ = demographic_inputs
    (folder / "enrollment_info.json").write_text(json.dumps({"good": record, "bad": record}))
    caplog.set_level(logging.WARNING)
    groups = _groups(track, demographic_inputs, ["good", "bad"])
    assert len(groups) == 2
    assert all(g == {"age_group": "unknown", "sex": "male"} for g in groups.values())
    warnings = _coverage_warnings(caplog)
    assert len(warnings) == 1
    assert "2/2 users have unknown age groups" in warnings[0]
    assert "Age-based fairness has no usable age groups" in warnings[0]


@pytest.mark.parametrize("track", ["prediction", "imputation", "forecasting"])
def test_absent_enrollment_record_is_reported_and_retained(track, demographic_inputs, caplog):
    """Missing participants are counted without removing them from the output."""
    caplog.set_level(logging.WARNING)
    groups = _groups(track, demographic_inputs, ["good", "absent"])
    assert groups["absent"] == {"age_group": "unknown", "sex": "unknown"}
    assert "1/2 users have unknown age groups" in _coverage_warnings(caplog)[0]


@pytest.mark.parametrize("track", ["prediction", "imputation", "forecasting"])
@pytest.mark.parametrize("payload", ['{"good":', "[]", "null", '"not an object"'])
def test_corrupt_enrollment_raises_instead_of_unknown(track, payload, demographic_inputs, caplog):
    """File-level errors must not be swallowed by per-user missing-age handling."""
    folder, output = demographic_inputs
    (folder / "enrollment_info.json").write_text(payload)
    caplog.set_level(logging.WARNING)
    with pytest.raises(ValueError, match="[Ee]nrollment"):
        _groups(track, demographic_inputs, ["good", "bad"])
    assert not _coverage_warnings(caplog)
    assert not (output / "_subgroups.json").exists()


@pytest.mark.parametrize("track", ["prediction", "imputation", "forecasting"])
def test_missing_enrollment_file_raises(track, demographic_inputs, caplog):
    """A missing file is an installation error, not a cohort of unknown ages."""
    folder, _ = demographic_inputs
    (folder / "enrollment_info.json").unlink()
    with pytest.raises(FileNotFoundError):
        _groups(track, demographic_inputs, ["good"])
    assert not _coverage_warnings(caplog)


@pytest.mark.parametrize("track", ["prediction", "imputation", "forecasting"])
@pytest.mark.parametrize("users", [[], ["good"]])
def test_no_unknown_ages_has_no_warning(track, users, demographic_inputs, caplog):
    """Valid and empty cohorts must not generate missing-age warnings."""
    caplog.set_level(logging.WARNING)
    groups = _groups(track, demographic_inputs, users)
    assert set(groups) == set(users)
    assert not _coverage_warnings(caplog)


def _age_error_message(caller):
    from downstream_evaluation.data.provider import age_target
    from downstream_evaluation.demo_covariates import build_demo_user_lookup_from_labels_df

    with pytest.raises(ValueError, match="Cannot calculate age target") as error:
        if caller == "target":
            age_target("bad")
        else:
            build_demo_user_lookup_from_labels_df(
                pd.DataFrame({"user_id": ["bad"], "age": [40.0]}), ["age"]
            )
    return str(error.value)


@pytest.mark.parametrize("caller", ["target", "linear_covariates"])
def test_age_only_enrollment_error_explains_migration(caller, demographic_inputs):
    """Age targets and Linear covariates explain why stored enrollment age is insufficient."""
    folder, _ = demographic_inputs
    (folder / "enrollment_info.json").write_text(
        json.dumps({"bad": {"age": 40, "createdOn": "2020-01-01"}})
    )
    message = _age_error_message(caller)
    assert "birth_year" in message
    assert "new directory" in message
    assert "labels.configure" in message


@pytest.mark.parametrize("caller", ["target", "linear_covariates"])
def test_single_missing_birth_year_does_not_advise_download(caller, demographic_inputs):
    """A gap in one record is not an outdated release, so a new download is not suggested."""
    message = _age_error_message(caller)
    assert "birth_year" in message
    assert "new directory" not in message
    assert "labels.configure" not in message
