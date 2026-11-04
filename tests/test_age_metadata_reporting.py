"""Explain why an age read fails when enrollment birth years are missing."""

import json

import pandas as pd
import pytest


@pytest.fixture
def age_inputs(age_label_environment):
    """Write a valid cohort with one missing birth year."""
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
                "age": {
                    uid: {"timestamps": ["2020-01-01"], "values": [40]} for uid in ("good", "bad")
                },
            }
        )
    )
    return folder


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
def test_age_only_enrollment_error_explains_migration(caller, age_inputs):
    """Age targets and Linear covariates explain why stored enrollment age is insufficient."""
    (age_inputs / "enrollment_info.json").write_text(
        json.dumps({"bad": {"age": 40, "createdOn": "2020-01-01"}})
    )
    message = _age_error_message(caller)
    assert "birth_year" in message
    assert "new directory" in message
    assert "labels.configure" in message


@pytest.mark.parametrize("caller", ["target", "linear_covariates"])
def test_single_missing_birth_year_does_not_advise_download(caller, age_inputs):
    """A gap in one record is not an outdated release, so a new download is not suggested."""
    message = _age_error_message(caller)
    assert "birth_year" in message
    assert "new directory" not in message
    assert "labels.configure" not in message
