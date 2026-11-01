"""Explicit-date age queries use enrolment metadata without changing stored labels."""

import json

import pandas as pd
import pytest

import labels
from labels import api


@pytest.fixture
def age_dataset(age_label_environment):
    """Provide different birth years and deliberately conflicting stored ages."""
    folder = age_label_environment
    (folder / "enrollment_info.json").write_text(
        json.dumps(
            {
                "known": {"birth_year": 1980, "age": 99, "createdOn": "2010-06-01"},
                "missing": {"age": 99},
                "legacy": {"birthdate": "2000-02-29"},
            }
        )
    )
    (folder / "last_labels.json").write_text(
        json.dumps({"age": {"known": {"timestamps": ["2015-06-01"], "values": [99]}}})
    )
    return folder.parent


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2019-12-31", 39),
        ("2020-01-01", 40),
        ("2020-12-31", 40),
        ("2021-01-01T00:30:00+14:00", 41),
    ],
)
def test_age_uses_supplied_calendar_year(age_dataset, date, expected):
    """Use the given date, including its timezone, rather than stored age or today."""
    result = labels.age_at("known", pd.Timestamp(date))
    assert result == expected
    assert isinstance(result, int)
    assert api.STORE._labels_index is None


def test_age_query_does_not_change_stored_label(age_dataset):
    """Adding dynamic age leaves the existing nearest-observation API unchanged."""
    query = pd.Timestamp("2020-06-01")
    before = labels.get_labels("known", query, "age")
    assert labels.age_at("known", query) == 40
    after = labels.get_labels("known", query, "age")
    assert after == before
    assert after.value == 99
    assert after.matched_timestamp == pd.Timestamp("2015-06-01")


@pytest.mark.parametrize("user", ["missing", "absent"])
def test_missing_birth_year_does_not_use_stored_age(age_dataset, user):
    """Missing metadata raises instead of substituting a static age."""
    with pytest.raises(KeyError):
        labels.age_at(user, pd.Timestamp("2020-01-01"))


@pytest.mark.parametrize("date", [None, pd.NaT, "2020-01-01", 2020])
def test_invalid_date_requires_an_explicit_timestamp(age_dataset, date):
    """Reject missing dates and values that would need implicit date interpretation."""
    with pytest.raises(ValueError, match="valid pandas.Timestamp"):
        labels.age_at("known", date)


def test_date_before_birth_year_is_rejected(age_dataset):
    """A query before the birth year cannot produce a valid calendar-year age."""
    with pytest.raises(ValueError, match="before.*birth year"):
        labels.age_at("known", pd.Timestamp("1979-12-31"))


@pytest.mark.parametrize("birth_year", [0, -1, 10000, "invalid", float("nan"), 1980.5, True])
def test_invalid_birth_year_is_rejected(birth_year):
    """Malformed enrolment metadata produces a clear error."""
    enrollment = api.EnrollmentIndex({"u": {"birth_year": birth_year}})
    with pytest.raises(ValueError, match="[Bb]irth year"):
        enrollment.age_at("u", pd.Timestamp("2020-01-01"))


def test_legacy_birthdate_uses_year_not_birthday(age_dataset):
    """Legacy enrolment records retain the same calendar-year arithmetic."""
    assert labels.age_at("legacy", pd.Timestamp("2020-01-01")) == 20
    assert labels.age_at("legacy", pd.Timestamp("2020-02-29")) == 20


def test_switching_datasets_refreshes_age(age_dataset, tmp_path, write_age_dataset):
    """An imported age function follows configuration without stale birth-year data."""
    from labels import age_at

    b = tmp_path / "b"
    folder = write_age_dataset(b)
    (folder / "enrollment_info.json").write_text(json.dumps({"known": {"birth_year": 1990}}))
    date = pd.Timestamp("2020-01-01")
    assert age_at("known", date) == 40
    labels.configure(b)
    assert age_at("known", date) == 30
    labels.configure(age_dataset)
    assert age_at("known", date) == 40
