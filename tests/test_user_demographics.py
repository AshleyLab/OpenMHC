"""One definition of each participant's fairness groups, shared by all tracks."""

import json
import logging

import pytest

import labels
from labels import api


def _write(folder, enrollment, sexes=None):
    (folder / "enrollment_info.json").write_text(json.dumps(enrollment))
    payload = {"BiologicalSex": sexes} if sexes is not None else {}
    (folder / "last_labels.json").write_text(json.dumps(payload))


def _sex(*values):
    return {"timestamps": [f"20{i:02d}-01-01" for i in range(len(values))], "values": list(values)}


def _coverage_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "unknown age groups" in r.getMessage()]


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (0, "unknown"),
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
        (100, "60+"),
    ],
)
def test_age_group_boundaries(label_schema_environment, age, expected):
    """Age is enrollment year minus birth year, and under-18 ages stay unknown."""
    _write(label_schema_environment, {"u": {"birth_year": 2000 - age, "createdOn": "2000-06-01"}})
    assert labels.user_demographics(["u"])["u"]["age_group"] == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((True,), "male"),
        ((False,), "female"),
        (("Male",), "male"),
        ((" female ",), "female"),
        (("Other",), "unknown"),
        ((None,), "unknown"),
        (("Male", "Female"), "female"),
    ],
)
def test_sex_group_uses_latest_label(label_schema_environment, values, expected):
    """Sex is the latest BiologicalSex value, and unrecognized values stay unknown."""
    _write(
        label_schema_environment,
        {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}},
        {"u": _sex(*values)},
    )
    assert labels.user_demographics(["u"])["u"]["sex"] == expected


def test_missing_metadata_keeps_users_and_reports_age_count(label_schema_environment, caplog):
    """Missing values become unknown without dropping anyone, and the age gap is reported."""
    caplog.set_level(logging.WARNING)
    _write(
        label_schema_environment,
        {"known": {"birth_year": 1980, "createdOn": "2020-01-01"}, "no_year": {"age": 40}},
        {"known": _sex("Male")},
    )
    assert labels.user_demographics(["no_year", "known", "absent"]) == {
        "absent": {"age_group": "unknown", "sex": "unknown"},
        "known": {"age_group": "40-49", "sex": "male"},
        "no_year": {"age_group": "unknown", "sex": "unknown"},
    }
    assert "2/3 users have unknown age groups" in _coverage_warnings(caplog)[0]


def test_complete_and_empty_cohorts_do_not_warn(label_schema_environment, caplog):
    caplog.set_level(logging.WARNING)
    _write(label_schema_environment, {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}})
    assert labels.user_demographics(["u"])["u"]["age_group"] == "40-49"
    assert labels.user_demographics([]) == {}
    assert not _coverage_warnings(caplog)


@pytest.mark.parametrize("name", ["enrollment_info.json", "last_labels.json"])
def test_broken_files_raise_instead_of_becoming_unknown(label_schema_environment, name):
    """A file problem is not a missing value, even for a cohort with no users."""
    _write(label_schema_environment, {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}})
    (label_schema_environment / name).write_text("{")
    with pytest.raises(ValueError):
        labels.user_demographics(["u"])


def test_follows_the_configured_dataset(label_schema_environment, tmp_path, write_label_schemas):
    first = label_schema_environment.parent
    _write(label_schema_environment, {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}})
    second = tmp_path / "second"
    _write(write_label_schemas(second), {"u": {"birth_year": 2000, "createdOn": "2020-01-01"}})
    labels.configure(second)
    assert labels.user_demographics(["u"])["u"]["age_group"] == "18-29"
    labels.configure(first)
    assert labels.user_demographics(["u"])["u"]["age_group"] == "40-49"


def test_explicit_paths_leave_the_configured_dataset_alone(
    label_schema_environment, tmp_path, write_label_schemas
):
    """Reading another dataset's files does not reconfigure or load the global store."""
    from openmhc._label_paths import LabelPaths

    _write(label_schema_environment, {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}})
    other = tmp_path / "other"
    _write(write_label_schemas(other), {"u": {"birth_year": 2000, "createdOn": "2020-01-01"}})
    active = api._PATHS
    groups = labels.user_demographics(["u"], paths=LabelPaths.resolve(other))
    assert groups["u"]["age_group"] == "18-29"
    assert api._PATHS is active
    assert api.STORE._enrollment is None
    assert api.STORE._labels_index is None


def test_all_tracks_return_the_same_groups(label_schema_environment, tmp_path):
    """Track-specific entry points are thin callers of the one definition."""
    from downstream_evaluation.evaluation.predictions_io import write_subgroup_map
    from openmhc._label_paths import LabelPaths

    _write(
        label_schema_environment,
        {
            "a": {"birth_year": 1980, "createdOn": "2020-01-01"},
            "b": {"birth_year": 1990, "createdOn": "2015-01-01"},
            "c": {"age": 40},
        },
        {"a": _sex("Male"), "b": _sex("Female")},
    )
    users = ["a", "b", "c", "absent"]
    expected = labels.user_demographics(users)
    write_subgroup_map(tmp_path / "predictions", users)
    assert json.loads((tmp_path / "predictions/_subgroups.json").read_text()) == expected
    paths = LabelPaths.resolve(label_schema_environment.parent)
    assert labels.user_demographics(users, paths=paths) == expected
