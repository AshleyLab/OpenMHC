"""One definition of each participant's age and sex groups."""

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


def _paths(reader, folder):
    """The configured dataset, or the same files named explicitly."""
    from openmhc._label_paths import LabelPaths

    return LabelPaths.resolve(folder.parent) if reader == "explicit_paths" else None


READERS = pytest.mark.parametrize("reader", ["configured", "explicit_paths"])


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
        ((1,), "unknown"),
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


@READERS
def test_missing_metadata_keeps_users_and_reports_age_count(
    reader, label_schema_environment, caplog
):
    """Missing values become unknown without dropping anyone, and the age gap is reported."""
    caplog.set_level(logging.WARNING)
    _write(
        label_schema_environment,
        {"known": {"birth_year": 1980, "createdOn": "2020-01-01"}, "no_year": {"age": 40}},
        {"known": _sex("Male")},
    )
    paths = _paths(reader, label_schema_environment)
    assert labels.user_demographics(["no_year", "known", "absent"], paths=paths) == {
        "absent": {"age_group": "unknown", "sex": "unknown"},
        "known": {"age_group": "40-49", "sex": "male"},
        "no_year": {"age_group": "unknown", "sex": "unknown"},
    }
    assert "2/3 users have unknown age groups" in _coverage_warnings(caplog)[0]


@READERS
def test_complete_and_empty_cohorts_do_not_warn(reader, label_schema_environment, caplog):
    caplog.set_level(logging.WARNING)
    _write(label_schema_environment, {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}})
    paths = _paths(reader, label_schema_environment)
    assert labels.user_demographics(["u"], paths=paths)["u"]["age_group"] == "40-49"
    assert labels.user_demographics([], paths=paths) == {}
    assert not _coverage_warnings(caplog)


@READERS
def test_partial_age_coverage_warns_once_and_counts_unique_users(
    reader, label_schema_environment, caplog
):
    """Duplicates count once, and sex and unknown ages are retained."""
    _write(
        label_schema_environment,
        {"good": {"birth_year": 1980, "createdOn": "2020-01-01"}, "bad": {"age": 40}},
        {"good": _sex(True), "bad": _sex(True)},
    )
    caplog.set_level(logging.WARNING)
    groups = labels.user_demographics(
        ["good", "bad", "good"], paths=_paths(reader, label_schema_environment)
    )
    assert groups == {
        "good": {"age_group": "40-49", "sex": "male"},
        "bad": {"age_group": "unknown", "sex": "male"},
    }
    warnings = _coverage_warnings(caplog)
    assert len(warnings) == 1
    assert "1/2 users have unknown age groups" in warnings[0]
    assert "birth_year" in warnings[0] and "createdOn" in warnings[0]
    assert "no usable age groups" not in warnings[0]


@READERS
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
def test_all_unknown_ages_are_reported_explicitly(reader, record, label_schema_environment, caplog):
    """Invalid individual records stay unknown, with an explicit all-unknown warning."""
    _write(
        label_schema_environment,
        {"good": record, "bad": record},
        {"good": _sex(True), "bad": _sex(True)},
    )
    caplog.set_level(logging.WARNING)
    groups = labels.user_demographics(
        ["good", "bad"], paths=_paths(reader, label_schema_environment)
    )
    assert len(groups) == 2
    assert all(group == {"age_group": "unknown", "sex": "male"} for group in groups.values())
    warnings = _coverage_warnings(caplog)
    assert len(warnings) == 1
    assert "2/2 users have unknown age groups" in warnings[0]
    assert "no usable age groups" in warnings[0]


@READERS
def test_a_cohort_with_no_sex_labels_is_reported(reader, label_schema_environment, caplog):
    """A labels file without BiologicalSex must not silently empty the sex cohorts."""
    caplog.set_level(logging.WARNING)
    enrollment = {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}}

    def sex_warnings():
        return [r for r in caplog.records if "usable BiologicalSex" in r.getMessage()]

    _write(label_schema_environment, enrollment)
    paths = _paths(reader, label_schema_environment)
    assert labels.user_demographics(["u"], paths=paths)["u"]["sex"] == "unknown"
    assert len(sex_warnings()) == 1

    caplog.clear()
    assert labels.user_demographics([], paths=paths) == {}
    assert not sex_warnings()


def test_partial_sex_coverage_does_not_warn(label_schema_environment, caplog):
    caplog.set_level(logging.WARNING)
    _write(
        label_schema_environment,
        {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}},
        {"u": _sex("Male")},
    )
    labels.user_demographics(["u", "absent"])
    assert not [r for r in caplog.records if "usable BiologicalSex" in r.getMessage()]


@pytest.mark.parametrize("name", ["enrollment_info.json", "last_labels.json"])
def test_broken_files_raise_instead_of_becoming_unknown(label_schema_environment, name):
    """A file problem is not a missing value, even for a cohort with no users."""
    _write(label_schema_environment, {"u": {"birth_year": 1980, "createdOn": "2020-01-01"}})
    (label_schema_environment / name).write_text("{")
    with pytest.raises(ValueError):
        labels.user_demographics(["u"])
    with pytest.raises(ValueError):
        labels.user_demographics([])


@READERS
@pytest.mark.parametrize("payload", ['{"good":', "[]", "null", '"not an object"'])
def test_corrupt_enrollment_raises_instead_of_unknown(
    reader, payload, label_schema_environment, caplog
):
    """File-level errors must not be swallowed by per-user missing-age handling."""
    _write(label_schema_environment, {})
    (label_schema_environment / "enrollment_info.json").write_text(payload)
    caplog.set_level(logging.WARNING)
    with pytest.raises(ValueError, match="[Ee]nrollment"):
        labels.user_demographics(["good"], paths=_paths(reader, label_schema_environment))
    assert not _coverage_warnings(caplog)


@READERS
def test_missing_enrollment_file_raises(reader, label_schema_environment, caplog):
    """A missing file is an installation error, not a cohort of unknown ages."""
    _write(label_schema_environment, {})
    (label_schema_environment / "enrollment_info.json").unlink()
    with pytest.raises(FileNotFoundError):
        labels.user_demographics(["good"], paths=_paths(reader, label_schema_environment))
    assert not _coverage_warnings(caplog)


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
