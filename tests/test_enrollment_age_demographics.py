"""Tracks 2/3 use released enrollment age independently of sample dates."""

import importlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecasting_evaluation.metrics.fairness_skill_score_summary import (
    bin_age as forecasting_bin_age,
)
from forecasting_evaluation.metrics.fairness_skill_score_summary import (
    load_user_demographics,
)
from imputation_evaluation.sensitivity import (
    bin_age,
    build_subgroup_mapping,
    get_user_demographics,
)
from labels import api


@pytest.fixture
def enrollment_store(tmp_path):
    """Create release-shaped ages with conflicting legacy fields and label ages."""
    enrollment = {
        "known": {"age": 35, "birth_year": 1900},
        "null": {"age": None, "birth_year": 1985},
        "legacy": {"birth_year": 1985},
        "unused": {"age": 65},
    }
    labels = {
        "age": {"known": {"timestamps": ["2020-01-01"], "values": [75]}},
        "BiologicalSex": {
            "known": {"timestamps": ["2020-01-01"], "values": [True]},
            "null": {"timestamps": ["2020-01-01"], "values": [False]},
        },
    }
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps(labels))
    enrollment_path.write_text(json.dumps(enrollment))
    return api.LabelsStore(labels_path, enrollment_path)


def test_imputation_reads_enrollment_age(enrollment_store):
    """Use enrollment age exclusively, preserving missing participants and sex."""
    result = get_user_demographics(enrollment_store, ["known", "null", "legacy", "absent"])
    assert result == {
        "known": {"age": 35, "sex": "male"},
        "null": {"age": None, "sex": "female"},
        "legacy": {"age": None, "sex": "unknown"},
        "absent": {"age": None, "sex": "unknown"},
    }


def test_imputation_groups_are_independent_of_sample_dates(enrollment_store):
    """Keep age groups fixed across dates and preserve the requested split membership."""
    dataset = {
        "user_id": ["known", "known", "null", "absent"],
        "date": ["2010-01-01", "2050-01-01", "2020-01-01", "2020-01-01"],
    }
    result = build_subgroup_mapping(
        dataset,
        {"train": [0], "val": [0], "test": [1, 2, 3]},
        [18, 30, 40, 50, 60],
        store=enrollment_store,
    )
    assert set(result) == {"val", "test"}
    assert result["val"][0] == result["test"][0] == {"age_group": "30-39", "sex": "male"}
    assert result["test"][1] == {"age_group": "unknown", "sex": "female"}
    assert result["test"][2] == {"age_group": "unknown", "sex": "unknown"}


@pytest.mark.parametrize("reference_date", ["2010-01-01", "2050-01-01"])
def test_forecasting_reads_enrollment_age(enrollment_store, reference_date):
    """Ignore legacy reference dates and birth years when grouping forecast users."""
    result = load_user_demographics(
        user_ids={"known", "null", "legacy", "absent"},
        labels_path=enrollment_store.labels_path,
        enrollment_path=enrollment_store.enrollment_path,
        reference_date=reference_date,
    )
    assert result == {
        "known": {"age_group": "30-39", "sex": "male"},
        "null": {"age_group": "unknown", "sex": "female"},
        "legacy": {"age_group": "unknown", "sex": "unknown"},
        "absent": {"age_group": "unknown", "sex": "unknown"},
    }


@pytest.mark.parametrize(
    "age, expected",
    [
        (None, "unknown"),
        (-1, "unknown"),
        (float("nan"), "unknown"),
        (float("inf"), "unknown"),
        (17, "unknown"),
        (18, "18-29"),
        (29.5, "18-29"),
        (30, "30-39"),
        (40, "40-49"),
        (50, "50-59"),
        (60, "60+"),
    ],
)
def test_tracks_agree_on_age_boundaries(age, expected):
    """Keep age boundaries and missing values consistent between tracks."""
    assert bin_age(age, [18, 30, 40, 50, 60]) == expected
    assert forecasting_bin_age(age) == expected


@pytest.mark.parametrize(
    "module_name",
    [
        "scripts.paper_results.compute_imputation_paper_metrics",
        "scripts.paper_results.imputation.bootstrap_imputation_draws",
        "scripts.paper_results.imputation.parity.produce_per_method_per_user_errors",
        "scripts.paper_results.imputation.parity.parity_locf",
    ],
)
def test_imputation_report_paths_use_same_age(module_name, enrollment_store, tmp_path, monkeypatch):
    """Exercise every offline mapping against a real sample manifest."""
    module = importlib.import_module(module_name)
    monkeypatch.setattr(api, "STORE", enrollment_store)
    pq.write_table(
        pa.table(
            {
                "sample_idx": [4, 9, 11],
                "user_id": ["known", "known", "null"],
                "date": ["2010-01-01", "2050-01-01", "2020-01-01"],
            }
        ),
        tmp_path / "manifest_test.parquet",
    )
    result = module._build_subgroup_mapping(tmp_path, "test", [18, 30, 40, 50, 60])
    assert result == {
        4: {"age_group": "30-39", "sex": "male"},
        9: {"age_group": "30-39", "sex": "male"},
        11: {"age_group": "unknown", "sex": "female"},
    }
