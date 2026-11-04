"""Direct Linear substitution preserves scores and participant alignment."""
from __future__ import annotations
import numpy as np
import pytest
from downstream_evaluation.evaluation.evaluator import _combine_with_fallback
from openmhc import PredictionResults


@pytest.mark.parametrize("ttype", ["binary", "ordinal", "regression", "multiclass"])
def test_direct_substitution_preserves_values_and_inputs(ttype):
    native = np.array([0.10, 0.40, 0.80, np.nan, np.inf, -np.inf])
    baseline = np.array([0.99, 0.99, 0.99, 0.20, 0.30, 0.15])
    original = native.copy()
    out, count = _combine_with_fallback(native, baseline, ttype)
    np.testing.assert_array_equal(out, [0.10, 0.40, 0.80, 0.20, 0.30, 0.15])
    np.testing.assert_array_equal(native, original)
    np.testing.assert_array_equal(baseline, [0.99, 0.99, 0.99, 0.20, 0.30, 0.15])
    assert count == 3
    # Routing must not depend on unrelated participants or their ordering.
    ix = [4, 1, 3]
    subset, subset_count = _combine_with_fallback(native[ix], baseline[ix], ttype)
    np.testing.assert_array_equal(subset, out[ix])
    assert subset_count == 2


@pytest.mark.parametrize("ttype", ["binary", "ordinal", "regression", "multiclass"])
def test_combine_all_fallback_no_rank(ttype):
    native = np.full(4, np.nan)
    baseline = np.array([.13, .81, .27, .81])
    out, count = _combine_with_fallback(native, baseline, ttype)
    np.testing.assert_array_equal(out, baseline)
    assert count == 4


def test_combine_no_nonfinite_is_identity():
    native = np.array([.13, .81, .27, .81])
    out, count = _combine_with_fallback(native, np.zeros(4), "binary")
    np.testing.assert_array_equal(out, native)
    assert count == 0


def test_wbmprobe_predict_aligns_and_nans_missing():
    """WBMProbe scatters SSL preds onto the daily cohort, NaN where no weekly embedding.

    The NaN positions are exactly what the harness fallback then substitutes with
    the Linear baseline. Exercised with mocked encoder/probe (no dataset needed).
    """
    import types

    from downstream_evaluation.models.wbm import WBMProbe

    m = WBMProbe("/tmp/x")
    # Daily cohort = u0..u3; weekly (encodable) cohort = u1, u3 only.
    m._ctx = types.SimpleNamespace(task="Diabetes", split="test", user_ids=["u0", "u1", "u2", "u3"])
    weekly = types.SimpleNamespace(user_ids=["u1", "u3"])
    m._weekly_td = lambda task, split: weekly
    # encode_cohort returns (embeddings, kept_mask); both weekly users encodable here.
    m._wbm = types.SimpleNamespace(
        encode_cohort=lambda task, td: (
            np.zeros((len(td.user_ids), 4)),
            np.ones(len(td.user_ids), dtype=bool),
        )
    )
    m._probe = types.SimpleNamespace(predict=lambda X: np.array([0.7, 0.9]))

    out = m.predict(None)

    assert out.shape == (4,)
    assert np.isnan(out[0]) and np.isnan(out[2])  # no weekly embedding → harness fallback
    assert out[1] == 0.7 and out[3] == 0.9  # weekly users keep the SSL probe output


def test_wbmprobe_predict_nans_unencodable_weekly_user():
    """A weekly-cohort user with no embedding is left NaN (→ Linear fallback), not raised.

    ``encode_cohort`` drops the un-encodable user (mask False) and returns only the rows it
    could pool, so the corresponding daily prediction stays NaN for the harness to fill.
    """
    import types

    from downstream_evaluation.models.wbm import WBMProbe

    m = WBMProbe("/tmp/x")
    m._ctx = types.SimpleNamespace(task="Diabetes", split="test", user_ids=["u1", "u3"])
    weekly = types.SimpleNamespace(user_ids=["u1", "u3"])
    m._weekly_td = lambda task, split: weekly
    # u1 has an embedding (kept), u3 has none (dropped) → one row of X, mask [True, False].
    m._wbm = types.SimpleNamespace(
        encode_cohort=lambda task, td: (np.zeros((1, 4)), np.array([True, False]))
    )
    m._probe = types.SimpleNamespace(predict=lambda X: np.array([0.7]))

    out = m.predict(None)

    assert out.shape == (2,)
    assert out[0] == 0.7  # encodable weekly user keeps its probe output
    assert np.isnan(out[1])  # un-encodable weekly user → NaN → harness fallback


def test_linear_probe_propagates_nan_for_unencodable_rows():
    """A non-finite embedding row predicts NaN (→ harness fallback) and is dropped from fit."""
    import openmhc

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8)).astype(np.float32)
    y = rng.integers(0, 2, size=20)
    x_fit = x.copy()
    x_fit[3] = np.nan  # a training participant the encoder could not represent

    probe = openmhc.LinearProbe("binary", n_components=5, seed=0).fit(x_fit, y)

    x_test = x[:4].copy()
    x_test[1] = np.nan
    out = probe.predict(x_test)

    assert np.isnan(out[1])  # non-finite embedding → NaN, so the harness scores it via fallback
    assert np.all(np.isfinite(np.delete(out, 1)))  # finite rows still score


def test_apply_demographics_fails_loud_on_absent_user():
    """A cohort user absent from the demographic lookup raises, not an all-zero row."""
    from downstream_evaluation.demo_covariates import apply_demographics

    covs = ["age", "BiologicalSex", "BMI_values"]
    lookup = {
        "u0": np.array([40.0, 1.0, 22.0], dtype=np.float32),
        "u1": np.array([55.0, 0.0, 28.0], dtype=np.float32),
    }
    x = np.zeros((2, 3), dtype=np.float32)

    # Fully covered cohort → covariates appended, no error.
    out = apply_demographics(x, ["u0", "u1"], "Diabetes", lookup, covs)
    assert out.shape == (2, 6)

    # An absent cohort user → loud failure, not a fabricated all-zero demographic row.
    with pytest.raises(ValueError, match="out of sync"):
        apply_demographics(x, ["u0", "u2"], "Diabetes", lookup, covs)


def test_missing_feature_fails_loud_not_zero_filled():
    """A cohort user with no extractable feature raises, rather than being scored as zeros.

    The silent zero-fill produced a finite prediction that bypassed the NaN->Linear
    fallback and under-counted ``n_fallback``; the guard now fails loudly instead.
    MultiRocket stands in for the shared pattern (tsfm/lsm2/grud guard identically; WBM
    instead leaves un-encodable users NaN so the harness fallback scores them).
    """
    from downstream_evaluation.models.multirocket import MultiRocket

    m = MultiRocket()
    m._pooled = {"u0": np.arange(5, dtype=np.float32), "u1": np.ones(5, dtype=np.float32)}

    # Fully covered cohort → aligned feature matrix, no error.
    x = m._features(["u0", "u1"])
    assert x.shape == (2, 5)
    assert np.array_equal(x[0], np.arange(5))

    # An uncovered cohort user → loud failure naming the gap, not a silent zero row.
    with pytest.raises(ValueError, match="zero-filled|out of sync"):
        m._features(["u0", "u2"])


def test_prediction_results_fallback_fields_default():
    """PredictionResults stays constructible from records alone (additive fields)."""
    pr = PredictionResults(records=[{"task": "t", "metric": "auprc", "value": 0.5}])
    assert pr.overall_fallback_rate == 0.0
    assert pr.fallback_rate == {}
    assert "overall_fallback_rate=0.0000" in repr(pr)


def test_prediction_results_fallback_fields_populated():
    pr = PredictionResults(
        records=[],
        overall_fallback_rate=0.25,
        fallback_rate={"task_a": 0.5},
    )
    assert pr.overall_fallback_rate == 0.25
    assert pr.fallback_rate == {"task_a": 0.5}
    # Fallback stays out of the metrics table so eval_<method>.csv is unaffected.
    assert "fallback" not in {c for c in pr.to_dataframe().columns}
