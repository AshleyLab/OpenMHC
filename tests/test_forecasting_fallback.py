"""Forecasting fallback rate: to_json preserves it and the uploader extracts it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from openmhc import ForecastingResults

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import upload_leaderboard_substrate as uploader  # noqa: E402


def test_to_json_preserves_fallback_rate(tmp_path):
    res = ForecastingResults(
        per_channel={"ch_0": {"mae": 1.0}},
        run_dir="/x",
        n_samples=10,
        overall_fallback_rate=0.0123,
        fallback_rate={"ch_0": 0.0},
    )
    p = tmp_path / "results.json"
    res.to_json(p)
    d = json.loads(p.read_text())
    # The rate survives the round-trip at the top level (unlike the old
    # behaviour, which dumped only per_channel and dropped it).
    assert d["overall_fallback_rate"] == pytest.approx(0.0123)
    assert d["fallback_rate"] == {"ch_0": 0.0}
    assert d["per_channel"] == {"ch_0": {"mae": 1.0}}
    # And the uploader extracts it from the exported file (the leaderboard path).
    assert uploader._worst_case_fallback_rate(p) == pytest.approx(0.0123)


def test_to_json_default_unknown(tmp_path):
    res = ForecastingResults(per_channel={})
    p = tmp_path / "results.json"
    res.to_json(p)
    assert json.loads(p.read_text())["overall_fallback_rate"] is None
    with pytest.raises(ValueError, match="fallback rate"):
        uploader._worst_case_fallback_rate(p)


@pytest.mark.parametrize('sub,total,expected', [
    (None, None, None), ([0, 0], [0, 0], None),
    ([0, 0], [3, 7], 0.0), ([1, 2], [3, 7], 0.3),
])
def test_forecasting_counters_distinguish_missing_empty_and_zero(sub, total, expected):
    """Zero requires a positive denominator and explicit substitution counters."""
    import numpy as np

    from forecasting_evaluation.config import ForecastingEvalConfig
    from forecasting_evaluation.evaluation.evaluator import ForecastingEvaluator

    evaluator = ForecastingEvaluator(ForecastingEvalConfig())
    result = evaluator._summarize_fallback(
        'm', None if sub is None else np.array(sub), None if total is None else np.array(total),
    )
    assert result['overall_fallback_rate'] == expected


@pytest.mark.parametrize('use_fallback', [False, True])
@pytest.mark.parametrize('public_api', [False, True])
def test_forecasting_resume_behavior_preserved_with_unknown_rate(
    tmp_path, monkeypatch, use_fallback, public_api
):
    """Preserve saved files, skips and sample counts; incomplete rates are unknown."""
    from types import SimpleNamespace

    import h5py
    import numpy as np

    import forecasting_evaluation.evaluation.evaluator as module
    from forecasting_evaluation.config import ForecastingConfig, ForecastingEvalConfig, OutputConfig

    config = ForecastingEvalConfig(
        experiment_name='synthetic',
        forecasting=ForecastingConfig(forecasting_length=2),
        output=OutputConfig(results_dir=str(tmp_path / 'out'), save_config=False),
    )
    cache = tmp_path / 'cache.h5'
    with h5py.File(cache, 'w') as handle:
        for i in range(2):
            handle.create_dataset(f'history_cf_rows/{i}', data=np.ones((2, 30), dtype=np.float32))
    context = SimpleNamespace(
        test_ds=[{'values': np.ones((30, 2)), 'channel_names': ['a', 'b']} for _ in range(2)],
        test_cache_path=cache,
        test_row_groups=[SimpleNamespace(
            user_id='u0', dataset_row_idx=0, windows=[SimpleNamespace(current_day=1)],
        )],
    )

    class Model:
        model_name = 'synthetic-model'
        calls = 0

        def predict(self, history, horizon):
            self.calls += 1
            point = np.ones((2, horizon), dtype=np.float32)
            if use_fallback and self.calls == 1:
                point[0] = np.nan
            return point

    model = Model()
    monkeypatch.setattr(module, 'create_forecasting_model', lambda *a, **k: model)
    if public_api:
        from forecasting_evaluation.runner import _CustomModelEvaluator

        evaluator = _CustomModelEvaluator(config, model)
    else:
        evaluator = module.ForecastingEvaluator(config)
    monkeypatch.setattr(evaluator, '_load_evaluation_data', lambda: context)
    first = evaluator.run()
    assert first['overall_fallback_rate'] == (0.5 if use_fallback else 0.0)
    assert first['prediction_samples'] == 1
    original = (Path(first['run_dir']) / 'u0.parquet').read_bytes()
    context.test_row_groups.append(SimpleNamespace(
        user_id='u1', dataset_row_idx=1, windows=[SimpleNamespace(current_day=1)],
    ))
    second = evaluator.run()
    third = evaluator.run()
    for result in [second, third]:
        assert result['overall_fallback_rate'] is None
        assert result['fallback_rate'] == {}
    # Keep main's meaning: prediction_samples counts newly written rows only.
    assert second['prediction_samples'] == 1
    assert third['prediction_samples'] == 0
    assert second['skipped_users'] == 1
    assert third['skipped_users'] == 2
    assert model.calls == 2
    assert (Path(first['run_dir']) / 'u0.parquet').read_bytes() == original
    path = tmp_path / 'results.json'
    ForecastingResults(per_channel={}, overall_fallback_rate=None).to_json(path)
    with pytest.raises(ValueError, match='fallback rate'):
        uploader._worst_case_fallback_rate(path)


def test_forecasting_default_result_cannot_be_submitted():
    """An object constructed without a measurement must not export a guessed zero."""
    result = ForecastingResults(per_channel={})
    assert result.overall_fallback_rate is None
    assert 'unknown' in repr(result)
    with pytest.raises(ValueError, match='measured'):
        result.to_submission_yaml('m', 't', 'url')
