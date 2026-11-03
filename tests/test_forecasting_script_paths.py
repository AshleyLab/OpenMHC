"""Exercise the paper scripts' demographic handoff without running bootstraps."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "build_forecasting_bootstrap_from_hf.py",
    "produce_forecasting_bootstrap_draws.py",
    "regenerate_fairness_mapd.py",
    "parity/parity_substrate.py",
    "run_paper_pipeline.py",
]


class HandoffReached(Exception):
    """Stop before the costly bootstrap, after checking its actual arguments."""


def demographic_handoff(script):
    """Compile the actual path-selection block through the fairness call.

    Earlier loading/reduction phases are outside this regression's scope.
    Keep the guard and call intact, including the parity script's else branch.
    """
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        start = next((i for i, stmt in enumerate(body) if isinstance(stmt, ast.ImportFrom)
                      and stmt.module in ("openmhc._label_paths", "labels.api")), None)
        if start is None:
            continue
        for end in range(start, len(body)):
            if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                   and x.func.id == "bootstrap_fair_skill_score" for x in ast.walk(body[end])):
                return compile(ast.Module(body=body[start:end + 1], type_ignores=[]),
                               str(script), "exec")
    raise AssertionError(f"No demographic handoff in {script}")


@pytest.mark.parametrize("name", SCRIPTS)
@pytest.mark.parametrize("mode", ["dataset", "file_overrides", "unset"])
def test_paper_script_demographic_paths(tmp_path, monkeypatch, name, mode):
    """Resolve current inputs and reject missing paths before the bootstrap."""
    from forecasting_evaluation.metrics import metric_spec
    from labels import api

    for key in ("MHC_DATA_DIR", "LABELS_DATA_PATH", "ENROLLMENT_DATA_PATH"):
        monkeypatch.delenv(key, raising=False)
    # Resolution must happen at call time, even if labels was imported earlier.
    active_paths = api._PATHS
    expected = (tmp_path / "labels/last_labels.json", tmp_path / "labels/enrollment_info.json")
    if mode == "dataset":
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path))
    elif mode == "file_overrides":
        expected = (tmp_path / "custom_labels.json", tmp_path / "custom_enrollment.json")
        monkeypatch.setenv("LABELS_DATA_PATH", str(expected[0]))
        monkeypatch.setenv("ENROLLMENT_DATA_PATH", str(expected[1]))

    def bootstrap(**kwargs):
        assert mode != "unset"
        assert (kwargs["labels_path"], kwargs["enrollment_path"]) == tuple(map(str, expected))
        raise HandoffReached

    def report(*args):
        assert mode == "unset"
        assert args[1] is False
        raise HandoffReached

    namespace = dict(
        bootstrap_fair_skill_score=bootstrap, rep=SimpleNamespace(check=report),
        models={}, _spec=metric_spec, BASELINE="seasonal_naive",
        CONTINUOUS_METRICS=["mae"], BINARY_METRICS=["auroc"], AGE_BINS=(40,),
        N_BOOT=2, SEED=42, CI_LEVEL=.95, substrate=None,
        bs={}, cfg={"baseline": "seasonal_naive"}, n_boot=2, seed=42,
        ci_level=.95, within="micro", bca_fairness=True, per_user_metrics=None,
        logger=logging.getLogger(__name__),
    )
    code = demographic_handoff(ROOT / "scripts/paper_results/forecasting" / name)
    error = SystemExit if mode == "unset" and not name.startswith("parity/") else HandoffReached
    with pytest.raises(error):
        exec(code, namespace)
    assert api._PATHS is active_paths


@pytest.mark.parametrize("override", ["labels_path", "enrollment_path", "both"])
def test_pipeline_config_overrides_environment(tmp_path, monkeypatch, override):
    """Keep each explicit sweep path ahead of environment-derived defaults."""
    import importlib.util

    import pandas as pd

    from forecasting_evaluation.metrics import bootstrap_fair_skill_score as fair
    from forecasting_evaluation.metrics import bootstrap_skill_rank as skill

    monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "dataset"))
    monkeypatch.delenv("LABELS_DATA_PATH", raising=False)
    monkeypatch.delenv("ENROLLMENT_DATA_PATH", raising=False)
    script = ROOT / "scripts/paper_results/forecasting/run_paper_pipeline.py"
    spec = importlib.util.spec_from_file_location("paper_pipeline_paths_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(skill, "bootstrap_skill_rank", lambda **kw: {
        "skill_scores": pd.DataFrame(), "avg_rankings": pd.DataFrame()})
    bs = {"fairness": True}
    expected = {"labels_path": str(tmp_path / "dataset/labels/last_labels.json"),
                "enrollment_path": str(tmp_path / "dataset/labels/enrollment_info.json")}
    for key in expected:
        if override in (key, "both"):
            bs[key] = expected[key] = str(tmp_path / key)

    def bootstrap(**kwargs):
        assert {key: kwargs[key] for key in expected} == expected
        raise HandoffReached

    monkeypatch.setattr(fair, "bootstrap_fair_skill_score", bootstrap)
    with pytest.raises(HandoffReached):
        module._phase_bootstrap({"output_root": str(tmp_path / "output"),
                                 "baseline": "seasonal_naive", "bootstrap": bs}, {}, False)
