"""Cache selection through the documented eval driver and downstream phase 0."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    """Load the actual entrypoint without running model evaluation at import time."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_script("eval_workflow_driver", "scripts/run_eval.py")
pipeline = load_script(
    "downstream_workflow_pipeline", "scripts/paper_results/downstream/run_paper_pipeline.py"
)
METHODS = ["lsm2", "wbm", "toto", "chronos2"]


def encoder_for(model):
    """Unwrap the WBM probe so tests inspect the encoder that uses the cache."""
    return getattr(model, "_wbm", model)


@pytest.mark.parametrize("method", [*METHODS, "hybrid"])
def test_driver_selects_fresh_method_cache(tmp_path, monkeypatch, method):
    """An exported root reaches the real constructor, with no cross-model collision."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path / "fresh"))
    model = encoder_for(driver.build_model(method, str(tmp_path / "data")))
    assert model._resolve_cache_dir() == tmp_path / "fresh" / model.name
    assert not (tmp_path / "fresh").exists()  # Constructor needs no extraction/download.


@pytest.mark.parametrize("method", METHODS)
def test_driver_keeps_default_cache_when_unset(tmp_path, monkeypatch, method):
    """An omitted override preserves the model's normal cache selection."""
    monkeypatch.delenv("CACHE_ROOT", raising=False)
    model = encoder_for(driver.build_model(method, str(tmp_path)))
    assert model._cache_dir is None


@pytest.mark.parametrize("selection", ["environment", "config", "unset"])
def test_phase0_cache_selection_reaches_all_encoder_constructors(tmp_path, monkeypatch, selection):
    """Follow phase-0 subprocess environments through the actual eval driver."""
    cfg = {
        "csvs_dir": str(tmp_path / "csv"),
        "predictions_dir": str(tmp_path / "predictions"),
        "data_dir": str(tmp_path / "data"),
        "version": "xs",
        "cache_root": None,
    }
    monkeypatch.delenv("CACHE_ROOT", raising=False)
    expected = None
    if selection != "unset":
        expected = tmp_path / "environment"
        monkeypatch.setenv("CACHE_ROOT", str(expected))
    if selection == "config":
        expected = tmp_path / "config"
        cfg["cache_root"] = str(expected)
    observed = []

    def run_driver(cmd, dry_run, env):
        assert Path(cmd[-1]) == ROOT / "scripts/run_eval.py"
        assert env["VERSION"] == "xs"
        method = env["METHOD"]
        assert env["OUT_CSV"] == str(tmp_path / "csv" / f"eval_{method}.csv")
        assert env["PREDICTIONS_DIR"] == cfg["predictions_dir"]
        with monkeypatch.context() as child:
            if env.get("CACHE_ROOT"):
                child.setenv("CACHE_ROOT", env["CACHE_ROOT"])
            else:
                child.delenv("CACHE_ROOT", raising=False)
            encoder = encoder_for(driver.build_model(method, env["MHC_DATA_DIR"]))
        if expected is None:
            assert encoder._cache_dir is None
        else:
            assert encoder._resolve_cache_dir() == expected / method
        observed.append(method)

    monkeypatch.setattr(pipeline, "_run", run_driver)
    pipeline._phase0_eval(cfg, METHODS, dry_run=False)
    assert observed == METHODS
