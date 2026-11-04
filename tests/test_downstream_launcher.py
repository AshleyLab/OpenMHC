"""Compose the PBS GPU launcher's default without downloading or running a model."""

import re
import subprocess
import sys
from pathlib import Path


def test_pbs_gpu_default_composes():
    """The launcher's default must select an available public method config."""
    root = Path(__file__).resolve().parents[1]
    launcher = root / "jobs/imperial/pbs/run_downstream_eval_gpu.pbs"
    default = re.search(r'export METHOD="\$\{METHOD:-([^}]+)\}"', launcher.read_text())
    assert default is not None
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "downstream_evaluation.hydra.cli",
            f"method={default[1]}",
            "--cfg",
            "job",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "type: lsm2" in result.stdout
