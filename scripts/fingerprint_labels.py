"""Print resolved label inputs and ordered metadata for consolidation checks.

Run with ``PYTHONPATH=src MHC_DATA_DIR=/path/to/data python scripts/fingerprint_labels.py``.
This reads files without changing the dataset or requiring its version marker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from downstream_evaluation.demo_covariates import _TASK_FEATURE_EXCLUSIONS_PATH, DEMO_ALIAS
from labels import api
from openmhc._constants import BENCHMARK_TASKS


def describe(path: Path | None) -> dict:
    """Record a resolved path and the hash of its bytes, if present."""
    if path is None:
        return {"path": None, "sha256": None}
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
    else:
        checksum = None
    return {"path": str(path), "sha256": checksum}


def main() -> None:
    """Emit a deterministic fingerprint (no timestamps or participant data)."""
    api.configure()
    names = (
        "LABELS_PATH",
        "CONTEXT_LABELS_PATH",
        "ENROLLMENT_PATH",
        "LABEL_VALIDITY_PATH",
        "HEALTHKIT_DAILY_PATH",
        "LABEL_TYPES_PATH",
        "ORDINAL_DICTIONARY_PATH",
        "VALIDITY_CONFIG_PATH",
    )
    result = {
        "files": {name: describe(getattr(api, name)) for name in names},
        "benchmark_exclusions_file": describe(_TASK_FEATURE_EXCLUSIONS_PATH),
        "LABEL_NAMES": api.LABEL_NAMES,
        "TARGET_NAMES": api.TARGET_NAMES,
        "CONTEXT_NAMES": api.CONTEXT_NAMES,
        "covariate_exclusions": {
            task: [
                cov
                for cov in ("age", "BiologicalSex", "BMI_values")
                if cov == task or cov in DEMO_ALIAS.get(task, ())
            ]
            for task in BENCHMARK_TASKS
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
