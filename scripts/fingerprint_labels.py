"""Print resolved label inputs and ordered metadata for consolidation checks.

Run with ``PYTHONPATH=src MHC_DATA_DIR=/path/to/data python scripts/fingerprint_labels.py``.
This reads files without changing the dataset or requiring its version marker.
Use this same script with each checkout's src on PYTHONPATH to compare main
before the resolver migration with the resolver implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

from downstream_evaluation.demo_covariates import _TASK_FEATURE_EXCLUSIONS_PATH, _kept_cov_indices
from downstream_evaluation.models.linear import _DEMO_COVARIATES
from labels import api
from openmhc._constants import BENCHMARK_TASKS

FILE_KEYS = {
    "labels": "LABELS_PATH",
    "context": "CONTEXT_LABELS_PATH",
    "enrollment": "ENROLLMENT_PATH",
    "validity": "LABEL_VALIDITY_PATH",
    "daily": "HEALTHKIT_DAILY_PATH",
    "devices": "USER_DEVICE_INFO_PATH",
    "label_types": "LABEL_TYPES_PATH",
    "ordinal_dictionary": "ORDINAL_DICTIONARY_PATH",
    "validity_config": "VALIDITY_CONFIG_PATH",
}


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
    if hasattr(api, "_PATHS"):
        paths = api._PATHS
        # Enumerate the actual resolver fields so new paths cannot be omitted.
        files = {FILE_KEYS[field.name]: describe(getattr(paths, field.name))
                 for field in fields(paths)}
    else:
        # The comparison baseline predates LabelPaths. Read its actual paths,
        # including the old repository-owned device path.
        from devices import api as devices

        files = {key: describe(getattr(devices if name == "devices" else api, key))
                 for name, key in FILE_KEYS.items()}
    result = {
        "files": files,
        "benchmark_exclusions_file": describe(_TASK_FEATURE_EXCLUSIONS_PATH),
        "LABEL_NAMES": api.LABEL_NAMES,
        "TARGET_NAMES": api.TARGET_NAMES,
        "CONTEXT_NAMES": api.CONTEXT_NAMES,
        "covariate_exclusions": {
            task: [
                cov
                for i, cov in enumerate(_DEMO_COVARIATES)
                if i not in _kept_cov_indices(task, _DEMO_COVARIATES)
            ]
            for task in BENCHMARK_TASKS
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
