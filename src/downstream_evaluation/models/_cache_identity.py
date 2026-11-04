"""Offline provenance checks for downstream embedding caches.

Local checkpoints are content-hashed. Remote references are compared verbatim;
pin their revision for reproducibility. Dataset fingerprints use canonical paths,
the release marker, and source file sizes/mtimes, without reading the large tensors.
Inputs must remain immutable during an evaluation run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_SCHEMA = 1
METADATA_FILE = "cache_identity.json"


def file_digest(path: Path) -> str:
    """Hash a file without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(reference: str, relative_to: Path | None = None) -> dict:
    """Match the caller's path semantics without fetching remote checkpoints."""
    reference = str(reference)
    if reference.startswith(("hf://", "wandb:")):
        return {"reference": reference}
    path = Path(reference)
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    if not path.exists():
        if (
            Path(reference).is_absolute()
            or reference.startswith(".")
            or path.suffix in {".ckpt", ".pt", ".pth", ".bin", ".safetensors"}
        ):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return {"reference": reference}  # Bare Hub model ID.
    path = path.resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return {
        "path": str(path),
        "sha256": {
            p.name if path.is_file() else str(p.relative_to(path)): file_digest(p) for p in files
        },
    }


def dataset_identity(data_dir: str | None, resolution: str) -> dict:
    """Fingerprint the dataset root and extraction inputs using inexpensive file stats."""
    from openmhc._evaluate import _DatasetPaths

    paths = _DatasetPaths.from_root(data_dir)
    root = paths.root.resolve()
    raw = paths.daily_hf if resolution == "minute" else paths.daily_hourly_hf
    files = set()
    for directory in (raw, paths.labels_dir, paths.splits_file.parent):
        files.update(p for p in directory.rglob("*") if p.is_file())
    for pattern in ("*.json", "*.parquet"):
        files.update((root / "processed").glob(pattern))
    entries = []
    for path in sorted(files):
        stat = path.stat()
        entries.append(
            (str(path.relative_to(paths.root)), str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        )
    return {
        "root": str(root),
        "resolution": resolution,
        "marker_sha256": file_digest(root / "dataset_version.json"),
        "sources_sha256": hashlib.sha256(json.dumps(entries).encode()).hexdigest(),
    }


def embedding_identity(
    model: str,
    checkpoint: str,
    data_dir: str | None,
    resolution: str,
    *,
    relative_to: Path | None = None,
    **settings,
) -> dict:
    """Describe the inputs and extraction settings that determine cached features."""
    return {
        "schema": CACHE_SCHEMA,
        "model": model,
        "checkpoint": checkpoint_identity(checkpoint, relative_to),
        "dataset": dataset_identity(data_dir, resolution),
        "settings": settings,
    }


def validate_loader(loader, data_dir: str | None, resolution: str) -> None:
    """Ensure an injected standard loader reads the dataset recorded in provenance."""
    if loader is None or not hasattr(loader, "_data_dir"):
        return
    from openmhc._evaluate import _DatasetPaths

    actual = _DatasetPaths.from_root(loader._data_dir).root.resolve()
    expected = _DatasetPaths.from_root(data_dir).root.resolve()
    if actual != expected or loader.resolution != resolution:
        raise ValueError("Injected loader dataset/resolution does not match the embedding inputs")


def validate_identity(actual: dict | None, expected: dict, path: Path) -> None:
    """Fail closed on missing provenance or any changed input/settings group."""
    if not isinstance(actual, dict) or actual != expected:
        changed = sorted(
            k for k in expected if not isinstance(actual, dict) or actual.get(k) != expected[k]
        )
        raise ValueError(
            f"Incompatible embedding cache {path}: missing or mismatched identity "
            f"({', '.join(changed)}). Use a fresh cache_dir to regenerate embeddings. "
            "Existing files have not been changed."
        )


def numpy_cache_ready(cache: Path, expected: dict, filenames: tuple[str, ...]) -> bool:
    """Validate a complete NumPy cache; return False only for an unused destination."""
    paths = [cache / name for name in (*filenames, METADATA_FILE)]
    if not any(path.exists() for path in paths):
        return False
    if not all(path.is_file() for path in paths):
        raise ValueError(f"Incomplete embedding cache {cache}; use a fresh cache_dir.")
    try:
        actual = json.loads((cache / METADATA_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid cache metadata at {cache}; use a fresh cache_dir.") from exc
    validate_identity(actual, expected, cache)
    return True


def write_numpy_cache(cache: Path, identity: dict, arrays: dict[str, np.ndarray]) -> None:
    """Write provenance last so interrupted extraction cannot appear complete."""
    cache.mkdir(parents=True, exist_ok=True)
    for filename, array in arrays.items():
        with (cache / filename).open("xb") as stream:
            np.save(stream, array)
    with (cache / METADATA_FILE).open("x") as stream:
        json.dump(identity, stream, sort_keys=True)
