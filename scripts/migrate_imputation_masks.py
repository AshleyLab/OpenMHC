"""Copy existing release masks to matching dataset samples without regenerating them.

Pass the original dataset root and the destination root. For a date-shifted
destination, also supply the private participant offset table. The offset table
is read locally and is never copied to the output. Output must be a new directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import numpy as np
import pandas as pd

from imputation_evaluation.config import DataConfig
from imputation_evaluation.data.data_loader import ImputationDataLoader
from imputation_evaluation.data.splits import load_split_file
from imputation_evaluation.masking.generator import ScenarioMasks
from openmhc._constants import MASKING_SCENARIOS


def sample_keys(root: Path, version: str) -> dict[str, list[tuple[str, str]]]:
    """Get split-local sample keys with the evaluator's canonical QA filters."""
    ds = datasets.load_from_disk(str(root / "processed/daily_hf"))
    filters = ImputationDataLoader(DataConfig())._build_filters()
    columns = list(dict.fromkeys(["user_id", "date"] + [f.required_column for f in filters]))
    ds = ds.select_columns(columns)
    kept = []
    for start in range(0, len(ds), 10000):
        batch = ds[start : start + 10000]
        valid = np.ones(len(batch["user_id"]), dtype=bool)
        for quality_filter in filters:
            valid &= np.asarray(quality_filter(batch))
        kept.extend((np.flatnonzero(valid) + start).tolist())
    ds = ds.select(kept)
    suffix = "_xs" if version == "xs" else ""
    splits = load_split_file(root / f"splits/sharable_users_seed42_2026{suffix}.json")
    keys = list(zip(ds["user_id"], ds["date"]))
    if len(set(keys)) != len(keys):
        raise ValueError("Dataset contains duplicate participant/date keys")
    return {
        split: [(str(user), str(date)) for user, date in keys if user in splits[source]]
        for split, source in [("val", "validation"), ("test", "test")]
    }


def remap_masks(
    masks: ScenarioMasks,
    source_keys: list[tuple[str, str]],
    target_keys: list[tuple[str, str]],
    offsets: dict[str, int] | None = None,
) -> ScenarioMasks:
    """Preserve mask rows exactly, mapping each to its destination participant/day."""
    indices = masks.indices
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(indices) != masks.n_applicable
        or len(set(indices.tolist())) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(source_keys))
    ):
        raise ValueError("Mask indices do not fit the source dataset")
    source = [source_keys[int(i)] for i in indices]
    users = np.asarray([k[0] for k in source], dtype=str)
    dates = np.asarray([k[1] for k in source], dtype=str)
    if (masks.user_ids is None) != (masks.dates is None):
        raise ValueError("Mask participant/date metadata is incomplete")
    if masks.user_ids is not None and not (
        np.array_equal(masks.user_ids, users) and np.array_equal(masks.dates, dates)
    ):
        raise ValueError("Mask participant/date metadata does not match the source dataset")
    if offsets is not None:
        if any(user not in offsets for user in users):
            raise ValueError("Participant missing from private offset table")
        days = np.asarray([7 * offsets[user] for user in users], dtype="timedelta64[D]")
        dates = (dates.astype("datetime64[D]") + days).astype(str)
    target_index = {key: i for i, key in enumerate(target_keys)}
    if len(target_index) != len(target_keys):
        raise ValueError("Duplicate target participant/date keys")
    keys = list(zip(users, dates))
    if any(key not in target_index for key in keys):
        raise ValueError("A mask's participant/date is absent from the destination dataset")
    return ScenarioMasks(
        indices=np.asarray([target_index[key] for key in keys], dtype=np.int64),
        masks_packed=masks.masks_packed.copy(),
        shape=masks.shape,
        user_ids=users,
        dates=dates,
    )


def subset_masks(masks: ScenarioMasks, target_keys: list[tuple[str, str]]) -> ScenarioMasks:
    """Extract XS rows from Full masks, preserving every selected mask bit."""
    if masks.user_ids is None or masks.dates is None:
        raise ValueError("Full masks must include participant/date metadata")
    target_index = {key: i for i, key in enumerate(target_keys)}
    if len(target_index) != len(target_keys):
        raise ValueError("Duplicate target participant/date keys")
    source_keys = list(zip(masks.user_ids, masks.dates))
    if len(source_keys) != masks.n_applicable or len(set(source_keys)) != len(source_keys):
        raise ValueError("Invalid Full mask participant/date metadata")
    positions = [i for i, key in enumerate(source_keys) if key in target_index]
    positions.sort(key=lambda i: target_index[source_keys[i]])
    selected = np.asarray(positions, dtype=np.int64)
    bytes_per_row = (int(np.prod(masks.shape[1:])) + 7) // 8
    packed_rows = masks.masks_packed.reshape(masks.n_applicable, bytes_per_row)
    return ScenarioMasks(
        indices=np.asarray([target_index[source_keys[i]] for i in positions], dtype=np.int64),
        masks_packed=packed_rows[selected].reshape(-1).copy(),
        shape=(len(positions), *masks.shape[1:]),
        user_ids=masks.user_ids[selected],
        dates=masks.dates[selected],
    )


def main() -> None:
    """Validate and migrate all twelve files into a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--target-data-dir", type=Path, required=True)
    parser.add_argument("--source-masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=("full", "xs"), required=True)
    parser.add_argument("--offsets", type=Path, help="Private user_id/offset_weeks Parquet")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    offsets = None
    if args.offsets:
        frame = pd.read_parquet(args.offsets)
        if (
            frame.user_id.isna().any()
            or frame.user_id.duplicated().any()
            or not pd.api.types.is_integer_dtype(frame.offset_weeks)
            or frame.offset_weeks.isna().any()
        ):
            raise ValueError("Expected unique participants and integer week offsets")
        offsets = dict(zip(frame.user_id.astype(str), frame.offset_weeks.astype(int)))
    source = sample_keys(args.source_data_dir, args.version)
    target = sample_keys(args.target_data_dir, args.version)
    prepared = {}
    for split in ("val", "test"):
        for scenario in MASKING_SCENARIOS:
            relative = Path(split) / f"{scenario}.npz"
            prepared[relative] = remap_masks(
                ScenarioMasks.load(args.source_masks / relative),
                source[split],
                target[split],
                offsets,
            )
    # Validate all inputs before writing any output.
    args.output.mkdir(parents=True)
    for relative, masks in prepared.items():
        destination = args.output / relative
        destination.parent.mkdir(exist_ok=True)
        masks.save(destination)
    print(f"Migrated {len(prepared)} mask files without changing their mask bits.")


if __name__ == "__main__":
    main()
