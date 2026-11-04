"""Demographic subgroup mapping for sensitivity analysis.

Builds per-sample demographic attributes (age group, biological sex) from
the HuggingFace dataset metadata and the Labels API, enabling subgroup-level
metric breakdowns in imputation evaluation.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def sample_subgroups(pairs_dir: Path, split: str) -> dict[int, dict[str, str]] | None:
    """Map each sample saved in a pairs directory to its user's fairness groups.

    Args:
        pairs_dir: Directory holding the split's sample manifest.
        split: Split name.

    Returns:
        ``{sample_idx: {"age_group": ..., "sex": ...}}``, or None without a manifest.
    """
    from imputation_evaluation.evaluation.pair_writer import load_sample_manifest
    from labels.api import user_demographics

    manifest = load_sample_manifest(pairs_dir, split)
    if manifest is None:
        return None
    sample_idxs = manifest.column("sample_idx").to_numpy()
    user_ids = manifest.column("user_id").to_pylist()
    groups = user_demographics(user_ids)
    logger.info("[split=%s] looked up demographics for %d users", split, len(groups))
    return {int(sidx): groups[uid] for sidx, uid in zip(sample_idxs, user_ids)}


def build_subgroup_mapping(
    hf_dataset,
    split_indices: dict[str, list[int]],
) -> dict[str, dict[int, dict[str, str]]]:
    """Build per-sample subgroup mapping for sensitivity analysis.

    For each sample in val/test splits, maps (split_local_index) to its
    demographic attributes (age_group, sex).

    Args:
        hf_dataset: HuggingFace dataset with a user_id column.
        split_indices: Maps split name -> list of global HF dataset indices.

    Returns:
        Dict: {split_name: {split_local_idx: {"age_group": "30-39", "sex": "male"}}}.
        Only includes val and test splits.
    """
    from labels.api import user_demographics

    # Pre-extract lightweight metadata columns once (avoids reading the 107 KB
    # "values" array per row, which under cgroups v1 faults ~46 GB of Arrow
    # page cache into memory for ~427 K samples).
    all_user_ids = list(hf_dataset["user_id"])

    # Collect unique user IDs across val+test splits
    eval_user_ids: set[str] = set()
    for split_name in ("val", "test"):
        if split_name in split_indices:
            for idx in split_indices[split_name]:
                eval_user_ids.add(all_user_ids[idx])

    unique_users = sorted(eval_user_ids)
    logger.info(f"Looking up demographics for {len(unique_users)} unique users...")
    groups = user_demographics(unique_users)

    # Build per-split mapping
    mapping: dict[str, dict[int, dict[str, str]]] = {}

    for split_name in ("val", "test"):
        if split_name not in split_indices:
            continue

        indices = split_indices[split_name]
        split_mapping: dict[int, dict[str, str]] = {}

        for local_idx, global_idx in enumerate(indices):
            split_mapping[local_idx] = groups[all_user_ids[global_idx]]

        mapping[split_name] = split_mapping

    return mapping


def format_subgroup_summary(mapping: dict[str, dict[int, dict[str, str]]]) -> str:
    """Format a human-readable summary of subgroup distributions.

    Args:
        mapping: Subgroup mapping as returned by build_subgroup_mapping().

    Returns:
        Multi-line string summarizing counts per attribute per group per split.
    """
    lines = ["Subgroup distributions:"]

    for split_name, split_mapping in sorted(mapping.items()):
        lines.append(f"\n  {split_name} ({len(split_mapping)} samples):")

        # Gather all attribute names from first entry
        if not split_mapping:
            lines.append("    (empty)")
            continue

        attrs = list(next(iter(split_mapping.values())).keys())

        for attr in attrs:
            counts = Counter(demo[attr] for demo in split_mapping.values())
            lines.append(f"    {attr}:")
            for group, count in sorted(counts.items()):
                pct = 100.0 * count / len(split_mapping)
                lines.append(f"      {group}: {count} ({pct:.1f}%)")

    return "\n".join(lines)
