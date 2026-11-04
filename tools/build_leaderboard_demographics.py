"""Build the leaderboard's one public demographics file.

Writes ``demographics.parquet`` (``user_id``, ``age_group``, ``sex``) and a
``demographics.meta.json`` provenance sidecar to ``--out-dir``, using
``labels.user_demographics`` on the selected dataset. Every track's cohort views
filter ``all`` rows by this file's membership.

It covers only the users already in published substrates, never the full cohort,
plus those of any ``--add-substrate`` file about to be uploaded. It stops if any of
them has an unknown age group, and does not upload: see
``docs/leaderboard-maintenance.md``.

Requires the ``[hf]`` extra (``pip install -e ".[hf]"``).

Usage:
    python tools/build_leaderboard_demographics.py \
        --data-dir /path/to/dataset --out-dir results/leaderboard_demographics
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"
TRACKS = ("downstream", "imputation", "forecasting")


def _users(path: Path) -> set[str]:
    return set(pd.read_parquet(path, columns=["user_id"])["user_id"].astype(str))


def published_users(repo_id: str, revision: str) -> set[str]:
    """Return the union of ``user_id`` across every published method substrate."""
    root = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{track}/*.parquet" for track in TRACKS],
            ignore_patterns=["*/bootstrap*/*"],
        )
    )
    users: set[str] = set()
    for track in TRACKS:
        for path in (root / track).glob("*.parquet"):
            users |= _users(path)
    return users


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build(
    data_dir: Path | None,
    out_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    add_substrates: Sequence[Path] = (),
) -> pd.DataFrame:
    """Write the demographics file and sidecar for the published users."""
    import labels
    from labels.api import AGE_GROUP_EDGES
    from openmhc._dataset import data_dir as resolve_data_dir
    from openmhc._label_paths import LabelPaths

    root = resolve_data_dir(data_dir)
    paths = LabelPaths.resolve(root)
    if {paths.labels.parent, paths.enrollment.parent} != {root / "labels"}:
        raise SystemExit(
            f"LABELS_DATA_PATH or ENROLLMENT_DATA_PATH overrides the labels under {root}. "
            "Unset them: the sidecar records only the dataset at this root."
        )
    substrate_revision = HfApi().dataset_info(repo_id, revision=revision).sha
    users = published_users(repo_id, substrate_revision)
    for path in add_substrates:
        users |= _users(path)
    if not users:
        raise SystemExit(f"No published substrates found in {repo_id} at {substrate_revision}")
    groups = labels.user_demographics(users, paths=paths)

    unknown = sum(row["age_group"] == "unknown" for row in groups.values())
    if unknown:
        raise SystemExit(
            f"{unknown}/{len(groups)} evaluated users have an unknown age group in {root}. "
            "The release must cover every evaluated user; nothing was written."
        )

    frame = pd.DataFrame(
        [{"user_id": user, **row} for user, row in sorted(groups.items())],
        columns=["user_id", "age_group", "sex"],
    )
    meta = {
        "age_rule": "createdOn.year - birth_year, one group per participant",
        "age_group_edges": list(AGE_GROUP_EDGES),
        "sex_rule": "latest BiologicalSex label",
        "n_users": len(frame),
        "code_commit": _git_commit(),
        "dataset_version": json.loads((root / "dataset_version.json").read_text()),
        "substrate_revision": substrate_revision,
        "added_substrates": sorted(path.name for path in add_substrates),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "demographics.parquet", index=False, compression="zstd")
    (out_dir / "demographics.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return frame


def main() -> None:
    """Build the demographics file from the command line."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data-dir", type=Path, default=None, help="Dataset root (default: MHC_DATA_DIR)."
    )
    p.add_argument("--out-dir", required=True, type=Path, help="Directory to write to.")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id.")
    p.add_argument("--revision", default=None, help="Substrate revision (default: latest).")
    p.add_argument(
        "--add-substrate",
        action="append",
        default=[],
        type=Path,
        help="Also cover this file's users, for a substrate about to be uploaded. Repeatable.",
    )
    args = p.parse_args()

    frame = build(
        args.data_dir,
        args.out_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        add_substrates=args.add_substrate,
    )
    print(f"Wrote {len(frame)} users to {args.out_dir / 'demographics.parquet'}")
    print(f"  age_group: {frame['age_group'].value_counts().sort_index().to_dict()}")
    print(f"  sex: {frame['sex'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
