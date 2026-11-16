"""Upload a method's predictions and measured fallback rate in one Hub commit.

Reads ``overall_fallback_rate`` from the adjacent ``<method>.parquet.meta.json``
provenance file, or from an explicitly supplied ``--results-json``. Track 1 uses
its pooled participant-task rate; imputation uses the maximum over scenario/split
cells. Missing or invalid measurements stop the upload. Existing display fields
are preserved unless replaced on the command line.

Requires ``pip install -e ".[hf]"`` and Hugging Face authentication.

Usage::

    python tools/upload_leaderboard_substrate.py \
        --dir results/leaderboard_downstream --method wbm --track downstream

    python tools/upload_leaderboard_substrate.py \
        --dir results/leaderboard_imputation --method locf --track imputation \
        --results-json runs/locf/results.json --name "LOCF" --type Statistical
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"


def _validate_rate(value: object, source: object) -> float:
    """Require a measured fraction, including a genuine zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: fallback rate must be a number in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{source}: fallback rate must be finite and in [0, 1]")
    return float(value)


def _check_identity(data: dict, method: str | None, track: str | None, source: Path) -> None:
    """Reject an explicitly identified result from a different method or track."""
    for key, expected in (("method", method), ("track", track)):
        if expected is not None and key in data and data[key] != expected:
            raise ValueError(f"{source}: {key} does not match {expected!r}")


def _worst_case_fallback_rate(
    results_json: Path, *, method: str | None = None, track: str | None = None
) -> float | None:
    """Read a pooled rate or the maximum over measured imputation cells.

    Missing measurements return None, never zero. Partially measured scenario
    maps are rejected because their maximum could understate fallback.
    """
    data = json.loads(results_json.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{results_json}: expected a result object with fallback metadata")
    _check_identity(data, method, track, results_json)
    config = data.get("config")
    if isinstance(config, dict):
        _check_identity(config, method, track, results_json)
    if "overall_fallback_rate" in data:
        return _validate_rate(data["overall_fallback_rate"], results_json)
    scenarios = data.get("scenarios", data)
    if not isinstance(scenarios, dict):
        raise ValueError(f"{results_json}: scenarios must be an object")
    cells = [
        metrics
        for splits in scenarios.values() if isinstance(splits, dict)
        for metrics in splits.values() if isinstance(metrics, dict)
    ]
    if not any("overall_fallback_rate" in cell for cell in cells):
        return None
    if any(
        not isinstance(splits, dict) or not splits
        or any(not isinstance(metrics, dict) for metrics in splits.values())
        for splits in scenarios.values()
    ):
        raise ValueError(f"{results_json}: incomplete scenario/split fallback measurements")
    return max(_validate_rate(cell.get("overall_fallback_rate"), results_json) for cell in cells)


def resolve_fallback_rate(
    src: Path, method: str, track: str, results_json: Path | None, expected_rate: float | None
) -> float:
    """Resolve measured metadata and reject contradictory sources or stale pairs."""
    rates = []
    provenance = Path(f"{src}.meta.json")
    if provenance.exists():
        meta = json.loads(provenance.read_text())
        if not isinstance(meta, dict):
            raise ValueError(f"{provenance}: expected a metadata object")
        _check_identity(meta, method, track, provenance)
        if "substrate_sha256" in meta:
            digest = hashlib.sha256()
            with src.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if meta["substrate_sha256"] != digest.hexdigest():
                raise ValueError(f"{provenance}: substrate checksum mismatch; regenerate the pair")
        if "overall_fallback_rate" in meta:
            rates.append(_validate_rate(meta["overall_fallback_rate"], provenance))
    if results_json is not None:
        if results_json.is_dir():
            results_json = results_json / "results.json"
        rate = _worst_case_fallback_rate(results_json, method=method, track=track)
        if rate is None:
            raise ValueError(f"{results_json}: missing measured fallback rate")
        rates.append(rate)
    if not rates:
        raise ValueError(
            "Missing measured fallback rate. Export the substrate with its .parquet.meta.json "
            "sidecar, or supply --results-json from the same evaluation. A manual rate alone "
            "is insufficient."
        )
    if expected_rate is not None:
        rates.append(_validate_rate(expected_rate, "--fallback-rate"))
    if any(not math.isclose(rate, rates[0], rel_tol=0, abs_tol=1e-12) for rate in rates[1:]):
        raise ValueError("Conflicting fallback rates; use files from the same evaluation")
    return rates[0]


def find_parquet(dir_path: Path, method: str) -> Path:
    """Locate the substrate parquet for ``method`` inside ``dir_path``.

    Prefers an exact stem match (e.g. ``method='mean'`` → ``mean.parquet``);
    falls back to a substring match on the filename; finally falls back to the
    sole parquet in the directory. Errors if the choice is ambiguous.

    The exact-stem step matters when method names share prefixes (e.g. ``mean``
    is a substring of ``temporal_mean`` and ``personalized_mean``); a pure
    substring match would mis-flag the lookup as ambiguous.
    """
    parquets = sorted(dir_path.glob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No .parquet files in {dir_path}")
    exact = [p for p in parquets if p.stem == method]
    if len(exact) == 1:
        return exact[0]
    named = [p for p in parquets if method in p.name]
    if len(named) == 1:
        return named[0]
    if len(parquets) == 1:
        return parquets[0]
    raise SystemExit(
        f"Ambiguous: {len(parquets)} parquet files in {dir_path}, "
        f"{len(named)} matching method '{method}'. Narrow it down."
    )


def validate_method_column(parquet_path: Path, method: str, *, column: str = "method") -> None:
    """Require the track's method identifier to match the upload name.

    The leaderboard concatenates every ``imputation/*.parquet`` and groups by the
    ``method`` column, so a substrate whose column disagrees with its upload name
    is mislabeled or collides with another method. The column defaults to
    ``"custom"`` when ``evaluate_imputation`` is run without ``method_name=``;
    this guard turns that silent footgun into an upfront error.
    """
    import pandas as pd

    values = sorted(pd.read_parquet(parquet_path, columns=[column])[column].astype(str).unique())
    if values != [method]:
        raise SystemExit(
            f"{parquet_path} has {column} column {values}, expected ['{method}']. "
            f"Re-run the track's evaluation with method_name='{method}' so the "
            f"`{column}` column matches the upload name; the leaderboard groups by that column."
        )


def validate_users_have_demographics(
    parquet_path: Path, repo_id: str, revision: str | None = None
) -> None:
    """Fail loudly if the parquet holds users absent from the demographics file.

    Cohort views filter by ``demographics.parquet``, so any other user would still
    count in the full-cohort view but silently drop out of every cohort.
    """
    import pandas as pd

    try:
        path = hf_hub_download(repo_id, "demographics.parquet", repo_type="dataset", revision=revision)
    except RepositoryNotFoundError:
        if revision is not None:
            raise
        print(f"{repo_id} does not exist yet, so it has no cohort views to check against.")
        return
    except EntryNotFoundError as exc:
        raise SystemExit(
            f"Could not get demographics.parquet from {repo_id} ({type(exc).__name__}). If it "
            "is not published yet, publish it first: see docs/leaderboard-maintenance.md."
        ) from exc
    known = set(pd.read_parquet(path, columns=["user_id"])["user_id"].astype(str))
    users = set(pd.read_parquet(parquet_path, columns=["user_id"])["user_id"].astype(str))
    missing = len(users - known)
    if missing:
        raise SystemExit(
            f"{parquet_path} has {missing} user(s) absent from demographics.parquet. They "
            "would be left out of every cohort view. Check the evaluation cohort. If they are "
            "expected, publish a file that covers them first: run "
            "tools/build_leaderboard_demographics.py with --add-substrate."
        )


@contextmanager
def _stage_substrate(src: Path):
    """Keep a private copy unchanged between validation and the Hub commit."""
    with tempfile.TemporaryDirectory(prefix="openmhc-upload-") as folder:
        staged = Path(folder) / src.name
        shutil.copyfile(src, staged)
        meta = Path(f"{src}.meta.json")
        if meta.exists():
            shutil.copyfile(meta, Path(f"{staged}.meta.json"))
        yield staged


def main() -> None:
    """Upload one method substrate parquet to the leaderboard dataset."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dir", required=True, type=Path, help="Directory containing the method's substrate parquet."
    )
    p.add_argument("--method", required=True, help="Method name (used as the destination filename).")
    p.add_argument("--track", default="imputation", choices=("imputation", "downstream", "forecasting"))
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id.")
    p.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the repo private (default: True).",
    )
    p.add_argument("--name", default=None, help="Display name (writes a <method>.meta.json sidecar).")
    p.add_argument("--type", dest="mtype", default=None, help="Method type, e.g. 'Deep Learning' (sidecar).")
    p.add_argument("--submitter", default=None, help="Submitter / team for attribution (sidecar).")
    p.add_argument(
        "--subtrack",
        default=None,
        help="Sub-track for grouping: 'single-day' or 'long-context' (sidecar).",
    )
    p.add_argument(
        "--fallback-rate", type=float, default=None,
        help="Optional consistency check against the measured rate; cannot supply or override it.",
    )
    p.add_argument(
        "--results-json", type=Path, default=None,
        help="Results from the same evaluation (a JSON file or directory containing results.json).",
    )
    args = p.parse_args()

    if not args.method or args.method in (".", "..") or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for c in args.method
    ):
        p.error("--method must be a filename identifier using letters, digits, '.', '_' or '-'")
    original = find_parquet(args.dir, args.method)
    with _stage_substrate(original) as src:
        _upload(args, src, p)


def _upload(args: argparse.Namespace, src: Path, parser: argparse.ArgumentParser) -> None:
    """Validate a staged substrate and publish it with the measured rate."""
    column = "model" if args.track == "forecasting" else "method"
    validate_method_column(src, args.method, column=column)
    try:
        fallback_rate = resolve_fallback_rate(
            src, args.method, args.track, args.results_json, args.fallback_rate
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    api = HfApi()
    try:
        revision = api.repo_info(args.repo_id, repo_type="dataset").sha
    except RepositoryNotFoundError:
        revision = None
    if revision is not None:
        validate_users_have_demographics(src, args.repo_id, revision=revision)

    dest = f"{args.track}/{args.method}.parquet"
    meta_dest = f"{args.track}/{args.method}.meta.json"
    meta = {"display_name": args.method, "type": "—", "submitter": "—", "subtrack": "other"}
    if revision is not None:
        try:
            existing_path = hf_hub_download(
                args.repo_id, meta_dest, repo_type="dataset", revision=revision
            )
        except EntryNotFoundError:
            pass
        else:
            existing = json.loads(Path(existing_path).read_text())
            if not isinstance(existing, dict):
                parser.error(f"{meta_dest}: existing metadata must be an object")
            meta.update(existing)
    meta.pop("overall_fallback_rate", None)
    meta.update({
        key: value for key, value in {
            "display_name": args.name, "type": args.mtype,
            "submitter": args.submitter, "subtrack": args.subtrack,
            "fallback_rate": fallback_rate,
        }.items() if value is not None
    })
    operations = [
        CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src)),
        CommitOperationAdd(
            path_in_repo=meta_dest,
            path_or_fileobj=json.dumps(meta, indent=2, allow_nan=False).encode("utf-8"),
        ),
    ]
    # Validate and prepare both files before any remote write. Pin the commit
    # to the revision used for demographics and display metadata checks.
    if revision is None:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=False)
        revision = api.repo_info(args.repo_id, repo_type="dataset").sha
    commit = api.create_commit(
        repo_id=args.repo_id, repo_type="dataset", operations=operations,
        parent_commit=revision,
        commit_message=f"Add/update {args.track} predictions and fallback metadata: {args.method}",
    )
    print(f"Uploaded {dest} and {meta_dest} with fallback_rate={fallback_rate:.12g}")
    print(commit.commit_url)


if __name__ == "__main__":
    main()
