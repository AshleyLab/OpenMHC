# OpenMHC leaderboard — maintainer runbook

Operational notes for the people who run the leaderboard. Submitters don't need
this — see the README "Submit to the Leaderboard" section and
`tools/leaderboard_docs/` instead.

## Two Hugging Face repos

The live leaderboard is two HF repos, separate from this code repo. Auth uses
standard `huggingface_hub` token discovery (`HF_TOKEN` env or
`huggingface-cli login`) with an org-write token on `MyHeartCounts`.

- **Space** `MyHeartCounts/OpenMHC` (`repo_type="space"`, Docker/FastAPI) — the
  live leaderboard *website* (https://myheartcounts-openmhc.hf.space). It
  self-renders HTML and computes the table in-process at startup. There is no
  static `leaderboard.json` and no `/api/data` endpoint.
- **Dataset** `MyHeartCounts/OpenMHC-leaderboard-data` (`repo_type="dataset"`,
  public) — the per-method substrate: `<track>/<method>.parquet` plus display
  sidecars `<track>/<method>.meta.json`
  (`{display_name, type, submitter, subtrack, fallback_rate}`). Track 2
  ships `imputation/<method>.parquet` (per-user errors); Track 1 ships
  `downstream/<method>.parquet` (per-user **prediction pairs** — its
  ranking/correlation metrics don't decompose per user, see
  `tools/leaderboard_docs/downstream/SCHEMA.md`). There is also an
  `imputation/bootstrap/` subdir (the per-draw CI reference frame) that is **not**
  used by the live point compute.

## How the Space computes

`leaderboard_compute.py` (in the Space) `snapshot_download`s
`imputation/*.parquet` + `*.meta.json`, then calls the canonical `openmhc`
reducers:

- **Skill** — `compute_per_task_paired_R(baseline_method="locf")` →
  `compute_skill_scores(mode="paired")`.
- **Rank** — `compute_average_rankings` → `aggregate_task_ranks_to_scopes`.

`HEADLINE_SCOPE="overall"`. Skill and rank filter to `subgroup_attr == "all"`. Cohort
views keep those rows for the participants in the cohort, using `demographics.parquet`.

## Adding / updating a method (no Space rebuild)

1. Upload the substrate to the **dataset** (`--track imputation` or `downstream`):

   ```bash
   python tools/upload_leaderboard_substrate.py \
       --dir <dir> --method <m> --track imputation \
       --name "<Display>" --type "<Type>" --submitter "<Team>" \
       --results-json <evaluation-dir>/results.json
   ```

   The uploader stages a private temporary copy of the prediction file and its
   provenance sidecar, validates that copy before any remote write, then publishes `<m>.parquet` and `<m>.meta.json` together in one commit. Existing
   display fields are preserved unless supplied on the command line. The commit
   is pinned to the revision used for the cohort and metadata checks; if another
   upload changes that revision, rerun the command to validate the new state.
   Temporary staging needs space for one prediction file and is removed after
   success or failure.

   For Track 1, omit `--results-json`: `evaluate_prediction(..., output_dir=...)`
   and the offline prediction-pair exporter write `<m>.parquet.meta.json` beside
   the predictions. The uploader reads its `overall_fallback_rate` and writes the
   leaderboard's `fallback_rate` field. This is the pooled count of substituted
   participant-task pairs divided by the total test pairs, using the same rule
   for every method. The offline exporter requires measured `fallback.json`
   counts matching the exported tasks and cohort sizes; it never substitutes
   zero for missing measurements.

   Imputation and forecasting can use `--results-json` from the same evaluation.
   Imputation reports the maximum across scenario/split cells; forecasting uses
   its pooled rate. A directory argument resolves to `results.json` inside it.
   Missing, nonnumeric, nonfinite, out-of-range or conflicting rates stop the
   upload. A measured zero is valid. `--fallback-rate` is retained only as an
   optional consistency check: it cannot supply or override a measurement.
   A measured zero requires explicit substitution counts and a positive
   denominator. Missing measurements or an empty cohort/target group have an
   unknown rate (`None` in Python, `null` in JSON), which cannot be submitted.
   Result objects constructed without measurements also default to `None`.

   Track 2's internal substitution opt-out remains supported. It does not collect
   target counts, so its rate is unknown; the public API's adapter supplies a
   fallback fill and records counts normally. Predictions and scoring are unchanged.
   Track 3's existing-file skip policy also remains unchanged. A resumed run's
   counters cover only newly generated predictions, so its full-run fallback rate
   is unknown. Use a complete recorded evaluation's metadata or a fresh run before
   publication; do not replace an unknown rate with zero.
   Explicit method or track identifiers in either metadata source must match
   the upload; older results without identifiers still require manual pairing.

   New Track-1 provenance sidecars include a prediction-file checksum, which the
   uploader verifies to detect replaced files. Older sidecars and explicit result
   files remain supported but do not prove file identity; keep them paired with
   their original evaluation output. A zero already emitted by older code from
   missing measurements cannot be authenticated by this uploader; recover the
   original recorded counts or rerun the evaluation before publishing it.
   In particular, Track-1 `to_json()` exports
   a metrics list, so use the adjacent provenance sidecar for uploading.

   Submitters using `results.to_submission_yaml(...)` also receive the measured
   rate in their generated metadata. Imputation packets require measurements for
   every scenario/split. Direct calls to the lower-level submission renderers now
   require a `fallback_rate` argument from the evaluation results.

   The `method` column is validated against `<m>` for `imputation` and
   `downstream`; forecasting's `model` column is checked against the same name.
   The upload stops if the file has users absent from
   `demographics.parquet`, because they would be left out of every cohort view.
2. The Space recomputes only on **restart** (in-process cache, no auto-refresh).
   Restart it to pick up the new/changed method. Uploading to the dataset does
   *not* trigger a Space rebuild; only pushing to the Space does.

   > **Track 1 (downstream)** is wired the same way (since 2026-07). The Space's
   > `compute_downstream_rows` runs `reduce_substrate_to_point`
   > (`downstream_evaluation/evaluation/bootstrap_skill_rank.py`) on the per-user
   > *pairs* — point skill / rank vs the `linear`
   > baseline — and reads `fallback_rate` from the sidecar. So a new downstream
   > substrate appears on the next restart, exactly like the other two tracks.
   > (It does not reuse the imputation per-cell-mean path.)

## The demographics file

`demographics.parquet` (`user_id`, `age_group`, `sex`) is the only source of cohort
membership. `tools/build_leaderboard_demographics.py` builds it with
`labels.user_demographics`, for the users already in published substrates, and stops
if any of them has an unknown age group. Rebuild it when a substrate adds users or the
age or sex rule changes, then upload both files in one commit. For a substrate with new
users, do this first and pass the file with `--add-substrate`, since the substrate
upload stops until the file covers them:

```bash
python tools/build_leaderboard_demographics.py --data-dir <dataset> --out-dir <dir> \
    [--add-substrate <m>.parquet]
```

```python
HfApi().upload_folder(
    folder_path="<dir>",
    repo_id="MyHeartCounts/OpenMHC-leaderboard-data",
    repo_type="dataset",
    allow_patterns=["demographics.parquet", "demographics.meta.json"],
    commit_message=...,
)
```

Restart the Space afterwards, as for a new method.

A submission merged as a Hub pull request skips the upload check. The Space logs a
warning at startup for users absent from the file. If it does, rebuild the file, which
now picks up the merged substrate, and restart.

Dataset visibility, if needed:
`HfApi().update_repo_settings(repo_id, private=False, repo_type="dataset")`
(the method is `update_repo_settings` — `update_repo_visibility` was renamed in
current `huggingface_hub`).

## Editing & deploying the Space (use HfApi, not git push)

1. Clone (ephemeral — re-clone each session):
   `git clone https://huggingface.co/spaces/MyHeartCounts/OpenMHC`.
   Files: `app.py`, `leaderboard_compute.py`, `Dockerfile`, `requirements.txt`,
   `logo.png`.
2. Edit locally and smoke-test in a slim venv that mirrors the Space (slim deps +
   `pip install -e . --no-deps`), e.g.
   `PYTHONPATH=<clone> python -c "import app; app.index()"`.
3. Deploy:

   ```python
   HfApi().upload_folder(
       folder_path=...,
       repo_id="MyHeartCounts/OpenMHC",
       repo_type="space",
       allow_patterns=[...changed files...],
       ignore_patterns=[".git/**", "**/__pycache__/**", "*.pyc"],
       commit_message=...,
   )
   ```

   This rebuilds the Docker Space.
4. Verify: poll `HfApi().space_info("MyHeartCounts/OpenMHC").runtime.stage`
   (`BUILDING` / `RUNNING_BUILDING` → `RUNNING`). On `BUILD_ERROR`, read the
   build logs via the Space's `logs/build` API endpoint. Then `curl` `/health`
   and `/`.

## Build / runtime gotchas

- `python:*-slim` has **no git** → the `Dockerfile` must `apt-get install git`
  before the `openmhc` VCS pip install.
- Install `openmhc` **`--no-deps`** plus a slim `requirements.txt` (no torch).
  The point-flow reducers are torch-free but still pull `scikit-learn`, `scipy`,
  and `datasets` (via `data.processing.hf_config`).
- Pin `pandas>=2.1,<3`: `compute_average_rankings` needs `future_stack` (≥2.1);
  pandas 3.0 is untested.
- Pin `openmhc` to a **commit SHA, not `@main`**: HF caches the Docker pip
  layer, so a bare `@main` silently serves stale code on rebuilds that don't
  touch the `Dockerfile`. Bump the pinned commit whenever the reducers change.
