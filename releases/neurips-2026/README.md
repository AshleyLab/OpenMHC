# OpenMHC v1.0.0 — NeurIPS 2026

This snapshot is prepared for the following software release:

| Item | Value |
| --- | --- |
| Python package version | `1.0.0` |
| Release tag | `v1.0.0` |
| Conference tag | `neurips-2026` |
| GitHub release title | OpenMHC v1.0.0 — NeurIPS 2026 |

When publishing, create both tags on the same merged commit. Keep both tags
fixed; subsequent software releases receive new version tags. After publication,
check out `v1.0.0` (or `neurips-2026`) before following the instructions below.
The package version is independent of the Dataverse dataset versions.

[manifest.yaml](manifest.yaml) records the Hugging Face revisions selected for
the NeurIPS 2026 release preparation: 14 OpenMHC model bundles, the upstream
Chronos-2 and Toto base models, the results dataset, and the leaderboard Space.
Use the full commit hashes in that file when fetching these artifacts.

The official NeurIPS paper link is **TODO until publication**. The dataset
selections are Full **1.4** (`10.7910/DVN/XNBITM`) and XS **1.2**
(`10.7910/DVN/RIKSK4`), carried forward from the release discussion. Dataverse
inventory and access checks are outside this snapshot's scope.

The `inspected_code_commit` field identifies the code inspected while preparing
these instructions. The eventual Git tag identifies the release commit,
including this directory; adding these files does not create or publish a tag.

## Load an OpenMHC model at its recorded revision

Run the examples from the repository root, with the relevant model dependencies
and the `hf` extra installed; see [installation instructions](../../docs/install.md).
The manifest is ordinary YAML, read here with PyYAML (included in the `hf` extra).

```python
from pathlib import Path

import yaml

release = yaml.safe_load(Path("releases/neurips-2026/manifest.yaml").read_text())
models = release["huggingface"]["models"]


def bundle_uri(repo_id):
    return f"hf://{repo_id}@{models[repo_id]['revision']}"
```

Pass that URI to the appropriate bundle loader. For example, in environments
with the corresponding model dependencies:

```python
from openmhc.encoders import WBM
from openmhc.forecasters import Chronos2Forecaster
from openmhc.imputers import BRITSImputer

wbm = WBM.from_release(bundle_uri("MyHeartCounts/openmhc-wbm-dp"))
brits = BRITSImputer.from_release(bundle_uri("MyHeartCounts/openmhc-brits-imp"))
chronos_ft = Chronos2Forecaster.from_release(
    bundle_uri("MyHeartCounts/openmhc-chronos2-fc")
)
```

Each URI pins the complete bundle, including its architecture manifest and
normalization/configuration files. The manifest's `files` lists identify the
files found at the recorded revisions. WBM still requires its CUDA/Mamba
dependencies. See the [imputer](../../docs/neural-imputers.md) and
[forecaster](../../docs/neural-forecasters.md) guides for the other wrappers.

For the Track 1 Hydra CLI, checkpoint references use the same `@commit` syntax.
For example, use the LSM-2 daily revision from the manifest:

```bash
mhc-downstream-eval method=lsm2 \
  'method.checkpoint="hf://MyHeartCounts/openmhc-lsm2-daily@aacb09cf634ad78977b8263e49cb70ebb677751f#loss=0.2706.ckpt"'
```

Configure the dataset root and evaluation settings separately using the
[Track 1 guide](../../src/downstream_evaluation/README.md).

## Pin upstream base models too

The upstream `amazon/chronos-2` and `Datadog/Toto-Open-Base-1.0` repositories are
recorded separately from the OpenMHC fine-tuned bundles. Download a recorded
snapshot and pass its local directory to a loader that accepts a pretrained
model path:

```python
from huggingface_hub import snapshot_download

base_id = "amazon/chronos-2"
base = models[base_id]
chronos_base_path = snapshot_download(
    repo_id=base_id,
    revision=base["revision"],
    allow_patterns=base["files"],
)
```

Toto forecasting also loads its base model when applying a fine-tuned `.ckpt`.
The public `TotoForecaster.from_release` wrapper currently does not expose a
base-model path override. To pin both components, download them into local
directories and use the forecasting CLI's explicit base-model setting. Run this
in the separate Toto environment described in the installation guide:

```python
for repo_id, directory in (
    ("Datadog/Toto-Open-Base-1.0", "artifacts/neurips-2026/toto-base"),
    ("MyHeartCounts/openmhc-toto-fc", "artifacts/neurips-2026/toto-ft"),
):
    entry = models[repo_id]
    snapshot_download(
        repo_id=repo_id,
        revision=entry["revision"],
        allow_patterns=entry["files"],
        local_dir=directory,
    )
```

```bash
mhc-forecast-eval model=toto \
  model.release_dir=artifacts/neurips-2026/toto-ft \
  model.toto.pretrained_model_name_or_path=artifacts/neurips-2026/toto-base
```

Supply the dataset root and the intended evaluation settings as described in
the [forecasting guide](../../src/forecasting_evaluation/README.md).

## Retrieve the recorded results

The results repository revision covers per-user results, metadata, demographic
grouping inputs, and bootstrap draws. Fetch individual files with that revision:

```python
from huggingface_hub import hf_hub_download

results = release["huggingface"]["results"]
wbm_results_path = hf_hub_download(
    repo_id=results["repo_id"],
    repo_type="dataset",
    revision=results["revision"],
    filename="downstream/wbm.parquet",
)
```

Paper-results scripts that accept `--revision` should receive the recorded
results hash. For example:

```bash
python scripts/paper_results/forecasting/build_forecasting_bootstrap_from_hf.py \
  --revision c4dc96b8dbf7af67704afaef179a2a613920952c \
  --out-dir artifacts/neurips-2026/forecasting-tables
```

The leaderboard entry pins the Space's source code separately:

```python
space = release["huggingface"]["leaderboard"]
leaderboard_source_path = snapshot_download(
    repo_id=space["repo_id"],
    repo_type="space",
    revision=space["revision"],
)
```

The hosted leaderboard can continue to change. Use the recorded results revision
for release comparisons; downloading the Space source alone does not pin data
that it fetches at runtime.

## Verification recorded for this snapshot

Every recorded Hugging Face commit resolved through the Hub API, and its file
listing contained the files named in the manifest. The 14 OpenMHC bundle JSON
manifests were read at those commits; their checkpoint, normalization and
Fourier-mode references were checked against the listings.

These are metadata checks. Checkpoint/result payload hashes, model loading,
benchmark reruns and historical training provenance were not independently
verified in this work. Recording an upstream revision identifies the snapshot
selected here; it does not establish which upstream revision a past run used.
Model packages and hardware requirements remain as documented in the install
guide; this manifest does not lock Python dependencies.

Hugging Face supports full commit hashes in
[`revision=`](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).
Keep this manifest fixed once the release is tagged. Future artifact selections
belong in a subsequent release.
