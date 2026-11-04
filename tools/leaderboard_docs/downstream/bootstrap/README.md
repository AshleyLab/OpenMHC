# Downstream bootstrap reference

`draws.parquet` contains the current Track-1 all-participant per-draw errors;
`draws.meta.json` records the sampling protocol and provenance. The live Space
computes point scores from `downstream/<method>.parquet`; it does not read this
bootstrap reference or display confidence intervals.

The reference covers eight methods, 32 tasks and the Linear baseline. Each of
1,000 replicates (seed 42) samples the entire declared test split of 3,569
participants once, sharing participant multiplicities across every task and
method. Each task retains its fixed eligibility rule; 1,762 distinct participants
contribute to at least one task. Resampled task sizes therefore vary. Undefined
single-class binary task draws are omitted jointly across methods and recorded
in the metadata. Draw -1 is the original-cohort point estimate.

Errors are `E = 1 - metric`: average precision for binary tasks, Spearman for
ordinal tasks and Pearson for regression. Skill uses clipped method/Linear error
ratios (0.01 to 100), a geometric mean within each domain, and an equal-weight
mean of the five domain skills. Average rank is also domain-balanced.

This reference supplies main skill/rank percentile confidence intervals. It
contains only `subgroup_attr="all"`, `subgroup_value="all"`; it does not supply
demographic subgroup or worst-subgroup confidence intervals.
The live worst-subgroup point score is computed separately from four cohort
views (age <40, age 40+, female, male), each with its own Linear baseline.

The draws were generated from the verified saved evaluation predictions. The
uploaded prediction-pair files serialize those predictions as float32, so a
bootstrap regenerated from the uploaded pairs need not match byte for byte.
Live point-score parity is checked at the leaderboard's display precision.
Missing WBM predictions are replaced with the corresponding Linear baseline
predictions; metadata records
`fallback_policy="direct_substitution"`.

Keep both reference files synchronized with the prediction files when publishing
a new Track-1 result revision.

## Loading

```python
from huggingface_hub import hf_hub_download
import pandas as pd, json

draws_path = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "downstream/bootstrap/draws.parquet",
    repo_type="dataset",
)
meta_path = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "downstream/bootstrap/draws.meta.json",
    repo_type="dataset",
)
draws = pd.read_parquet(draws_path)
meta = json.loads(open(meta_path).read())
print(meta["seed"], meta["n_boot"], len(meta["methods"]))
```

See [`SCHEMA.md`](SCHEMA.md) for the full column spec.

## Uploaded with

`tools/upload_leaderboard_bootstrap.py --track downstream` in the
[code repo](https://github.com/AshleyLab/myheartcounts-dataset).
