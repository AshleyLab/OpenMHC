# Context API

Discoverable accessor for context (covariate) labels — participant-level
variables that aren't prediction targets, e.g. `field_smokingHistory`, `field_race`,
`field_device_iphone`.

## Public surface
- `context.get_context(health_code, timestamp, context_label, enforce_type=True, return_valid_only=True) -> LabelResult`
- `context.CONTEXT_NAMES`: list of available context-label names

## Why a separate accessor?

Context labels are merged into the same in-memory index as prediction targets
and would be queryable via `labels.get_labels()` directly. `get_context()`
adds a guardrail: it raises `ValueError` if the label is actually a prediction
target, so accidental category mix-ups fail loudly instead of silently.

## Usage

```python
import pandas as pd
from labels import configure
from context import get_context, CONTEXT_NAMES

configure(data_dir="/path/to/dataset")

# Look up the user's smoking status nearest to the query date
result = get_context(
    health_code="user-123",
    timestamp=pd.Timestamp("2020-01-01"),
    context_label="field_smokingHistory",
)
print(result.matched_timestamp, result.value)
```

## Data sources

Reads the selected dataset's `labels/context_labels.json` via the shared
`LabelsStore` in `labels.api`. Context names and types come from that dataset's
schemas. Call `labels.configure()` before inspecting `CONTEXT_NAMES` or
switching datasets; a first context query can configure from `MHC_DATA_DIR`.
See the [Labels API data sources](../labels/README.md#data-sources) for dataset
paths and overrides. No additional data files.

## Variable dictionary

See [`data/labels/survey_documentation/INDEX.md`](../../data/labels/survey_documentation/INDEX.md) for variable definitions, original survey options, release mappings, observed values and iOS sources. The documented Full/XS release contains 129 context variables.
