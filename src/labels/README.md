# Labels API

This module exposes a fast, in-memory lookup for label values with automatic type enforcement.

## Public surface
- `labels.configure(data_dir=None)`: select a dataset and refresh schemas and cached payloads.
- `labels.age_at(health_code, date)`: calculate calendar-year age at an explicit `pandas.Timestamp`.
- `labels.age_at_enrollment(health_code)`: calculate calendar-year age at enrollment `createdOn`.
- `labels.get_labels(health_code: str, timestamp: pandas.Timestamp, label: str, enforce_type: bool = True, return_valid_only: bool = True) -> LabelResult`
- `labels.LABEL_NAMES`: list of available labels (including `"age"`).
- `labels.LABEL_TYPES`: dict mapping label names to their semantic types (`binary`, `ordinal`, `categorical`, `multi_categorical`, `continuous`).
- `labels.get_labels_statistics() -> pandas.DataFrame`: returns statistics for all labels as a DataFrame.
- `labels.print_labels_statistics()`: prints a table with statistics for all labels.
- `LabelResult`: carries `matched_timestamp` and `value`.
- `LabelTypeError`: raised when a value cannot be converted to its expected type.
- `LabelValueError`: raised when a value is None or NaN.

## Related accessors
- [`context.get_context`](../context/README.md) — discoverable wrapper that validates the label is a context variable (not a prediction target).

## Data sources
- `last_labels.json`: per-label → healthCode → `timestamps`/`values`.
- `context_labels.json`: context-label payload merged into the same lookup index when available.
- `enrollment_info.json`: per-healthCode metadata with de-identified `birth_year` and enrollment `createdOn` for calendar-year age calculations.
- `label_types.json`: maps each label to its semantic type (`binary`, `ordinal`, `categorical`, `multi_categorical`, `continuous`).

## Variable dictionary
See [`data/labels/survey_documentation/INDEX.md`](../../data/labels/survey_documentation/INDEX.md) for variable definitions, original survey options, release mappings, observed values and iOS sources.

All three schemas (`label_types.json`, `ordinal_dictionary.json`, `validity_config.json`) load together from `<dataset_root>/labels/`, with no repository fallback. Reading these restricted Dataverse files requires dataset access. Select the dataset with `data_dir=` or `MHC_DATA_DIR`; unset legacy schema overrides that point elsewhere, since configuration rejects them.
```bash
export MHC_DATA_DIR=/path/to/openmhc-dataset
```

Call `labels.configure(data_dir=...)` before inspecting metadata or switching datasets. Metadata containers are empty until configuration; a first label query can configure from `MHC_DATA_DIR`.

Optional payload overrides include `LABELS_DATA_PATH` and `ENROLLMENT_DATA_PATH`; use files from the same dataset.

## Usage
```python
import pandas as pd
from labels import (
    configure,
    age_at,
    age_at_enrollment,
    get_labels,
    LABEL_NAMES,
    LABEL_TYPES,
    LabelTypeError,
    LabelValueError,
    get_labels_statistics,
    print_labels_statistics,
)

configure(data_dir="/path/to/dataset")

# lookup nearest label value (returns enforced type)
result = get_labels(
    health_code="user-123",
    timestamp=pd.Timestamp("2020-01-01T00:05:00"),
    label="Diabetes",
)
print(result.matched_timestamp, result.value)  # value is bool

# look up the stored age label (returns float)
age = get_labels(
    health_code="user-123",
    timestamp=pd.Timestamp("2020-01-14"),
    label="age",
)
print(age.value)  # e.g., 25.0

# Calculate calendar-year age at an explicit date or at enrollment
print(age_at("user-123", pd.Timestamp("2020-01-14")))
print(age_at_enrollment("user-123"))

# handle type enforcement errors
try:
    result = get_labels(health_code, timestamp, label)
except LabelValueError as e:
    print(f"Missing or NaN value: {e}")
except LabelTypeError as e:
    print(f"Type conversion failed: {e}")

# get statistics as DataFrame for programmatic use
stats_df = get_labels_statistics()
print(stats_df.head())

# print statistics table for all labels
print_labels_statistics()
```

## Type Enforcement

By default, `get_labels()` enforces type conversion based on `label_types.json`:

| Type | Output | Conversion rules |
|------|--------|------------------|
| `binary` | `bool` | "Male"→True, "Female"→False, 0/1→bool, "true"/"false"→bool |
| `ordinal` | `int` | String/float with integer value → int |
| `categorical` | `int` | String/float with integer value → int |
| `multi_categorical` | `tuple[int, ...]` | List of ints (multi-select survey answers) → sorted tuple of ints |
| `continuous` | `float` | Any numeric → float. GoSleepTime/WakeUpTime: datetime string → decimal hours (e.g., "23:30" → 23.5) |

**Exceptions:**
- `LabelValueError`: Raised if value is `None` or `NaN`
- `LabelTypeError`: Raised if value cannot be converted to expected type

**Disable type enforcement:**
```python
# Get raw value without type conversion
result = get_labels(health_code, timestamp, label, enforce_type=False)
```

**Label type mapping:**
```python
from labels import LABEL_TYPES, configure

configure(data_dir="/path/to/dataset")

# Check a label's type
print(LABEL_TYPES["BiologicalSex"])  # "binary"
print(LABEL_TYPES["happiness"])       # "continuous"
print(LABEL_TYPES["field_race"])      # "multi_categorical"
print(LABEL_TYPES["WeightKilograms"]) # "continuous"
```

## Label Statistics

The `print_labels_statistics()` function outputs a table with statistics for all available labels:

```
Label                     Type         Min          Max          Median       Unique
-------------------------------------------------------------------------------------
BiologicalSex             str          N/A          N/A          N/A          2
Diabetes                  bool         0.00         1.00         0.00         2
DiastolicBloodPressure    float        40.00        120.00       75.00        79
GoSleepTime               str          N/A          N/A          N/A          16574
Hdl                       float        0.50         5.00         2.83         82
HeightCentimeters         float        100.00       230.00       175.26       46
Hypertension              bool         0.00         1.00         0.00         2
Ldl                       float        0.50         14.93        5.38         234
SystolicBloodPressure     float        70.00        200.00       120.00       114
TotalCholesterol          float        2.00         19.98        9.38         270
WakeUpTime                str          N/A          N/A          N/A          16795
WeightKilograms           float        30.00        300.00       77.11        366
atwork                    str          1.00         4.00         1.00         4
feel_worthwhile1          float        0.00         10.00        8.00         11
feel_worthwhile2          float        0.00         10.00        8.00         11
feel_worthwhile3          float        0.00         10.00        4.00         11
feel_worthwhile4          float        0.00         10.00        1.00         11
happiness                 float        0.00         10.00        8.00         11
heart_disease             float        1.00         11.00        10.00        11
phys_activity             str          1.00         6.00         3.00         6
satisfiedwith_life        float        0.00         10.00        8.00         11
sleep_diagnosis1          bool         0.00         1.00         0.00         2
sleep_diagnosis2          float        1.00         8.00         1.00         8
sleep_time                float        0.00         24.00        8.00         22
sleep_time1               float        0.00         24.00        7.00         25
vascular                  float        1.00         8.00         7.00         8
vigorous_act              float        0.00         1440.00      30.00        265
work                      bool         0.00         1.00         1.00         2
```

## Behavior
- For labels in `labels.json`, the closest timestamp is chosen; ties favor the earlier point.
- Age lookups return the stored `age` label value from `last_labels.json`.
- Calculated age is `reference_date.year - birth_year`. Track 1 targets and Linear age covariates use the existing age-label timestamp; age subgroups in all three tracks use enrollment `createdOn`. This counts calendar years, not completed birthdays.

Previously measured local throughput (not remeasured for this change):
- Label lookups: ~762,936.5 per second
- Age lookups: ~729,164.1 per second

## Tests
Run from repo root:
```bash
PYTHONPATH=src pytest tests/test_labels_api.py
```
