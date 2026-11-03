# Watch_HeartRateVariabilitySDNN

**Benchmark column**: `Watch_HeartRateVariabilitySDNN`
**Raw identifier**: `HKQuantityTypeIdentifierHeartRateVariabilitySDNN`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1353 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 529 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierHeartRateVariabilitySDNN`
- Expected unit: milliseconds (ms)
- Formula: `[HKUnit secondUnitWithMetricPrefix:HKMetricPrefixMilli]`
- Source device: Apple Watch (primary source for HRV measurement)

## Observed values

**Total observations**: 197,454 — **type-enforced**: 197,454 (**unique**: 172,403) — raw Python types seen: `float` (197,454).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 5.18 |
| q25 | 22.63 |
| median | 29.36 |
| mean | 33.48 |
| q75 | 38.99 |
| max | 200 |
| std | 18.98 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `19.30` | 11 |
| `20.64` | 11 |
| `24.77` | 9 |
| `28.69` | 9 |
| `24.80` | 9 |
| `27.31` | 9 |
| `25.91` | 9 |
| `30.20` | 9 |
| `24.71` | 9 |
| `22.58` | 9 |
| `28.00` | 9 |
| `22.16` | 9 |
| `19.81` | 8 |
| `21.90` | 8 |
| `23.91` | 8 |
| `26.84` | 8 |
| `22.01` | 8 |
| `25.68` | 8 |
| `29.06` | 8 |
| `23.53` | 8 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Heart Rate Variability SDNN (Standard Deviation of NN intervals) identifier stable throughout app history

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- SDNN (Standard Deviation of Normal-to-Normal intervals) measures variability in time between heartbeats
- Indicates autonomic nervous system activity; higher HRV typically associated with better cardiovascular fitness
- Collected from Apple Watch during sleep and rest periods
- Part of MyHeart Counts cardiovascular research monitoring
