# Watch_BasalEnergyBurned

**Benchmark column**: `Watch_BasalEnergyBurned`
**Raw identifier**: `HKQuantityTypeIdentifierBasalEnergyBurned`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1348 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 523 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierBasalEnergyBurned`
- Expected unit: kilocalories (kcal)
- Formula: `[HKUnit smallCalorieUnit]`
- Source device: Apple Watch (primary source for basal metabolic rate estimation)

## Observed values

**Total observations**: 269,912 — **type-enforced**: 269,912 (**unique**: 254,001) — raw Python types seen: `float` (269,912).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 500.3 |
| q25 | 1544 |
| median | 1831 |
| mean | 1958 |
| q75 | 2211 |
| max | 4500 |
| std | 675.8 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1697` | 112 |
| `1529` | 105 |
| `3148` | 102 |
| `3138` | 92 |
| `1524` | 83 |
| `1816` | 76 |
| `1340` | 76 |
| `1508` | 73 |
| `1779` | 73 |
| `1534` | 72 |
| `1513` | 71 |
| `1774` | 69 |
| `2814` | 66 |
| `1662` | 66 |
| `1759` | 64 |
| `1633` | 63 |
| `1692` | 63 |
| `1635` | 59 |
| `1506` | 59 |
| `1504` | 57 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Basal energy identifier stable throughout app history; reverted mobility changes did not affect registration

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Basal energy burned represents resting metabolic rate (calories burned at rest)
- Calculated based on age, weight, height, and biological sex
- Apple Watch continuously estimates basal energy using motion and heart rate data
- Part of comprehensive energy expenditure tracking for cardiovascular health assessment in MyHeart Counts
