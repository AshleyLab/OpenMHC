# Watch_VO2Max

**Benchmark column**: `Watch_VO2Max`
**Raw identifier**: `HKQuantityTypeIdentifierVO2Max`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1363 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 538 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierVO2Max`
- Expected unit: milliliters per kilogram per minute (ml/kg/min)
- Formula: `[[HKUnit literUnitWithMetricPrefix:HKMetricPrefixMilli] unitDividedByUnit:[[HKUnit gramUnitWithMetricPrefix:HKMetricPrefixKilo] unitMultipliedByUnit:[HKUnit minuteUnit]]]`
- Source device: Apple Watch (primary source for VO2 Max estimation)

## Observed values

**Total observations**: 68,782 — **type-enforced**: 68,782 (**unique**: 32,919) — raw Python types seen: `float` (68,782).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 14.00 |
| q25 | 27.71 |
| median | 33.92 |
| mean | 33.75 |
| q75 | 39.34 |
| max | 59.89 |
| std | 8.18 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `14.00` | 127 |
| `25.92` | 27 |
| `36.20` | 26 |
| `37.50` | 26 |
| `27.68` | 26 |
| `37.20` | 26 |
| `27.08` | 26 |
| `31.67` | 25 |
| `31.08` | 25 |
| `26.92` | 25 |
| `29.42` | 24 |
| `26.59` | 23 |
| `35.12` | 23 |
| `30.42` | 23 |
| `38.30` | 22 |
| `25.81` | 22 |
| `35.00` | 22 |
| `27.83` | 22 |
| `25.26` | 22 |
| `27.97` | 22 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: VO2 Max identifier stable; reverted mobility field additions but VO2Max registration unchanged

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- VO2 Max is Apple Watch Series 3+ metric calculated from workouts and activity patterns
- Represents maximum oxygen utilization capacity, key cardiovascular fitness indicator
- Part of continuous health monitoring for MyHeart Counts research
