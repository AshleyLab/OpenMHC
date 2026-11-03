# Watch_RestingHeartRate

**Benchmark column**: `Watch_RestingHeartRate`
**Raw identifier**: `HKQuantityTypeIdentifierRestingHeartRate`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1351 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 527 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierRestingHeartRate`
- Expected unit: beats per minute (bpm)
- Formula: `[[HKUnit countUnit] unitDividedByUnit:[HKUnit secondUnit]]`
- Source device: Apple Watch (primary source for this metric)

## Observed values

**Total observations**: 205,617 — **type-enforced**: 205,617 (**unique**: 1,271) — raw Python types seen: `float` (205,617).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 31.50 |
| q25 | 56.00 |
| median | 62.00 |
| mean | 62.74 |
| q75 | 68.50 |
| max | 175 |
| std | 9.64 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `60.00` | 9,072 |
| `59.00` | 8,129 |
| `58.00` | 7,108 |
| `63.00` | 6,941 |
| `57.00` | 6,815 |
| `61.00` | 6,800 |
| `62.00` | 6,381 |
| `56.00` | 6,296 |
| `66.00` | 6,168 |
| `67.00` | 6,128 |
| `64.00` | 5,945 |
| `68.00` | 5,569 |
| `55.00` | 5,563 |
| `65.00` | 5,470 |
| `54.00` | 5,056 |
| `69.00` | 5,055 |
| `53.00` | 4,837 |
| `52.00` | 4,381 |
| `70.00` | 4,323 |
| `71.00` | 3,896 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Resting heart rate identifier stable since early releases; reverted mobility changes in Sept 2020 but resting heart rate registration unchanged

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency during app initialization
- Resting heart rate is Apple Watch-specific metric typically collected during sleep or inactivity periods
- Part of continuous cardiovascular monitoring for MyHeart Counts research study
