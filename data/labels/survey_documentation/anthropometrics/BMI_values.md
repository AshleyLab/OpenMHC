# BMI_values

**Benchmark column**: `BMI_values`
**Raw identifier**: `HKQuantityTypeIdentifierBodyMassIndex` OR computed from `HKQuantityTypeIdentifierBodyMass` + `HKQuantityTypeIdentifierHeight`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: N/A (not directly registered; computed from Body Mass and Height)
- Height registered: Line 1349 (in `healthKitQuantityTypesToRead` method)
- Weight registered: Line 1346 (in `healthKitQuantityTypesToRead` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Computed from Body Mass (kg) and Height (m) automatically via HealthKit data.

## Answer options / Units
- Computed as: Weight (kg) / Height (m²)
- Expected unit: count per square meter (kg/m²)
- Source variables: `HKQuantityTypeIdentifierBodyMass` and `HKQuantityTypeIdentifierHeight`
- Source devices: iPhone (user-entered or synced) and Apple Watch

## Observed values

**Total observations**: 10,047 — **type-enforced**: 10,047 (**unique**: 2,477) — raw Python types seen: `float` (10,047).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 11.74 |
| q25 | 23.33 |
| median | 26.11 |
| mean | 27.24 |
| q75 | 30.04 |
| max | 58.04 |
| std | 5.85 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `25.10` | 40 |
| `22.96` | 35 |
| `23.01` | 34 |
| `25.83` | 34 |
| `24.37` | 32 |
| `23.67` | 32 |
| `25.85` | 31 |
| `23.73` | 29 |
| `25.10` | 27 |
| `27.26` | 27 |
| `22.81` | 27 |
| `22.89` | 27 |
| `22.38` | 27 |
| `23.49` | 27 |
| `26.54` | 26 |
| `26.50` | 26 |
| `23.57` | 26 |
| `25.09` | 26 |
| `27.32` | 25 |
| `27.20` | 25 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: BMI is derived metric; no standalone HealthKit identifier for BMI registered. Both Height and Body Mass identifiers are stable throughout app history

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency for both Height and Body Mass
- BMI computed from two fundamental measurements collected via HealthKit
- Used in daily insights to categorize weight status (see APHDailyInsights.m reference to "Optimal: BMI of 18.5 to 24.9")
- Primary indicator for cardiovascular risk assessment in MyHeart Counts research
- Automatically updated whenever either Height or Weight changes in HealthKit
