# WeightKilograms

**Benchmark column**: `WeightKilograms`
**Raw identifier**: `HKQuantityTypeIdentifierBodyMass`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1346 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 524 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors and manually entered by user.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierBodyMass`
- Expected unit: kilograms (kg)
- Formula: `[HKUnit gramUnitWithMetricPrefix:HKMetricPrefixKilo]`
- Source devices: iPhone (user-entered), Apple Watch (synced from health data)

## Observed values

**Total observations**: 10,126 — **type-enforced**: 10,126 (**unique**: 314) — raw Python types seen: `float` (10,126).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 31.30 |
| q25 | 70.31 |
| median | 81.65 |
| mean | 84.09 |
| q75 | 95.25 |
| max | 223.6 |
| std | 20.87 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `74.84` | 231 |
| `81.65` | 199 |
| `77.11` | 198 |
| `83.91` | 195 |
| `72.57` | 187 |
| `86.18` | 181 |
| `68.04` | 177 |
| `79.38` | 177 |
| `90.72` | 175 |
| `70.31` | 168 |
| `88.45` | 141 |
| `95.25` | 134 |
| `63.50` | 131 |
| `92.99` | 128 |
| `79.83` | 121 |
| `65.77` | 119 |
| `99.79` | 114 |
| `78.02` | 114 |
| `80.74` | 114 |
| `76.20` | 106 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Body mass identifier stable throughout app history; core metric for all health calculations

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Users can input weight through Health app on iPhone or Apple Health integration
- Weight data used for BMI calculation, basal metabolic rate estimation, and activity intensity adjustments
- Critical metric for MyHeart Counts cardiovascular health assessment and personalized recommendations
- Cross-referenced with Height to compute BMI and other health indicators
