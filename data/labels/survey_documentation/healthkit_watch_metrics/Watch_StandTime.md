# Watch_StandTime

**Benchmark column**: `Watch_StandTime`
**Raw identifier**: `HKQuantityTypeIdentifierAppleStandTime`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1377 (registered in `healthKitQuantityTypesToRead` method, iOS 13.0+ only)
- Unit configuration: Line 557 (in `researcherSpecifiedUnits` method, iOS 13.0+ only)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierAppleStandTime`
- Expected unit: minutes
- Source device: Apple Watch (tracks stand hours throughout the day)
- Availability: iOS 13.0+ only (conditional registration in `researcherSpecifiedUnits`)

## Observed values

**Total observations**: 134,955 — **type-enforced**: 134,955 (**unique**: 7,112) — raw Python types seen: `float` (134,955).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.02 |
| q25 | 1.15 |
| median | 1.71 |
| mean | 1.82 |
| q75 | 2.36 |
| max | 14.00 |
| std | 0.94 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1.50` | 194 |
| `1.62` | 183 |
| `1.77` | 179 |
| `1.43` | 178 |
| `1.45` | 177 |
| `1.25` | 177 |
| `1.23` | 177 |
| `1.15` | 176 |
| `1.05` | 171 |
| `1.30` | 170 |
| `1.27` | 169 |
| `1.38` | 167 |
| `1.37` | 166 |
| `1.12` | 166 |
| `1.32` | 165 |
| `1.20` | 164 |
| `1.57` | 164 |
| `1.52` | 163 |
| `1.48` | 162 |
| `1.40` | 161 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Stand time metric added for iOS 13.0+ support; reverted mobility changes in Sept 2020 maintained this support

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Apple Watch tracks stand time as minutes user spent standing during each hour
- Complements Move Ring and Exercise Ring for overall activity monitoring
- Part of cardiovascular health tracking in MyHeart Counts research
- iOS 13.0+ feature conditional availability ensures backward compatibility
