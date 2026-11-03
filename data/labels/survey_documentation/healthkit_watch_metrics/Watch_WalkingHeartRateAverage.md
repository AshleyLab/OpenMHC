# Watch_WalkingHeartRateAverage

**Benchmark column**: `Watch_WalkingHeartRateAverage`
**Raw identifier**: `HKQuantityTypeIdentifierWalkingHeartRateAverage`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1352 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 528 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierWalkingHeartRateAverage`
- Expected unit: beats per minute (bpm)
- Formula: `[[HKUnit countUnit] unitDividedByUnit:[HKUnit secondUnit]]`
- Source device: Apple Watch (primary source for walking heart rate)

## Observed values

**Total observations**: 196,697 — **type-enforced**: 196,697 (**unique**: 28,268) — raw Python types seen: `float` (196,697).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 45.00 |
| q25 | 87.75 |
| median | 96.17 |
| mean | 96.61 |
| q75 | 104.8 |
| max | 202 |
| std | 13.38 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `90.00` | 349 |
| `97.00` | 310 |
| `105` | 296 |
| `98.00` | 285 |
| `97.50` | 269 |
| `104` | 234 |
| `90.50` | 217 |
| `89.50` | 211 |
| `89.00` | 207 |
| `91.00` | 206 |
| `105.5` | 195 |
| `106` | 191 |
| `99.00` | 190 |
| `82.00` | 187 |
| `113` | 186 |
| `83.00` | 182 |
| `96.00` | 178 |
| `96.50` | 176 |
| `104.5` | 174 |
| `112` | 164 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Walking heart rate average identifier stable; reverted mobility additions but registration maintained

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Walking heart rate average measures cardiovascular response during normal walking activity
- Apple Watch Series 5+ metric; provides insight into aerobic fitness level
- Collected from daily walking patterns and light activity
- Used as baseline indicator for cardiovascular health assessment in MyHeart Counts study
