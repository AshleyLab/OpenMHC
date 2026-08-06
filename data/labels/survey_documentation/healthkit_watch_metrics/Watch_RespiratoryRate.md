# Watch_RespiratoryRate

**Benchmark column**: `Watch_RespiratoryRate`
**Raw identifier**: `HKQuantityTypeIdentifierRespiratoryRate`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1362 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 537 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierRespiratoryRate`
- Expected unit: breaths per minute
- Formula: `[[HKUnit countUnit] unitDividedByUnit:[HKUnit secondUnit]]`
- Source device: Apple Watch (via motion and heart rate sensors)

## Observed values

**Total observations**: 54,959 — **type-enforced**: 54,959 (**unique**: 2,797) — raw Python types seen: `float` (54,959).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 5.00 |
| q25 | 14.00 |
| median | 15.86 |
| mean | 16.24 |
| q75 | 18.00 |
| max | 39.00 |
| std | 3.16 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `15.00` | 3,320 |
| `17.00` | 2,687 |
| `14.00` | 2,606 |
| `14.50` | 2,530 |
| `13.00` | 2,497 |
| `15.50` | 2,329 |
| `13.50` | 2,188 |
| `16.00` | 2,095 |
| `16.50` | 2,017 |
| `17.50` | 1,706 |
| `12.50` | 1,279 |
| `18.00` | 1,224 |
| `19.00` | 1,092 |
| `18.50` | 1,033 |
| `11.50` | 963 |
| `12.00` | 947 |
| `20.00` | 787 |
| `20.50` | 737 |
| `19.50` | 716 |
| `21.00` | 610 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Respiratory rate identifier stable; reverted mobility additions maintained this registration

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Respiratory rate measured via Apple Watch motion and optical sensors
- Typically collected during rest periods; Apple Watch Series 6+ provides more accurate readings
- Important cardiovascular health metric; elevated respiratory rate may indicate stress or poor fitness
- Included in MyHeart Counts comprehensive vital signs monitoring for research
