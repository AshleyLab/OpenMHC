# age

**Benchmark column**: `age`
**Raw identifier**: `heartAgeDataAge`
**Obj-C constant**: `kHeartAgeTestDataAge`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 62
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 175, 193, 209
- UI question: `APHHeartAgeTaskViewController.m` lines 153-162
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> What is your age?

## Answer options
| Value | Label |
|-------|-------|
| 18–150 | Integer (years) |

**Input format**: Numeric (integer), minimum 18, maximum 150 years

## Observed values

**Total observations**: 11,893 — **type-enforced**: 11,893 (**unique**: 72) — raw Python types seen: `int` (11,893).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 18.00 |
| q25 | 30.00 |
| median | 39.00 |
| mean | 41.90 |
| q75 | 52.00 |
| max | 90.00 |
| std | 15.38 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `30.00` | 445 |
| `33.00` | 352 |
| `31.00` | 348 |
| `32.00` | 345 |
| `36.00` | 332 |
| `35.00` | 321 |
| `29.00` | 303 |
| `40.00` | 300 |
| `27.00` | 289 |
| `25.00` | 286 |
| `34.00` | 284 |
| `39.00` | 283 |
| `26.00` | 283 |
| `28.00` | 283 |
| `43.00` | 279 |
| `37.00` | 272 |
| `44.00` | 270 |
| `41.00` | 260 |
| `38.00` | 257 |
| `45.00` | 251 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- .h/.m commits: `06a6f76` (MHX-640 Added NSLocalizedString), `0869e98` (Squashed commit)
- View controller commits: `dbdd5a0` (MHC-508), `06a6f76`, `a2c3b1e` (MHC-86)
- Recent material change: `dbdd5a0` (2024, MHC-508 UK Heart Risk Task unit localization)

## Notes
- This is the self-reported age entered by the user in the initial demographics form. It is used as a primary input to the Framingham risk calculation.
- Can be pre-populated from HealthKit birth date when "Are you submitting your own heart risk data?" is answered YES (line 684 in view controller).
- The actual age appears twice: once in the demographic step and modified/corrected in the summary results before final submission (lines 500-507).
