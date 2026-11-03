# TotalCholesterol

**Benchmark column**: `TotalCholesterol`
**Raw identifier**: `heartAgeDataTotalCholesterol`
**Obj-C constant**: `kHeartAgeTestDataTotalCholesterol`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 63
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 194, 215, 218, 284, 409
- UI question: `APHHeartAgeTaskViewController.m` lines 234-244
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Total Cholesterol

## Answer options
| Value | Label |
|-------|-------|
| 80–400 | Numeric, unit localized (mg/dL or mmol/L) |

**Input format**: Numeric (integer or decimal depending on locale), minimum 80, maximum 400. Unit is localized via `HKUnit.localizedCholesterolUnit`.

## Observed values

**Total observations**: 3,558 — **type-enforced**: 3,558 (**unique**: 289) — raw Python types seen: `float` (3,558).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 1.40 |
| q25 | 3.75 |
| median | 4.55 |
| mean | 4.70 |
| q75 | 5.30 |
| max | 19.98 |
| std | 1.55 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `3.36` | 174 |
| `2.07` | 100 |
| `4.65` | 87 |
| `5.17` | 82 |
| `3.88` | 63 |
| `4.40` | 58 |
| `4.14` | 57 |
| `4.53` | 51 |
| `3.62` | 50 |
| `4.91` | 47 |
| `4.78` | 42 |
| `4.89` | 38 |
| `7.21` | 37 |
| `4.81` | 37 |
| `4.99` | 36 |
| `4.27` | 36 |
| `5.43` | 34 |
| `4.55` | 33 |
| `3.10` | 32 |
| `4.76` | 31 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508 UK localization), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (2024, MHC-508)

## Notes
- Total cholesterol is a primary input to the Framingham 10-year risk calculation.
- Used as log-transformed value in coefficient calculations (lines 194, 215, 218).
- Also used in lifetime risk factor categorization (line 409): thresholds at 180, 200, 240 mg/dL.
- Optimal total cholesterol for 10-year risk is 170 (defined at line 101 in `.m` file).
- Appears on the "Cholesterol & Glucose" form step, marked as required.
