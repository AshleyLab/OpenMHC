# SystolicBloodPressure

**Benchmark column**: `SystolicBloodPressure`
**Raw identifier**: `heartAgeDataSystolicBloodPressure`
**Obj-C constant**: `kHeartAgeTestDataSystolicBloodPressure`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 66
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 196, 197, 227, 230, 233, 236, 288, 410
- UI question: `APHHeartAgeTaskViewController.m` lines 302-312
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Systolic Blood Pressure

## Answer options
| Value | Label |
|-------|-------|
| 90–200 | Integer (mmHg) |

**Input format**: Numeric (integer), minimum 90, maximum 200 mmHg. Unit: mmHg.

## Observed values

**Total observations**: 3,691 — **type-enforced**: 3,691 (**unique**: 88) — raw Python types seen: `float` (3,691).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 70.00 |
| q25 | 112 |
| median | 120 |
| mean | 120.8 |
| q75 | 128 |
| max | 199 |
| std | 13.21 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `120` | 526 |
| `110` | 296 |
| `130` | 194 |
| `118` | 166 |
| `128` | 135 |
| `125` | 128 |
| `124` | 106 |
| `122` | 105 |
| `140` | 103 |
| `100` | 100 |
| `116` | 98 |
| `115` | 97 |
| `117` | 93 |
| `112` | 82 |
| `135` | 81 |
| `114` | 69 |
| `121` | 65 |
| `90.00` | 62 |
| `127` | 60 |
| `132` | 55 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update), `34c0781` (MHC-178 identifier fix)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Systolic BP is a primary Framingham input. It is used differently depending on hypertension treatment status.
- Two separate calculations: treated systolic BP (when `kHeartAgeTestDataHypertension` = 1, line 196) and untreated systolic BP (when hypertension = 0, line 197).
- Used as log-transformed value in Framingham coefficients (lines 227, 230, 233, 236).
- Also used in lifetime risk categorization (line 410): thresholds at 120, 140, 160 mmHg.
- Optimal systolic BP for 10-year risk is 110 mmHg (defined at line 103 in `.m` file).
- Appears on the "Blood pressure" form step, marked as required.
- Form step description: "Blood pressure (typically shown as systolic over diastolic)" (line 299).
