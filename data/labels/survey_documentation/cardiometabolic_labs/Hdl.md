# Hdl

**Benchmark column**: `Hdl`
**Raw identifier**: `heartAgeDataHdl`
**Obj-C constant**: `kHeartAgeTestDataHDL`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 64
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 195, 221, 224, 286
- UI question: `APHHeartAgeTaskViewController.m` lines 246-256
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> HDL Cholesterol

## Answer options
| Value | Label |
|-------|-------|
| 10–140 | Numeric, unit localized (mg/dL or mmol/L) |

**Input format**: Numeric (integer or decimal depending on locale), minimum 10, maximum 140. Unit is localized via `HKUnit.localizedCholesterolUnit`.

## Observed values

**Total observations**: 3,342 — **type-enforced**: 3,342 (**unique**: 130) — raw Python types seen: `float` (3,342).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.26 |
| q25 | 1.06 |
| median | 1.32 |
| mean | 1.40 |
| q75 | 1.68 |
| max | 5.00 |
| std | 0.55 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1.55` | 123 |
| `1.29` | 117 |
| `2.07` | 113 |
| `1.16` | 106 |
| `1.03` | 101 |
| `1.14` | 86 |
| `1.42` | 84 |
| `1.24` | 80 |
| `1.01` | 77 |
| `0.52` | 76 |
| `1.27` | 76 |
| `1.09` | 74 |
| `1.37` | 72 |
| `1.22` | 71 |
| `0.98` | 69 |
| `1.11` | 68 |
| `1.19` | 67 |
| `1.45` | 67 |
| `1.68` | 67 |
| `1.32` | 63 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508 UK localization), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (2024, MHC-508)

## Notes
- HDL ("good cholesterol") is inversely associated with cardiovascular risk; higher HDL is protective.
- Appears in Framingham coefficients as log-transformed value (lines 195, 221, 224).
- Optimal HDL for 10-year risk calculation is 50 (defined in lookup at line 102 in `.m` file).
- Input appears on the "Cholesterol & Glucose" form step, marked as required (not optional).
