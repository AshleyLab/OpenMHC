# Ldl

**Benchmark column**: `Ldl`
**Raw identifier**: `heartAgeDataLdl`
**Obj-C constant**: `kHeartAgeTestDataLDL`
**Role**: target
**Type**: continuous

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 65
- Used in Framingham calculation: Not directly used in coefficient calculation; no `kHeartAgeTestDataLDL` reference in Framingham method
- UI question: `APHHeartAgeTaskViewController.m` lines 265-275
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> LDL Cholesterol (optional)

## Answer options
| Value | Label |
|-------|-------|
| 0–1000 | Numeric, unit localized (mg/dL or mmol/L) |

**Input format**: Numeric (integer or decimal depending on locale), minimum 0, maximum 1000. Unit is localized. Marked as OPTIONAL.

## Observed values

**Total observations**: 2,755 — **type-enforced**: 2,755 (**unique**: 247) — raw Python types seen: `float` (2,755).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.23 |
| q25 | 2.04 |
| median | 2.61 |
| mean | 2.74 |
| q75 | 3.26 |
| max | 14.60 |
| std | 1.09 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `2.59` | 59 |
| `2.66` | 40 |
| `2.07` | 39 |
| `2.33` | 39 |
| `2.46` | 38 |
| `2.84` | 38 |
| `2.74` | 38 |
| `2.51` | 38 |
| `1.81` | 37 |
| `3.10` | 36 |
| `2.43` | 35 |
| `3.36` | 34 |
| `2.97` | 32 |
| `2.20` | 32 |
| `2.09` | 31 |
| `2.25` | 30 |
| `3.05` | 30 |
| `3.13` | 30 |
| `2.56` | 30 |
| `2.53` | 30 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update)
- Recent material change: `dbdd5a0` (2HC-508)

## Notes
- LDL ("bad cholesterol") is captured for informational and research purposes, but is NOT used in the Framingham 10-year risk or heart age calculation.
- Marked as optional with helper text: "The items below are optional. If you do not know the values you need to enter 0." (line 260).
- Does not appear in the Framingham coefficients lookup or calculation.
- The app notes in the "Learn More" section that "the risk score and heart age can be affected in people taking cholesterol medications" and that results do not apply to people with LDL > 190 (lines 118–120).
