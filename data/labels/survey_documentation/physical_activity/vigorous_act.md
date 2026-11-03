# vigorous_act

**Benchmark column**: `vigorous_act`
**Raw identifier**: `vigorous_act`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~137
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> Overall, how many minutes of vigorous activity do you get in a week?

**Detail**: A person doing vigorous-intensity activity, such as running, usually cannot say more than a few words without pausing for a breath.

## Answer options
Continuous integer input (slider).

| Constraint | Value |
|-----------|-------|
| Data Type | integer |
| Min Value | 0 |
| Max Value | 2000 |
| Unit | minutes per week |
| UI Hint | slide |

## Observed values

**Total observations**: 9,633 — **type-enforced**: 9,633 (**unique**: 138) — raw Python types seen: `float` (9,633).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 4.00 |
| median | 30.00 |
| mean | 71.70 |
| q75 | 90.00 |
| max | 1440 |
| std | 124.2 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `0` | 2,134 |
| `30.00` | 1,073 |
| `60.00` | 929 |
| `10.00` | 601 |
| `20.00` | 516 |
| `120` | 470 |
| `15.00` | 400 |
| `5.00` | 369 |
| `90.00` | 333 |
| `180` | 303 |
| `100` | 225 |
| `150` | 199 |
| `300` | 170 |
| `45.00` | 168 |
| `200` | 155 |
| `240` | 118 |
| `40.00` | 117 |
| `50.00` | 117 |
| `2.00` | 100 |
| `1.00` | 97 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Paired with `moderate_act` (context variable) to measure overall weekly activity intensity distribution.
- Used in coaching logic to determine activity level and personalized interventions.
- No derived targets depend directly on vigorous_act; however, vigorous activity is often weighted more heavily in cardiovascular health assessments.
- Related to `phys_activity` (leisure time activity ordinal response) which provides categorical context.
