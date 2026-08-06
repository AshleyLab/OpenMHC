# sleep_time

**Benchmark column**: `field_sleep_time`
**Raw identifier**: `sleep_time`
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~167
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> How much sleep do you think you need every night to be rested? 
> (in hours)

## Answer options
Continuous integer input (slider).

| Constraint | Value |
|-----------|-------|
| Data Type | integer |
| Min Value | 0 |
| Max Value | 24 |
| Unit | hours |
| UI Hint | slide |

## Observed values

**Total observations**: 9,883 — **type-enforced**: 9,883 (**unique**: 16) — raw Python types seen: `float` (9,883).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 7.00 |
| median | 8.00 |
| mean | 7.69 |
| q75 | 8.00 |
| max | 24.00 |
| std | 1.22 |

**Top 16 most frequent values**:

| value | count |
|------:|------:|
| `8.00` | 4,836 |
| `7.00` | 2,716 |
| `9.00` | 953 |
| `6.00` | 821 |
| `10.00` | 272 |
| `5.00` | 157 |
| `12.00` | 41 |
| `4.00` | 35 |
| `24.00` | 16 |
| `11.00` | 15 |
| `3.00` | 9 |
| `13.00` | 5 |
| `0` | 2 |
| `2.00` | 2 |
| `16.00` | 2 |
| `1.00` | 1 |

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Measures perceived sleep need (subjective): "how much sleep do you think you need" rather than actual sleep obtained.
- Complements `sleep_time1` (actual weekday sleep duration): comparison between perceived need and actual sleep quantity can indicate sleep satisfaction/deficit.
- One of three sleep-related continuous variables: `sleep_time` (perceived need), `sleep_time1` (weekday actual), and potentially `sleep_time2` if it exists (not found in this survey).
- Note: `WakeUpTime` and `GoSleepTime` are NOT in this survey; they may be derived from HealthKit sleep analysis or tracked in another survey (see HealthKit integration notes).
- Used to assess sleep hygiene and awareness: large gaps between perceived need and actual sleep may indicate sleep dissatisfaction or insomnia.
- Informs coaching: respondents with high perceived need relative to actual sleep may benefit from sleep hygiene or activity-adjustment recommendations.
