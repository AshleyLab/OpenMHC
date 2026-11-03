# happiness

**Benchmark column**: `happiness`
**Raw identifier**: `happiness`
**Role**: target
**Type**: continuous (0–10)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check_coaching.json`
- Line: ~9
- Survey: `daily_check` (coaching variant)

## Question
> Happiness

**Detail**: This question asks about how you felt yesterday on a scale from 0 to 10. Zero means you did not experience the feeling "at all" yesterday while 10 means you experienced the feeling "all of the time" yesterday.

## Answer options
| Value | Label |
|-------|-------|
| 0–10 | Slider (1-point increments) |

## Observed values

**Total observations**: 27,428 — **type-enforced**: 27,428 (**unique**: 11) — raw Python types seen: `float` (27,428).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 7.00 |
| median | 8.00 |
| mean | 7.53 |
| q75 | 9.00 |
| max | 10.00 |
| std | 1.96 |

**Top 11 most frequent values**:

| value | count |
|------:|------:|
| `8.00` | 7,116 |
| `9.00` | 5,855 |
| `7.00` | 4,591 |
| `10.00` | 3,511 |
| `6.00` | 2,421 |
| `5.00` | 1,871 |
| `4.00` | 840 |
| `3.00` | 502 |
| `2.00` | 333 |
| `0` | 218 |
| `1.00` | 170 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-07-25 from `data/labels/last_labels.json` (md5 `521c157a…`) and `data/labels/context_labels.json` (md5 `220f5946…`)._

## Git history (file-level)
- Commits: 4 (daily_check_coaching.json)
- Recent material change: `50743bd` (MHC-126 - Create Alternative Daily survey 2 - change the happiness prompt on coaching daily survey)
- Notes: Added in coaching variant; the happiness prompt was updated in MHC-126

## Notes
This is the target variable in the coaching variant of the daily check-in survey. It measures subjective happiness/well-being on a 0–10 scale for the prior day, collected daily via a slider. This is distinct from other well-being variables in different surveys (e.g., `feel_worthwhile1`, `feel_worthwhile2` from the well-being survey). The coaching survey captures happiness as a key outcome for evaluating intervention impact.
