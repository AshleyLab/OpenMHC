# ethnicity

**Benchmark column**: `field_ethnicity`
**Raw identifier**: `ethnicity` (as in survey JSON)
**Role**: context
**Type**: ordinal

## Released Full/XS encoding

`field_ethnicity` is a single-choice Hispanic/Latino-origin question. The release uses three categories:

| Released code | Exact dictionary label | Original codes mapped here |
|---|---|---|
| 1 | No, not Spanish/Hispanic/Latino | 1 |
| 2 | Yes, Mexican, Mexican American, or Chicano | 3 |
| 3 | Yes, other Spanish/Hispanic/Latina | 2 (Puerto Rican), 4 (Cuban), 5 (other Spanish/Hispanic/Latina) |

Puerto Rican and Cuban responses are grouped with the existing other Hispanic/Latino category. Missing responses remain missing.

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~214
- Survey identifier: `risk_factors_SchemaV2`

## Question
> Are you Spanish/Hispanic/Latino? Choose the best answer.

## Original survey answer options
| Value | Label |
|-------|-------|
| 1 | No, not Spanish/Hispanic/Latino |
| 2 | Yes, Puerto Rican |
| 3 | Yes, Mexican, Mexican American, or Chicano |
| 4 | Yes, Cuban |
| 5 | Yes, other Spanish/Hispanic/Latina |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: false

## Observed values (released data)

Responses are available for **6,682 participants in Full** and **337 in XS**.

| Hispanic/Latino-origin category | Full participants | XS participants |
|---|---:|---:|
| No, not Spanish/Hispanic/Latino | 6,192 | 305 |
| Yes, Mexican, Mexican American, or Chicano | 170 | 13 |
| Yes, other Spanish/Hispanic/Latina | 320 | 19 |

_Source: Full/XS `data/labels/context_labels.json` and `ordinal_dictionary.json`, checked 29 October 2026._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: c1833d4 (2024) [MHC-780] Add PAH response to vascular survey
- Notes: No targeted changes to ethnicity identifier

## Notes
Context variable providing demographic ethnicity/Hispanic origin information for cardiovascular health study participants.
