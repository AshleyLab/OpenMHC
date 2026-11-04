# race

**Benchmark column**: `field_race`
**Raw identifier**: `race` (as in survey JSON)
**Role**: context
**Type**: multi_categorical

## Released Full/XS encoding

`field_race` is released with four categories:

| Released code | Category | Original codes mapped here |
|---|---|---|
| 1 | White | 1 (White) |
| 2 | Black | 2 (Black, African-American, or Negro) |
| 3 | Asian | 5–10 (Asian Indian, Chinese, Filipino, Japanese, Korean, Vietnamese) |
| 4 | Other | 3, 4, 11, 12 (American Indian, Alaska Native, Pacific Islander, Some other race) |

Multiple selections are preserved, with codes sorted and duplicates removed after mapping. Missing responses remain missing.

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json`
- Line: ~255
- Survey identifier: `risk_factors_SchemaV2`

## Question
> What is your race? Choose one or more races to indicate what you consider yourself to be.

## Original survey answer options
| Value | Label |
|-------|-------|
| 1 | White |
| 2 | Black, African-American, or Negro |
| 3 | American Indian |
| 4 | Alaska Native |
| 5 | Asian Indian |
| 6 | Chinese |
| 7 | Filipino |
| 8 | Japanese |
| 9 | Korean |
| 10 | Vietnamese |
| 11 | Pacific islander |
| 12 | Some other race |

**Data type**: integer
**UI Hint**: MultiValueConstraints
**Allow multiple**: true (multi-select question)

## Observed values (released data)

Race responses are available for **6,708 participants in Full** and **340 in XS**.

| Race category | Full participants | XS participants |
|---|---:|---:|
| White | 5,765 | 287 |
| Black | 250 | 12 |
| Asian | 577 | 29 |
| Other | 378 | 22 |

Participants may select multiple categories, so the category counts overlap.

_Source: Full/XS `data/labels/context_labels.json` and `ordinal_dictionary.json`, checked 29 October 2026._

## Git history (file-level)
- Commits touching cardio_CVhealth_survey.json: 8
- Most recent material change: c1833d4 (2024) [MHC-780] Add PAH response to vascular survey
- Notes: No targeted changes to race identifier

## Notes
Context variable providing self-identified racial/ethnic categories following standard demographic survey conventions. Multi-select question allowing respondents to indicate multiple races.
