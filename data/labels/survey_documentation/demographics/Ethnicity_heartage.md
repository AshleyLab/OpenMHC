# Ethnicity_heartage

**Benchmark column**: `field_Ethnicity_heartage`
**Raw identifier**: `heartAgeDataEthnicity`
**Obj-C constant**: `kHeartAgeTestDataEthnicity`
**Role**: context
**Type**: categorical

## Released Full/XS encoding

`field_Ethnicity_heartage` is a single-choice question. Alaska Native, American Indian and Pacific Islander responses are mapped to Other. The release retains six categories:

| Released dictionary code | Stored category name |
|---|---|
| 0 | Asian |
| 1 | Black |
| 2 | Hispanic |
| 3 | White |
| 4 | Other |
| 5 | I prefer not to indicate an ethnicity |

Responses are stored as category-name strings; the codes above are dictionary codes. Missing responses remain missing.

## Source
- Obj-C constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m` line 48
- Used in Framingham calculation: `APHHeartAgeAndRiskFactors.m` lines 178, 180–184, 187, 263, 277, 316, 363
- UI question: `APHHeartAgeTaskViewController.m` lines 195-210
- Survey: Heart Age / Framingham Risk form (identifier: `heart_risk_and_age`)

## Question
> Ethnicity

## Original survey answer options
| Value | Label |
|-------|-------|
| I prefer not to indicate an ethnicity | I prefer not to indicate an ethnicity |
| Alaska Native | Alaska Native |
| American Indian | American Indian |
| Asian | Asian |
| Black | Black |
| Hispanic | Hispanic |
| Pacific Islander | Pacific Islander |
| White | White |
| Other | Other |

**Notes on values**: Nine localized string options are presented to the user. However, only two categories are used internally for the Framingham calculation: "African-American" (when user selects "Black") and "Other" (all other selections, including prefer-not-to-answer).

## Observed values (released data)

Responses are available for **3,769 participants in Full** and **182 in XS**.

| Ethnicity category | Full participants | XS participants |
|---|---:|---:|
| Asian | 233 | 14 |
| Black | 109 | 3 |
| Hispanic | 195 | 12 |
| White | 3,104 | 148 |
| Other | 109 | 5 |
| I prefer not to indicate an ethnicity | 19 | 0 |

_Source: Full/XS `data/labels/context_labels.json` and `ordinal_dictionary.json`, checked 29 October 2026._

## Git history
- .h/.m commits: `06a6f76`, `0869e98`
- View controller commits: `dbdd5a0` (MHC-508), `eaf8632` (MHC-709 UI update), `06a6f76`
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- Ethnicity is **contextual** to the Framingham calculation; it determines which set of coefficients are used.
- The user-visible options (9 categories) are mapped internally to only 2 Framingham categories (lines 180–184):
  - "Black" → `kLookupEthnicityAfricanAmerican` ("African-American")
  - All others → `kLookupEthnicityOther` ("Other")
- The Framingham model stratifies by African-American vs. Other race/ethnicity, reflecting different baseline risks and coefficient values documented in epidemiological literature.
- Appears as a single-choice question on its own form step, marked as required.
- The "Black" category is the only one explicitly recognized by the Framingham lookup; all other selections collapse into the "Other" category.
