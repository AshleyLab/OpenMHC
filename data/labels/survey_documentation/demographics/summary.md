# Demographics

User-identifying traits: age, sex/gender, ethnicity/race, education, and innate physiological characteristics (skin type). Variables here are collected across multiple sources (Heart Age form, CVhealth survey, HealthKit characteristics) but are grouped together because they describe *who the participant is*. Geographic variables are documented in [geography](../geography/summary.md).

## Variables (7 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [age](age.md) | target | continuous | Heart Age form | Age entered by user for risk calculation |
| [BiologicalSex](BiologicalSex.md) | target | binary | Heart Age form | Male / Female (Framingham stratification) |
| [ethnicity](ethnicity.md) | context | ordinal | cardio_CVhealth_survey.json | Hispanic/Latino origin (3 released categories) |
| [Ethnicity_heartage](Ethnicity_heartage.md) | context | categorical | Heart Age form | Heart-age ethnicity (6 released categories) |
| [race](race.md) | context | multi_categorical | cardio_CVhealth_survey.json | Self-identified race (4 released categories, multi-select) |
| [education](education.md) | context | ordinal | cardio_CVhealth_survey.json | Highest education level (7 options) |
| [FitzpatrickSkinType](FitzpatrickSkinType.md) | context | categorical → ordinal (released bucketed) | HealthKit (`HKCharacteristicTypeIdentifierFitzpatrickSkinType`) | Fitzpatrick skin type (raw I-VI categorical; released as bucketed `{light, medium, dark}` ordinal). Benchmark column: `field_FitzpatrickSkinType`. Earlier revisions misspelled this as `FrickSkinType`. |

## Notes

- `ethnicity`, `Ethnicity_heartage`, and `race` are separate survey questions. Their linked pages document the original options and release mappings.
