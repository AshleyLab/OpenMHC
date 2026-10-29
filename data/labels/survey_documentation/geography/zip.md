# zip

`field_zip` is removed from the released Full/XS datasets and replaced by `field_state` and `field_ruca_primary_mean`.

**Original field**: `field_zip`
**Raw identifier**: `zip`
**Role**: context
**Original type**: categorical

## Released Full/XS fields

Three-digit US ZIP prefixes are matched to five-digit ZIP areas in the 2020 USDA ZIP–RUCA reference.

| Released field | Type | Transformation |
|---|---|---|
| `field_state` | categorical | Two-letter abbreviation of the state or territory with the most eligible ZIP areas sharing the prefix. Each ZIP area counts once. |
| `field_ruca_primary_mean` | continuous | Unweighted arithmetic mean of primary RUCA codes across eligible ZIP areas sharing the prefix, excluding code `99`, rounded to the nearest integer with `.5` rounded upward. |

Only reference rows classified as **ZIP Code Area** are eligible; post-office/large-volume-customer entries are excluded. State ties use a reproducible selection per prefix (seed 42). The RUCA mean uses all eligible areas in the prefix, including those outside the selected state.

Missing values, one- or two-digit prefixes, unmatched prefixes, and UK/HK tokens produce missing derived fields. Short prefixes are not padded with zeros. The separate `field_country` is unchanged.

## Original survey source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json`
- Line: ~177
- Survey: `satisfied_SchemaV3` (Wellbeing and Risk Perception)

## Question
> Please enter your postcode.

## Answer options
Free-text string entry (no enumerated options)

**Description**: This information will help us understand how heart health varies by region.

## Observed values (released data)

| Released field | Full participants with a value | XS participants with a value |
|---|---:|---:|
| `field_state` | 5,791 | 290 |
| `field_ruca_primary_mean` | 5,791 | 290 |

_Source: Full/XS `data/labels/context_labels.json`, checked 29 October 2026._

## Original survey constraints
- **Data type**: string
- **Constraint type**: StringConstraints
- **UI hint**: numberfield
- **Min value**: 0 (nominal)
- **Max value**: 99999 (nominal)

## Original survey conditional logic
- **Displayed when**: Country selection = "UK"
- **Follows**: country selection in survey flow
- **Skips to**: riskfactors1 after entry

## Related variable
- **numericZip**: US equivalent for numeric zipcode entry (displayed when country = "US")
  - numericZip uses IntegerConstraints with range 0-99999
  - zip uses StringConstraints to allow alphanumeric UK postcodes

## Git history (file-level)
- Recent change: `5ff65f1` (2020-04-03) [MHC-756] Update for postal code
- Commits affecting file: 5
- Notes: UK postcode field. Updated in commit 5ff65f1 to improve postal code handling. Initial implementation 2015.

## Original survey notes
- Context variable for geographic granularity within UK
- Collected only from UK residents (conditional on country = "UK")
- String type to accommodate UK postcode format (e.g., "SW1A 1AA")
- Used for regional health analysis, deprivation mapping, and geographic variation studies
- Distinct from numericZip which serves US residents
