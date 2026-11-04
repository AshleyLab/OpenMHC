# Geography

Released geographic fields are `field_country`, `field_state`, and `field_ruca_primary_mean`. See [ZIP transformations](zip.md) for how the latter two are derived.

## Released variables

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [field_country](country.md) | context | categorical | cardio_wellbeing_survey.json | Country of residence (US / UK / HK) |
| [field_state](zip.md) | context | categorical | Derived from US ZIP prefix | Inferred state or territory abbreviation |
| [field_ruca_primary_mean](zip.md) | context | integer-valued | Derived from US ZIP prefix | Rounded mean primary RUCA code |

## Notes

- The original `country` and `zip` documentation pages were moved from `demographics/` to `geography/`. The retained `zip.md` page documents the source field and its transformation; `field_zip` itself is absent from the released Full/XS datasets.
