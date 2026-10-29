# Geography

Released geographic fields are `field_country`, `field_state`, and `field_ruca_primary_mean`. See [ZIP transformations](zip.md) for how the latter two are derived.

## Released variables

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [field_country](country.md) | context | categorical | cardio_wellbeing_survey.json | Country of residence (US / UK / HK) |
| [field_state](zip.md) | context | categorical | Derived from US ZIP prefix | Inferred state or territory abbreviation |
| [field_ruca_primary_mean](zip.md) | context | continuous | Derived from US ZIP prefix | Rounded mean primary RUCA code |

## Notes

- `country` and `zip` previously lived in `demographics/`; they were split out into `geography/` so the demographics group cleanly isolates *who the participant is* from *where they are*.
