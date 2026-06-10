---
tags:
- relbench
- relational-deep-learning
pretty_name: rel-trial
---

# rel-trial

ClinicalTrials.gov clinical trials: studies, outcomes, adverse events, eligibilities, sponsors, conditions, and facilities.

## Schema

```mermaid
erDiagram
    conditions_studies {
        key id PK
        key nct_id FK
        key condition_id FK
        datetime date
    }
    interventions {
        key intervention_id PK
    }
    drop_withdrawals {
        key id PK
        key nct_id FK
        datetime date
    }
    outcome_analyses {
        key id PK
        key nct_id FK
        key outcome_id FK
        datetime date
    }
    sponsors_studies {
        key id PK
        key nct_id FK
        key sponsor_id FK
        datetime date
    }
    facilities_studies {
        key id PK
        key nct_id FK
        key facility_id FK
        datetime date
    }
    eligibilities {
        key id PK
        key nct_id FK
        datetime date
    }
    interventions_studies {
        key id PK
        key nct_id FK
        key intervention_id FK
        datetime date
    }
    outcomes {
        key id PK
        key nct_id FK
        datetime date
    }
    facilities {
        key facility_id PK
    }
    reported_event_totals {
        key id PK
        key nct_id FK
        datetime date
    }
    sponsors {
        key sponsor_id PK
    }
    studies {
        key nct_id PK
        datetime start_date
    }
    conditions {
        key condition_id PK
    }
    designs {
        key id PK
        key nct_id FK
        datetime date
    }
    conditions_studies }o--|| studies : nct_id
    conditions_studies }o--|| conditions : condition_id
    drop_withdrawals }o--|| studies : nct_id
    outcome_analyses }o--|| studies : nct_id
    outcome_analyses }o--|| outcomes : outcome_id
    sponsors_studies }o--|| studies : nct_id
    sponsors_studies }o--|| sponsors : sponsor_id
    facilities_studies }o--|| studies : nct_id
    facilities_studies }o--|| facilities : facility_id
    eligibilities }o--|| studies : nct_id
    interventions_studies }o--|| studies : nct_id
    interventions_studies }o--|| interventions : intervention_id
    outcomes }o--|| studies : nct_id
    reported_event_totals }o--|| studies : nct_id
    designs }o--|| studies : nct_id
```

Splits: validation `2020-01-01`, test `2021-01-01` (rows up to a split's timestamp are the inputs for that split).

## Tasks

| task | kind | type | description |
|---|---|---|---|
| `condition-sponsor-run` | forecast | link_prediction | Predict whether this condition will have which sponsors. |
| `eligibilities-adult` | autocomplete | binary_classification | Predict the `adult` column of the `eligibilities` table. |
| `eligibilities-child` | autocomplete | binary_classification | Predict the `child` column of the `eligibilities` table. |
| `site-sponsor-run` | forecast | link_prediction | Predict whether this sponsor will have a trial in a facility. |
| `site-success` | forecast | regression | Predict the success rate of a trial site in the next 1 year. |
| `studies-enrollment` | autocomplete | regression | Predict the `enrollment` column of the `studies` table. |
| `studies-has_dmc` | autocomplete | binary_classification | Predict the `has_dmc` column of the `studies` table. |
| `study-adverse` | forecast | regression | Predict the number of affected patients with severe advsere events/death for the trial in the next 1 year. |
| `study-outcome` | forecast | binary_classification | Predict if the trials in the next 1 year will achieve its primary outcome. |

## Loading

```python
import relbench
ds = relbench.load_dataset("rel-trial")
task = relbench.load_task("rel-trial", "<task>")
```

Manifest layout (`manifest.yaml` + plain parquet); see the RelBench [CONTRIBUTING guide](https://github.com/snap-stanford/relbench/blob/main/CONTRIBUTING.md).
