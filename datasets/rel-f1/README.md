---
tags:
- relbench
- relational-deep-learning
pretty_name: rel-f1
---

# rel-f1

Formula 1 motorsport database: races, drivers, constructors, circuits, race results, qualifying, and championship standings.

## Schema

![schema diagram](schema.svg)

Open [`schema.svg`](schema.svg) for a zoomable view of the foreign-key graph (PK = primary key, FK = foreign key).

Splits: validation `2005-01-01`, test `2010-01-01` (rows up to a split's timestamp are the inputs for that split).

## Tasks

| task | kind | type | description |
|---|---|---|---|
| `driver-circuit-compete` | forecast | link_prediction | Predict on which circuits a driver will compete in the next 1 year. |
| `driver-dnf` | forecast | binary_classification | Predict the if each driver will DNF (not finish) a race in the next 1 month. |
| `driver-position` | forecast | regression | Predict the average finishing position of each driver all races in the next 2 months. |
| `driver-top3` | forecast | binary_classification | Predict if each driver will qualify in the top-3 for a race within the next 1 month. |
| `qualifying-position` | autocomplete | regression | Predict the `position` column of the `qualifying` table. |
| `results-position` | autocomplete | regression | Predict the `position` column of the `results` table. |

## Loading

```python
import relbench
ds = relbench.load_dataset("rel-f1")
task = relbench.load_task("rel-f1", "<task>")
```

Manifest layout (`manifest.yaml` + plain parquet); see the RelBench [CONTRIBUTING guide](https://github.com/snap-stanford/relbench/blob/main/CONTRIBUTING.md).
