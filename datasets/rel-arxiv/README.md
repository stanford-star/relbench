---
tags:
- relbench
- relational-deep-learning
pretty_name: rel-arxiv
---

# rel-arxiv

arXiv scholarly papers: papers, authors, citations, and subject categories.

## Schema

```mermaid
erDiagram
    paperCategories {
        key Paper_ID FK
        key Category_ID FK
        datetime Submission_Date
    }
    authors {
        key Author_ID PK
    }
    categories {
        key Category_ID PK
    }
    citations {
        key Paper_ID FK
        key References_Paper_ID FK
        datetime Submission_Date
    }
    papers {
        key Paper_ID PK
        datetime Submission_Date
    }
    paperAuthors {
        key Paper_ID FK
        key Author_ID FK
        datetime Submission_Date
    }
    paperCategories }o--|| papers : Paper_ID
    paperCategories }o--|| categories : Category_ID
    citations }o--|| papers : Paper_ID
    citations }o--|| papers : References_Paper_ID
    paperAuthors }o--|| papers : Paper_ID
    paperAuthors }o--|| authors : Author_ID
```

Splits: validation `2022-01-01`, test `2023-01-01` (rows up to a split's timestamp are the inputs for that split).

## Tasks

| task | kind | type | description |
|---|---|---|---|
| `author-category` | external | multiclass_classification | Predict the primary research category in which an author will publish papers in the next six months. |
| `author-publication` | forecast | regression | Predict how many papers an author will publish in the next six months. |
| `paper-citation` | forecast | binary_classification | Predict if a paper gets cited in the next 6 months. |
| `paper-paper-cocitation` | forecast | link_prediction | Predict which other papers will be cited together with a given paper in the next six months. |

## Loading

```python
import relbench
ds = relbench.load_dataset("rel-arxiv")
task = relbench.load_task("rel-arxiv", "<task>")
```

Manifest layout (`manifest.yaml` + plain parquet); see the RelBench [CONTRIBUTING guide](https://github.com/snap-stanford/relbench/blob/main/CONTRIBUTING.md).
