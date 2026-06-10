---
tags:
- relbench
- relational-deep-learning
pretty_name: rel-ratebeer
---

# rel-ratebeer

RateBeer reviews: users, beers, brewers, places, and time-stamped ratings.

## Schema

```mermaid
erDiagram
    brewers {
        key brewer_id PK
        key country_id FK
        key state_id FK
        key type_id FK
    }
    beer_styles {
        key style_id PK
    }
    beers {
        key beer_id PK
        key brewer_id FK
        key style_id FK
        datetime created_at
    }
    countries {
        key country_id PK
    }
    place_types {
        key type_id PK
    }
    users {
        key user_id PK
        datetime created_at
    }
    places {
        key place_id PK
        key state_id FK
        key type_id FK
        key country_id FK
    }
    beer_ratings {
        key rating_id PK
        key user_id FK
        key beer_id FK
        key availability_id FK
        datetime created_at
    }
    place_ratings {
        key rating_id PK
        key place_id FK
        key user_id FK
        datetime created_at
    }
    favorites {
        key favorite_id PK
        key user_id FK
        key beer_id FK
        datetime created_at
    }
    availability {
        key avail_id PK
        key beer_id FK
        key place_id FK
        key country_id FK
        key user_id FK
    }
    beer_upcs {
        key beer_id FK
    }
    states {
        key state_id PK
        key country_id FK
    }
    brewers }o--|| countries : country_id
    brewers }o--|| states : state_id
    brewers }o--|| place_types : type_id
    beers }o--|| brewers : brewer_id
    beers }o--|| beer_styles : style_id
    places }o--|| states : state_id
    places }o--|| place_types : type_id
    places }o--|| countries : country_id
    beer_ratings }o--|| users : user_id
    beer_ratings }o--|| beers : beer_id
    beer_ratings }o--|| availability : availability_id
    place_ratings }o--|| places : place_id
    place_ratings }o--|| users : user_id
    favorites }o--|| users : user_id
    favorites }o--|| beers : beer_id
    availability }o--|| beers : beer_id
    availability }o--|| places : place_id
    availability }o--|| countries : country_id
    availability }o--|| users : user_id
    beer_upcs }o--|| beers : beer_id
    states }o--|| countries : country_id
```

Splits: validation `2018-09-01`, test `2020-01-01` (rows up to a split's timestamp are the inputs for that split).

## Tasks

| task | kind | type | description |
|---|---|---|---|
| `beer-churn` | external | binary_classification | Predict whether a beer will receive a rating in the next 90 days. |
| `beer_ratings-total_score` | external | regression | regression task on `beer_ratings`. |
| `brewer-dormant` | external | binary_classification | Predict whether a brewer will release zero beers in the next 365 days. |
| `user-beer-favorite` | external | link_prediction | Predict the list of distinct beers each active user will add to their favorites in the next 90 days. |
| `user-beer-liked` | external | link_prediction | Predict the list of distinct beers each active user rates at least 4.0 / 5.0 in the next 90 days. |
| `user-churn` | external | binary_classification | Predict whether a user will give a beer rating in the next 90 days. |
| `user-count` | external | regression | Predict the number of beer ratings that a user will give in the next 90 days. |
| `user-place-liked` | external | link_prediction | Predict the list of distinct places each active user rates at least 80.0 / 100.0 in the next 90 days. |

## Loading

```python
import relbench
ds = relbench.load_dataset("rel-ratebeer")
task = relbench.load_task("rel-ratebeer", "<task>")
```

Manifest layout (`manifest.yaml` + plain parquet); see the RelBench [CONTRIBUTING guide](https://github.com/snap-stanford/relbench/blob/main/CONTRIBUTING.md).
