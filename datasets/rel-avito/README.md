---
tags:
- relbench
- relational-deep-learning
pretty_name: rel-avito
---

# rel-avito

Avito online classifieds: users, ads, search queries, and impression / click / visit streams.

## Schema

```mermaid
erDiagram
    VisitStream {
        key UserID FK
        key AdID FK
        datetime ViewDate
    }
    AdsInfo {
        key AdID PK
        key LocationID FK
        key CategoryID FK
    }
    SearchStream {
        key SearchID FK
        key AdID FK
        datetime SearchDate
    }
    SearchInfo {
        key SearchID PK
        key UserID FK
        key LocationID FK
        key CategoryID FK
        datetime SearchDate
    }
    Category {
        key CategoryID PK
    }
    PhoneRequestsStream {
        key UserID FK
        key AdID FK
        datetime PhoneRequestDate
    }
    UserInfo {
        key UserID PK
    }
    Location {
        key LocationID PK
    }
    VisitStream }o--|| UserInfo : UserID
    VisitStream }o--|| AdsInfo : AdID
    AdsInfo }o--|| Location : LocationID
    AdsInfo }o--|| Category : CategoryID
    SearchStream }o--|| SearchInfo : SearchID
    SearchStream }o--|| AdsInfo : AdID
    SearchInfo }o--|| UserInfo : UserID
    SearchInfo }o--|| Location : LocationID
    SearchInfo }o--|| Category : CategoryID
    PhoneRequestsStream }o--|| UserInfo : UserID
    PhoneRequestsStream }o--|| AdsInfo : AdID
```

Splits: validation `2015-05-08`, test `2015-05-14` (rows up to a split's timestamp are the inputs for that split).

## Tasks

| task | kind | type | description |
|---|---|---|---|
| `ad-ctr` | forecast | regression | Assuming the ad will be clicked in the next 4 days, predict the Click-Through- Rate (CTR) for each ad. |
| `searchinfo-isuserloggedon` | autocomplete | binary_classification | Predict the `IsUserLoggedOn` column of the `SearchInfo` table. |
| `searchstream-click` | autocomplete | binary_classification | Predict the `IsClick` column of the `SearchStream` table. |
| `user-ad-visit` | forecast | link_prediction | Predict the distinct list of ads a user will visit in the next 4 days. |
| `user-clicks` | forecast | binary_classification | Predict whether the each customer will click on more than one ads in the next 4 days. |
| `user-visits` | forecast | binary_classification | Predict whether each customer will visit more than one ad in the next 4 days. |

## Loading

```python
import relbench
ds = relbench.load_dataset("rel-avito")
task = relbench.load_task("rel-avito", "<task>")
```

Manifest layout (`manifest.yaml` + plain parquet); see the RelBench [CONTRIBUTING guide](https://github.com/snap-stanford/relbench/blob/main/CONTRIBUTING.md).
