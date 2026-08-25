import dataclasses
import random
import string

import numpy as np
import pandas as pd
import pytest

from relbench.base import Database, Dataset, Table
from relbench.load import build_task, load_dataset
from relbench.manifest import DatasetManifest, TableSpec, TaskManifest


def _random_string(rng, min_length, max_length):
    length = rng.randint(min_length, max_length)
    return "".join(rng.choice(string.ascii_letters) for _ in range(length))


class FakeDataset(Dataset):
    def __init__(
        self,
        num_products=30,
        num_customers=100,
        num_reviews=600,
        num_relations=20,
        seed=42,
    ):
        self.seed = seed
        self.num_products = num_products
        self.num_customers = num_customers
        self.num_reviews = num_reviews
        self.num_relations = num_relations

        min_timestamp = pd.Timestamp(0, unit="D")
        max_timestamp = pd.Timestamp(2 * (num_reviews - 1), unit="D")
        self.val_timestamp = min_timestamp + 0.8 * (max_timestamp - min_timestamp)
        self.test_timestamp = min_timestamp + 0.9 * (max_timestamp - min_timestamp)

    def make_db(self):
        rng = random.Random(self.seed)
        nprng = np.random.RandomState(self.seed)
        num_products = self.num_products
        num_customers = self.num_customers
        num_reviews = self.num_reviews
        num_relations = self.num_relations

        product_df = pd.DataFrame(
            {
                "product_id": [f"product_id_{i}" for i in range(num_products)],
                "category": [None, [], ["toy", "health"]] * (num_products // 3),
                "title": [_random_string(rng, 5, 15) for _ in range(num_products)],
                "price": nprng.rand(num_products) * 10,
            }
        )
        customer_df = pd.DataFrame(
            {
                "customer_id": [f"customer_id_{i}" for i in range(num_customers)],
                "age": nprng.randint(10, 50, size=(num_customers,)),
                "gender": ["male", "female"] * (num_customers // 2),
            }
        )
        review_df = pd.DataFrame(
            {
                "customer_id": [
                    f"customer_id_{rng.randint(0, num_customers + 5)}"
                    for _ in range(num_reviews)
                ],
                "product_id": [
                    f"product_id_{rng.randint(0, num_products - 1)}"
                    for _ in range(num_reviews)
                ],
                "review_time": pd.to_datetime(2 * np.arange(num_reviews), unit="D"),
                "rating": nprng.randint(1, 6, size=(num_reviews,)),
            }
        )
        review_df["review"] = review_df["rating"].map(
            lambda x: "positive" if x > 3 else "negative"
        )
        relations_df = pd.DataFrame(
            {
                "customer_id": [
                    f"customer_id_{rng.randint(0, num_customers + 5)}"
                    for _ in range(num_relations)
                ],
                "product_id": [
                    f"product_id_{rng.randint(0, num_products - 1)}"
                    for _ in range(num_relations)
                ],
            }
        )
        return Database(
            table_dict={
                "product": Table(
                    df=product_df, fkey_col_to_pkey_table={}, pkey_col="product_id"
                ),
                "customer": Table(
                    df=customer_df, fkey_col_to_pkey_table={}, pkey_col="customer_id"
                ),
                "review": Table(
                    df=review_df,
                    fkey_col_to_pkey_table={
                        "customer_id": "customer",
                        "product_id": "product",
                    },
                    time_col="review_time",
                ),
                "relations": Table(
                    df=relations_df,
                    fkey_col_to_pkey_table={
                        "customer_id": "customer",
                        "product_id": "product",
                    },
                ),
            }
        )


_CHURN_SQL = """
SELECT
    t.timestamp AS timestamp,
    past.customer_id AS customer_id,
    CAST(CASE WHEN COUNT(fut.review_time) = 0 THEN 1 ELSE 0 END AS BIGINT) AS churn
FROM timestamps t
JOIN review past
    ON past.review_time <= t.timestamp AND past.customer_id IS NOT NULL
LEFT JOIN review fut
    ON fut.customer_id = past.customer_id
    AND fut.review_time > t.timestamp
    AND fut.review_time <= t.timestamp + INTERVAL '{timedelta}'
GROUP BY t.timestamp, past.customer_id
""".strip()

_LTV_SQL = """
SELECT
    t.timestamp AS timestamp,
    past.customer_id AS customer_id,
    CAST(COUNT(fut.review_time) AS DOUBLE) AS ltv
FROM timestamps t
JOIN review past
    ON past.review_time <= t.timestamp AND past.customer_id IS NOT NULL
LEFT JOIN review fut
    ON fut.customer_id = past.customer_id
    AND fut.review_time > t.timestamp
    AND fut.review_time <= t.timestamp + INTERVAL '{timedelta}'
GROUP BY t.timestamp, past.customer_id
""".strip()

_PURCHASE_SQL = """
SELECT
    t.timestamp AS timestamp,
    r.customer_id AS customer_id,
    LIST(DISTINCT r.product_id) AS product_id
FROM timestamps t
JOIN review r
    ON r.review_time > t.timestamp
    AND r.review_time <= t.timestamp + INTERVAL '{timedelta}'
    AND r.customer_id IS NOT NULL AND r.product_id IS NOT NULL
GROUP BY t.timestamp, r.customer_id
""".strip()

CHURN = TaskManifest(
    name="user-churn",
    kind="forecast",
    task_type="binary_classification",
    entity_table="customer",
    entity_col="customer_id",
    target_col="churn",
    time_col="timestamp",
    timedelta="100 days",
    sql=_CHURN_SQL,
)
LTV = TaskManifest(
    name="user-ltv",
    kind="forecast",
    task_type="regression",
    entity_table="customer",
    entity_col="customer_id",
    target_col="ltv",
    time_col="timestamp",
    timedelta="100 days",
    sql=_LTV_SQL,
)
PURCHASE = TaskManifest(
    name="user-item-purchase",
    kind="forecast",
    task_type="recommendation",
    src_entity_table="customer",
    src_entity_col="customer_id",
    dst_entity_table="product",
    dst_entity_col="product_id",
    target_col="product_id",
    time_col="timestamp",
    timedelta="100 days",
    eval_k=10,
    sql=_PURCHASE_SQL,
)
RATING = TaskManifest(
    name="review-rating",
    kind="autocomplete",
    task_type="regression",
    entity_table="review",
    target_col="rating",
    remove_columns=[["review", "review"]],
)
TASKS = (CHURN, LTV, PURCHASE, RATING)


@pytest.fixture(scope="session", autouse=True)
def _offline():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("relbench.hf._repo_has", lambda *a, **k: False)
        mp.setattr("relbench.hf.list_task_names", lambda *a, **k: [])
        mp.setattr("relbench.hf.load_core_regression_stds", lambda *a, **k: {})
        yield


@pytest.fixture
def fake_dataset():
    return FakeDataset


@pytest.fixture
def task_manifests():
    return TASKS


@pytest.fixture(scope="session")
def dataset():
    return FakeDataset()


@pytest.fixture(scope="session")
def churn_task(dataset):
    return build_task(dataset, CHURN)


@pytest.fixture(scope="session")
def purchase_task(dataset):
    return build_task(dataset, PURCHASE)


@pytest.fixture
def churn_manifest():
    return dataclasses.replace(CHURN)


@pytest.fixture
def purchase_manifest():
    return dataclasses.replace(PURCHASE)


@pytest.fixture
def rating_manifest():
    return dataclasses.replace(RATING)


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("fakeds")
    ds = FakeDataset()
    db = ds.get_db(upto_test_timestamp=False)
    (root / "db").mkdir()
    tables = {}
    for name, table in db.table_dict.items():
        table.df.to_parquet(root / "db" / f"{name}.parquet")
        tables[name] = TableSpec(
            pkey=table.pkey_col,
            time_col=table.time_col,
            fkeys=dict(table.fkey_col_to_pkey_table),
        )
    DatasetManifest(
        name="fakeds",
        val_timestamp=str(ds.val_timestamp),
        test_timestamp=str(ds.test_timestamp),
        tables=tables,
    ).save(root / "manifest.yaml")
    for tm in TASKS:
        tm.save(root / "tasks" / tm.name / "manifest.yaml")
    loaded = load_dataset(root)
    for tm in TASKS:
        task = loaded.load_task(tm.name, regenerate=True)
        db = task.get_db(upto_test_timestamp=False)
        for split in ("train", "val", "test"):
            task.get_table(split, mask_input_cols=False, db=db).df.to_parquet(
                root / "tasks" / tm.name / f"{split}.parquet"
            )
    return root
