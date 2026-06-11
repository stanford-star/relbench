r"""Shared test fixtures.

Provides the synthetic ``FakeDataset`` (the in-memory test database) and new-system
(manifest-driven) tasks built on it via :func:`relbench.load.build_task`, replacing the
removed legacy dataset/task classes. A `forecast` churn (entity) task and purchase (link)
task over the fake schema (customer / product / review) give the modeling tests entity and
link task tables.
"""

import random
import string

import numpy as np
import pandas as pd
import pytest

from relbench.base import Database, Dataset, Table
from relbench.load import build_task
from relbench.manifest import TaskManifest


def _generate_random_string(min_length: int, max_length: int) -> str:
    length = random.randint(min_length, max_length)
    random_string = "".join(random.choice(string.ascii_letters) for _ in range(length))
    return random_string


class FakeDataset(Dataset):
    r"""A small synthetic dataset (customers / products / reviews) for tests."""

    def __init__(
        self,
        num_products: int = 30,
        num_customers: int = 100,
        num_reviews: int = 600,
        num_relations: int = 20,
    ):
        self.num_products = num_products
        self.num_customers = num_customers
        self.num_reviews = num_reviews
        self.num_relations = num_relations

        min_timestamp = pd.Timestamp(0, unit="D")
        max_timestamp = pd.Timestamp(2 * (num_reviews - 1), unit="D")
        self.val_timestamp = min_timestamp + 0.8 * (max_timestamp - min_timestamp)
        self.test_timestamp = min_timestamp + 0.9 * (max_timestamp - min_timestamp)
        super().__init__()

    def make_db(self) -> Database:
        num_products = self.num_products
        num_customers = self.num_customers
        num_reviews = self.num_reviews
        num_relations = self.num_relations
        product_df = pd.DataFrame(
            {
                "product_id": [f"product_id_{i}" for i in range(num_products)],
                "category": [None, [], ["toy", "health"]] * (num_products // 3),
                "title": [_generate_random_string(5, 15) for _ in range(num_products)],
                "price": np.random.rand(num_products) * 10,
            }
        )
        customer_df = pd.DataFrame(
            {
                "customer_id": [f"customer_id_{i}" for i in range(num_customers)],
                "age": np.random.randint(10, 50, size=(num_customers,)),
                "gender": ["male", "female"] * (num_customers // 2),
            }
        )
        # Add some dangling foreign keys:
        review_df = pd.DataFrame(
            {
                "customer_id": [
                    f"customer_id_{random.randint(0, num_customers+5)}"
                    for _ in range(num_reviews)
                ],
                "product_id": [
                    f"product_id_{random.randint(0, num_products-1)}"
                    for _ in range(num_reviews)
                ],
                "review_time": pd.to_datetime(2 * np.arange(num_reviews), unit="D"),
                "rating": np.random.randint(1, 6, size=(num_reviews,)),
            }
        )
        review_df["review"] = review_df["rating"].apply(
            lambda x: "positive" if x > 3 else "negative"
        )
        relations_df = pd.DataFrame(
            {
                "customer_id": [
                    f"customer_id_{random.randint(0, num_customers+5)}"
                    for _ in range(num_relations)
                ],
                "product_id": [
                    f"product_id_{random.randint(0, num_products-1)}"
                    for _ in range(num_relations)
                ],
            }
        )

        return Database(
            table_dict={
                "product": Table(
                    df=product_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="product_id",
                ),
                "customer": Table(
                    df=customer_df,
                    fkey_col_to_pkey_table={},
                    pkey_col="customer_id",
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

_CHURN = TaskManifest(
    name="churn", kind="forecast", task_type="binary_classification",
    entity_table="customer", entity_col="customer_id", target_col="churn",
    time_col="timestamp", timedelta="100 days", num_eval_timestamps=1, sql=_CHURN_SQL,
)
_PURCHASE = TaskManifest(
    name="purchase", kind="forecast", task_type="link_prediction",
    src_entity_table="customer", src_entity_col="customer_id",
    dst_entity_table="product", dst_entity_col="product_id", target_col="product_id",
    time_col="timestamp", timedelta="100 days", eval_k=10, num_eval_timestamps=1,
    sql=_PURCHASE_SQL,
)


@pytest.fixture
def fake_dataset():
    r"""Return the :class:`FakeDataset` class (call it to build a synthetic dataset)."""
    return FakeDataset


@pytest.fixture
def make_churn_task():
    r"""Return a builder: dataset -> forecast binary-classification (churn) task."""
    return lambda dataset: build_task(dataset, _CHURN)


@pytest.fixture
def make_purchase_task():
    r"""Return a builder: dataset -> forecast link-prediction (purchase) task."""
    return lambda dataset: build_task(dataset, _PURCHASE)
