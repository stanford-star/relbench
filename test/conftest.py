r"""Shared test fixtures.

Provides new-system (manifest-driven) tasks built directly on the in-memory
``FakeDataset`` via :func:`relbench.load.build_task`, replacing the removed legacy task
classes. A `forecast` churn (entity) task and purchase (link) task over the fake schema
(customer / product / review) give the modeling tests entity and link task tables.
"""

import pytest

from relbench.load import build_task
from relbench.manifest import TaskManifest

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
def make_churn_task():
    r"""Return a builder: dataset -> forecast binary-classification (churn) task."""
    return lambda dataset: build_task(dataset, _CHURN)


@pytest.fixture
def make_purchase_task():
    r"""Return a builder: dataset -> forecast link-prediction (purchase) task."""
    return lambda dataset: build_task(dataset, _PURCHASE)
