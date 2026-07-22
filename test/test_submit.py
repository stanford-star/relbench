r"""Tests for the prediction-table evaluator and leaderboard tooling.

Network-free: a synthetic :class:`FakeDataset` (from ``conftest``) is serialized to a temp
directory as a manifest + parquet RelBench dataset, with three tasks (binary
classification, regression and link prediction) so ``relbench.load.load_task`` -- which the
evaluator calls by name -- finds it on disk. The hosted regression-std lookup is
monkeypatched out, and ``evaluate_submission`` is exercised in-process (``num_workers=1``)
so monkeypatched dataset resolution is visible.
"""

import random

import numpy as np
import pandas as pd
import pytest
from conftest import _CHURN, _PURCHASE, FakeDataset

from relbench.base import TaskType
from relbench.load import load_task
from relbench.manifest import DatasetManifest, TableSpec, TaskManifest
from relbench.submit import (
    LEADERBOARD_TASKS,
    evaluate_submission,
    evaluate_task,
    write_prediction_table,
)

# A regression task over the fake schema: future-review count per customer (numeric target).
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

_LTV = TaskManifest(
    name="ltv",
    kind="forecast",
    task_type="regression",
    entity_table="customer",
    entity_col="customer_id",
    target_col="ltv",
    time_col="timestamp",
    timedelta="100 days",
    num_eval_timestamps=1,
    sql=_LTV_SQL,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_core_std_network(monkeypatch):
    r"""Keep regression NMAE std resolution offline (fall back to train-split std)."""
    monkeypatch.setattr("relbench.hf.load_core_regression_stds", lambda *a, **k: {})


def _write_fake_dataset(root):
    r"""Serialize a seeded FakeDataset to ``root`` as a manifest + parquet dataset with
    user-churn (clf), user-ltv (regression) and user-item-purchase (link) tasks."""
    random.seed(0)
    np.random.seed(0)
    ds = FakeDataset()
    db = ds.get_db(upto_test_timestamp=False)
    (root / "db").mkdir(parents=True, exist_ok=True)
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
    TaskManifest(**{**_CHURN.__dict__, "name": "user-churn"}).save(
        root / "tasks" / "user-churn" / "manifest.yaml"
    )
    TaskManifest(**{**_LTV.__dict__, "name": "user-ltv"}).save(
        root / "tasks" / "user-ltv" / "manifest.yaml"
    )
    TaskManifest(**{**_PURCHASE.__dict__, "name": "user-item-purchase"}).save(
        root / "tasks" / "user-item-purchase" / "manifest.yaml"
    )
    return root


@pytest.fixture(scope="module")
def fake_ds_dir(tmp_path_factory):
    return _write_fake_dataset(tmp_path_factory.mktemp("fakeds"))


def _entity_pred(df, task, scale=1.0):
    r"""Deterministic per-entity prediction (the test split has a single timestamp, so
    the entity id alone is the key); order-independent, so write- and direct-eval paths
    agree."""
    ent = df[task.entity_col].astype("int64").to_numpy()
    return (((ent * 2654435761) % 997) / 997.0) * scale


def _link_pred(df, task):
    r"""Deterministic per-source top-eval_k destination ids, shape (N, eval_k)."""
    src = df[task.src_entity_col].astype("int64").to_numpy()
    return (src[:, None] + np.arange(task.eval_k)[None, :]) % task.num_dst_nodes


# --------------------------------------------------------------------------- #
# Round-trip tests: write_prediction_table -> evaluate_task == task.evaluate(pred)
# --------------------------------------------------------------------------- #
def test_roundtrip_binary_classification(fake_ds_dir, tmp_path):
    task = load_task(str(fake_ds_dir), "user-churn")
    assert task.task_type == TaskType.BINARY_CLASSIFICATION
    masked = task.get_table("test", mask_input_cols=True)
    gt = task.get_table("test", mask_input_cols=False)

    csv = tmp_path / "churn.csv"
    write_prediction_table(task, _entity_pred(masked.df, task), csv)

    # CSV is key columns + the target column.
    cols = list(pd.read_csv(csv).columns)
    assert cols == [task.entity_col, task.time_col, task.target_col]

    got = evaluate_task("fakeds/user-churn", csv, dataset=str(fake_ds_dir))
    want = task.evaluate(_entity_pred(gt.df, task), target_table=gt)
    assert set(got) == {"roc_auc"}
    assert got["roc_auc"] == pytest.approx(want["roc_auc"])


def test_roundtrip_regression(fake_ds_dir, tmp_path):
    task = load_task(str(fake_ds_dir), "user-ltv")
    assert task.task_type == TaskType.REGRESSION
    masked = task.get_table("test", mask_input_cols=True)
    gt = task.get_table("test", mask_input_cols=False)

    csv = tmp_path / "ltv.csv"
    write_prediction_table(task, _entity_pred(masked.df, task, scale=5.0), csv)

    got = evaluate_task("fakeds/user-ltv", csv, dataset=str(fake_ds_dir))
    want = task.evaluate(_entity_pred(gt.df, task, scale=5.0), target_table=gt)
    assert set(got) == {"nmae"}
    assert np.isfinite(got["nmae"])
    assert got["nmae"] == pytest.approx(want["nmae"])


def test_roundtrip_link_prediction(fake_ds_dir, tmp_path):
    task = load_task(str(fake_ds_dir), "user-item-purchase")
    assert task.task_type == TaskType.LINK_PREDICTION
    masked = task.get_table("test", mask_input_cols=True)
    gt = task.get_table("test", mask_input_cols=False)

    csv = tmp_path / "purchase.csv"
    write_prediction_table(task, _link_pred(masked.df, task), csv)

    cols = list(pd.read_csv(csv).columns)
    assert cols == [task.src_entity_col, task.time_col, task.dst_entity_col]

    got = evaluate_task("fakeds/user-item-purchase", csv, dataset=str(fake_ds_dir))
    want = task.evaluate(_link_pred(gt.df, task), target_table=gt)
    assert set(got) == {"link_prediction_map"}
    assert 0.0 <= got["link_prediction_map"] <= 1.0
    assert got["link_prediction_map"] == pytest.approx(want["link_prediction_map"])


# --------------------------------------------------------------------------- #
# Validation failures
# --------------------------------------------------------------------------- #
def _valid_clf_csv(fake_ds_dir, csv):
    task = load_task(str(fake_ds_dir), "user-churn")
    masked = task.get_table("test", mask_input_cols=True)
    write_prediction_table(task, _entity_pred(masked.df, task), csv)
    return task


def test_validation_missing_rows(fake_ds_dir, tmp_path):
    csv = tmp_path / "churn.csv"
    _valid_clf_csv(fake_ds_dir, csv)
    df = pd.read_csv(csv)
    df.iloc[:-1].to_csv(csv, index=False)  # drop one row
    with pytest.raises(ValueError, match="missing"):
        evaluate_task("fakeds/user-churn", csv, dataset=str(fake_ds_dir))


def test_validation_extra_rows(fake_ds_dir, tmp_path):
    csv = tmp_path / "churn.csv"
    task = _valid_clf_csv(fake_ds_dir, csv)
    df = pd.read_csv(csv)
    extra = df.iloc[[0]].copy()
    extra[task.entity_col] = df[task.entity_col].max() + 12345  # unknown entity id
    pd.concat([df, extra], ignore_index=True).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="extra"):
        evaluate_task("fakeds/user-churn", csv, dataset=str(fake_ds_dir))


def test_validation_duplicate_keys(fake_ds_dir, tmp_path):
    csv = tmp_path / "churn.csv"
    _valid_clf_csv(fake_ds_dir, csv)
    df = pd.read_csv(csv)
    pd.concat([df, df.iloc[[0]]], ignore_index=True).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_task("fakeds/user-churn", csv, dataset=str(fake_ds_dir))


def test_validation_prob_out_of_range(fake_ds_dir, tmp_path):
    csv = tmp_path / "churn.csv"
    task = _valid_clf_csv(fake_ds_dir, csv)
    df = pd.read_csv(csv)
    df.loc[0, task.target_col] = 2.0  # invalid probability
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="must lie in"):
        evaluate_task("fakeds/user-churn", csv, dataset=str(fake_ds_dir))


def test_validation_link_list_too_long(fake_ds_dir, tmp_path):
    csv = tmp_path / "purchase.csv"
    task = load_task(str(fake_ds_dir), "user-item-purchase")
    masked = task.get_table("test", mask_input_cols=True)
    write_prediction_table(task, _link_pred(masked.df, task), csv)
    df = pd.read_csv(csv)
    import json

    df.loc[0, task.dst_entity_col] = json.dumps(list(range(task.eval_k + 5)))
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="eval_k"):
        evaluate_task("fakeds/user-item-purchase", csv, dataset=str(fake_ds_dir))


# --------------------------------------------------------------------------- #
# evaluate_submission
# --------------------------------------------------------------------------- #
def test_evaluate_submission(fake_ds_dir, tmp_path, monkeypatch):
    # Resolve any dataset name to the on-disk fake dataset (in-process num_workers=1).
    monkeypatch.setattr(
        "relbench.load._resolve_dataset_dir", lambda *a, **k: fake_ds_dir
    )

    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()

    churn = load_task(str(fake_ds_dir), "user-churn")
    write_prediction_table(
        churn,
        _entity_pred(churn.get_table("test", mask_input_cols=True).df, churn),
        pred_dir / "rel-amazon__user-churn.csv",
    )
    purchase = load_task(str(fake_ds_dir), "user-item-purchase")
    write_prediction_table(
        purchase,
        _link_pred(purchase.get_table("test", mask_input_cols=True).df, purchase),
        pred_dir / "rel-amazon__user-item-purchase.csv",
    )

    result = evaluate_submission(pred_dir, num_workers=1, verbose=True)

    # Structured return shape.
    assert set(result) == {"tasks", "families", "validated", "extra_files"}
    assert set(result["families"]) == set(LEADERBOARD_TASKS)

    clf = result["tasks"]["rel-amazon/user-churn"]
    rec = result["tasks"]["rel-amazon/user-item-purchase"]
    assert clf["status"] == "ok" and clf["family"] == "classification"
    assert clf["metric_name"] == "roc_auc" and np.isfinite(clf["metric"])
    assert rec["status"] == "ok" and rec["family"] == "recommendation"
    assert rec["metric_name"] == "link_prediction_map" and np.isfinite(rec["metric"])

    # Per-task metric matches the standalone evaluator.
    direct = evaluate_task(
        "rel-amazon/user-churn",
        pred_dir / "rel-amazon__user-churn.csv",
        dataset=str(fake_ds_dir),
    )
    assert clf["metric"] == pytest.approx(direct["roc_auc"])

    # Completeness: families are independent and none is fully present here.
    fams = result["families"]
    assert fams["classification"]["num_valid"] == 1
    assert fams["classification"]["num_total"] == 12
    assert fams["classification"]["complete"] is False
    assert "rel-amazon/item-churn" in fams["classification"]["missing"]
    assert fams["recommendation"]["num_valid"] == 1
    assert fams["recommendation"]["num_total"] == 10
    assert fams["recommendation"]["complete"] is False
    assert fams["regression"]["num_valid"] == 0
    assert fams["regression"]["complete"] is False
    assert result["validated"] == []
    assert "rejected" in fams["classification"]["verdict"]

    # Aggregate is the mean over valid tasks (one task here).
    assert fams["classification"]["aggregate"] == pytest.approx(clf["metric"])


def test_evaluate_submission_reports_task_errors(fake_ds_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "relbench.load._resolve_dataset_dir", lambda *a, **k: fake_ds_dir
    )
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()

    churn = load_task(str(fake_ds_dir), "user-churn")
    csv = pred_dir / "rel-amazon__user-churn.csv"
    write_prediction_table(
        churn,
        _entity_pred(churn.get_table("test", mask_input_cols=True).df, churn),
        csv,
    )
    # Corrupt: drop a row so validation fails and the task is recorded as an error.
    df = pd.read_csv(csv)
    df.iloc[:-1].to_csv(csv, index=False)

    result = evaluate_submission(pred_dir, num_workers=1, verbose=False)
    entry = result["tasks"]["rel-amazon/user-churn"]
    assert entry["status"] == "error"
    assert entry["metric"] is None
    assert "missing" in entry["error"]
    assert "rel-amazon/user-churn" in result["families"]["classification"]["invalid"]
    assert result["validated"] == []
