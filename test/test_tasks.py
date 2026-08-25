import dataclasses

import numpy as np
import pandas as pd
import pytest

from relbench.base import TaskType
from relbench.load import build_task


def _lists(df):
    df = df.copy()
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (list, np.ndarray))).any():
            df[col] = df[col].map(list)
    return df


def test_entity_task_splits(churn_task, dataset):
    assert churn_task.task_type == TaskType.BINARY_CLASSIFICATION
    assert [m.__name__ for m in churn_task.metrics] == ["roc_auc"]
    train = churn_task.get_table("train")
    val = churn_task.get_table("val")
    test = churn_task.get_table("test")
    t = churn_task.time_col
    assert train.df[t].nunique() >= 3
    assert train.df[t].max() <= dataset.val_timestamp - churn_task.timedelta
    assert val.df[t].unique().tolist() == [dataset.val_timestamp]
    assert test.df[t].unique().tolist() == [dataset.test_timestamp]
    assert churn_task.target_col in train.df and churn_task.target_col in val.df
    assert list(test.df.columns) == [t, churn_task.entity_col]
    assert set(train.df[churn_task.target_col]) <= {0, 1}
    assert train.df[churn_task.entity_col].between(0, dataset.num_customers - 1).all()
    assert train.df.index.tolist() == list(range(len(train)))


def test_recommendation_task_splits(purchase_task, dataset):
    assert purchase_task.task_type == TaskType.RECOMMENDATION
    assert [m.__name__ for m in purchase_task.metrics] == ["map"]
    train = purchase_task.get_table("train")
    dst = train.df[purchase_task.dst_entity_col]
    assert dst.map(len).gt(0).all()
    assert dst.map(lambda v: list(v) == sorted(v)).all()
    assert all(0 <= i < dataset.num_products for row in dst for i in row)
    assert train.df[purchase_task.src_entity_col].lt(dataset.num_customers).all()


def test_tables_are_deterministic(churn_task, purchase_task):
    for task in (churn_task, purchase_task):
        a, b = task.get_table("val").df, task.get_table("val").df
        pd.testing.assert_frame_equal(_lists(a), _lists(b))


def test_evaluate_entity(churn_task):
    val = churn_task.get_table("val")
    target = val.df[churn_task.target_col].to_numpy(dtype=float)
    assert churn_task.evaluate(target, val) == {"roc_auc": 1.0}
    with pytest.raises(ValueError, match="length"):
        churn_task.evaluate(target[:-1], val)


def test_evaluate_recommendation(purchase_task):
    val = purchase_task.get_table("val")
    k = purchase_task.eval_k
    perfect = np.full((len(val), k), -1)
    for i, row in enumerate(val.df[purchase_task.dst_entity_col]):
        perfect[i, : min(len(row), k)] = list(row)[:k]
    assert purchase_task.evaluate(perfect, val)["map"] == pytest.approx(1.0)
    assert purchase_task.evaluate(np.full((len(val), k), -1), val)["map"] == 0.0
    with pytest.raises(ValueError, match="shape"):
        purchase_task.evaluate(perfect[:, :-1], val)


def test_timedelta_larger_than_split_gap_raises(dataset, churn_manifest):
    with pytest.raises(ValueError, match="timedelta"):
        build_task(dataset, dataclasses.replace(churn_manifest, timedelta="1000 days"))


def test_stats(churn_task):
    stats = churn_task.stats()
    assert set(stats) == {"train", "val", "test", "total"}
    assert {"num_positives", "num_negatives"} <= set(stats["total"])
    assert 0 <= stats["total"]["ratio_train_test_entity_overlap"] <= 1


def test_autocomplete_task(dataset, rating_manifest):
    tm = dataclasses.replace(rating_manifest, task_type="multiclass_classification")
    task = build_task(dataset, tm)
    assert task.metrics == []
    db = task.get_db(upto_test_timestamp=False)
    assert "review" not in db.table_dict["review"].df.columns
    assert "rating" not in db.table_dict["review"].df.columns
    assert "review" in dataset.get_db().table_dict["review"].df.columns
    assert task.entity_col == "primary_key" and task.time_col == "review_time"
    pks = db.table_dict["review"].df["primary_key"]
    tables = [
        task.get_table(split, mask_input_cols=False, db=db)
        for split in ("train", "val", "test")
    ]
    assert all(t.df["rating"].notna().all() for t in tables)
    assert tables[2].df["primary_key"].max() == pks.max()
    assert pd.concat([t.df for t in tables])["primary_key"].isin(pks).all()
    assert tables[0].df["review_time"].lt(dataset.val_timestamp).all()
    assert tables[1].df["review_time"].gt(dataset.val_timestamp).all()
    assert tables[1].df["review_time"].lt(dataset.test_timestamp).all()
    assert tables[2].df["review_time"].gt(dataset.test_timestamp).all()
    assert list(task.get_table("test", db=db).df.columns) == [
        "review_time",
        "primary_key",
    ]

    regression = build_task(dataset, rating_manifest)
    assert [m.__name__ for m in regression.metrics] == ["nmae"]
    assert regression.nmae_std > 0


def _db_columns(task, table):
    return list(task.get_db().table_dict[table].df.columns)


def test_forecast_task_honors_remove_columns(dataset, churn_manifest):
    tm = dataclasses.replace(churn_manifest, remove_columns=[["review", "review"]])
    task = build_task(dataset, tm)
    assert task.remove_columns == [("review", "review")]
    assert "review" not in _db_columns(task, "review")
    assert "rating" in _db_columns(task, "review")
    assert "churn" in task.get_table("train").df.columns


def test_external_task_honors_remove_columns(dataset, churn_manifest):
    tm = dataclasses.replace(
        churn_manifest, kind="external", sql=None, remove_columns=[["review", "rating"]]
    )
    assert "rating" not in _db_columns(build_task(dataset, tm), "review")


def test_remove_columns_are_scoped_per_task(dataset, churn_manifest):
    first = build_task(
        dataset,
        dataclasses.replace(churn_manifest, remove_columns=[["review", "review"]]),
    )
    second = build_task(
        dataset,
        dataclasses.replace(churn_manifest, remove_columns=[["review", "rating"]]),
    )
    assert "review" not in _db_columns(first, "review")
    assert "rating" not in _db_columns(second, "review")
    assert "review" in _db_columns(second, "review")


def test_stale_remove_columns_entry_warns(dataset, churn_manifest):
    tm = dataclasses.replace(
        churn_manifest, remove_columns=[["nosuchtable", "col"], ["review", "nosuchcol"]]
    )
    task = build_task(dataset, tm)
    with pytest.warns(UserWarning, match="nosuchtable"):
        with pytest.warns(UserWarning, match="nosuchcol"):
            task.get_db()


def test_external_task_rejects_regenerate(dataset, churn_manifest):
    tm = dataclasses.replace(churn_manifest, kind="external", sql=None)
    with pytest.raises(ValueError, match="not regenerable"):
        build_task(dataset, tm, regenerate=True)


def test_task_repr(churn_task, dataset):
    assert repr(churn_task) == "ForecastEntityTask('user-churn', dataset=FakeDataset())"
    assert repr(dataset.get_db()).startswith(
        "Database(product=30, customer=100, review="
    )
