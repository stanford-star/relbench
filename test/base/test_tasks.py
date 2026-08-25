import dataclasses

import pandas as pd

from relbench.base import AutoCompleteTask, TaskType


def test_autocomplete_task(fake_dataset):
    dataset = fake_dataset()
    assert dataset is not None
    task = AutoCompleteTask(
        dataset=dataset,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        entity_table="review",
        target_col="rating",
        remove_columns=[("review", "review")],
    )
    # ensure columns are removed correctly, in the task's view of the database
    for table, column in task.remove_columns:
        assert column not in task.get_db().table_dict[table].df.columns
    # the dataset itself is untouched -- removals are per task
    assert "review" in dataset.get_db().table_dict["review"].df.columns

    # ensure all entities are present. The synthetic entity table has no primary key
    # of its own, so the task's view of the database is where `primary_key` is added.
    db = task.get_db(upto_test_timestamp=False)
    pks = db.table_dict["review"].df.get("primary_key")
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")
    assert test_table.df.primary_key.max() == pks.max()
    task_table_full = pd.concat(
        [train_table.df, val_table.df, test_table.df], ignore_index=True
    )
    assert task_table_full.primary_key.isin(pks).all()

    # ensure the task can be constructed multiple times on the same database
    task = AutoCompleteTask(
        dataset=dataset,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        entity_table="review",
        target_col="rating",
        remove_columns=[],
    )
    # ensure the review column is present as it was not removed this time
    assert "review" in task.get_db().table_dict["review"].df.columns


def _db_columns(task, table):
    return list(task.get_db().table_dict[table].df.columns)


def test_forecast_entity_task_honors_remove_columns(fake_dataset, churn_manifest):
    """`remove_columns` is not an autocomplete-only field.

    A forecast task whose label is derived from a database column has to hide that
    column too, and `task.get_db()` is what applies it.
    """
    from relbench.load import build_task

    tm = dataclasses.replace(churn_manifest, remove_columns=[["review", "review"]])
    task = build_task(fake_dataset(), tm)
    assert task.remove_columns == [("review", "review")]
    assert "review" not in _db_columns(task, "review")
    # only the named column goes; the rest of the table is untouched
    assert "rating" in _db_columns(task, "review")
    # removal applies to the graph, not to the labels
    assert "churn" in task.get_table("train").df.columns


def test_link_task_honors_remove_columns(fake_dataset, purchase_manifest):
    from relbench.load import build_task

    tm = dataclasses.replace(purchase_manifest, remove_columns=[["review", "rating"]])
    task = build_task(fake_dataset(), tm)
    assert "rating" not in _db_columns(task, "review")
    assert "review" in _db_columns(task, "review")


def test_external_task_honors_remove_columns(fake_dataset, churn_manifest):
    from relbench.load import build_task

    tm = dataclasses.replace(
        churn_manifest, kind="external", sql=None, remove_columns=[["review", "rating"]]
    )
    task = build_task(fake_dataset(), tm)
    assert "rating" not in _db_columns(task, "review")


def test_remove_columns_are_scoped_per_task(fake_dataset, churn_manifest):
    """Two tasks over one dataset each see their own removals.

    Each task applies its own `remove_columns` in `task.get_db()`; the dataset is never
    modified, so tasks cannot leak removals into one another regardless of load order.
    """
    from relbench.load import build_task

    dataset = fake_dataset()
    first = build_task(
        dataset,
        dataclasses.replace(churn_manifest, remove_columns=[["review", "review"]]),
    )
    assert "review" not in _db_columns(first, "review")
    second = build_task(
        dataset,
        dataclasses.replace(churn_manifest, remove_columns=[["review", "rating"]]),
    )
    assert "rating" not in _db_columns(second, "review")
    assert "review" in _db_columns(second, "review")  # first task's removal is gone
    third = build_task(dataset, dataclasses.replace(churn_manifest, remove_columns=[]))
    assert {"review", "rating"} <= set(_db_columns(third, "review"))


def test_stale_remove_columns_entry_warns(fake_dataset, churn_manifest, capsys):
    """A `remove_columns` pair naming something absent warns instead of raising."""
    from relbench.load import build_task

    tm = dataclasses.replace(
        churn_manifest,
        remove_columns=[["nosuchtable", "col"], ["review", "nosuchcol"]],
    )
    task = build_task(fake_dataset(), tm)
    task.get_db()
    out = capsys.readouterr().out
    assert "nosuchtable" in out
    assert "nosuchcol" in out
