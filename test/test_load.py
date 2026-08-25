import numpy as np
import pandas as pd
import pytest

from relbench.hf import RELBENCH_HF, resolve_repo
from relbench.load import load_dataset, train_std


def _lists(df):
    df = df.copy()
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (list, np.ndarray))).any():
            df[col] = df[col].map(list)
    return df


def test_load_dataset_from_local_dir(dataset_dir, dataset, task_manifests):
    ds = load_dataset(dataset_dir)
    assert ds.get_task_names() == sorted(tm.name for tm in task_manifests)
    assert (ds.val_timestamp, ds.test_timestamp) == (
        dataset.val_timestamp,
        dataset.test_timestamp,
    )
    db = ds.get_db()
    assert db.max_timestamp <= ds.test_timestamp
    for table in db.table_dict.values():
        if table.pkey_col is not None:
            assert table.df[table.pkey_col].tolist() == list(range(len(table)))
    with pytest.raises(ValueError, match="user-churn"):
        ds.load_task("no-such-task")


def test_load_task_from_directory(dataset_dir):
    ds = load_dataset(dataset_dir)
    by_name = ds.load_task("user-churn").get_table("val").df
    by_path = ds.load_task(dataset_dir / "tasks" / "user-churn").get_table("val").df
    pd.testing.assert_frame_equal(by_name, by_path)


@pytest.mark.parametrize(
    "name", ["user-churn", "user-ltv", "user-item-purchase", "review-rating"]
)
def test_hosted_labels_match_regeneration(dataset_dir, name):
    ds = load_dataset(dataset_dir)
    hosted = ds.load_task(name)
    regenerated = ds.load_task(name, regenerate=True)
    db = regenerated.get_db(upto_test_timestamp=False)
    for split in ("train", "val", "test"):
        a = hosted.get_table(split, mask_input_cols=False).df
        b = regenerated.get_table(split, mask_input_cols=False, db=db).df
        pd.testing.assert_frame_equal(_lists(a), _lists(b))


def test_hosted_task_matches_in_memory_task(dataset_dir, churn_task):
    hosted = load_dataset(dataset_dir).load_task("user-churn")
    pd.testing.assert_frame_equal(
        hosted.get_table("val").df, churn_task.get_table("val").df
    )


def test_regression_task_nmae(dataset_dir):
    task = load_dataset(dataset_dir).load_task("user-ltv")
    assert [m.__name__ for m in task.metrics] == ["nmae"]
    assert task.nmae_std == pytest.approx(train_std(task))
    val = task.get_table("val")
    target = val.df[task.target_col].to_numpy()
    assert task.evaluate(target, val)["nmae"] == 0.0
    assert task.evaluate(target + task.nmae_std, val)["nmae"] == pytest.approx(1.0)


def test_local_dataset_never_touches_hub(dataset_dir, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("hub lookup for a local dataset")

    monkeypatch.setattr("relbench.hf.find_task_dir", boom)
    monkeypatch.setattr("relbench.hf.list_task_names", boom)
    monkeypatch.setattr("relbench.hf.load_core_regression_stds", boom)
    ds = load_dataset(dataset_dir)
    assert ds.is_local
    assert "user-ltv" in ds.get_task_names()
    task = ds.load_task("user-ltv")
    assert task.nmae_std == pytest.approx(train_std(task))
    with pytest.raises(ValueError) as err:
        ds.load_task("nope")
    assert "hosted" not in str(err.value)


def test_hosted_recommendation_test_table_is_masked(dataset_dir):
    task = load_dataset(dataset_dir).load_task("user-item-purchase")
    assert list(task.get_table("test").df.columns) == [
        task.time_col,
        task.src_entity_col,
    ]


def test_bad_local_path_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.yaml"):
        load_dataset(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="manifest.yaml"):
        load_dataset(str(tmp_path))


def test_version():
    import relbench

    assert relbench.__version__.split(".")[0].isdigit()


def test_resolve_repo():
    assert resolve_repo("org/name") == ("org/name", "")
    assert resolve_repo("org/name/a/b") == ("org/name", "a/b")
    assert resolve_repo("rel-f1") == (RELBENCH_HF, "rel-f1")
    with pytest.raises(ValueError):
        resolve_repo("/")
