import json

import numpy as np
import pandas as pd
import pytest

from relbench.load import load_dataset
from relbench.submit import (
    LEADERBOARD_TASKS,
    evaluate_submission,
    evaluate_task,
    write_prediction_table,
)


def _entity_pred(df, task, scale=1.0):
    ent = df[task.entity_col].to_numpy(dtype="int64")
    return (((ent * 2654435761) % 997) / 997.0) * scale


def _rec_pred(df, task):
    n_dst = len(task.get_db().table_dict[task.dst_entity_table])
    src = df[task.src_entity_col].to_numpy(dtype="int64")
    return (src[:, None] + np.arange(task.eval_k)[None, :]) % n_dst


def _pred(df, task):
    if task.task_type.value == "recommendation":
        return _rec_pred(df, task)
    return _entity_pred(
        df, task, scale=5.0 if task.task_type.value == "regression" else 1.0
    )


@pytest.fixture(scope="module")
def tasks(dataset_dir):
    ds = load_dataset(dataset_dir)
    return {name: ds.load_task(name) for name in ds.get_task_names()}


@pytest.mark.parametrize(
    "name, metric",
    [("user-churn", "roc_auc"), ("user-ltv", "nmae"), ("user-item-purchase", "map")],
)
def test_roundtrip(dataset_dir, tasks, tmp_path, name, metric):
    task = tasks[name]
    masked = task.get_table("test", mask_input_cols=True)
    gt = task.get_table("test", mask_input_cols=False)
    csv = tmp_path / f"{name}.csv"
    write_prediction_table(task, _pred(masked.df, task), csv)

    key_cols = (
        [task.src_entity_col, task.time_col, task.dst_entity_col]
        if metric == "map"
        else [task.entity_col, task.time_col, task.target_col]
    )
    assert list(pd.read_csv(csv).columns) == key_cols

    got = evaluate_task(f"fakeds/{name}", csv, dataset=str(dataset_dir))
    want = task.evaluate(_pred(gt.df, task), target_table=gt)
    assert set(got) == {metric}
    assert np.isfinite(got[metric])
    assert got[metric] == pytest.approx(want[metric])


@pytest.fixture
def churn_csv(dataset_dir, tasks, tmp_path):
    task = tasks["user-churn"]
    csv = tmp_path / "churn.csv"
    write_prediction_table(task, _entity_pred(task.get_table("test").df, task), csv)
    return task, csv


def _drop_row(df, task):
    return df.iloc[:-1]


def _extra_row(df, task):
    extra = df.iloc[[0]].copy()
    extra[task.entity_col] = df[task.entity_col].max() + 12345
    return pd.concat([df, extra], ignore_index=True)


def _duplicate_row(df, task):
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def _bad_prob(df, task):
    df.loc[0, task.target_col] = 2.0
    return df


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (_drop_row, "missing"),
        (_extra_row, "extra"),
        (_duplicate_row, "duplicate"),
        (_bad_prob, "must lie in"),
    ],
)
def test_validation_failures(dataset_dir, churn_csv, corrupt, match):
    task, csv = churn_csv
    corrupt(pd.read_csv(csv), task).to_csv(csv, index=False)
    with pytest.raises(ValueError, match=match):
        evaluate_task("fakeds/user-churn", csv, dataset=str(dataset_dir))


def test_validation_link_list_too_long(dataset_dir, tasks, tmp_path):
    task = tasks["user-item-purchase"]
    csv = tmp_path / "purchase.csv"
    write_prediction_table(task, _rec_pred(task.get_table("test").df, task), csv)
    df = pd.read_csv(csv)
    df.loc[0, task.dst_entity_col] = json.dumps(list(range(task.eval_k + 5)))
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="eval_k"):
        evaluate_task("fakeds/user-item-purchase", csv, dataset=str(dataset_dir))


@pytest.fixture
def submission(dataset_dir, tasks, tmp_path, monkeypatch):
    monkeypatch.setattr("relbench.submit._prefetch_hf_data", lambda *a, **k: None)
    monkeypatch.setattr(
        "relbench.submit._fetch_eval_files", lambda *a, **k: dataset_dir
    )
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    for name, fname in [
        ("user-churn", "rel-amazon__user-churn.csv"),
        ("user-item-purchase", "rel-amazon__user-item-purchase.csv"),
    ]:
        task = tasks[name]
        write_prediction_table(
            task, _pred(task.get_table("test").df, task), pred_dir / fname
        )
    (pred_dir / "notes.txt").write_text("not a prediction table")
    return pred_dir


def test_evaluate_submission(dataset_dir, submission):
    result = evaluate_submission(submission, num_workers=1, verbose=True)
    assert set(result) == {"tasks", "families", "validated", "extra_files"}
    assert result["extra_files"] == ["notes.txt"]
    assert set(result["families"]) == set(LEADERBOARD_TASKS)

    clf = result["tasks"]["rel-amazon/user-churn"]
    rec = result["tasks"]["rel-amazon/user-item-purchase"]
    assert (clf["status"], clf["family"], clf["metric_name"]) == (
        "ok",
        "classification",
        "roc_auc",
    )
    assert (rec["status"], rec["family"], rec["metric_name"]) == (
        "ok",
        "recommendation",
        "map",
    )
    direct = evaluate_task(
        "rel-amazon/user-churn",
        submission / "rel-amazon__user-churn.csv",
        dataset=str(dataset_dir),
    )
    assert clf["metric"] == pytest.approx(direct["roc_auc"])

    fams = result["families"]
    assert (
        fams["classification"]["num_valid"],
        fams["classification"]["num_total"],
    ) == (1, 12)
    assert "rel-amazon/item-churn" in fams["classification"]["missing"]
    assert (
        fams["recommendation"]["num_valid"],
        fams["recommendation"]["num_total"],
    ) == (1, 10)
    assert fams["regression"]["num_valid"] == 0
    assert fams["classification"]["aggregate"] == pytest.approx(clf["metric"])
    assert not any(f["complete"] for f in fams.values())
    assert result["validated"] == []


def test_evaluate_submission_reports_task_errors(submission):
    csv = submission / "rel-amazon__user-churn.csv"
    pd.read_csv(csv).iloc[:-1].to_csv(csv, index=False)
    result = evaluate_submission(submission, num_workers=1, verbose=False)
    entry = result["tasks"]["rel-amazon/user-churn"]
    assert entry["status"] == "error" and entry["metric"] is None
    assert "missing" in entry["error"]
    assert "rel-amazon/user-churn" in result["families"]["classification"]["invalid"]
    assert result["validated"] == []
