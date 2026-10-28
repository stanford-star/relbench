import pandas as pd
import pytest

from relbench.manifest import DatasetManifest, TableSpec, TaskManifest
from relbench.validate import ERROR, main, validate_dataset

TABLES = {
    "users": TableSpec(pkey="uid"),
    "events": TableSpec(pkey="eid", time_col="t", fkeys={"uid": "users"}),
}
CHURN = TaskManifest(
    name="churn",
    kind="forecast",
    task_type="binary_classification",
    entity_table="users",
    entity_col="uid",
    target_col="churn",
    time_col="timestamp",
    timedelta="1 day",
    sql="SELECT 1",
)
EVENT_X = TaskManifest(
    name="event-x",
    kind="autocomplete",
    task_type="regression",
    entity_table="events",
    target_col="x",
)


def _users():
    return pd.DataFrame({"uid": [0, 1, 2], "name": ["a", "b", "c"]})


def _events():
    return pd.DataFrame(
        {
            "eid": range(6),
            "uid": pd.array([0, 1, 2, 0, None, 1], dtype="Int64"),
            "t": pd.date_range("2020-01-01", periods=6, freq="D"),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def _churn(split):
    ts = {"train": "2020-01-02", "val": "2020-01-04", "test": "2020-01-05"}[split]
    return pd.DataFrame(
        {"timestamp": pd.to_datetime([ts, ts]), "uid": [0, 1], "churn": [0, 1]}
    )


def _event_x(split):
    rows = {"train": [0, 1, 2], "val": [3, 4], "test": [5]}[split]
    return _events().iloc[rows][["t", "eid", "x"]].reset_index(drop=True)


@pytest.fixture
def make(tmp_path):
    def make(tables=None, users=None, events=None, val="2020-01-03", test="2020-01-05"):
        (tmp_path / "db").mkdir(exist_ok=True)
        (_users() if users is None else users).to_parquet(
            tmp_path / "db" / "users.parquet"
        )
        (_events() if events is None else events).to_parquet(
            tmp_path / "db" / "events.parquet"
        )
        DatasetManifest(
            name="tiny",
            val_timestamp=val,
            test_timestamp=test,
            tables=TABLES if tables is None else tables,
        ).save(tmp_path / "manifest.yaml")
        for tm, labels in [(CHURN, _churn), (EVENT_X, _event_x)]:
            tm.save(tmp_path / "tasks" / tm.name / "manifest.yaml")
            for split in ("train", "val", "test"):
                labels(split).to_parquet(
                    tmp_path / "tasks" / tm.name / f"{split}.parquet"
                )
        return tmp_path

    return make


def _errors(root, **kwargs):
    return [i for i in validate_dataset(root, **kwargs) if i.level == ERROR]


def _only(root, where, match, **kwargs):
    errors = _errors(root, **kwargs)
    assert len(errors) == 1, errors
    assert errors[0].where == where and match in errors[0].message, errors[0]
    return errors[0]


def test_valid_tiny_dataset(make):
    assert validate_dataset(make()) == []


def test_valid_generated_dataset(dataset_dir):
    assert validate_dataset(dataset_dir) == []


def test_missing_manifest(tmp_path):
    _only(tmp_path, "manifest.yaml", "not found")


def test_missing_table_parquet(make):
    root = make()
    (root / "db" / "users.parquet").unlink()
    _only(root, "table 'users'", "missing parquet db/users.parquet")


def test_manifest_column_not_in_parquet(make):
    tables = dict(TABLES, events=TableSpec(pkey="eid", time_col="ts", fkeys={}))
    _only(
        make(tables=tables),
        "table 'events'",
        "time_col column 'ts' not in parquet",
        tasks=False,
    )


def test_fkey_to_unknown_table(make):
    tables = dict(
        TABLES, events=TableSpec(pkey="eid", time_col="t", fkeys={"uid": "people"})
    )
    _only(make(tables=tables), "events.uid", "unknown table 'people'")


def test_fkey_to_table_without_pkey(make):
    tables = dict(TABLES, users=TableSpec())
    errors = _errors(make(tables=tables))
    assert any(e.where == "events.uid" and "no pkey" in e.message for e in errors)


def test_pkey_nulls_and_duplicates(make):
    users = pd.DataFrame({"uid": pd.array([0, 0, None], dtype="Int64")})
    errors = _errors(make(users=users), tasks=False)
    msgs = {e.message for e in errors if e.where == "users.uid"}
    assert "primary key has 1 null values" in msgs
    assert "primary key has 1 duplicate values, e.g. 0 (x2)" in msgs


def test_pkey_not_consecutive(make):
    users = pd.DataFrame({"uid": [0, 2, 1]})
    root = make(users=users)
    _only(root, "users.uid", "not 0..n-1 in row order (2 rows differ, first at row 1)")
    assert _errors(root, key_contract=False) == []


def test_string_keys_need_reindex_but_pass_raw(make):
    users = pd.DataFrame({"uid": ["u0", "u1", "u2"]})
    events = _events().assign(uid=["u0", "u1", "u2", "u0", None, "u1"])
    root = make(users=users, events=events)
    where = {e.where for e in _errors(root, tasks=False)}
    assert where == {"users.uid", "events.uid"}
    assert _errors(root, tasks=False, key_contract=False) == []


def test_dangling_fkey(make):
    events = _events().assign(uid=pd.array([0, 7, 2, 9, None, 7], dtype="Int64"))
    e = _only(make(events=events), "events.uid", "3 of 5 non-null values")
    assert "do not resolve to users.uid" in e.message
    assert "7" in e.message and "9" in e.message


def test_fkey_type_mismatch(make):
    events = _events().assign(uid=["0", "1", "2", "0", None, "1"])
    _only(make(events=events), "events.uid", "does not match users.uid type")


def test_time_column_string(make):
    events = _events().assign(t=_events()["t"].astype(str))
    e = _only(make(events=events), "events.t", "expected a timestamp")
    assert "all values parse" in e.message
    events.loc[2, "t"] = "not a date"
    e = _only(make(events=events), "events.t", "1 values do not parse")
    assert "'not a date'" in e.message


def test_time_column_not_sorted(make):
    events = _events().assign(t=_events()["t"][[1, 0, 2, 3, 4, 5]].to_numpy())
    root = make(events=events)
    _only(root, "events.t", "not sorted by time (1 out of order)", tasks=False)
    assert _errors(root, tasks=False, key_contract=False) == []


def test_split_timestamps_out_of_order(make):
    errors = _errors(make(val="2020-01-05", test="2020-01-03"), tasks=False)
    assert errors[0].where == "manifest.yaml"
    assert "is not before test_timestamp" in errors[0].message


def test_split_timestamp_outside_data(make):
    _only(
        make(val="2020-01-03", test="2021-01-01"),
        "manifest.yaml",
        "test_timestamp 2021-01-01 00:00:00 is outside the data",
        tasks=False,
    )


def test_task_missing_columns(make):
    root = make()
    _churn("val").drop(columns=["churn"]).to_parquet(
        root / "tasks" / "churn" / "val.parquet"
    )
    _only(root, "task 'churn' val", "missing columns ['churn']")


def test_task_entity_not_in_entity_table(make):
    root = make()
    _churn("test").assign(uid=[1, 5]).to_parquet(
        root / "tasks" / "churn" / "test.parquet"
    )
    _only(root, "task 'churn' test.uid", "1 of 2 non-null values do not resolve")


def test_task_train_labels_after_val(make):
    root = make()
    _churn("val").to_parquet(root / "tasks" / "churn" / "train.parquet")
    _only(root, "task 'churn' train", "past val_timestamp")


def test_task_forecast_window_leaks(make):
    root = make()
    _churn("val").assign(timestamp=pd.Timestamp("2020-01-03")).to_parquet(
        root / "tasks" / "churn" / "train.parquet"
    )
    _only(root, "task 'churn' train", "+ timedelta 1 day passes val_timestamp")


def test_task_unknown_entity_table(make):
    root = make()
    TaskManifest(**{**CHURN.__dict__, "entity_table": "people"}).save(
        root / "tasks" / "churn" / "manifest.yaml"
    )
    _only(root, "task 'churn'", "entity table 'people' is not in the dataset")


def test_autocomplete_target_missing(make):
    root = make()
    TaskManifest(**{**EVENT_X.__dict__, "target_col": "y"}).save(
        root / "tasks" / "event-x" / "manifest.yaml"
    )
    errors = _errors(root)
    assert any(
        e.where == "task 'event-x'" and "'y' is not in table 'events'" in e.message
        for e in errors
    )


def test_reports_all_issues_at_once(make):
    users = pd.DataFrame({"uid": pd.array([0, 0, 2], dtype="Int64")})
    events = _events().assign(uid=pd.array([0, 8, 2, 0, None, 1], dtype="Int64"))
    root = make(users=users, events=events)
    _churn("test").assign(uid=[1, 5]).to_parquet(
        root / "tasks" / "churn" / "test.parquet"
    )
    where = {e.where for e in _errors(root)}
    assert {"users.uid", "events.uid", "task 'churn' test.uid"} <= where


def test_cli(make, capsys):
    root = make()
    assert main([str(root)]) == 0
    assert "0 error(s), 0 warning(s)" in capsys.readouterr().out
    (root / "db" / "users.parquet").unlink()
    assert main([str(root)]) == 1
    assert "ERROR   table 'users': missing parquet" in capsys.readouterr().out
