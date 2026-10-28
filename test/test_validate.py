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


def test_valid_tiny_dataset(make):
    assert validate_dataset(make()) == []


def test_valid_generated_dataset(dataset_dir):
    assert validate_dataset(dataset_dir) == []


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
