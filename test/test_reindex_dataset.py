import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest

from relbench.base import Database, Table
from relbench.load import load_dataset

SCRIPT = Path(__file__).resolve().parents[1] / "byod" / "reindex_dataset.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("reindex_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _string_keyed_db():
    users = pd.DataFrame({"uid": ["u2", "u0", "u1"], "age": [30, 20, 40]})
    events = pd.DataFrame(
        {
            "eid": ["e1", "e0", "e2"],
            "uid": ["u1", "u9", None],
            "t": pd.to_datetime(["2020-01-02", "2020-01-01", "2020-01-03"]),
        }
    )
    return Database(
        {
            "users": Table(users, {}, pkey_col="uid"),
            "events": Table(events, {"uid": "users"}, pkey_col="eid", time_col="t"),
        }
    )


def test_reindex_maps_keys_and_nulls_dangling(script):
    db = _string_keyed_db()
    index_maps, dangling = script.reindex(db)
    users, events = db.table_dict["users"].df, db.table_dict["events"].df
    assert users["uid"].tolist() == [0, 1, 2]
    assert index_maps["users"].to_dict() == {"u2": 0, "u0": 1, "u1": 2}
    assert events["t"].is_monotonic_increasing
    assert events["eid"].tolist() == [0, 1, 2]
    assert events["uid"].tolist()[0] is pd.NA and events["uid"].tolist()[1] == 2
    assert dangling == {"users": 0, "events": 1}


def test_reindex_rejects_duplicate_keys(script):
    db = Database({"t": Table(pd.DataFrame({"id": ["a", "a"]}), {}, pkey_col="id")})
    with pytest.raises(RuntimeError, match="duplicates"):
        script.reindex(db)


def test_reindex_is_identity_on_normalized_data(script, dataset, dataset_dir, tmp_path):
    db = dataset.get_db(upto_test_timestamp=False)
    before = {name: t.df.copy() for name, t in db.table_dict.items()}
    index_maps, dangling = script.reindex(db)
    for name, t in db.table_dict.items():
        pd.testing.assert_frame_equal(t.df, before[name])
    assert all(script.is_identity(m) for m in index_maps.values())
    out = tmp_path / "copy"
    shutil.copytree(dataset_dir, out)
    script.main.__globals__["sys"].argv = ["reindex_dataset.py", str(out)]
    script.main()
    for name in db.table_dict:
        pd.testing.assert_frame_equal(
            pd.read_parquet(out / "db" / f"{name}.parquet"),
            pd.read_parquet(dataset_dir / "db" / f"{name}.parquet"),
        )
    assert set(load_dataset(out).get_db().table_dict) == set(db.table_dict)
