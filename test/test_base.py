import pandas as pd
import pytest

from relbench.base import Database, Table


def test_table_upto_and_bounds():
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "x": [1, 2, 3],
        }
    )
    table = Table(df=df, fkey_col_to_pkey_table={}, time_col="t")
    assert len(table) == 3
    assert table.min_timestamp == pd.Timestamp("2020-01-01")
    assert table.max_timestamp == pd.Timestamp("2020-01-03")
    assert table.upto(pd.Timestamp("2020-01-02")).df["x"].tolist() == [1, 2]

    timeless = Table(df=df, fkey_col_to_pkey_table={})
    assert timeless.upto(pd.Timestamp("2020-01-01")) is timeless
    with pytest.raises(ValueError):
        timeless.min_timestamp


def test_reindex_pkeys_and_fkeys(dataset):
    db = dataset.make_db()
    db.reindex_pkeys_and_fkeys()
    customer = db.table_dict["customer"].df
    assert customer["customer_id"].tolist() == list(range(dataset.num_customers))
    review = db.table_dict["review"].df
    assert review["customer_id"].isna().any()
    assert review["customer_id"].dropna().between(0, dataset.num_customers - 1).all()
    assert review["product_id"].between(0, dataset.num_products - 1).all()


def test_duplicate_pkey_raises():
    table = Table(df=pd.DataFrame({"id": ["a", "a"]}), fkey_col_to_pkey_table={})
    table.pkey_col = "id"
    with pytest.raises(RuntimeError, match="duplicated"):
        Database({"t": table}).reindex_pkeys_and_fkeys()


def test_get_db_is_pure_and_hides_rows_after_test_timestamp(dataset):
    db = dataset.get_db()
    full = dataset.get_db(upto_test_timestamp=False)
    assert db.max_timestamp <= dataset.test_timestamp < full.max_timestamp
    assert len(full.table_dict["review"]) == dataset.num_reviews
    assert len(db.table_dict["review"]) < dataset.num_reviews
    assert db.min_timestamp == full.min_timestamp
    again = dataset.get_db()
    for name, table in db.table_dict.items():
        pd.testing.assert_frame_equal(table.df, again.table_dict[name].df)
        if table.pkey_col is not None:
            assert table.df[table.pkey_col].tolist() == list(range(len(table)))
