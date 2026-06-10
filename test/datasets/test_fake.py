import copy

from relbench.datasets.fake import FakeDataset


def test_fake_reviews_dataset(make_churn_task):
    dataset = FakeDataset()
    assert dataset.get_db().max_timestamp <= dataset.test_timestamp
    assert str(dataset) == "FakeDataset()"

    task = make_churn_task(dataset)

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    for table in [train_table, val_table]:
        assert set(table.df.columns) >= {
            "timestamp",
            "customer_id",
            "churn",
        }

    test_table = task.get_table("test")  # test split masks the target column
    assert set(test_table.df.columns) == {
        "timestamp",
        "customer_id",
    }


def test_reindex():
    dataset = FakeDataset()
    db = dataset.make_db()
    db_indexed = copy.deepcopy(db)
    db_indexed.reindex_pkeys_and_fkeys()
    for table_name in db.table_dict.keys():
        table = db.table_dict[table_name]
        table_indexed = db_indexed.table_dict[table_name]
        if table.pkey_col is not None:
            arr = table.df[table.pkey_col].apply(lambda x: int(x.split("_")[-1]))
            arr_indexed = table_indexed.df[table.pkey_col]
            assert (arr == arr_indexed).all()
        for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
            num_pkeys = len(db.table_dict[pkey_table])
            arr = table.df[fkey_col].apply(lambda x: int(x.split("_")[-1]))
            mask = arr < num_pkeys
            arr_indexed = table_indexed.df[fkey_col]
            assert (arr[mask] == arr_indexed[mask]).all()
