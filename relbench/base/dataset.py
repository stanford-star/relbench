import warnings

import numpy as np
import pandas as pd

from .database import Database


class Dataset:
    r"""A dataset is a database with validation and test timestamps defined for it.

    Attributes:
        val_timestamp: Rows upto this timestamp (inclusive) can be input for validation.
        test_timestamp: Rows upto this timestamp (inclusive) can be input for testing.

    Validation split of a task involves predicting the target variable for a
    time period after val_timestamp (exclusive) using data upto val_timestamp.
    Similarly for test_timestamp.

    ``get_db`` is a pure function of the underlying data: it holds no state, caches
    nothing, and never depends on which tasks have been loaded. Materializing a
    database is expensive, so **callers should keep the object they get back** rather
    than calling ``get_db`` repeatedly. For a view with the columns a task must not
    see removed, use :meth:`relbench.base.BaseTask.get_db`.
    """

    # To be set by subclass.
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def validate_and_correct_db(self, db):
        r"""Validate and correct input db in-place.

        Removing rows after test_timestamp can result in dangling foreign keys.
        """
        # Validate that all primary keys are consecutively index.

        for table_name, table in db.table_dict.items():
            if table.pkey_col is not None:
                ser = table.df[table.pkey_col]
                if not (ser.values == np.arange(len(ser))).all():
                    raise RuntimeError(
                        f"The primary key column {table.pkey_col} of table "
                        f"{table_name} is not consecutively indexed (0..n-1 in row "
                        "order), which RelBench requires. Hosted datasets are; for your "
                        "own data run `python byod/reindex_dataset.py <dataset_dir>` "
                        "(see byod/README.md)."
                    )

        # Discard any foreign keys that are larger than primary key table as
        # dangling foreign keys (represented as None).
        for table_name, table in db.table_dict.items():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                num_pkeys = len(db.table_dict[pkey_table_name])
                mask = table.df[fkey_col] >= num_pkeys
                if mask.any():
                    table.df.loc[mask, fkey_col] = None

    def get_db(self, upto_test_timestamp: bool = True) -> Database:
        r"""Build and return the database object.

        Args:
            upto_test_timestamp: If True, only return rows upto test_timestamp.

        Returns:
            Database: A freshly built database object.

        `upto_test_timestamp` is True by default to prevent test leakage.

        Nothing is cached: every call rebuilds the database. Hold on to the returned
        object instead of calling this repeatedly.
        """
        db = self.make_db()
        db.reindex_pkeys_and_fkeys()

        if upto_test_timestamp:
            db = db.upto(self.test_timestamp)

        self.validate_and_correct_db(db)
        return db

    def make_db(self) -> Database:
        r"""Make the database object from scratch, i.e. using raw data sources.

        To be implemented by subclass.
        """
        raise NotImplementedError


def drop_columns(db: Database, remove_columns) -> Database:
    r"""Drop ``(table, column)`` pairs from ``db`` in place, and return it.

    These are the columns a task must not see -- the ones its label is, or is derived
    from. Applied per task in :meth:`relbench.base.BaseTask.get_db`, never stored on
    the dataset, so two tasks over one dataset cannot disturb each other.
    """
    for table, remove_col in remove_columns:
        if table not in db.table_dict:
            warnings.warn(
                f"Table {table} not in the database. "
                f"Skipping removal of column {remove_col}.",
                stacklevel=2,
            )
            continue
        if remove_col in db.table_dict[table].df.columns:
            db.table_dict[table].df = db.table_dict[table].df.drop(columns=[remove_col])
        else:
            warnings.warn(
                f"Column {remove_col} not found in table {table}. "
                "Skipping removal from this table.",
                stacklevel=2,
            )
    return db
