import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

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
    """

    # To be set by subclass.
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp

    # For predict column task.
    target_col: Optional[str]
    entity_table: Optional[str]
    remove_columns: list[tuple[str, str]]

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ) -> None:
        r"""Create a dataset object.

        Args:
            cache_dir: A directory for caching the database object. If specified,
                we will either process and cache the file (if not available) or use
                the cached file. If None, we will not use cached file and re-process
                everything from scratch without saving the cache.
        """

        self.cache_dir = cache_dir

        self.target_col = None
        self.entity_table = None
        self.remove_columns = []

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
                        f"{table_name} is not consecutively index."
                    )

        # Discard any foreign keys that are larger than primary key table as
        # dangling foreign keys (represented as None).
        for table_name, table in db.table_dict.items():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                num_pkeys = len(db.table_dict[pkey_table_name])
                mask = table.df[fkey_col] >= num_pkeys
                if mask.any():
                    table.df.loc[mask, fkey_col] = None

    @lru_cache(maxsize=None)
    def get_db(self, upto_test_timestamp=True) -> Database:
        r"""Return the database object.

        The returned database object is cached in memory.

        Args:
            upto_test_timestamp: If True, only return rows upto test_timestamp.

        Returns:
            Database: The database object.

        `upto_test_timestamp` is True by default to prevent test leakage.
        """

        db_path = f"{self.cache_dir}/db"
        if self.cache_dir and Path(db_path).exists() and any(Path(db_path).iterdir()):
            print(f"Loading Database object from {db_path}...")
            tic = time.time()
            db = Database.load(db_path)
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

        else:
            print("Making Database object from scratch...")
            print(
                "(For datasets prepared by the RelBench team, use "
                "`relbench.load_dataset(name)`.)"
            )
            tic = time.time()
            db = self.make_db()
            db.reindex_pkeys_and_fkeys()
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

            if self.cache_dir:
                print(f"Caching Database object to {db_path}...")
                tic = time.time()
                db.save(db_path)
                toc = time.time()
                print(f"Done in {toc - tic:.2f} seconds.")

        if upto_test_timestamp:
            db = db.upto(self.test_timestamp)

        self.validate_and_correct_db(db)

        if self.target_col or self.remove_columns:
            # Get the modified db with the target column removed
            db = self.get_modified_db(db)

        return db

    def get_modified_db(self, db) -> Database:
        r"""Get the modified db with the target column and any ``remove_columns``
        dropped.

        The target column is saved to `db.table_dict[table_name].removed_cols`
        and the column is dropped from the table. Only autocomplete tasks set
        ``target_col`` here (their label *is* a column of the entity table);
        ``remove_columns`` applies to every task kind and is handled below,
        independently.
        Args:
            db: The database object.

        Returns:
            Database: The modified database object.
        """

        # Remove the target column from the source entity table
        # Ensure the entity table has a primary key if not add one
        if self.target_col:
            table_name = self.entity_table
            col = self.target_col

            if db.table_dict[table_name].pkey_col is None:
                db.table_dict[table_name].pkey_col = "primary_key"
                db.table_dict[table_name].df["primary_key"] = np.arange(
                    len(db.table_dict[table_name].df)
                )

            if col not in db.table_dict[table_name].df.columns:
                raise ValueError(f"Column {col} not found in table {table_name}.")
            if col in db.table_dict[table_name].fkey_col_to_pkey_table.keys():
                raise ValueError(
                    f"Column {col} is a foreign key in table {table_name}. Only feature columns can be removed."
                )
            if col == db.table_dict[table_name].pkey_col:
                raise ValueError(
                    f"Column {col} is the primary key in table {table_name}. Only feature columns can be removed."
                )

            # save the columns to be dropped
            id_keys = []
            if db.table_dict[table_name].pkey_col:
                id_keys.append(db.table_dict[table_name].pkey_col)
            else:
                # add primary key to table_name if it doesn't have one
                db.table_dict[table_name].df["primary_key"] = np.arange(
                    len(db.table_dict[table_name].df)
                )
                id_keys.append("primary_key")
                db.table_dict[table_name].pkey_col = "primary_key"

            # Save the target column to be dropped
            db.table_dict[table_name].removed_cols = db.table_dict[table_name].df[
                id_keys + [col]
            ]
            # drop the columns
            db.table_dict[table_name].df = db.table_dict[table_name].df.drop(
                columns=[col]
            )

        # Remove any other columns marked for removal. Deliberately outside the
        # `target_col` branch: a forecast or external task also has to hide a column its
        # label was derived from, and it has no `target_col` on the dataset to gate on.
        for table, remove_col in self.remove_columns:
            if table not in db.table_dict:
                print(
                    f"Table {table} not in the database. "
                    "Skipping removal of column {remove_col}."
                )
                continue
            if remove_col in db.table_dict[table].df.columns:
                # If the column is in the table, remove it
                db.table_dict[table].df = db.table_dict[table].df.drop(
                    columns=[remove_col]
                )
            else:
                print(
                    f"Column {remove_col} not found in table {table}. "
                    "Skipping removal from this table."
                )

        return db

    def make_db(self) -> Database:
        r"""Make the database object from scratch, i.e. using raw data sources.

        To be implemented by subclass.
        """
        raise NotImplementedError
