from typing import List, Optional

import numpy as np
import pandas as pd

from relbench.metrics import make_nmae, roc_auc

from .database import Database
from .dataset import Dataset
from .table import Table, is_time_sorted
from .task_base import TaskType, _sort_deterministically
from .task_entity import EntityTask


class AutoCompleteTask(EntityTask):
    r"""Auto complete column task on a dataset. Predict all values in the target column.

    The task is constructed by specifying the entity table, entity column, time column, and target column.
    The target column is removed from the entity table by :meth:`get_db` and kept on
    the task, which is what the label table is built from.

    The entity table needs to have a time column by which the data is split into training and validation set.
    An entity table without a primary key is keyed by row position, so it must be
    sorted by time (rows after ``test_timestamp`` last) for those positions to mean
    the same thing in every view of the database.

    Args:
        dataset: The dataset object.
        task_type: The type of the task.
        entity_table: The name of the entity table.
        target_col: The name of the target column to be predicted.
        remove_columns: List of columns, table pairs to remove from the graph.
    """

    timedelta = pd.Timedelta(seconds=1)
    entity_col: str

    def __init__(
        self,
        dataset: Dataset,
        task_type: TaskType,
        entity_table: str,
        target_col: str,
        remove_columns: Optional[List[tuple]] = None,
        nmae_std: Optional[float] = None,
    ):
        super().__init__(dataset, remove_columns=remove_columns)

        self.task_type = task_type
        self.entity_table = entity_table
        self.target_col = target_col

        db = self.get_db()
        self.entity_col = db.table_dict[entity_table].pkey_col
        assert self.entity_col is not None
        self.time_col = db.table_dict[self.entity_table].time_col

        if self.task_type == TaskType.REGRESSION:
            # NMAE = MAE / std(train target, ddof=1). Resolved once, here: the metric
            # reads the attribute, so evaluating repeatedly costs nothing extra.
            self.nmae_std = (
                float(nmae_std)
                if nmae_std is not None
                else float(self.get_table("train").df[self.target_col].std(ddof=1))
            )
            self.metrics = [make_nmae(lambda: self.nmae_std)]
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            self.metrics = [roc_auc]
            self.num_classes = 2
        else:
            # Multiclass / multilabel tasks are definable and usable, but RelBench
            # provides no evaluator for them -- bring your own via
            # ``task.evaluate(pred, metrics=[...])``.
            self.metrics = []

    def hidden_columns(self) -> List[tuple]:
        return [*self.remove_columns, (self.entity_table, self.target_col)]

    def get_db(self, upto_test_timestamp: bool = True) -> Database:
        r"""The database with the target column (this task's label) taken out.

        The target column is moved off the entity table and kept on the task as
        ``self._removed_cols``, an ``[entity_col, target_col]`` frame that
        ``make_table`` joins back to build labels. Nothing is written to the dataset,
        so other tasks over the same dataset are unaffected.
        """
        db = super().get_db(upto_test_timestamp=upto_test_timestamp)
        table = db.table_dict[self.entity_table]
        col = self.target_col

        if table.pkey_col is None:
            if table.time_col is not None and not is_time_sorted(
                table.df[table.time_col]
            ):
                raise RuntimeError(
                    f"table '{self.entity_table}' has no primary key and is not sorted "
                    f"by '{table.time_col}', so its rows have no identity that survives "
                    "the time cuts; sort it by time (byod/reindex_dataset.py does) "
                    "before defining an autocomplete task on it."
                )
            table.pkey_col = "primary_key"
            table.df["primary_key"] = np.arange(len(table.df))

        if col not in table.df.columns:
            raise ValueError(f"Column {col} not found in table {self.entity_table}.")
        if col in table.fkey_col_to_pkey_table:
            raise ValueError(
                f"Column {col} is a foreign key in table {self.entity_table}. "
                "Only feature columns can be removed."
            )
        if col == table.pkey_col:
            raise ValueError(
                f"Column {col} is the primary key in table {self.entity_table}. "
                "Only feature columns can be removed."
            )

        self._removed_cols = table.df[[table.pkey_col, col]]
        table.df = table.df.drop(columns=[col])
        return db

    def filter_dangling_entities(self, table: Table, db: Database) -> Table:
        num_entities = len(db.table_dict[self.entity_table])
        filter_mask = table.df[self.entity_col] >= num_entities

        if filter_mask.any():
            table.df = table.df[~filter_mask]

        return table

    def _get_table(self, split: str, db: Optional[Database] = None) -> Table:
        r"""The label table of ``split``: every row of the entity table whose time falls
        in the split's window, with the target column as the label.

        train: ``time <= val_timestamp``; val: ``(val_timestamp, test_timestamp]``;
        test: ``> test_timestamp``. ``make_table`` selects ``(start, end]``, so the
        train window starts just before the earliest row in the database.
        """
        full_db = db if db is not None else self.get_db(upto_test_timestamp=False)
        db = full_db if split == "test" else full_db.upto(self.dataset.test_timestamp)

        if split == "train":
            start = db.min_timestamp - pd.Timedelta(1, "ns")
            end = self.dataset.val_timestamp
        elif split == "val":
            start, end = self.dataset.val_timestamp, self.dataset.test_timestamp
        elif split == "test":
            db_max_timestamp = db.max_timestamp
            if db_max_timestamp <= self.dataset.test_timestamp:
                raise RuntimeError("the database has no rows after test_timestamp")
            start, end = self.dataset.test_timestamp, db_max_timestamp
        else:
            raise ValueError(f"unknown split: {split!r}")

        table = self.make_table(db, pd.DatetimeIndex([start, end]))
        table = self.filter_dangling_entities(table, full_db)
        return _sort_deterministically(table, self)

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        entity_table = db.table_dict[self.entity_table].df  # noqa: F841
        entity_table_removed_cols = self._removed_cols  # noqa: F841

        entity_col = db.table_dict[self.entity_table].pkey_col

        # Calculate minimum and maximum timestamps from timestamp_df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        min_timestamp = timestamp_df["timestamp"].min()
        max_timestamp = timestamp_df["timestamp"].max()

        import duckdb  # lazy: keeps `import relbench` importable where duckdb isn't (Pyodide)

        df = duckdb.sql(f"""
            SELECT
                entity_table.{self.time_col},
                entity_table.{entity_col},
                entity_table_removed_cols.{self.target_col}
            FROM
                entity_table
            LEFT JOIN
                entity_table_removed_cols
            ON
                entity_table.{entity_col} = entity_table_removed_cols.{entity_col}
            WHERE
                entity_table.{self.time_col} > '{min_timestamp}' AND
                entity_table.{self.time_col} <= '{max_timestamp}'
            """).df()

        # remove rows where self.target_col is nan
        df = df.dropna(subset=[self.target_col])

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                entity_col: self.entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )
