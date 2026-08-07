from enum import Enum
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .database import Database
from .dataset import Dataset, drop_columns
from .table import Table


def sort_labels(df: pd.DataFrame, sort_cols) -> pd.DataFrame:
    r"""Order a label frame by ``sort_cols``, ignoring unorderable columns.

    This is the canonical order of a label table, and it is applied **when labels are
    written**: a hosted ``train/val/test.parquet`` is expected to already be in this
    order, so loading one does not re-sort it. Generators must call this before
    writing.

    ``make_table`` does not promise an order, and the duckdb queries behind it return
    the same rows in a different order on every run -- so regeneration sorts too, and
    the two agree.
    """

    def _sortable(col) -> bool:
        # A link task's destination column holds lists, which are unorderable (and are
        # not part of the row's identity anyway -- the (time, source) pair is).
        return (
            col is not None
            and col in df.columns
            and not df[col].map(lambda v: isinstance(v, (list, np.ndarray))).any()
        )

    # A link task's destination lists are sets as far as evaluation is concerned
    # (`np.isin`), and duckdb aggregates them in a different order every run -- so
    # order them too, or the same labels still differ cell by cell.
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (list, np.ndarray))).any():
            df[col] = df[col].map(
                lambda v: (
                    type(v)(sorted(v))
                    if isinstance(v, list)
                    else np.sort(v) if isinstance(v, np.ndarray) else v
                )
            )

    cols = [col for col in sort_cols if _sortable(col)]
    if not cols:
        return df
    return df.sort_values(cols, kind="stable").reset_index(drop=True)


def label_sort_cols(task) -> list:
    r"""The canonical sort key of ``task``'s label table, in order.

    Pinned explicitly -- time first, then entity, then link source and destination -- so
    that every producer (regeneration, the generators, the migration tool) sorts
    identically. Deriving it from an unordered mapping would let two paths disagree the
    moment a label table has more than one key column.
    """
    return [
        getattr(task, "time_col", None),
        getattr(task, "entity_col", None),
        getattr(task, "src_entity_col", None),
        getattr(task, "dst_entity_col", None),
    ]


def _sort_deterministically(table: Table, task) -> Table:
    r"""Apply :func:`sort_labels` to a freshly generated label table."""
    table.df = sort_labels(table.df, label_sort_cols(task))
    return table


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
        MULTILABEL_CLASSIFICATION: Multi-label classification task.
        RECOMMENDATION: Recommendation task."
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    RECOMMENDATION = "recommendation"


class BaseTask:
    r"""Base class for a task on a dataset.

    Attributes:
        task_type: The type of the task.
        timedelta: The prediction task at `timestamp` is over the time window
            (timestamp, timestamp + timedelta].
        num_eval_timestamps: The number of evaluation time windows. e.g., test
            time windows are (test_timestamp, test_timestamp + timedelta] ...
            (test_timestamp + (num_eval_timestamps - 1) * timedelta, test_timestamp
            + num_eval_timestamps * timedelta].
        metrics: The metrics to evaluate this task on.

    Inherited by EntityTask and RecommendationTask.
    """

    # To be set by subclass.
    task_type: TaskType
    timedelta: pd.Timedelta
    num_eval_timestamps: int = 1
    metrics: List[Callable[[NDArray, NDArray], float]]

    def __init__(
        self,
        dataset: Dataset,
        remove_columns: Optional[List[tuple]] = None,
    ):
        r"""Create a task object.

        Args:
            dataset: The dataset object on which the task is defined.
            remove_columns: ``(table, column)`` pairs this task must not see -- the
                database columns its label is derived from. Applied by
                :meth:`get_db`, per task; the dataset is never modified.
        """
        self.dataset = dataset
        self.remove_columns = [tuple(pair) for pair in (remove_columns or [])]

        time_diff = self.dataset.test_timestamp - self.dataset.val_timestamp
        if time_diff < self.timedelta:
            raise ValueError(
                f"timedelta cannot be larger than the difference between val "
                f"and test timestamps (timedelta: {self.timedelta}, time "
                f"diff: {time_diff})."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={repr(self.dataset)})"

    def get_db(self, upto_test_timestamp: bool = True) -> Database:
        r"""The database as this task is allowed to see it.

        The dataset's database with this task's ``remove_columns`` dropped. Pure and
        uncached, like :meth:`Dataset.get_db` -- keep the returned object rather than
        calling this repeatedly.
        """
        return drop_columns(
            self.dataset.get_db(upto_test_timestamp=upto_test_timestamp),
            self.remove_columns,
        )

    def _split_db(self, split: str, db: Optional[Database] = None) -> Database:
        r"""The database to generate ``split``'s labels from.

        Test labels live after ``test_timestamp``, so they need the full database;
        train/val labels must not see past it.

        ``db``, when given, must be this task's *full* (``upto_test_timestamp=False``)
        database -- it is filtered here as needed, but never re-filtered for columns.
        Passing it lets a caller build the database once and generate every split from
        it, instead of rebuilding it per split.
        """
        if db is None:
            return self.get_db(upto_test_timestamp=split != "test")
        return db if split == "test" else db.upto(self.dataset.test_timestamp)

    def make_table(
        self,
        db: Database,
        timestamps: "pd.Series[pd.Timestamp]",
    ) -> Table:
        r"""Make a table using the task definition.

        Args:
            db: The database object to use for (historical) ground truth.
            timestamps: Collection of timestamps to compute labels for. A label can be
            computed for a timestamp using historical data
            upto this timestamp in the database.

        To be implemented by subclass. The table rows need not be ordered
        deterministically.
        """

        raise NotImplementedError

    def _get_table(self, split: str, db: Optional[Database] = None) -> Table:
        r"""Helper function to get a table for a split."""

        db = self._split_db(split, db)

        if split == "train":
            start = self.dataset.val_timestamp - self.timedelta
            end = db.min_timestamp
            freq = -self.timedelta

        elif split == "val":
            if self.dataset.val_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "val timestamp + timedelta is larger than max timestamp! "
                    "This would cause val labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.val_timestamp
            end = min(
                self.dataset.val_timestamp
                + self.timedelta * (self.num_eval_timestamps - 1),
                self.dataset.test_timestamp - self.timedelta,
            )
            freq = self.timedelta

        elif split == "test":
            # Read once: `max_timestamp` is uncached and scans every table.
            db_max_timestamp = db.max_timestamp
            if self.dataset.test_timestamp + self.timedelta > db_max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.test_timestamp
            end = min(
                self.dataset.test_timestamp
                + self.timedelta * (self.num_eval_timestamps - 1),
                db_max_timestamp - self.timedelta,
            )
            freq = self.timedelta

        timestamps = pd.date_range(start=start, end=end, freq=freq)

        if split == "train" and len(timestamps) < 3:
            raise RuntimeError(
                f"The number of training time frames is too few. "
                f"({len(timestamps)} given)"
            )

        table = self.make_table(db, timestamps)
        # Dangling-entity filtering is defined against the database upto
        # test_timestamp for every split, including test.
        table = self.filter_dangling_entities(
            table, db if split != "test" else db.upto(self.dataset.test_timestamp)
        )

        return _sort_deterministically(table, self)

    def get_table(
        self,
        split: str,
        mask_input_cols: Optional[bool] = None,
        db: Optional[Database] = None,
    ) -> Table:
        r"""Get a table for a split.

        Args:
            split: The split to get the table for. One of "train", "val", or "test".
            mask_input_cols: If True, keep only the input columns in the table. If
                None, mask the input columns only for the test split. This helps
                prevent data leakage.
            db: Optional pre-built database to generate labels from, so several
                splits can share one build. It must be *this task's* full database,
                i.e. ``task.get_db(upto_test_timestamp=False)`` -- it is used as
                given, so a dataset-level database would not have this task's
                ``remove_columns`` applied.

        Returns:
            The task table for the split.

        Nothing is cached: keep the table you get back. For a task built with
        ``regenerate=True`` each call re-runs the label SQL, so pass ``db`` and hold
        the result.
        """
        if mask_input_cols is None:
            mask_input_cols = split == "test"

        table = self._get_table(split, db)
        if mask_input_cols:
            table = self._mask_input_cols(table)
        return table

    def _mask_input_cols(self, table: Table) -> Table:
        input_cols = [
            table.time_col,
            *table.fkey_col_to_pkey_table.keys(),
        ]
        input_cols = [col for col in input_cols if col is not None]
        return Table(
            df=table.df[input_cols],
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    def filter_dangling_entities(self, table: Table, db: Database) -> Table:
        r"""Filter out dangling entities from a table, against ``db``.

        Implemented by EntityTask and RecommendationTask.
        """
        raise NotImplementedError

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ):
        r"""Evaluate predictions on the task.

        Args:
            pred: Predictions as a numpy array.
            target_table: The target table. If None, use the test table.
            metrics: The metrics to evaluate the prediction table. If None, use
                the default metrics for the task.

        Implemented by EntityTask and RecommendationTask.
        """
        raise NotImplementedError
