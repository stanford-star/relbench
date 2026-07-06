import time
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from numpy.typing import NDArray

from .database import Database
from .dataset import Dataset
from .table import Table


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
        MULTILABEL_CLASSIFICATION: Multi-label classification task.
        LINK_PREDICTION: Link prediction task."
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    LINK_PREDICTION = "link_prediction"


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
        cache_dir: Optional[str] = None,
    ):
        r"""Create a task object.

        Args:
            dataset: The dataset object on which the task is defined.
            cache_dir: A directory for caching the task table objects. If specified,
                we will either process and cache the file (if not available) or use
                the cached file. If None, we will not use cached file and re-process
                everything from scratch without saving the cache.
        """
        self.dataset = dataset
        self.cache_dir = cache_dir

        time_diff = self.dataset.test_timestamp - self.dataset.val_timestamp
        if time_diff < self.timedelta:
            raise ValueError(
                f"timedelta cannot be larger than the difference between val "
                f"and test timestamps (timedelta: {self.timedelta}, time "
                f"diff: {time_diff})."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset={repr(self.dataset)})"

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

    def _get_table(self, split: str) -> Table:
        r"""Helper function to get a table for a split."""
        db = self.dataset.get_db(upto_test_timestamp=split != "test")

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
            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta is larger than max timestamp! "
                    "This would cause test labels to be generated with "
                    "insufficient aggregation time."
                )

            start = self.dataset.test_timestamp
            end = min(
                self.dataset.test_timestamp
                + self.timedelta * (self.num_eval_timestamps - 1),
                db.max_timestamp - self.timedelta,
            )
            freq = self.timedelta

        timestamps = pd.date_range(start=start, end=end, freq=freq)

        if split == "train" and len(timestamps) < 3:
            raise RuntimeError(
                f"The number of training time frames is too few. "
                f"({len(timestamps)} given)"
            )

        table = self.make_table(db, timestamps)
        table = self.filter_dangling_entities(table)

        return table

    @lru_cache(maxsize=None)
    def get_table(self, split, mask_input_cols=None):
        r"""Get a table for a split.

        Args:
            split: The split to get the table for. One of "train", "val", or "test".
            mask_input_cols: If True, keep only the input columns in the table. If
                None, mask the input columns only for the test split. This helps
                prevent data leakage.

        Returns:
            The task table for the split.

        The table is cached in memory.
        """
        if mask_input_cols is None:
            mask_input_cols = split == "test"

        table_path = f"{self.cache_dir}/{split}.parquet"
        if self.cache_dir and Path(table_path).exists():
            table = Table.load(table_path)
        else:
            print(f"Making task table for {split} split from scratch...")
            # print(
            #     "(You can also use `get_task(..., download=True)` "
            #     "for tasks prepared by the RelBench team.)"
            # )
            tic = time.time()
            table = self._get_table(split)
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

            if self.cache_dir:
                table.save(table_path)

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

    def filter_dangling_entities(self, table: Table) -> Table:
        r"""Filter out dangling entities from a table.

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
