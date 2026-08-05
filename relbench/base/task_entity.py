from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .database import Database
from .table import Table
from .task_base import BaseTask, TaskType


class EntityTask(BaseTask):
    r"""A node prediction task on a dataset.

    Attributes:
        entity_col: The entity column.
        entity_table: The entity table.
        time_col: The time column.
        target_col: The target column.

    Other attributes are inherited from BaseTask.
    """

    entity_col: str
    entity_table: str
    time_col: str
    target_col: str
    task_type: TaskType
    timedelta: pd.Timedelta
    metrics: List[Callable[[NDArray, NDArray], float]]
    num_eval_timestamps: int = 1

    def filter_dangling_entities(self, table: Table, db: Database) -> Table:
        num_entities = len(db.table_dict[self.entity_table])
        filter_mask = table.df[self.entity_col] >= num_entities

        if filter_mask.any():
            table.df = table.df[~filter_mask]

        return table

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = self.metrics

        if target_table is None:
            target_table = self.get_table("test", mask_input_cols=False)

        target = target_table.df[self.target_col].to_numpy()
        if len(pred) != len(target):
            raise ValueError(
                f"The length of pred and target must be the same (got "
                f"{len(pred)} and {len(target)}, respectively)."
            )

        return {fn.__name__: fn(target, pred) for fn in metrics}

    def stats(self) -> Dict[str, Dict[str, Any]]:
        r"""Get train / val / test table statistics for each timestamp
        and the whole table, including number of rows and number of entities.
        Tasks with different task types have different statistics computed:

        BINARY_CLASSIFICATION: Number of positives and negatives.
        REGRESSION: Minimum, maximum, mean, median, quantile 25 and,
            quantile 75 of the target values.
        MULTILABEL_CLASSIFICATION: Mean, minimum and maximum number of
            classes per entity. Number and index of classes having minimum
            and maximum number of classes.
        """
        res = {}
        for split in ["train", "val", "test"]:
            table = self.get_table(split, mask_input_cols=False)
            timestamps = table.df[self.time_col].unique()
            split_stats = {}
            for timestamp in timestamps:
                temp_df = table.df[table.df[self.time_col] == timestamp]
                stats = {
                    "num_rows": len(temp_df),
                    "num_unique_entities": temp_df[self.entity_col].nunique(),
                }
                self._set_stats(temp_df, stats)
                split_stats[str(timestamp)] = stats
            split_stats["total"] = {
                "num_rows": len(table.df),
                "num_unique_entities": table.df[self.entity_col].nunique(),
            }
            self._set_stats(table.df, split_stats["total"])
            res[split] = split_stats
        total_df = pd.concat(
            [
                table.df
                for table in [
                    self.get_table(split, mask_input_cols=False)
                    for split in ["train", "val", "test"]
                ]
                if table is not None
            ]
        )
        res["total"] = {}
        self._set_stats(total_df, res["total"])
        train_uniques = set(self.get_table("train").df[self.entity_col].unique())
        test_uniques = set(
            self.get_table("test", mask_input_cols=False).df[self.entity_col].unique()
        )
        ratio_train_test_entity_overlap = len(
            train_uniques.intersection(test_uniques)
        ) / len(test_uniques)
        res["total"][
            "ratio_train_test_entity_overlap"
        ] = ratio_train_test_entity_overlap
        return res

    def _set_stats(self, df: pd.DataFrame, stats: dict[str, Any]) -> None:
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            self._set_binary_stats(df, stats)
        elif self.task_type == TaskType.REGRESSION:
            self._set_regression_stats(df, stats)
        elif self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            self._set_multilabel_stats(df, stats)
        else:
            raise ValueError(f"Unsupported task type {self.task_type}")

    def _set_binary_stats(self, df: pd.DataFrame, stats: dict[str, Any]) -> None:
        stats["num_positives"] = (df[self.target_col] == 1).sum()
        stats["num_negatives"] = (df[self.target_col] == 0).sum()

    def _set_regression_stats(self, df: pd.DataFrame, stats: dict[str, Any]) -> None:
        stats["min_target"] = df[self.target_col].min()
        stats["max_target"] = df[self.target_col].max()
        stats["mean_target"] = df[self.target_col].mean()
        quantiles = df[self.target_col].quantile([0.25, 0.5, 0.75])
        stats["quantile_25_target"] = quantiles.iloc[0]
        stats["median_target"] = quantiles.iloc[1]
        stats["quantile_75_target"] = quantiles.iloc[2]

    def _set_multilabel_stats(self, df: pd.DataFrame, stats: dict[str, Any]) -> None:
        arr = np.array([row for row in df[self.target_col]])
        arr_row = arr.sum(1)
        stats["mean_num_classes_per_entity"] = round(arr_row.mean(), 4)
        stats["max_num_classes_per_entity"] = arr_row.max()
        stats["min_num_classes_per_entity"] = arr_row.min()
        arr_class = arr.sum(0)
        max_num_class_idx = arr_class.argmax()
        stats["max_num_class_idx"] = max_num_class_idx
        stats["max_num_class_num"] = arr_class[max_num_class_idx]
        min_num_class_idx = arr_class.argmin()
        stats["min_num_class_idx"] = min_num_class_idx
        stats["min_num_class_num"] = arr_class[min_num_class_idx]
