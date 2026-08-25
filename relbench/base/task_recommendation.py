from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .database import Database
from .dataset import Dataset
from .table import Table
from .task_base import BaseTask, TaskType


class RecommendationTask(BaseTask):
    r"""A recommendation task on a dataset.

    Attributes:
        src_entity_col: The source entity column.
        src_entity_table: The source entity table.
        dst_entity_col: The destination entity column.
        dst_entity_table: The destination entity table.
        time_col: The time column.
        eval_k: k for eval@k metrics.

    Other attributes are inherited from BaseTask.
    """

    src_entity_col: str
    src_entity_table: str
    dst_entity_col: str
    dst_entity_table: str
    time_col: str
    eval_k: int
    task_type: TaskType
    timedelta: pd.Timedelta
    metrics: List[Callable[[NDArray, NDArray], float]]
    num_eval_timestamps: int = 1

    def __init__(
        self,
        dataset: Dataset,
        remove_columns: Optional[List[tuple]] = None,
    ):
        if self.num_eval_timestamps != 1:
            raise NotImplementedError(
                "RecommendationTask currently only supports num_eval_timestamps=1."
            )
        super().__init__(dataset, remove_columns)

    def filter_dangling_entities(self, table: Table, db: Database) -> Table:
        num_src_nodes = len(db.table_dict[self.src_entity_table])
        num_dst_nodes = len(db.table_dict[self.dst_entity_table])

        # filter dangling destination entities from a list
        table.df[self.dst_entity_col] = table.df[self.dst_entity_col].apply(
            lambda x: [i for i in x if i < num_dst_nodes]
        )

        # filter dangling source entities and empty list (after above filtering)
        filter_mask = (table.df[self.src_entity_col] >= num_src_nodes) | (
            ~table.df[self.dst_entity_col].map(bool)
        )

        if filter_mask.any():
            table.df = table.df[~filter_mask]
            table.df = table.df.reset_index(drop=True)

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

        expected_pred_shape = (len(target_table), self.eval_k)
        if pred.shape != expected_pred_shape:
            raise ValueError(
                f"The shape of pred must be {expected_pred_shape}, but "
                f"{pred.shape} given."
            )

        if len(target_table) == 0:
            return {fn.__name__: float("nan") for fn in metrics}

        pred_isin_list = []
        dst_count_list = []
        for true_dst_nodes, pred_dst_nodes in zip(
            target_table.df[self.dst_entity_col],
            pred,
        ):
            pred_isin_list.append(
                np.isin(np.array(pred_dst_nodes), np.array(true_dst_nodes))
            )
            dst_count_list.append(len(true_dst_nodes))
        pred_isin = np.stack(pred_isin_list)
        dst_count = np.array(dst_count_list)

        return {fn.__name__: fn(pred_isin, dst_count) for fn in metrics}

    def stats(self) -> Dict[str, Dict[str, int]]:
        r"""Get train / val / test table statistics for each timestamp and the whole
        table, including number of unique source entities, number of unique destination
        entities, number of destination entities and number of rows."""

        res = {}
        for split in ["train", "val", "test"]:
            split_stats = {}
            table = self.get_table(split, mask_input_cols=False)
            timestamps = table.df[self.time_col].unique()
            for timestamp in timestamps:
                temp_df = table.df[table.df[self.time_col] == timestamp]
                (
                    num_unique_src_entities,
                    num_unique_dst_entities,
                    num_dst_entities,
                    num_rows,
                ) = self._get_stats(temp_df)
                split_stats[str(timestamp)] = {
                    "num_unique_src_entities": num_unique_src_entities,
                    "num_unique_dst_entities": num_unique_dst_entities,
                    "num_dst_entities": num_dst_entities,
                    "num_rows": num_rows,
                }

            (
                num_unique_src_entities,
                num_unique_dst_entities,
                num_dst_entities,
                num_rows,
            ) = self._get_stats(table.df)
            split_stats["total"] = {
                "num_unique_src_entities": num_unique_src_entities,
                "num_unique_dst_entities": num_unique_dst_entities,
                "num_dst_entities": num_dst_entities,
                "num_rows": num_rows,
            }
            res[split] = split_stats
        total_df = pd.concat(
            [
                table.df
                for table in [
                    self.get_table("train"),
                    self.get_table("val"),
                    self.get_table("test"),
                ]
                if table is not None
            ]
        )
        num_unique_src_entities, num_unique_dst_entities, num_dst_entities, num_rows = (
            self._get_stats(total_df)
        )
        res["total"] = {
            "num_unique_src_entities": num_unique_src_entities,
            "num_unique_dst_entities": num_unique_dst_entities,
            "num_dst_entities": num_dst_entities,
            "num_rows": num_rows,
        }
        train_uniques = set(self.get_table("train").df[self.src_entity_col].unique())
        test_uniques = set(self.get_table("test").df[self.src_entity_col].unique())
        ratio_train_test_entity_overlap = len(
            train_uniques.intersection(test_uniques)
        ) / len(test_uniques)
        res["total"][
            "ratio_train_test_entity_overlap"
        ] = ratio_train_test_entity_overlap
        return res

    def _get_stats(self, df: pd.DataFrame) -> List[int]:
        num_unique_src_entities = df[self.src_entity_col].nunique()
        num_unique_dst_entities = len(
            set(value for row in df[self.dst_entity_col] for value in row)
        )
        num_dst_entities = sum(len(row) for row in df[self.dst_entity_col])
        num_rows = len(df)
        return (
            num_unique_src_entities,
            num_unique_dst_entities,
            num_dst_entities,
            num_rows,
        )
