from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from relbench.base import BaseTask, Dataset, RecommendationTask, Table, TaskType
from relbench.metrics import (
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
)


def _to_unix_seconds(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts, utc=True)
    return (ts.astype("int64").to_numpy(copy=False) // 1_000_000_000).astype(
        np.int64, copy=False
    )


def _tgb_eval_hits_and_mrr(
    y_pred_pos: np.ndarray,
    y_pred_neg: np.ndarray,
    *,
    k_value: int,
) -> dict[str, float]:
    r"""Compute one-vs-many Hits@k and MRR with tie-aware ranking."""
    y_pred_pos = np.asarray(y_pred_pos).reshape(-1, 1)
    y_pred_neg = np.asarray(y_pred_neg)
    if y_pred_neg.ndim != 2 or y_pred_neg.shape[0] != y_pred_pos.shape[0]:
        raise ValueError(
            "Expected y_pred_neg with shape (N, num_neg) matching y_pred_pos (N,). "
            f"Got y_pred_pos={y_pred_pos.shape}, y_pred_neg={y_pred_neg.shape}."
        )

    optimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits_k = (ranking_list <= int(k_value)).astype(np.float32).mean().item()
    mrr = (1.0 / ranking_list.astype(np.float32)).mean().item()
    return {f"hits@{int(k_value)}": hits_k, "mrr": mrr}


@dataclass(frozen=True)
class TGBLinkPredSpec:
    r"""Specification for a TGB link prediction task over a single event table."""

    event_table: str
    src_col: str = "src_id"
    dst_col: str = "dst_id"
    time_col: str = "event_ts"


class TGBOneVsManyLinkPredTask(BaseTask):
    r"""TGB link prediction with official one-vs-many MRR/Hits@k evaluation."""

    task_type = TaskType.LINK_PREDICTION
    timedelta = pd.Timedelta(seconds=1)
    metrics = []
    num_eval_timestamps = 1

    def __init__(
        self,
        dataset: Dataset,
        *,
        spec: TGBLinkPredSpec,
        k_value: int = 10,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.spec = spec
        self.k_value = int(k_value)

        self.time_col = spec.time_col
        self.src_entity_col = spec.src_col
        self.dst_entity_col = spec.dst_col

        db = dataset.get_db()
        if spec.event_table not in db.table_dict:
            raise ValueError(
                f"Event table '{spec.event_table}' not found in dataset db. "
                f"Available tables: {sorted(db.table_dict.keys())}"
            )
        event_tbl = db.table_dict[spec.event_table]
        self.src_entity_table = event_tbl.fkey_col_to_pkey_table.get(spec.src_col)
        self.dst_entity_table = event_tbl.fkey_col_to_pkey_table.get(spec.dst_col)
        if self.src_entity_table is None or self.dst_entity_table is None:
            raise ValueError(
                f"Event table '{spec.event_table}' must have fkeys for "
                f"'{spec.src_col}' and '{spec.dst_col}'. Got {event_tbl.fkey_col_to_pkey_table}."
            )

        m = re.fullmatch(r"events_edge_type_(\d+)", spec.event_table)
        self.edge_type_id: Optional[int] = int(m.group(1)) if m else None

        super().__init__(dataset, cache_dir=cache_dir)

    def make_table(self, db, timestamps):  # pragma: no cover
        raise RuntimeError(
            "TGBOneVsManyLinkPredTask expects precomputed task tables "
            "(train/val/test.parquet) and overrides _get_table()."
        )

    def filter_dangling_entities(self, table: Table) -> Table:
        if self.src_entity_table:
            table.df = table.df[
                table.df[self.src_entity_col]
                < len(self.dataset.get_db().table_dict[self.src_entity_table])
            ]
        if self.dst_entity_table:
            table.df = table.df[
                table.df[self.dst_entity_col]
                < len(self.dataset.get_db().table_dict[self.dst_entity_table])
            ]
        table.df = table.df.reset_index(drop=True)
        return table

    def _get_table(self, split: str) -> Table:
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unknown split '{split}'.")
        table_path = f"{self.cache_dir}/{split}.parquet"
        if not self.cache_dir or not Path(table_path).exists():
            raise RuntimeError(
                "Exact TGB parity requires precomputed task tables. "
                f"Missing {table_path}. Use download=True or place the parquet files in cache."
            )

        table = Table.load(table_path)
        return self.filter_dangling_entities(table)

    def _negatives_path(self, split: str) -> Path:
        if split not in ["val", "test"]:
            raise ValueError("Negative samples are only defined for val/test splits.")
        if self.dataset.cache_dir is None:
            raise RuntimeError("Dataset has no cache_dir; cannot locate TGB negatives.")
        return Path(self.dataset.cache_dir) / "negatives" / f"{split}_ns.pkl"

    @lru_cache(maxsize=1)
    def _load_negatives(self, split: str) -> dict[Any, Any]:
        path = self._negatives_path(split)
        if not path.exists():
            raise RuntimeError(
                f"Missing TGB negative samples at {path}. "
                "To match TGB exactly, include the official negative sample pickle "
                "in the dataset download under `negatives/`."
            )
        with path.open("rb") as f:
            return pickle.load(f)

    def _mapping_paths(self) -> dict[str, Path]:
        if self.dataset.cache_dir is None:
            raise RuntimeError("Dataset has no cache_dir; cannot locate mapping files.")
        base = Path(self.dataset.cache_dir) / "mappings"
        return {
            "node_type": base / "node_type.npy",
            "local_id": base / "local_id.npy",
        }

    @lru_cache(maxsize=None)
    def _load_global_to_local(self) -> tuple[np.ndarray, np.ndarray]:
        paths = self._mapping_paths()
        node_type_path = paths["node_type"]
        local_id_path = paths["local_id"]
        if not node_type_path.exists() or not local_id_path.exists():
            raise RuntimeError(
                "Missing heterogeneous mapping files. Expected:\n"
                f"- {node_type_path}\n- {local_id_path}\n"
                "These are required to map TGB global ids to RelBench per-type local ids."
            )
        node_type = np.load(node_type_path)
        local_id = np.load(local_id_path)
        return node_type.astype(np.int64, copy=False), local_id.astype(
            np.int64, copy=False
        )

    @lru_cache(maxsize=None)
    def _load_local_to_global(self, node_type_id: int) -> np.ndarray:
        if self.dataset.cache_dir is None:
            raise RuntimeError("Dataset has no cache_dir; cannot locate mapping files.")
        p = (
            Path(self.dataset.cache_dir)
            / "mappings"
            / f"globals_type_{int(node_type_id)}.npy"
        )
        if not p.exists():
            raise RuntimeError(
                f"Missing mapping file {p}. This is required to map local ids "
                "back to TGB global ids for negative-sample lookup."
            )
        return np.load(p).astype(np.int64, copy=False)

    def _node_type_id_from_table(self, table_name: str) -> Optional[int]:
        m = re.fullmatch(r"nodes_type_(\d+)", str(table_name))
        return int(m.group(1)) if m else None

    def _bipartite_offset(self) -> Optional[int]:
        if (
            self.src_entity_table == "src_nodes"
            and self.dst_entity_table == "dst_nodes"
        ):
            return len(self.dataset.get_db().table_dict["src_nodes"])
        return None

    def _src_local_to_global(self, src_local: np.ndarray) -> np.ndarray:
        src_type = self._node_type_id_from_table(self.src_entity_table)
        if src_type is None:
            return src_local.astype(np.int64, copy=False)
        globals_ = self._load_local_to_global(src_type)
        return globals_[src_local.astype(np.int64, copy=False)]

    def _dst_global_to_local(self, dst_global: np.ndarray) -> np.ndarray:
        dst_type = self._node_type_id_from_table(self.dst_entity_table)
        if dst_type is None:
            offset = self._bipartite_offset()
            if offset is None:
                return dst_global.astype(np.int64, copy=False)
            dst_global = dst_global.astype(np.int64, copy=False)
            out = dst_global - int(offset)
            if (out < 0).any():
                raise RuntimeError(
                    "Bipartite negatives contain ids outside destination range."
                )
            return out.astype(np.int64, copy=False)
        node_type, local_id = self._load_global_to_local()
        dst_global = dst_global.astype(np.int64, copy=False)
        bad = node_type[dst_global] != dst_type
        if bad.any():
            raise RuntimeError(
                "Negative samples contain destination nodes of unexpected type."
            )
        return local_id[dst_global].astype(np.int64, copy=False)

    def get_negative_dsts_local(
        self, *, split: str, table: Optional[Table] = None
    ) -> list[np.ndarray]:
        r"""Return negative destination ids (local to dst entity table) for each row.

        This is intended to help users reproduce the TGB evaluation protocol, i.e.,
        score the true destination vs the provided negatives.
        """
        if split not in ["val", "test"]:
            raise ValueError("Negatives are only defined for val/test splits.")
        if table is None:
            table = self.get_table(split, mask_input_cols=False)

        df = table.df
        ts_s = _to_unix_seconds(df[self.time_col])
        src_local = df[self.src_entity_col].astype("int64").to_numpy()
        src_global = self._src_local_to_global(src_local)
        neg_dict = self._load_negatives(split)

        negs_local: list[np.ndarray] = []
        if self.edge_type_id is None:
            dst_local = df[self.dst_entity_col].astype("int64").to_numpy()
            offset = self._bipartite_offset()
            if offset is None:
                dst_key = dst_local
            else:
                dst_key = dst_local + int(offset)
            for s, d, t in zip(src_global.tolist(), dst_key.tolist(), ts_s.tolist()):
                negs_g = np.asarray(neg_dict[(s, d, t)], dtype=np.int64)
                negs_local.append(self._dst_global_to_local(negs_g))
        else:
            et = int(self.edge_type_id)
            for t, s in zip(ts_s.tolist(), src_global.tolist()):
                negs_g = np.asarray(neg_dict[(t, s, et)], dtype=np.int64)
                negs_local.append(self._dst_global_to_local(negs_g))
        return negs_local

    def evaluate(
        self,
        pred: Any,
        target_table: Optional[Table] = None,
        metrics: Optional[list] = None,
    ) -> dict[str, float]:
        r"""Evaluate predictions using TGB's one-vs-many MRR/Hits@k.

        Expected `pred` formats:
        - dict with keys `y_pred_pos` (shape [N]) and `y_pred_neg` (shape [N, K])
        - numpy array of shape [N, 1+K] where pred[:, 0] is positive score
        """
        if target_table is None:
            target_table = self.get_table("test", mask_input_cols=False)

        if isinstance(pred, dict):
            y_pred_pos = np.asarray(pred.get("y_pred_pos"))
            y_pred_neg = np.asarray(pred.get("y_pred_neg"))
        else:
            arr = np.asarray(pred)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(
                    "For TGBOneVsManyLinkPredTask, pred must be a dict with "
                    "'y_pred_pos'/'y_pred_neg' or an array shaped (N, 1+K)."
                )
            y_pred_pos = arr[:, 0]
            y_pred_neg = arr[:, 1:]

        if y_pred_pos.shape[0] != len(target_table):
            raise ValueError(
                f"Prediction length {y_pred_pos.shape[0]} does not match target table rows {len(target_table)}."
            )

        return _tgb_eval_hits_and_mrr(y_pred_pos, y_pred_neg, k_value=self.k_value)


class TGBNextLinkPredTask(RecommendationTask):
    r"""TGB-style "next" link prediction task in RelBench recommendation format.

    This task is backed by precomputed task tables (train/val/test.parquet) in
    RelBench RecommendationTask format:
    - columns: `timestamp`, `src_id`, and a list-valued destination column
      (e.g. `dst_id` or `label_id`)

    Notes:
    - This task is intended to validate RelBench baselines (e.g.
      `examples/gnn_recommendation.py`, `examples/tgn_attention_recommendation.py`)
      on the translated TGB exports.
    - We do not currently support building these tables from scratch, since
      exact parity with TGB-style "next interaction" semantics is exporter-defined.
    """

    task_type = TaskType.LINK_PREDICTION
    timedelta = pd.Timedelta(seconds=1)
    num_eval_timestamps = 1
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]

    def __init__(
        self,
        dataset: Dataset,
        *,
        tgb_task_name: str,
        eval_k: int = 10,
        src_entity_col: str = "src_id",
        dst_entity_col: str = "dst_id",
        time_col: str = "timestamp",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.eval_k = int(eval_k)

        self.time_col = str(time_col)
        self.src_entity_col = str(src_entity_col)
        self.dst_entity_col = str(dst_entity_col)

        src_table = None
        dst_table = None
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            for split in ["train", "val", "test"]:
                p = cache_path / f"{split}.parquet"
                if p.exists():
                    tbl = Table.load(p)
                    src_table = tbl.fkey_col_to_pkey_table.get(self.src_entity_col)
                    dst_table = tbl.fkey_col_to_pkey_table.get(self.dst_entity_col)
                    break

        if src_table is None or dst_table is None:
            task_name = str(tgb_task_name)
            m = re.fullmatch(r"type(\d+)-type(\d+)-next", task_name)
            if m is not None:
                src_table = f"nodes_type_{int(m.group(1))}"
                dst_table = f"nodes_type_{int(m.group(2))}"
            elif task_name == "src-dst-next":
                db = dataset.get_db()
                if "src_nodes" in db.table_dict and "dst_nodes" in db.table_dict:
                    src_table = "src_nodes"
                    dst_table = "dst_nodes"
                else:
                    src_table = "nodes"
                    dst_table = "nodes"
            elif task_name == "node-label-next":
                src_table = "nodes"
                dst_table = "labels"
            else:
                raise ValueError(
                    f"Unable to infer src/dst entity tables for task_name={task_name!r}. "
                    "Provide cached task tables with fkey metadata."
                )

        self.src_entity_table = str(src_table)
        self.dst_entity_table = str(dst_table)

        if dataset.cache_dir is None:
            raise RuntimeError(
                "TGBNextLinkPredTask requires dataset.cache_dir to validate entity tables."
            )
        db_dir = Path(dataset.cache_dir) / "db"
        src_path = db_dir / f"{self.src_entity_table}.parquet"
        dst_path = db_dir / f"{self.dst_entity_table}.parquet"
        if not src_path.exists():
            raise ValueError(
                f"src_entity_table='{self.src_entity_table}' not found in dataset db at {src_path}."
            )
        if not dst_path.exists():
            raise ValueError(
                f"dst_entity_table='{self.dst_entity_table}' not found in dataset db at {dst_path}."
            )

        super().__init__(dataset, cache_dir=cache_dir)

    def make_table(self, db, timestamps):  # pragma: no cover
        raise RuntimeError(
            "TGBNextLinkPredTask expects precomputed task tables "
            "(train/val/test.parquet)."
        )


@dataclass(frozen=True)
class TGBNodePropSpec:
    r"""Specification for a TGB node property prediction task."""

    label_events_table: str = "label_events"
    label_items_table: str = "label_event_items"
    labels_table: str = "labels"
    node_col: str = "src_id"
    label_event_id_col: str = "label_event_id"
    label_id_col: str = "label_id"
    label_weight_col: str = "label_weight"
    time_col: str = "label_ts"


def _tgb_nodeprop_ndcg_at_k(
    *,
    topk_label_ids: np.ndarray,  # [N, K]
    topk_scores: np.ndarray,  # [N, K] (unused except for ordering)
    true_label_ids: list[np.ndarray],
    true_label_w: list[np.ndarray],
    k: int,
) -> float:
    """Compute NDCG@k with the same gain convention as sklearn.ndcg_score."""
    k = int(k)
    if topk_label_ids.ndim != 2 or topk_scores.ndim != 2:
        raise ValueError("topk_label_ids/topk_scores must be 2D arrays (N,K).")
    if topk_label_ids.shape != topk_scores.shape:
        raise ValueError("topk_label_ids and topk_scores must have the same shape.")

    n = int(topk_label_ids.shape[0])
    if len(true_label_ids) != n or len(true_label_w) != n:
        raise ValueError("true_label_ids/true_label_w must be lists of length N.")

    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)

    ndcgs: list[float] = []
    for i in range(n):
        ids = np.asarray(true_label_ids[i], dtype=np.int64)
        rel = np.asarray(true_label_w[i], dtype=np.float64)
        if ids.size == 0:
            ndcgs.append(0.0)
            continue

        rel_sorted = np.sort(rel)[::-1]
        rel_top = rel_sorted[:k]
        idcg = ((np.exp2(rel_top) - 1.0) * discounts[: rel_top.shape[0]]).sum()
        if idcg <= 0:
            ndcgs.append(0.0)
            continue

        rel_map = {int(l): float(w) for l, w in zip(ids.tolist(), rel.tolist())}
        pred_ids = topk_label_ids[i, :k]
        gains = np.fromiter(
            (np.exp2(rel_map.get(int(l), 0.0)) - 1.0 for l in pred_ids.tolist()),
            dtype=np.float64,
        )
        dcg = (gains * discounts[: gains.shape[0]]).sum()
        ndcgs.append(float(dcg / idcg))

    return float(np.mean(ndcgs)) if ndcgs else 0.0


class TGBNodePropNDCGTask(BaseTask):
    r"""TGB-style node property prediction with official NDCG@10 evaluation.

    This task assumes the `tgbn-*` export schema includes:
    - `label_events(label_event_id, src_id, label_ts)`
    - `label_event_items(label_event_id, label_id, label_weight)`
    - `labels(label_id, ...)`

    Important: Exact parity with TGB requires scoring *all* labels (or at least
    retrieving the top-k labels under that full scoring) for each label event.
    """

    task_type = TaskType.MULTILABEL_CLASSIFICATION
    timedelta = pd.Timedelta(seconds=1)
    metrics = []
    num_eval_timestamps = 1

    def __init__(
        self,
        dataset: Dataset,
        *,
        spec: TGBNodePropSpec = TGBNodePropSpec(),
        k: int = 10,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.spec = spec
        self.k = int(k)

        db = dataset.get_db()
        if spec.label_events_table not in db.table_dict:
            raise ValueError(f"Missing table {spec.label_events_table} in dataset db.")
        if spec.label_items_table not in db.table_dict:
            raise ValueError(f"Missing table {spec.label_items_table} in dataset db.")
        if spec.labels_table not in db.table_dict:
            raise ValueError(f"Missing table {spec.labels_table} in dataset db.")

        label_events = db.table_dict[spec.label_events_table]
        self.entity_table = label_events.fkey_col_to_pkey_table.get(spec.node_col)
        if self.entity_table is None:
            raise ValueError(
                f"Expected {spec.label_events_table}.{spec.node_col} to be a foreign key. "
                f"Got {label_events.fkey_col_to_pkey_table}."
            )

        self.time_col = spec.time_col
        self.entity_col = spec.node_col
        self.label_event_id_col = spec.label_event_id_col

        super().__init__(dataset, cache_dir=cache_dir)

    def make_table(self, db, timestamps):  # pragma: no cover
        raise RuntimeError(
            "TGBNodePropNDCGTask expects precomputed task tables (train/val/test.parquet) "
            "and overrides _get_table()."
        )

    def filter_dangling_entities(self, table: Table) -> Table:
        db = self.dataset.get_db()
        num_nodes = len(db.table_dict[self.entity_table])
        bad = table.df[self.entity_col] >= num_nodes
        if bad.any():
            table.df = table.df[~bad]
        return table

    @lru_cache(maxsize=1)
    def _label_csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CSR arrays over label_event_id -> (label_id, label_weight)."""
        db = self.dataset.get_db(upto_test_timestamp=False)
        items = (
            db.table_dict[self.spec.label_items_table]
            .df[
                [
                    self.spec.label_event_id_col,
                    self.spec.label_id_col,
                    self.spec.label_weight_col,
                ]
            ]
            .copy()
        )

        event_ids = items[self.spec.label_event_id_col].astype("int64").to_numpy()
        order = np.argsort(event_ids, kind="mergesort")
        event_ids = event_ids[order]
        label_ids = items[self.spec.label_id_col].astype("int64").to_numpy()[order]
        label_w = items[self.spec.label_weight_col].astype("float64").to_numpy()[order]

        num_events = len(db.table_dict[self.spec.label_events_table])
        counts = np.bincount(event_ids, minlength=num_events).astype(
            np.int64, copy=False
        )
        indptr = np.empty(num_events + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        return indptr, label_ids, label_w

    def _truth_for_events(
        self, label_event_ids: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        indptr, label_ids, label_w = self._label_csr()
        true_ids: list[np.ndarray] = []
        true_w: list[np.ndarray] = []
        for e in label_event_ids.astype(np.int64, copy=False).tolist():
            e = int(e)
            start = int(indptr[e])
            end = int(indptr[e + 1])
            true_ids.append(np.asarray(label_ids[start:end], dtype=np.int64))
            true_w.append(np.asarray(label_w[start:end], dtype=np.float64))
        return true_ids, true_w

    def evaluate(
        self,
        pred: Any,
        target_table: Optional[Table] = None,
        metrics: Optional[list] = None,
    ) -> dict[str, float]:
        if target_table is None:
            target_table = self.get_table("test", mask_input_cols=False)

        if isinstance(pred, dict):
            y_pred = np.asarray(pred.get("y_pred"))
        else:
            y_pred = np.asarray(pred)

        if y_pred.ndim != 2:
            raise ValueError("Expected predictions with shape (N, num_labels).")
        if y_pred.shape[0] != len(target_table):
            raise ValueError(
                f"Prediction rows {y_pred.shape[0]} != target rows {len(target_table)}."
            )

        k = int(self.k)
        topk = np.argpartition(-y_pred, kth=min(k - 1, y_pred.shape[1] - 1), axis=1)[
            :, :k
        ]
        topk_scores = np.take_along_axis(y_pred, topk, axis=1)
        order = np.argsort(-topk_scores, axis=1, kind="mergesort")
        topk = np.take_along_axis(topk, order, axis=1)
        topk_scores = np.take_along_axis(topk_scores, order, axis=1)

        label_event_ids = (
            target_table.df[self.label_event_id_col].astype("int64").to_numpy()
        )
        true_ids, true_w = self._truth_for_events(label_event_ids)
        ndcg = _tgb_nodeprop_ndcg_at_k(
            topk_label_ids=topk,
            topk_scores=topk_scores,
            true_label_ids=true_ids,
            true_label_w=true_w,
            k=k,
        )
        return {f"ndcg@{k}": float(ndcg)}
