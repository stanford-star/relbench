"""
Temporal recommendation baseline: TGN + attention (TransformerConv).

This script ports the core TGB2 "TGN + GraphAttentionEmbedding" training recipe
to RelBench's RecommendationTask interface.

Key idea:
- Use a *history* event table from the RelBench database (not the task tables,
  since task tables contain *future* targets by construction).
- Train a Temporal Graph Network (TGN) on observed (src, dst, time) interactions.
- Evaluate at RelBench split cutoffs by producing top-k dst predictions per src.

This is intended as an *example* script (for small/medium tasks). For large
dst spaces, full ranking can be expensive.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear
from torch_geometric.nn import TransformerConv
from torch_geometric.seed import seed_everything
from torch_geometric.utils import scatter
from tqdm import tqdm

from relbench import load_dataset, load_task
from relbench.base import Dataset, RecommendationTask, Table, TaskType


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.lin = Linear(1, self.out_channels)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = int(raw_msg_dim + 2 * memory_dim + time_dim)

    def forward(
        self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor
    ) -> Tensor:
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int) -> Tensor:
        if msg.size(0) == 0:
            return msg.new_zeros((dim_size, msg.size(-1)))
        max_t = scatter(t, index, dim=0, dim_size=dim_size, reduce="max")
        is_max = t == max_t[index]
        pos = torch.arange(t.size(0), device=t.device, dtype=torch.long)
        pos = torch.where(is_max, pos, pos.new_full(pos.shape, -1))
        argmax = scatter(pos, index, dim=0, dim_size=dim_size, reduce="max")

        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax >= 0
        out[mask] = msg[argmax[mask]]
        return out


class LastNeighborLoader:
    """Keeps the last K neighbors per node (undirected)."""

    def __init__(
        self, num_nodes: int, size: int, *, device: Optional[torch.device] = None
    ):
        self.size = int(size)
        self.neighbors = torch.empty(
            (num_nodes, self.size), dtype=torch.long, device=device
        )
        self.e_id = torch.empty((num_nodes, self.size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        self.reset_state()

    def reset_state(self) -> None:
        self.cur_e_id = 0
        self.e_id.fill_(-1)

    def __call__(self, n_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]
        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor) -> None:
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(
            self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
        ).repeat(2)
        self.cur_e_id += src.numel()

        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, : self.size], dense_neighbors], dim=-1
        )

        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)


class TGNMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        *,
        message_module: torch.nn.Module,
        aggregator_module: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.raw_msg_dim = int(raw_msg_dim)
        self.memory_dim = int(memory_dim)
        self.time_dim = int(time_dim)

        # NOTE: This is a memory-safe variant intended for large RelBench datasets.
        # The original TGN reference implementation keeps a Python dict of message
        # queues per node. For million-scale node counts (e.g., rel-hm), that is
        # not feasible. Since this example uses a LastAggregator, we stream-update
        # memory on each interaction (online update), avoiding per-node queues.
        self.msg_module = message_module
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.memory_updater = GRUCell(self.msg_module.out_channels, memory_dim)

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer("_tmp", torch.empty(0, dtype=torch.long))

    def reset_state(self) -> None:
        self.memory.zero_()
        self.last_update.zero_()

    def detach(self) -> None:
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> tuple[Tensor, Tensor]:
        return self.memory[n_id], self.last_update[n_id]

    def update_state(
        self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor
    ) -> None:
        # Stream update both endpoints per interaction (undirected view).
        if src.numel() == 0:
            return

        # Update src nodes.
        t_rel_s = t - self.last_update[src]
        t_enc_s = self.time_enc(t_rel_s.to(torch.float32))
        msg_s = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc_s)
        self.memory[src] = self.memory_updater(msg_s, self.memory[src])
        self.last_update[src] = torch.maximum(self.last_update[src], t)

        # Update dst nodes.
        t_rel_d = t - self.last_update[dst]
        t_enc_d = self.time_enc(t_rel_d.to(torch.float32))
        msg_d = self.msg_module(self.memory[dst], self.memory[src], raw_msg, t_enc_d)
        self.memory[dst] = self.memory_updater(msg_d, self.memory[dst])
        self.last_update[dst] = torch.maximum(self.last_update[dst], t)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, msg_dim: int, time_enc: TimeEncoder
    ):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = int(msg_dim + time_enc.out_channels)
        if out_channels % 2 != 0:
            raise ValueError(
                "out_channels must be divisible by 2 (TransformerConv heads=2)."
            )
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(
        self, x: Tensor, last_update: Tensor, edge_index: Tensor, t: Tensor, msg: Tensor
    ) -> Tensor:
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


@dataclass(frozen=True)
class _EventSpec:
    table: str
    src_col: str
    dst_col: str
    time_col: str


def _infer_event_spec(
    db,
    task: RecommendationTask,
    *,
    task_name: str,
    event_table: Optional[str],
) -> _EventSpec:
    candidates: list[_EventSpec] = []
    for name, table in db.table_dict.items():
        if event_table is not None and name != event_table:
            continue
        if table.time_col is None:
            continue
        fks = table.fkey_col_to_pkey_table
        if task.src_entity_table != task.dst_entity_table:
            src_cols = [c for c, t in fks.items() if t == task.src_entity_table]
            dst_cols = [c for c, t in fks.items() if t == task.dst_entity_table]
            if not src_cols or not dst_cols:
                continue
            candidates.append(
                _EventSpec(
                    table=name,
                    src_col=src_cols[0],
                    dst_col=dst_cols[0],
                    time_col=table.time_col,
                )
            )
        else:
            # If src/dst entity tables are the same (homogeneous recommendation),
            # we must choose two *distinct* FK columns pointing to that table.
            cols = [c for c, t in fks.items() if t == task.src_entity_table]
            if len(cols) < 2:
                continue

            # Prefer canonical naming if present.
            if "src_id" in cols and "dst_id" in cols:
                candidates.append(
                    _EventSpec(
                        table=name,
                        src_col="src_id",
                        dst_col="dst_id",
                        time_col=table.time_col,
                    )
                )
                continue
            if "dst_id" in cols and "src_id" in cols:
                candidates.append(
                    _EventSpec(
                        table=name,
                        src_col="src_id",
                        dst_col="dst_id",
                        time_col=table.time_col,
                    )
                )
                continue

            # Otherwise, pick a stable pair based on name hints.
            def _col_score(col: str) -> tuple[int, int]:
                col_l = col.lower()
                is_src = 1 if "src" in col_l else 0
                is_dst = 1 if "dst" in col_l else 0
                return (is_dst, is_src)  # prefer dst-like columns later

            cols_sorted = sorted(cols, key=_col_score)
            src_col = cols_sorted[0]
            dst_col = next((c for c in cols_sorted[1:] if c != src_col), None)
            if dst_col is None:
                continue
            candidates.append(
                _EventSpec(
                    table=name,
                    src_col=src_col,
                    dst_col=dst_col,
                    time_col=table.time_col,
                )
            )

    if not candidates:
        raise RuntimeError(
            "Could not infer an event table with FK columns to both the task's "
            f"src table ({task.src_entity_table}) and dst table ({task.dst_entity_table}) "
            "and a time column. Specify one via --event_table."
        )

    # Heuristic: prefer tables whose name matches task name tokens (e.g.
    # `user-post-comment` should pick `comments` over `votes`).
    tokens = [t for t in str(task_name).lower().split("-") if t]
    action_token = tokens[-1] if tokens else ""

    def _score(spec: _EventSpec) -> tuple[int, int, int, int]:
        name_l = spec.table.lower()
        # Prioritize the single-word suffix (RelBench convention):
        # <src>-<dst>-<single_word>
        action_hit = 1 if action_token and action_token in name_l else 0
        token_hits = sum(1 for tok in tokens if tok in name_l)
        # Prefer canonical src/dst columns when multiple choices exist.
        col_hint = int(spec.src_col == "src_id") + int(spec.dst_col == "dst_id")
        return (action_hit, token_hits, col_hint, len(db.table_dict[spec.table]))

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _to_unix_seconds(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts, utc=True)
    ns = ts.astype("int64").to_numpy()
    return (ns // 1_000_000_000).astype(np.int64)


def _load_relbench_table_metadata(path: Path) -> tuple[dict[str, str], Optional[str]]:
    pf = pq.ParquetFile(path)
    md = pf.schema_arrow.metadata or {}

    fkeys = {}
    time_col = None
    if b"fkey_col_to_pkey_table" in md:
        fkeys = json.loads(md[b"fkey_col_to_pkey_table"].decode("utf-8"))
    if b"time_col" in md:
        time_col = json.loads(md[b"time_col"].decode("utf-8"))
    return fkeys, time_col


def _infer_event_spec_from_cache(
    dataset: Dataset,
    task: RecommendationTask,
    *,
    task_name: str,
    event_table: Optional[str],
) -> _EventSpec:
    db_dir = Path(str(dataset.cache_dir)) / "db"
    if not db_dir.exists():
        raise RuntimeError(f"Dataset cache_dir is missing db/: {db_dir}")

    candidates: list[_EventSpec] = []
    for p in db_dir.glob("*.parquet"):
        name = p.stem
        if event_table is not None and name != event_table:
            continue
        fkeys, time_col = _load_relbench_table_metadata(p)
        if time_col is None:
            continue

        if task.src_entity_table != task.dst_entity_table:
            src_cols = [c for c, t in fkeys.items() if t == task.src_entity_table]
            dst_cols = [c for c, t in fkeys.items() if t == task.dst_entity_table]
            if not src_cols or not dst_cols:
                continue
            candidates.append(
                _EventSpec(
                    table=name,
                    src_col=src_cols[0],
                    dst_col=dst_cols[0],
                    time_col=time_col,
                )
            )
        else:
            cols = [c for c, t in fkeys.items() if t == task.src_entity_table]
            if len(cols) < 2:
                continue
            if "src_id" in cols and "dst_id" in cols:
                candidates.append(
                    _EventSpec(
                        table=name,
                        src_col="src_id",
                        dst_col="dst_id",
                        time_col=time_col,
                    )
                )
                continue

            def _col_score(col: str) -> tuple[int, int]:
                col_l = col.lower()
                is_src = 1 if "src" in col_l else 0
                is_dst = 1 if "dst" in col_l else 0
                return (is_dst, is_src)

            cols_sorted = sorted(cols, key=_col_score)
            src_col = cols_sorted[0]
            dst_col = next((c for c in cols_sorted[1:] if c != src_col), None)
            if dst_col is None:
                continue
            candidates.append(
                _EventSpec(
                    table=name, src_col=src_col, dst_col=dst_col, time_col=time_col
                )
            )

    if not candidates:
        raise RuntimeError(
            "Could not infer an event table with FK columns to both the task's "
            f"src table ({task.src_entity_table}) and dst table ({task.dst_entity_table}) "
            "and a time column. Specify one via --event_table."
        )

    tokens = [t for t in str(task_name).lower().split("-") if t]
    action_token = tokens[-1] if tokens else ""

    def _score(spec: _EventSpec) -> tuple[int, int, int]:
        name_l = spec.table.lower()
        action_hit = 1 if action_token and action_token in name_l else 0
        token_hits = sum(1 for tok in tokens if tok in name_l)
        col_hint = int(spec.src_col == "src_id") + int(spec.dst_col == "dst_id")
        return (action_hit, token_hits, col_hint)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


@dataclass(frozen=True)
class _JoinedLabelEventSpec:
    label_events_table: str = "label_events"
    label_event_items_table: str = "label_event_items"
    label_event_id_col: str = "label_event_id"
    src_col: str = "src_id"
    dst_col: str = "label_id"
    time_col: str = "label_ts"


def _table_num_rows_from_cache(dataset: Dataset, table_name: str) -> int:
    p = Path(str(dataset.cache_dir)) / "db" / f"{table_name}.parquet"
    if not p.exists():
        raise RuntimeError(f"Missing table parquet for {table_name}: {p}")
    return int(pq.ParquetFile(p).metadata.num_rows)


def _load_last_events_before(
    dataset: Dataset,
    *,
    table: str,
    src_col: str,
    dst_col: str,
    time_col: str,
    cutoff_s: int,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_events = int(max_events)
    if max_events <= 0:
        raise ValueError("max_events must be > 0")

    path = Path(str(dataset.cache_dir)) / "db" / f"{table}.parquet"
    pf = pq.ParquetFile(path)
    cols = pf.schema.names
    time_idx = cols.index(time_col) if time_col in cols else None

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    t_chunks: list[np.ndarray] = []
    total = 0

    def _stat_to_unix_seconds(x) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        try:
            return int(pd.to_datetime(x, utc=True).timestamp())
        except Exception:
            return None

    for rg in range(pf.num_row_groups - 1, -1, -1):
        take_all = False
        if time_idx is not None:
            try:
                col_meta = pf.metadata.row_group(rg).column(time_idx)
                stats = col_meta.statistics
                min_s = (
                    _stat_to_unix_seconds(getattr(stats, "min", None))
                    if stats is not None
                    else None
                )
                max_s = (
                    _stat_to_unix_seconds(getattr(stats, "max", None))
                    if stats is not None
                    else None
                )
                if min_s is not None and min_s >= cutoff_s:
                    continue
                take_all = max_s is not None and max_s < cutoff_s
            except Exception:
                take_all = False

        tbl = pf.read_row_group(
            rg, columns=[src_col, dst_col, time_col], use_threads=True
        ).to_pandas()
        tbl = tbl.dropna()
        if tbl.shape[0] == 0:
            continue
        t_np = _to_unix_seconds(tbl[time_col])
        if take_all:
            mask = slice(None)
        else:
            mask = t_np < int(cutoff_s)
            if not np.any(mask):
                continue

        src_np = tbl[src_col].astype("int64").to_numpy(copy=False)[mask]
        dst_np = tbl[dst_col].astype("int64").to_numpy(copy=False)[mask]
        t_np = t_np[mask]

        src_chunks.append(src_np)
        dst_chunks.append(dst_np)
        t_chunks.append(t_np)
        total += int(t_np.shape[0])
        if total >= max_events:
            break

    if total == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    src_np = np.concatenate(src_chunks[::-1], axis=0)
    dst_np = np.concatenate(dst_chunks[::-1], axis=0)
    t_np = np.concatenate(t_chunks[::-1], axis=0)

    if t_np.shape[0] > max_events:
        src_np = src_np[-max_events:]
        dst_np = dst_np[-max_events:]
        t_np = t_np[-max_events:]

    order = np.argsort(t_np, kind="mergesort")
    return src_np[order], dst_np[order], t_np[order]


def _load_last_joined_label_events_before(
    dataset: Dataset,
    spec: _JoinedLabelEventSpec,
    *,
    cutoff_s: int,
    max_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_events = int(max_events)
    if max_events <= 0:
        raise ValueError("max_events must be > 0")

    db_dir = Path(str(dataset.cache_dir)) / "db"
    label_events_path = db_dir / f"{spec.label_events_table}.parquet"
    label_items_path = db_dir / f"{spec.label_event_items_table}.parquet"

    le_df = pq.read_table(
        label_events_path,
        columns=[spec.label_event_id_col, spec.src_col, spec.time_col],
        use_threads=True,
    ).to_pandas()
    le_df = le_df.dropna()
    le_ids = le_df[spec.label_event_id_col].astype("int64").to_numpy(copy=False)
    le_src = le_df[spec.src_col].astype("int64").to_numpy(copy=False)
    le_t = _to_unix_seconds(le_df[spec.time_col])

    if le_ids.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    if (le_ids == np.arange(le_ids.size)).all():
        idx = None
    else:
        max_id = int(le_ids.max())
        idx = np.full((max_id + 1,), -1, dtype=np.int64)
        idx[le_ids] = np.arange(le_ids.size, dtype=np.int64)

    pf = pq.ParquetFile(label_items_path)
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    t_chunks: list[np.ndarray] = []
    total = 0

    for rg in range(pf.num_row_groups - 1, -1, -1):
        items = pf.read_row_group(
            rg, columns=[spec.label_event_id_col, spec.dst_col], use_threads=True
        ).to_pandas()
        items = items.dropna()
        if items.shape[0] == 0:
            continue
        item_le_ids = (
            items[spec.label_event_id_col].astype("int64").to_numpy(copy=False)
        )
        item_dst = items[spec.dst_col].astype("int64").to_numpy(copy=False)

        if idx is None:
            le_pos = item_le_ids
        else:
            bad = item_le_ids >= idx.shape[0]
            if np.any(bad):
                keep = ~bad
                item_le_ids = item_le_ids[keep]
                item_dst = item_dst[keep]
            le_pos = idx[item_le_ids]
            keep = le_pos >= 0
            if not np.any(keep):
                continue
            le_pos = le_pos[keep]
            item_dst = item_dst[keep]

        item_t = le_t[le_pos]
        keep = item_t < int(cutoff_s)
        if not np.any(keep):
            continue
        le_pos = le_pos[keep]
        item_dst = item_dst[keep]
        item_t = item_t[keep]
        item_src = le_src[le_pos]

        src_chunks.append(item_src)
        dst_chunks.append(item_dst)
        t_chunks.append(item_t)
        total += int(item_t.shape[0])
        if total >= max_events:
            break

    if total == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    src_np = np.concatenate(src_chunks[::-1], axis=0)
    dst_np = np.concatenate(dst_chunks[::-1], axis=0)
    t_np = np.concatenate(t_chunks[::-1], axis=0)

    if t_np.shape[0] > max_events:
        src_np = src_np[-max_events:]
        dst_np = dst_np[-max_events:]
        t_np = t_np[-max_events:]

    order = np.argsort(t_np, kind="mergesort")
    return src_np[order], dst_np[order], t_np[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-race-compete")
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--event_table", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--mem_dim", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_neg_train", type=int, default=50)
    parser.add_argument("--eval_dst_block_size", type=int, default=50000)
    parser.add_argument(
        "--max_train_events",
        type=int,
        default=0,
        help="Cap number of training events before val cutoff (0 disables).",
    )
    parser.add_argument(
        "--max_history_events",
        type=int,
        default=0,
        help="Cap history events used to build memory for eval (0 disables).",
    )
    parser.add_argument(
        "--max_val_rows",
        type=int,
        default=0,
        help="Cap number of val rows evaluated (0 disables).",
    )
    parser.add_argument(
        "--max_test_rows",
        type=int,
        default=0,
        help="Cap number of test rows evaluated (0 disables).",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional directory to save checkpoints/metrics.",
    )
    parser.add_argument(
        "--save_every_epoch", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.set_num_threads(1)

    try:
        dataset: Dataset = load_dataset(args.dataset)
        task: RecommendationTask = load_task(args.dataset, args.task)
    except Exception:
        if bool(args.download):
            print(
                "Download failed. If you already have the dataset cached, re-run with --no-download.",
                file=sys.stderr,
            )
        raise
    if task.task_type != TaskType.LINK_PREDICTION:
        raise ValueError(
            f"Task {args.dataset}/{args.task} is not a link prediction task."
        )

    val_ts = int(pd.to_datetime(dataset.val_timestamp, utc=True).timestamp())
    test_ts = int(pd.to_datetime(dataset.test_timestamp, utc=True).timestamp())

    # Use task-provided counts (they respect Dataset.get_db(upto_test_timestamp=True),
    # which may filter time-stamped entity tables like `races` in rel-f1).
    num_src = int(task.num_src_nodes)
    num_dst = int(task.num_dst_nodes)
    if task.src_entity_table != task.dst_entity_table:
        num_nodes = num_src + num_dst
    else:
        num_nodes = max(num_src, num_dst)

    max_train = int(args.max_train_events) if args.max_train_events else 0
    max_hist = int(args.max_history_events) if args.max_history_events else 0
    cap_val = max(max_train, max_hist) if (max_train > 0 and max_hist > 0) else 0
    cap_test = max_hist if max_hist > 0 else 0

    joined_spec = _JoinedLabelEventSpec()
    db_dir = Path(str(dataset.cache_dir)) / "db"
    has_label_tables = (
        db_dir / f"{joined_spec.label_events_table}.parquet"
    ).exists() and (db_dir / f"{joined_spec.label_event_items_table}.parquet").exists()

    direct_spec: Optional[_EventSpec] = None
    use_joined = False
    try:
        direct_spec = _infer_event_spec_from_cache(
            dataset, task, task_name=args.task, event_table=args.event_table
        )
    except RuntimeError:
        if has_label_tables and args.event_table is None:
            use_joined = True
        else:
            raise

    if use_joined:
        if cap_val <= 0:
            cap_val = int(
                pq.ParquetFile(
                    db_dir / f"{joined_spec.label_event_items_table}.parquet"
                ).metadata.num_rows
            )
        if cap_test <= 0:
            cap_test = cap_val

        src_val_np, dst_val_np, t_val_np = _load_last_joined_label_events_before(
            dataset, joined_spec, cutoff_s=val_ts, max_events=cap_val
        )
        src_test_np, dst_test_np, t_test_np = _load_last_joined_label_events_before(
            dataset, joined_spec, cutoff_s=test_ts, max_events=cap_test
        )
        print(
            f"[event_table] <joined:{joined_spec.label_events_table}+{joined_spec.label_event_items_table}> "
            f"(src_col={joined_spec.src_col}, dst_col={joined_spec.dst_col}, time_col={joined_spec.time_col})"
        )
    else:
        assert direct_spec is not None
        if cap_val > 0:
            src_val_np, dst_val_np, t_val_np = _load_last_events_before(
                dataset,
                table=direct_spec.table,
                src_col=direct_spec.src_col,
                dst_col=direct_spec.dst_col,
                time_col=direct_spec.time_col,
                cutoff_s=val_ts,
                max_events=cap_val,
            )
        else:
            path = db_dir / f"{direct_spec.table}.parquet"
            df = pq.read_table(
                path,
                columns=[
                    direct_spec.src_col,
                    direct_spec.dst_col,
                    direct_spec.time_col,
                ],
                use_threads=True,
            ).to_pandas()
            df = df.dropna()
            src_val_np = df[direct_spec.src_col].astype("int64").to_numpy()
            dst_val_np = df[direct_spec.dst_col].astype("int64").to_numpy()
            t_val_np = _to_unix_seconds(df[direct_spec.time_col])
            order = np.argsort(t_val_np, kind="mergesort")
            src_val_np, dst_val_np, t_val_np = (
                src_val_np[order],
                dst_val_np[order],
                t_val_np[order],
            )
            mask = t_val_np < int(val_ts)
            src_val_np, dst_val_np, t_val_np = (
                src_val_np[mask],
                dst_val_np[mask],
                t_val_np[mask],
            )

        if cap_test > 0:
            src_test_np, dst_test_np, t_test_np = _load_last_events_before(
                dataset,
                table=direct_spec.table,
                src_col=direct_spec.src_col,
                dst_col=direct_spec.dst_col,
                time_col=direct_spec.time_col,
                cutoff_s=test_ts,
                max_events=cap_test,
            )
        else:
            src_test_np, dst_test_np, t_test_np = src_val_np, dst_val_np, t_val_np
        print(
            f"[event_table] {direct_spec.table} "
            f"(src_col={direct_spec.src_col}, dst_col={direct_spec.dst_col}, time_col={direct_spec.time_col})"
        )

    if task.src_entity_table != task.dst_entity_table:
        dst_val_global_np = dst_val_np + num_src
        dst_test_global_np = dst_test_np + num_src
    else:
        dst_val_global_np = dst_val_np
        dst_test_global_np = dst_test_np

    val_stream = (
        torch.from_numpy(src_val_np).to(device=device, dtype=torch.long),
        torch.from_numpy(dst_val_global_np).to(device=device, dtype=torch.long),
        torch.from_numpy(t_val_np).to(device=device, dtype=torch.long),
        torch.zeros((int(t_val_np.shape[0]), 0), device=device, dtype=torch.float32),
    )
    test_stream = (
        torch.from_numpy(src_test_np).to(device=device, dtype=torch.long),
        torch.from_numpy(dst_test_global_np).to(device=device, dtype=torch.long),
        torch.from_numpy(t_test_np).to(device=device, dtype=torch.long),
        torch.zeros((int(t_test_np.shape[0]), 0), device=device, dtype=torch.float32),
    )

    rng = np.random.default_rng(int(args.seed))

    def _subset_table(table: Table, *, max_rows: int) -> Table:
        if not max_rows or int(max_rows) <= 0 or table.df.shape[0] <= int(max_rows):
            return table
        idx = rng.choice(table.df.shape[0], size=int(max_rows), replace=False)
        df_sub = table.df.iloc[idx].reset_index(drop=True)
        return Table(
            df=df_sub,
            fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
            pkey_col=table.pkey_col,
            time_col=table.time_col,
        )

    def build_pred_at(
        timestamp_s: int, *, split: str, max_rows: int
    ) -> tuple[np.ndarray, Table]:
        target_table = task.get_table(split, mask_input_cols=False)
        if len(target_table.df) == 0:
            return np.empty((0, int(task.eval_k)), dtype=np.int64), target_table

        target_table = _subset_table(target_table, max_rows=max_rows)
        target_ts = int(
            pd.to_datetime(target_table.df[task.time_col].iloc[0], utc=True).timestamp()
        )
        if target_ts != timestamp_s:
            raise RuntimeError("This example assumes a single timestamp per split.")

        if timestamp_s == val_ts:
            src, dst, t, msg = val_stream
        elif timestamp_s == test_ts:
            src, dst, t, msg = test_stream
        else:
            raise RuntimeError(
                "This example only supports the val/test split timestamps."
            )

        memory.eval()
        gnn.eval()
        memory.reset_state()
        neighbor_loader.reset_state()

        max_hist = int(args.max_history_events) if args.max_history_events else 0
        src_h = src[-max_hist:] if max_hist > 0 else src
        dst_h = dst[-max_hist:] if max_hist > 0 else dst
        t_h = t[-max_hist:] if max_hist > 0 else t
        msg_h = msg[-max_hist:] if max_hist > 0 else msg

        for start in range(0, t_h.numel(), args.batch_size):
            end = min(start + args.batch_size, t_h.numel())
            src_b, dst_b, t_b = src_h[start:end], dst_h[start:end], t_h[start:end]
            msg_b = msg_h[start:end]
            memory.update_state(src_b, dst_b, t_b, msg_b)
            neighbor_loader.insert(src_b, dst_b)

        with torch.no_grad():
            src_ids = torch.from_numpy(
                target_table.df[task.src_entity_col].astype("int64").to_numpy()
            ).to(device)
            k = int(task.eval_k)
            block = int(args.eval_dst_block_size)

            if num_dst <= block:
                # Small/medium destination space: embed all dst nodes at once.
                if task.src_entity_table != task.dst_entity_table:
                    dst_ids = (
                        torch.arange(num_dst, device=device, dtype=torch.long) + num_src
                    )
                else:
                    dst_ids = torch.arange(num_dst, device=device, dtype=torch.long)

                n_id_seed = torch.cat([src_ids, dst_ids]).unique()
                n_id, edge_index, e_id = neighbor_loader(n_id_seed)
                assoc = torch.empty(num_nodes, device=device, dtype=torch.long)
                assoc[n_id] = torch.arange(n_id.size(0), device=device)

                z, last_update = memory(n_id)
                z = gnn(z, last_update, edge_index, t_h[e_id], msg_h[e_id])
                src_emb = z[assoc[src_ids]]
                dst_emb = z[assoc[dst_ids]]

                topk_scores = src_emb.new_full((src_emb.size(0), k), float("-inf"))
                topk_idx = torch.full(
                    (src_emb.size(0), k), -1, device=device, dtype=torch.long
                )

                for start in range(0, dst_emb.size(0), block):
                    end = min(start + block, dst_emb.size(0))
                    scores = src_emb @ dst_emb[start:end].t()  # [B, block]
                    cand_scores, cand_idx = torch.topk(
                        scores, k=min(k, end - start), dim=1
                    )
                    cand_idx = cand_idx + start

                    merged_scores = torch.cat([topk_scores, cand_scores], dim=1)
                    merged_idx = torch.cat([topk_idx, cand_idx], dim=1)
                    topk_scores, sel = torch.topk(merged_scores, k=k, dim=1)
                    topk_idx = torch.gather(merged_idx, 1, sel)
            else:
                # Large destination space: compute dst embeddings in blocks to reduce peak memory.
                src_seed = src_ids.unique()
                n_id_s, edge_index_s, e_id_s = neighbor_loader(src_seed)
                assoc = torch.empty(num_nodes, device=device, dtype=torch.long)
                assoc[n_id_s] = torch.arange(n_id_s.size(0), device=device)
                z_s, last_update_s = memory(n_id_s)
                z_s = gnn(z_s, last_update_s, edge_index_s, t_h[e_id_s], msg_h[e_id_s])
                src_emb = z_s[assoc[src_ids]]

                topk_scores = src_emb.new_full((src_emb.size(0), k), float("-inf"))
                topk_idx = torch.full(
                    (src_emb.size(0), k), -1, device=device, dtype=torch.long
                )

                for start in range(0, num_dst, block):
                    end = min(start + block, num_dst)
                    if task.src_entity_table != task.dst_entity_table:
                        dst_block_ids = (
                            torch.arange(start, end, device=device, dtype=torch.long)
                            + num_src
                        )
                    else:
                        dst_block_ids = torch.arange(
                            start, end, device=device, dtype=torch.long
                        )

                    n_id_d, edge_index_d, e_id_d = neighbor_loader(dst_block_ids)
                    assoc[n_id_d] = torch.arange(n_id_d.size(0), device=device)
                    z_d, last_update_d = memory(n_id_d)
                    z_d = gnn(
                        z_d, last_update_d, edge_index_d, t_h[e_id_d], msg_h[e_id_d]
                    )
                    dst_emb = z_d[assoc[dst_block_ids]]

                    scores = src_emb @ dst_emb.t()  # [B, block]
                    cand_scores, cand_idx = torch.topk(
                        scores, k=min(k, end - start), dim=1
                    )
                    cand_idx = cand_idx + start

                    merged_scores = torch.cat([topk_scores, cand_scores], dim=1)
                    merged_idx = torch.cat([topk_idx, cand_idx], dim=1)
                    topk_scores, sel = torch.topk(merged_scores, k=k, dim=1)
                    topk_idx = torch.gather(merged_idx, 1, sel)

            return topk_idx.detach().cpu().numpy(), target_table

    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=0,
        memory_dim=args.mem_dim,
        time_dim=args.time_dim,
        message_module=IdentityMessage(0, args.mem_dim, args.time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    gnn = GraphAttentionEmbedding(
        in_channels=args.mem_dim,
        out_channels=args.emb_dim,
        msg_dim=0,
        time_enc=memory.time_enc,
    ).to(device)

    neighbor_loader = LastNeighborLoader(
        num_nodes, size=args.num_neighbors, device=device
    )
    optimizer = torch.optim.Adam(
        list(memory.parameters()) + list(gnn.parameters()),
        lr=args.lr,
    )

    def train_epoch() -> float:
        memory.train()
        gnn.train()
        memory.reset_state()
        neighbor_loader.reset_state()

        total_loss = 0.0
        total_events = 0
        max_train = int(args.max_train_events) if args.max_train_events else 0
        src_all, dst_all, t_all, msg_all = val_stream
        src_tr = src_all[-max_train:] if max_train > 0 else src_all
        dst_tr = dst_all[-max_train:] if max_train > 0 else dst_all
        t_tr = t_all[-max_train:] if max_train > 0 else t_all
        msg_tr = msg_all[-max_train:] if max_train > 0 else msg_all

        for start in tqdm(
            range(0, t_tr.numel(), args.batch_size), desc="train", leave=False
        ):
            end = min(start + args.batch_size, t_tr.numel())
            src_b, pos_dst_b, t_b = (
                src_tr[start:end],
                dst_tr[start:end],
                t_tr[start:end],
            )
            msg_b = msg_tr[start:end]

            # negatives are sampled from dst-type only
            if task.src_entity_table != task.dst_entity_table:
                neg_dst_b = torch.randint(
                    num_src,
                    num_src + num_dst,
                    (pos_dst_b.size(0), args.num_neg_train),
                    device=device,
                )
            else:
                neg_dst_b = torch.randint(
                    0, num_dst, (pos_dst_b.size(0), args.num_neg_train), device=device
                )

            n_id = torch.cat([src_b, pos_dst_b, neg_dst_b.reshape(-1)]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc = torch.empty(num_nodes, device=device, dtype=torch.long)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, t_tr[e_id], msg_tr[e_id])

            src_z = z[assoc[src_b]]
            pos_z = z[assoc[pos_dst_b]]
            neg_z = z[assoc[neg_dst_b]]

            pos_score = (src_z * pos_z).sum(dim=-1)  # [B]
            neg_score = (src_z.unsqueeze(1) * neg_z).sum(dim=-1)  # [B, N]

            loss = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            memory.update_state(src_b, pos_dst_b, t_b, msg_b)
            neighbor_loader.insert(src_b, pos_dst_b)
            memory.detach()

            total_loss += float(loss.detach()) * (end - start)
            total_events += end - start

        return total_loss / max(total_events, 1)

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "task": args.task,
                    "event_table": args.event_table,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "num_neighbors": args.num_neighbors,
                    "mem_dim": args.mem_dim,
                    "time_dim": args.time_dim,
                    "emb_dim": args.emb_dim,
                    "lr": args.lr,
                    "num_neg_train": args.num_neg_train,
                    "eval_dst_block_size": args.eval_dst_block_size,
                    "max_train_events": args.max_train_events,
                    "max_history_events": args.max_history_events,
                    "max_val_rows": args.max_val_rows,
                    "max_test_rows": args.max_test_rows,
                    "seed": args.seed,
                    "device": device_str,
                    "val_ts": val_ts,
                    "test_ts": test_ts,
                },
                indent=2,
            )
        )

    print(f"[src_table] {task.src_entity_table} (num_src={num_src:,})")
    print(f"[dst_table] {task.dst_entity_table} (num_dst={num_dst:,})")
    print(
        f"[streams] val_events={val_stream[2].numel():,} "
        f"test_events={test_stream[2].numel():,} "
        f"train_events={(min(max_train, int(val_stream[2].numel())) if max_train > 0 else int(val_stream[2].numel())):,} "
        f"hist_events={(min(max_hist, int(val_stream[2].numel())) if max_hist > 0 else int(val_stream[2].numel())):,}"
    )

    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch()
        val_pred, val_table = build_pred_at(
            val_ts, split="val", max_rows=int(args.max_val_rows)
        )
        val_metrics = task.evaluate(val_pred, val_table)
        tune = "link_prediction_map"
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | val={val_metrics}")
        if tune in val_metrics and val_metrics[tune] >= best_val:
            best_val = float(val_metrics[tune])
            best_state = {"memory": memory.state_dict(), "gnn": gnn.state_dict()}
            if run_dir is not None:
                torch.save(best_state, run_dir / "best.pt")
        if run_dir is not None and args.save_every_epoch:
            torch.save(
                {"memory": memory.state_dict(), "gnn": gnn.state_dict()},
                run_dir / f"epoch_{epoch:02d}.pt",
            )
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"epoch": epoch, "loss": loss, "val": val_metrics})
                    + "\n"
                )

    if best_state is not None:
        memory.load_state_dict(best_state["memory"])
        gnn.load_state_dict(best_state["gnn"])

    val_pred, val_table = build_pred_at(
        val_ts, split="val", max_rows=int(args.max_val_rows)
    )
    val_metrics = task.evaluate(val_pred, val_table)

    test_table = task.get_table("test", mask_input_cols=False)
    if len(test_table.df) == 0:
        test_metrics = "<skipped: empty test split>"
    else:
        test_pred, test_table = build_pred_at(
            test_ts, split="test", max_rows=int(args.max_test_rows)
        )
        test_metrics = task.evaluate(test_pred, test_table)

    print(f"Best val:  {val_metrics}")
    print(f"Best test: {test_metrics}")


if __name__ == "__main__":
    main()
