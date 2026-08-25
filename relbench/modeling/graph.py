import os
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_frame import TensorFrame, stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType, compute_col_stats
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.modeling.utils import remove_pkey_fkey, to_unix_time


def drop_tensor_frame_columns(
    tf: TensorFrame, col_stats: Dict[str, Dict[StatType, Any]], cols
) -> Tuple[TensorFrame, Dict[str, Dict[StatType, Any]]]:
    r"""Return ``tf`` and ``col_stats`` without the columns named in ``cols``.

    A table left with no feature columns gets a constant ``__const__`` column, as
    :func:`make_pkey_fkey_graph` does for feature-less tables.
    """
    cols = set(cols)
    feat_dict, col_names_dict = {}, {}
    for stype_, names in tf.col_names_dict.items():
        keep = [i for i, name in enumerate(names) if name not in cols]
        if not keep:
            continue
        feat = tf.feat_dict[stype_]
        feat_dict[stype_] = feat if len(keep) == len(names) else feat[:, keep]
        col_names_dict[stype_] = [names[i] for i in keep]
    col_stats = {name: s for name, s in col_stats.items() if name not in cols}
    if not feat_dict:
        feat_dict = {stype.numerical: torch.ones(tf.num_rows, 1)}
        col_names_dict = {stype.numerical: ["__const__"]}
        col_stats = {
            "__const__": compute_col_stats(
                pd.Series(np.ones(tf.num_rows)), stype.numerical
            )
        }
    return TensorFrame(feat_dict, col_names_dict, y=tf.y), col_stats


def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
    remove_columns=None,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
    foreign key relationships, together with the column stats of each table.

    Args:
        db: A database object containing a set of tables.
        col_to_stype_dict: Column to stype for
            each table.
        text_embedder_cfg: Text embedder config.
        cache_dir: A directory for storing materialized tensor
            frames. If specified, we will either cache the file or use the
            cached file. If not specified, we will not use cached file and
            re-process everything from scratch without saving the cache.
        remove_columns: ``(table, column)`` pairs to leave out of the node features
            (a task's :meth:`~relbench.base.BaseTask.hidden_columns`). They are dropped
            *after* materialization, so a cache built from the dataset-level database
            (``dataset.get_db()``) can be shared by every task on that dataset.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
    """
    data = HeteroData()
    col_stats_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    remove_columns = [tuple(pair) for pair in (remove_columns or [])]

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        tf, col_stats = dataset.tensor_frame, dataset.col_stats
        drop = {col for t, col in remove_columns if t == table_name}
        if drop:
            tf, col_stats = drop_tensor_frame_columns(tf, col_stats, drop)
        data[table_name].tf = tf
        col_stats_dict[table_name] = col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

    return data, col_stats_dict


class AttachTargetTransform:
    r"""Attach the target label to the heterogeneous mini-batch.

    The batch consists of disjoint subgraphs loaded via temporal sampling. The same
    input node can occur multiple times with different timestamps, and thus different
    subgraphs and labels. Hence labels cannot be stored in the graph object directly,
    and must be attached to the batch after the batch is created.
    """

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: HeteroData) -> HeteroData:
        batch[self.entity].y = self.target[batch[self.entity].input_id]
        return batch


class NodeTrainTableInput(NamedTuple):
    r"""Training table input for node prediction.

    - nodes is a Tensor of node indices.
    - time is a Tensor of node timestamps.
    - target is a Tensor of node labels.
    - transform attaches the target to the batch.
    """

    nodes: Tuple[NodeType, Tensor]
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]


def get_node_train_table_input(
    table: Table,
    task: EntityTask,
) -> NodeTrainTableInput:
    r"""Get the training table input for node prediction."""

    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        else:
            target = torch.from_numpy(
                table.df[task.target_col].values.astype(target_type)
            )
        transform = AttachTargetTransform(task.entity_table, target)

    return NodeTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
    )


class LinkTrainTableInput(NamedTuple):
    r"""Training table input for recommendation.

    - src_nodes is a Tensor of source node indices.
    - dst_nodes is PyTorch sparse tensor in csr format.
        dst_nodes[src_node_idx] gives a tensor of destination node
        indices for src_node_idx. None for a masked (test) table, which
        carries no destination lists.
    - num_dst_nodes is the total number of destination nodes.
        (used to perform negative sampling).
    - src_time is a Tensor of time for src_nodes
    """

    src_nodes: Tuple[NodeType, Tensor]
    dst_nodes: Optional[Tuple[NodeType, Tensor]]
    num_dst_nodes: int
    src_time: Optional[Tensor]


def num_dst_nodes(db: Database, task: RecommendationTask) -> int:
    r"""Number of destination entities of ``task`` in ``db``.

    Used for negative sampling and to size the score matrix. It lives here rather than
    on the task because only the modeling path needs it, and because it is a property of
    a particular database -- pass the one you already built.
    """
    return len(db.table_dict[task.dst_entity_table])


def get_link_train_table_input(
    table: Table,
    task: RecommendationTask,
    num_dst_nodes: int,
) -> LinkTrainTableInput:
    r"""Get the training table input for recommendation.

    ``num_dst_nodes`` sizes the destination space; see :func:`num_dst_nodes`.
    """

    src_node_idx: Tensor = torch.from_numpy(
        table.df[task.src_entity_col].astype(int).values
    )
    dst_nodes = None
    if task.dst_entity_col in table.df:
        exploded = table.df[task.dst_entity_col].explode()
        coo_indices = torch.from_numpy(
            np.stack([exploded.index.values, exploded.values.astype(int)])
        )
        sparse_coo = torch.sparse_coo_tensor(
            coo_indices,
            torch.ones(coo_indices.size(1), dtype=bool),
            (len(src_node_idx), num_dst_nodes),
        )
        dst_nodes = (task.dst_entity_table, sparse_coo.to_sparse_csr())

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    return LinkTrainTableInput(
        src_nodes=(task.src_entity_table, src_node_idx),
        dst_nodes=dst_nodes,
        num_dst_nodes=num_dst_nodes,
        src_time=time,
    )
