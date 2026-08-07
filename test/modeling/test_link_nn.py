from typing import Dict, Tuple

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import NodeType

from relbench.base import RecommendationTask, TaskType
from relbench.modeling.graph import (
    get_link_train_table_input,
    make_pkey_fkey_graph,
    num_dst_nodes,
)
from relbench.modeling.loader import LinkNeighborLoader
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE
from relbench.modeling.utils import get_stype_proposal, to_unix_time


@pytest.mark.parametrize(
    "share_same_time",
    [True, False],
)
def test_link_train_fake_product_dataset(
    tmp_path, share_same_time, make_purchase_task, fake_dataset
):
    dataset = fake_dataset()

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        get_stype_proposal(dataset.get_db()),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(8), batch_size=None
        ),
        cache_dir=tmp_path,
    )
    node_to_col_names_dict = {
        node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
    }
    encoder = HeteroEncoder(64, node_to_col_names_dict, col_stats_dict)
    gnn = HeteroGraphSAGE(data.node_types, data.edge_types, 64)

    # Ensure that neighbor loading works on train/val/test splits ############
    task: RecommendationTask = make_purchase_task(dataset)
    assert task.task_type == TaskType.RECOMMENDATION

    # Ensure that stats computation works on train/val/test splits ###########
    stats = task.stats()
    assert len(stats) == 4
    assert len(stats["train"]) >= 2
    assert len(next(iter(stats["train"].values()))) == 4
    assert len(stats["val"]) == 2
    assert len(next(iter(stats["val"].values()))) == 4
    assert len(stats["test"]) == 2
    assert len(next(iter(stats["test"].values()))) == 4
    assert len(stats["total"].values()) == 5

    n_dst_nodes = num_dst_nodes(dataset.get_db(), task)
    train_table = task.get_table("train")
    train_table_input = get_link_train_table_input(train_table, task, n_dst_nodes)
    # Test get_link_train_table_input
    for index, row in train_table.df.iterrows():
        assert set(row[task.dst_entity_col]) == set(
            train_table_input.dst_nodes[1][index].indices()[0].numpy()
        )
        assert row[task.src_entity_col] == train_table_input.src_nodes[1][index]
        assert (
            to_unix_time(pd.Series([row[task.time_col]]))[0]
            == train_table_input.src_time[index]
        )

    batch_size = 16
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[-1, -1],
        time_attr="time",
        src_nodes=train_table_input.src_nodes,
        dst_nodes=train_table_input.dst_nodes,
        num_dst_nodes=train_table_input.num_dst_nodes,
        src_time=train_table_input.src_time,
        share_same_time=share_same_time,
        batch_size=batch_size,
        # if share_same_time is True, we use sampler, so shuffle must be set False
        shuffle=not share_same_time,
        drop_last=not share_same_time,
    )

    batch = next(iter(train_loader))
    src_batch, batch_pos_dst, batch_neg_dst = batch
    src_seed_time = src_batch[task.src_entity_table].seed_time
    pos_dst_seed_time = batch_pos_dst[task.dst_entity_table].seed_time
    neg_dst_seed_time = batch_neg_dst[task.dst_entity_table].seed_time
    assert len(src_seed_time) == batch_size
    assert len(pos_dst_seed_time) == batch_size
    assert len(neg_dst_seed_time) == batch_size
    if share_same_time:
        shared_time = src_seed_time[0]
        assert (shared_time == src_seed_time).all()
        assert (shared_time == pos_dst_seed_time).all()
        assert (shared_time == neg_dst_seed_time).all()

    eval_loaders_dict: Dict[str, Tuple[NeighborLoader, NeighborLoader]] = {}
    for split in ["val", "test"]:
        timestamp = dataset.val_timestamp if split == "val" else dataset.test_timestamp
        seed_time = int(timestamp.timestamp())
        target_table = task.get_table(split)
        src_node_indices = torch.from_numpy(target_table.df[task.src_entity_col].values)
        src_loader = NeighborLoader(
            data,
            num_neighbors=[-1, -1],
            time_attr="time",
            input_nodes=(task.src_entity_table, src_node_indices),
            input_time=torch.full(
                size=(len(src_node_indices),), fill_value=seed_time, dtype=torch.long
            ),
            batch_size=32,
            shuffle=False,
        )
        dst_loader = NeighborLoader(
            data,
            num_neighbors=[-1, -1],
            time_attr="time",
            input_nodes=task.dst_entity_table,
            input_time=torch.full(
                size=(n_dst_nodes,), fill_value=seed_time, dtype=torch.long
            ),
            batch_size=32,
            shuffle=False,
        )
        eval_loaders_dict[split] = (src_loader, dst_loader)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gnn.parameters()),
        lr=0.01,
    )

    def model_forward(batch: HeteroData, node_type: NodeType):
        x_dict = encoder(batch.tf_dict)
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        batch_size = batch[node_type].batch_size
        return x_dict[node_type][:batch_size]

    # Training
    encoder.train()
    gnn.train()
    for batch in train_loader:
        src_batch, batch_pos_dst, batch_neg_dst = batch
        # [batch_size, emb_dim]
        x_src = model_forward(src_batch, task.src_entity_table)
        x_pos_dst = model_forward(batch_pos_dst, task.dst_entity_table)
        x_neg_dst = model_forward(batch_neg_dst, task.dst_entity_table)

        # [batch_size, ]
        pos_score = torch.sum(x_src * x_pos_dst, dim=1)
        if share_same_time:
            # [batch_size, batch_size]
            neg_score = x_src @ x_neg_dst.t()
            # [batch_size, 1]
            pos_score = pos_score.view(-1, 1)
        else:
            # [batch_size, ]
            neg_score = torch.sum(x_src * x_neg_dst, dim=1)
        optimizer.zero_grad()
        # BPR loss
        diff_score = pos_score - neg_score
        loss = F.softplus(-diff_score).mean()
        loss.backward()
        optimizer.step()

    # Validation
    encoder.eval()
    gnn.eval()

    for split in ["val", "test"]:
        with torch.no_grad():
            dst_embs = []
            src_loader, dst_loader = eval_loaders_dict[split]
            for batch in dst_loader:
                emb = model_forward(batch, task.dst_entity_table)
                dst_embs.append(emb)
            dst_emb = torch.cat(dst_embs, dim=0)
            del dst_embs

            pred_index_mat_list = []
            for batch in src_loader:
                emb = model_forward(batch, task.src_entity_table)
                _, pred_index_mat = torch.topk(emb @ dst_emb.t(), k=task.eval_k, dim=1)
                pred_index_mat_list.append(pred_index_mat)
            pred = torch.cat(pred_index_mat_list, dim=0).numpy()

        if split == "val":
            task.evaluate(pred, task.get_table("val"))
        else:
            task.evaluate(pred)
