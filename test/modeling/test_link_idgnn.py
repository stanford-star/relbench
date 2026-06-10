from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.base.task_base import TaskType
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import SparseTensor
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE
from relbench.modeling.utils import get_stype_proposal


def test_link_train_fake_product_dataset(tmp_path, make_purchase_task, fake_dataset):
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
    channels = 64
    encoder = HeteroEncoder(channels, node_to_col_names_dict, col_stats_dict)
    gnn = HeteroGraphSAGE(data.node_types, data.edge_types, channels)
    head = MLP(channels, out_channels=1, num_layers=1)
    id_awareness = torch.nn.Embedding(1, channels)

    # Ensure that neighbor loading works on train/val/test splits ############
    task = make_purchase_task(dataset)
    assert task.task_type == TaskType.LINK_PREDICTION

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    loader_dict: Dict[str, NeighborLoader] = {}
    dst_nodes_dict: Dict[str, Tuple[NodeType, Tensor]] = {}
    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        table_input = get_link_train_table_input(table, task)
        dst_nodes_dict[split] = table_input.dst_nodes
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[8, 8],
            time_attr="time",
            input_nodes=table_input.src_nodes,
            input_time=table_input.src_time,
            subgraph_type="bidirectional",
            batch_size=16,
            temporal_strategy="uniform",
            shuffle=split == "train",
            num_workers=1,
            persistent_workers=True,
        )

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(gnn.parameters())
        + list(head.parameters())
        + list(id_awareness.parameters()),
        lr=0.01,
    )
    entity_table = table_input.src_nodes[0]
    dst_table = table_input.dst_nodes[0]

    # Training
    encoder.train()
    gnn.train()
    head.train()
    id_awareness.train()
    train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1])
    for batch in loader_dict["train"]:
        batch_size = batch[entity_table].batch_size
        x_dict = encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][:batch_size] += id_awareness.weight
        x_dict = gnn(
            x_dict,
            batch.edge_index_dict,
        )
        out = head(x_dict[dst_table]).flatten()

        # Get ground-truth
        input_id = batch[entity_table].input_id
        src_batch, dst_index = train_sparse_tensor[input_id]

        # Get target label
        target = torch.isin(
            batch[dst_table].batch + batch_size * batch[dst_table].n_id,
            src_batch + batch_size * dst_index,
        ).float()

        # Optimization
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        optimizer.step()

    # Validation
    encoder.eval()
    gnn.eval()
    head.eval()
    id_awareness.eval()
    for split in ["val", "test"]:
        pred_list = []
        for batch in loader_dict[split]:
            with torch.no_grad():
                x_dict = encoder(batch.tf_dict)
                # Add ID-awareness to the root node
                batch_size = batch[entity_table].batch_size
                x_dict[entity_table][:batch_size] += id_awareness.weight
                x_dict = gnn(
                    x_dict,
                    batch.edge_index_dict,
                )
                out = head(x_dict[dst_table]).flatten()
                batch_size = batch[entity_table].batch_size
                scores = torch.zeros(batch_size, task.num_dst_nodes, device=out.device)
                scores[batch[dst_table].batch, batch[dst_table].n_id] = torch.sigmoid(
                    out
                )
                _, pred_mini = torch.topk(scores, k=task.eval_k, dim=1)
                pred_list.append(pred_mini)
        pred = torch.cat(pred_list, dim=0).numpy()
        if split == "val":
            task.evaluate(pred, val_table)
        else:
            task.evaluate(pred)
