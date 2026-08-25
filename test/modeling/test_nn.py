import pytest
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP

from relbench.modeling.graph import (
    get_link_train_table_input,
    get_node_train_table_input,
    num_dst_nodes,
)
from relbench.modeling.loader import LinkNeighborLoader, SparseTensor
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE

CHANNELS = 32


def _models(data, col_stats):
    encoder = HeteroEncoder(
        CHANNELS,
        {node_type: data[node_type].tf.col_names_dict for node_type in data.node_types},
        col_stats,
    )
    gnn = HeteroGraphSAGE(data.node_types, data.edge_types, CHANNELS)
    return encoder, gnn


def _embed(encoder, gnn, batch, node_type):
    x_dict = gnn(encoder(batch.tf_dict), batch.edge_index_dict)
    return x_dict[node_type][: batch[node_type].batch_size]


def _step(optimizer, loss):
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_entity_train_step_and_eval(graph, churn_task):
    data, col_stats = graph
    encoder, gnn = _models(data, col_stats)
    head = MLP(CHANNELS, out_channels=1, num_layers=1)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *gnn.parameters(), *head.parameters()], lr=0.01
    )
    entity = churn_task.entity_table

    def loader(split):
        inp = get_node_train_table_input(churn_task.get_table(split), churn_task)
        return NeighborLoader(
            data,
            num_neighbors=[-1, -1],
            time_attr="time",
            input_nodes=inp.nodes,
            input_time=inp.time,
            transform=inp.transform,
            batch_size=64,
            shuffle=split == "train",
        )

    batch = next(iter(loader("train")))
    batch_size = batch[entity].batch_size
    assert batch[entity].y.size() == (batch_size,)
    assert batch[entity].seed_time.size() == (batch_size,)
    assert (
        batch["review"].time <= batch[entity].seed_time[batch["review"].batch]
    ).all()
    pred = head(_embed(encoder, gnn, batch, entity)).squeeze(-1)
    _step(optimizer, F.binary_cross_entropy_with_logits(pred, batch[entity].y.float()))

    val = churn_task.get_table("val")
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader("val"):
            preds.append(
                head(_embed(encoder, gnn, batch, entity)).squeeze(-1).sigmoid()
            )
            targets.append(batch[entity].y)
    assert torch.cat(targets).tolist() == val.df[churn_task.target_col].tolist()
    assert 0 <= churn_task.evaluate(torch.cat(preds).numpy(), val)["roc_auc"] <= 1

    assert not hasattr(next(iter(loader("test")))[entity], "y")


@pytest.mark.parametrize("share_same_time", [True, False])
def test_link_train_step_and_eval(graph, purchase_task, dataset, share_same_time):
    data, col_stats = graph
    encoder, gnn = _models(data, col_stats)
    optimizer = torch.optim.Adam([*encoder.parameters(), *gnn.parameters()], lr=0.01)
    src, dst = purchase_task.src_entity_table, purchase_task.dst_entity_table
    n_dst = num_dst_nodes(dataset.get_db(), purchase_task)
    inp = get_link_train_table_input(
        purchase_task.get_table("train"), purchase_task, n_dst
    )
    batch_size = 16
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[-1, -1],
        time_attr="time",
        src_nodes=inp.src_nodes,
        dst_nodes=inp.dst_nodes,
        num_dst_nodes=inp.num_dst_nodes,
        src_time=inp.src_time,
        share_same_time=share_same_time,
        batch_size=batch_size,
        shuffle=not share_same_time,
        drop_last=not share_same_time,
    )
    src_batch, pos_batch, neg_batch = next(iter(loader))
    seed_times = [
        b[t].seed_time
        for b, t in [(src_batch, src), (pos_batch, dst), (neg_batch, dst)]
    ]
    assert all(t.size() == (batch_size,) for t in seed_times)
    assert torch.equal(seed_times[0], seed_times[1]) and torch.equal(
        seed_times[0], seed_times[2]
    )
    if share_same_time:
        assert (seed_times[0] == seed_times[0][0]).all()
    x_src = _embed(encoder, gnn, src_batch, src)
    x_pos = _embed(encoder, gnn, pos_batch, dst)
    x_neg = _embed(encoder, gnn, neg_batch, dst)
    pos_score = (x_src * x_pos).sum(dim=1)
    neg_score = x_src @ x_neg.t() if share_same_time else (x_src * x_neg).sum(dim=1)
    _step(optimizer, F.softplus(-(pos_score.view(-1, 1) - neg_score)).mean())

    val = purchase_task.get_table("val")
    seed_time = int(dataset.val_timestamp.timestamp())
    src_loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        time_attr="time",
        input_nodes=(
            src,
            torch.from_numpy(val.df[purchase_task.src_entity_col].values),
        ),
        input_time=torch.full((len(val),), seed_time, dtype=torch.long),
        batch_size=128,
    )
    dst_loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        time_attr="time",
        input_nodes=dst,
        input_time=torch.full((n_dst,), seed_time, dtype=torch.long),
        batch_size=128,
    )
    with torch.no_grad():
        dst_emb = torch.cat([_embed(encoder, gnn, b, dst) for b in dst_loader])
        pred = torch.cat(
            [
                torch.topk(
                    _embed(encoder, gnn, b, src) @ dst_emb.t(),
                    k=purchase_task.eval_k,
                    dim=1,
                ).indices
                for b in src_loader
            ]
        )
    assert pred.size() == (len(val), purchase_task.eval_k)
    assert 0 <= purchase_task.evaluate(pred.numpy(), val)["map"] <= 1


def test_idgnn_train_step(graph, purchase_task, dataset):
    data, col_stats = graph
    encoder, gnn = _models(data, col_stats)
    head = MLP(CHANNELS, out_channels=1, num_layers=1)
    id_awareness = torch.nn.Embedding(1, CHANNELS)
    optimizer = torch.optim.Adam(
        [
            *encoder.parameters(),
            *gnn.parameters(),
            *head.parameters(),
            *id_awareness.parameters(),
        ],
        lr=0.01,
    )
    src, dst = purchase_task.src_entity_table, purchase_task.dst_entity_table
    n_dst = num_dst_nodes(dataset.get_db(), purchase_task)
    inp = get_link_train_table_input(
        purchase_task.get_table("train"), purchase_task, n_dst
    )
    loader = NeighborLoader(
        data,
        num_neighbors=[8, 8],
        time_attr="time",
        input_nodes=inp.src_nodes,
        input_time=inp.src_time,
        subgraph_type="bidirectional",
        batch_size=16,
        shuffle=True,
    )
    sparse = SparseTensor(inp.dst_nodes[1])
    batch = next(iter(loader))
    batch_size = batch[src].batch_size
    x_dict = encoder(batch.tf_dict)
    x_dict[src][:batch_size] += id_awareness.weight
    out = head(gnn(x_dict, batch.edge_index_dict)[dst]).flatten()
    src_batch, dst_index = sparse[batch[src].input_id]
    target = torch.isin(
        batch[dst].batch + batch_size * batch[dst].n_id,
        src_batch + batch_size * dst_index,
    ).float()
    assert out.size() == target.size() == (batch[dst].num_nodes,)
    _step(optimizer, F.binary_cross_entropy_with_logits(out, target))

    scores = torch.zeros(batch_size, n_dst)
    scores[batch[dst].batch, batch[dst].n_id] = out.detach().sigmoid()
    assert torch.topk(scores, k=purchase_task.eval_k, dim=1).indices.size() == (
        batch_size,
        purchase_task.eval_k,
    )


def test_forward_with_isolated_nodes(fake_dataset, make_graph):
    dataset = fake_dataset(num_customers=50, num_reviews=1)
    data, col_stats = make_graph(dataset.get_db())
    encoder, gnn = _models(data, col_stats)
    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        time_attr="time",
        input_nodes=("customer", torch.arange(50)),
        input_time=torch.zeros(50, dtype=torch.long),
        batch_size=50,
    )
    assert _embed(encoder, gnn, next(iter(loader)), "customer").size() == (50, CHANNELS)
