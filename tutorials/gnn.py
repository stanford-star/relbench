import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Training a GNN on RelBench

        A Graph Neural Network baseline for a RelBench entity task, with PyTorch Geometric
        (graph) and PyTorch Frame (tabular features).

        > **Download and run locally.** This notebook needs `torch` and `torch_geometric`
        > and is best on a GPU. Install with `pip install relbench[full]`, then open it with
        > `marimo edit gnn.py` (or run it as a script: `python gnn.py`).
        """
    )
    return


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from torch_frame.config.text_embedder import TextEmbedderConfig
    from torch_frame.testing.text_embedder import HashTextEmbedder
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import MLP

    import relbench
    from relbench.modeling.graph import (
        get_node_train_table_input,
        make_pkey_fkey_graph,
    )
    from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE
    from relbench.modeling.utils import get_stype_proposal

    return (
        F,
        HashTextEmbedder,
        HeteroEncoder,
        HeteroGraphSAGE,
        MLP,
        NeighborLoader,
        TextEmbedderConfig,
        get_node_train_table_input,
        get_stype_proposal,
        make_pkey_fkey_graph,
        relbench,
        torch,
    )


@app.cell
def _(mo):
    mo.md("## Load the dataset and a (binary) entity task")
    return


@app.cell
def _(relbench):
    dataset = relbench.load_dataset("rel-f1")
    task = relbench.load_task("rel-f1", "driver-dnf")  # will a driver DNF soon?
    db = dataset.get_db()
    return db, task


@app.cell
def _(mo):
    mo.md(
        "## Build the heterogeneous graph\n\n"
        "`make_pkey_fkey_graph` turns the relational database into a PyG `HeteroData` graph "
        "using the foreign-key edges; a text embedder encodes string columns."
    )
    return


@app.cell
def _(HashTextEmbedder, TextEmbedderConfig, db, get_stype_proposal, make_pkey_fkey_graph):
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        get_stype_proposal(db),
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=HashTextEmbedder(8), batch_size=None
        ),
        cache_dir=None,
    )
    return col_stats_dict, data


@app.cell
def _(HeteroEncoder, HeteroGraphSAGE, MLP, col_stats_dict, data, torch):
    channels = 64
    node_cols = {nt: data[nt].tf.col_names_dict for nt in data.node_types}
    encoder = HeteroEncoder(channels, node_cols, col_stats_dict)
    gnn = HeteroGraphSAGE(data.node_types, data.edge_types, channels)
    head = MLP(channels, out_channels=1, num_layers=1)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gnn.parameters()) + list(head.parameters()),
        lr=0.01,
    )
    return encoder, gnn, head, optimizer


@app.cell
def _(NeighborLoader, data, get_node_train_table_input, task):
    loaders = {}
    for split in ["train", "val", "test"]:
        ti = get_node_train_table_input(task.get_table(split), task=task)
        loaders[split] = NeighborLoader(
            data,
            num_neighbors=[-1, -1],
            time_attr="time",
            input_nodes=ti.nodes,
            input_time=ti.time,
            transform=ti.transform,
            batch_size=256,
            shuffle=split == "train",
        )
    entity = task.entity_table
    return entity, loaders


@app.cell
def _(mo):
    mo.md("## Train a few epochs")
    return


@app.cell
def _(F, encoder, entity, gnn, head, loaders, optimizer, torch):
    def train():
        for epoch in range(1, 4):
            encoder.train(), gnn.train(), head.train()
            for batch in loaders["train"]:
                seed = batch[entity].batch_size
                x = encoder(batch.tf_dict)
                x = gnn(
                    x,
                    batch.edge_index_dict,
                    batch.num_sampled_nodes_dict,
                    batch.num_sampled_edges_dict,
                )
                pred = head(x[entity][:seed]).squeeze(-1)
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(pred, batch[entity].y.float())
                loss.backward()
                optimizer.step()
            print(f"epoch {epoch}: train loss {loss.item():.4f}")

    train()
    return


@app.cell
def _(encoder, entity, gnn, head, loaders, task, torch):
    def predict():
        encoder.eval(), gnn.eval(), head.eval()
        preds = []
        for batch in loaders["test"]:
            seed = batch[entity].batch_size
            with torch.no_grad():
                x = encoder(batch.tf_dict)
                x = gnn(
                    x,
                    batch.edge_index_dict,
                    batch.num_sampled_nodes_dict,
                    batch.num_sampled_edges_dict,
                )
                preds.append(head(x[entity][:seed]).squeeze(-1).sigmoid().cpu())
        return torch.cat(preds).numpy()

    test_metrics = task.evaluate(predict())
    test_metrics
    return


if __name__ == "__main__":
    app.run()
