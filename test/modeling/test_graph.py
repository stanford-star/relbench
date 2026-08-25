import torch
from torch_frame import TensorFrame, stype

from relbench.modeling.graph import (
    get_link_train_table_input,
    get_node_train_table_input,
    num_dst_nodes,
)
from relbench.modeling.utils import to_unix_time


def test_graph_structure(graph, dataset):
    data, col_stats = graph
    data.validate()
    assert set(data.node_types) == {"customer", "review", "product", "relations"}
    assert set(col_stats) == set(data.node_types)
    assert data["customer"].num_nodes == dataset.num_customers
    assert data["product"].num_nodes == dataset.num_products
    assert data["review"].num_nodes == len(dataset.get_db().table_dict["review"])
    for node_type in data.node_types:
        assert isinstance(data[node_type].tf, TensorFrame)
    assert "__const__" in data["relations"].tf.col_names_dict[stype.numerical]
    assert data["review"].time.dtype == torch.int64
    assert not hasattr(data["customer"], "time")

    assert len(data.edge_types) == 8
    for src, rel, dst in data.edge_types:
        edge_index = data[src, rel, dst].edge_index
        assert edge_index.size(0) == 2
        assert edge_index[0].max() < data[src].num_nodes
        assert edge_index[1].max() < data[dst].num_nodes
        reverse = (
            (dst, f"rev_{rel}", src)
            if not rel.startswith("rev_")
            else (dst, rel[4:], src)
        )
        assert reverse in data.edge_types
    review = dataset.get_db().table_dict["review"].df
    assert (
        data["review", "f2p_customer_id", "customer"].num_edges
        == review["customer_id"].notna().sum()
    )


def test_graph_cache_roundtrip(graph, dataset, cache_dir, make_graph):
    data, _ = graph
    assert sorted(p.name for p in cache_dir.iterdir()) == [
        "customer.pt",
        "product.pt",
        "relations.pt",
        "review.pt",
    ]
    again, _ = make_graph(dataset.get_db(), cache_dir)
    for node_type in data.node_types:
        assert again[node_type].tf.num_rows == data[node_type].tf.num_rows
        assert again[node_type].tf.col_names_dict == data[node_type].tf.col_names_dict


def test_remove_columns_from_cached_graph(graph, dataset, cache_dir, make_graph):
    full, _ = graph
    hidden = [
        ("review", "rating"),
        ("product", "category"),
        ("product", "title"),
        ("product", "price"),
    ]
    data, col_stats = make_graph(dataset.get_db(), cache_dir, remove_columns=hidden)
    data.validate()
    assert data["review"].tf.col_names_dict[stype.categorical] == ["review"]
    assert data["review"].tf.feat_dict[stype.categorical].size() == (540, 1)
    assert data["review"].tf.col_names_dict[stype.timestamp] == ["review_time"]
    assert "rating" not in col_stats["review"] and "review" in col_stats["review"]
    assert data["product"].tf.col_names_dict == {stype.numerical: ["__const__"]}
    assert data["product"].num_nodes == full["product"].num_nodes
    assert set(col_stats["product"]) == {"__const__"}
    assert data["customer"].tf.col_names_dict == full["customer"].tf.col_names_dict
    assert full["review"].tf.col_names_dict[stype.categorical] == ["rating", "review"]


def test_node_train_table_input(churn_task):
    table = churn_task.get_table("train")
    inp = get_node_train_table_input(table, churn_task)
    assert inp.nodes[0] == churn_task.entity_table
    assert inp.nodes[1].tolist() == table.df[churn_task.entity_col].tolist()
    assert inp.time.tolist() == to_unix_time(table.df[churn_task.time_col]).tolist()
    assert inp.target.dtype == torch.float64
    assert inp.target.tolist() == table.df[churn_task.target_col].tolist()
    masked = get_node_train_table_input(churn_task.get_table("test"), churn_task)
    assert masked.target is None and masked.transform is None


def test_link_train_table_input(purchase_task, dataset):
    table = purchase_task.get_table("train")
    n_dst = num_dst_nodes(dataset.get_db(), purchase_task)
    assert n_dst == dataset.num_products
    inp = get_link_train_table_input(table, purchase_task, n_dst)
    assert inp.src_nodes[0] == purchase_task.src_entity_table
    assert inp.dst_nodes[0] == purchase_task.dst_entity_table
    assert inp.src_nodes[1].tolist() == table.df[purchase_task.src_entity_col].tolist()
    assert (
        inp.src_time.tolist() == to_unix_time(table.df[purchase_task.time_col]).tolist()
    )
    assert inp.dst_nodes[1].size() == (len(table), n_dst)
    for i, row in enumerate(table.df[purchase_task.dst_entity_col].head(20)):
        assert inp.dst_nodes[1][i].indices()[0].tolist() == sorted(row)
    masked = get_link_train_table_input(
        purchase_task.get_table("test"), purchase_task, n_dst
    )
    assert masked.dst_nodes is None and masked.src_time is not None
