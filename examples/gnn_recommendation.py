import argparse
import copy
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from model import Model
from text_embedder import GloveTextEmbedding
from torch import Tensor
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench import load_dataset
from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import LinkNeighborLoader
from relbench.modeling.utils import get_stype_proposal
from relbench.submit import evaluate_task, write_prediction_table

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-hm")
parser.add_argument("--task", type=str, default="user-item-purchase")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--eval_epochs_interval", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--gnn", type=str, default="sage", choices=["sage", "gat"])
parser.add_argument("--gat_heads", type=int, default=4)
parser.add_argument("--gat_dropout", type=float, default=0.0)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
# Use the same seed time across the mini-batch and share the negatives
parser.add_argument("--share_same_time", action="store_true", default=True)
parser.add_argument(
    "--no-share_same_time", dest="share_same_time", action="store_false"
)
# Whether to use shallow embedding on dst nodes or not.
parser.add_argument("--use_shallow", action="store_true", default=True)
parser.add_argument("--no-use_shallow", dest="use_shallow", action="store_false")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

try:
    dataset: Dataset = load_dataset(args.dataset)
    task: RecommendationTask = dataset.load_task(args.task)
except Exception:
    if bool(args.download):
        print(
            "Download failed. If you already have the dataset cached, re-run with --no-download.",
            file=sys.stderr,
        )
    raise
tune_metric = "link_prediction_map"
assert task.task_type == TaskType.LINK_PREDICTION

# Materialize the database once and reuse it: `get_db` is uncached, and this is the
# task's view of it (the columns the task must not see are already dropped).
db = task.get_db()

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(db)
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device="cpu"), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

num_neighbors = [int(args.num_neighbors // 2**i) for i in range(args.num_layers)]

train_table_input = get_link_train_table_input(task.get_table("train"), task)
train_loader = LinkNeighborLoader(
    data=data,
    num_neighbors=num_neighbors,
    time_attr="time",
    src_nodes=train_table_input.src_nodes,
    dst_nodes=train_table_input.dst_nodes,
    num_dst_nodes=train_table_input.num_dst_nodes,
    src_time=train_table_input.src_time,
    share_same_time=args.share_same_time,
    batch_size=args.batch_size,
    temporal_strategy=args.temporal_strategy,
    # if share_same_time is True, we use sampler, so shuffle must be set False
    shuffle=not args.share_same_time,
    num_workers=args.num_workers,
)

eval_loaders_dict: Dict[str, Tuple[NeighborLoader, NeighborLoader]] = {}
for split in ["val", "test"]:
    timestamp = dataset.val_timestamp if split == "val" else dataset.test_timestamp
    seed_time = int(timestamp.timestamp())
    target_table = task.get_table(split)
    if target_table.df.shape[0] == 0:
        warnings.warn(f"Split '{split}' table is empty; skipping {split} evaluation.")
        continue
    src_node_indices = torch.from_numpy(target_table.df[task.src_entity_col].values)
    src_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=(task.src_entity_table, src_node_indices),
        input_time=torch.full(
            size=(len(src_node_indices),), fill_value=seed_time, dtype=torch.long
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    dst_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        input_nodes=task.dst_entity_table,
        input_time=torch.full(
            size=(task.num_dst_nodes,), fill_value=seed_time, dtype=torch.long
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    eval_loaders_dict[split] = (src_loader, dst_loader)

model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=args.channels,
    aggr=args.aggr,
    norm="layer_norm",
    shallow_list=[task.dst_entity_table] if args.use_shallow else [],
    gnn=args.gnn,
    gat_heads=args.gat_heads,
    gat_dropout=args.gat_dropout,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(train_loader), args.max_steps_per_epoch)
    for batch in tqdm(train_loader, total=total_steps):
        src_batch, batch_pos_dst, batch_neg_dst = batch
        src_batch, batch_pos_dst, batch_neg_dst = (
            src_batch.to(device),
            batch_pos_dst.to(device),
            batch_neg_dst.to(device),
        )
        x_src = model(src_batch, task.src_entity_table)
        x_pos_dst = model(batch_pos_dst, task.dst_entity_table)
        x_neg_dst = model(batch_neg_dst, task.dst_entity_table)

        # [batch_size, ]
        pos_score = torch.sum(x_src * x_pos_dst, dim=1)
        if args.share_same_time:
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

        loss_accum += float(loss) * x_src.size(0)
        count_accum += x_src.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    if count_accum == 0:
        warnings.warn(
            f"Did not sample a single '{task.dst_entity_table}' "
            f"node in any mini-batch. Try to increase the number "
            f"of layers/hops and re-try. If you run into memory "
            f"issues with deeper nets, decrease the batch size."
        )

    return loss_accum / count_accum if count_accum > 0 else float("nan")


@torch.no_grad()
def test(src_loader: NeighborLoader, dst_loader: NeighborLoader) -> np.ndarray:
    model.eval()

    dst_embs: list[Tensor] = []
    for batch in tqdm(dst_loader):
        batch = batch.to(device)
        emb = model(batch, task.dst_entity_table).detach()
        dst_embs.append(emb)
    dst_emb = torch.cat(dst_embs, dim=0)
    del dst_embs

    pred_index_mat_list: list[Tensor] = []
    for batch in tqdm(src_loader):
        batch = batch.to(device)
        emb = model(batch, task.src_entity_table)
        _, pred_index_mat = torch.topk(emb @ dst_emb.t(), k=task.eval_k, dim=1)
        pred_index_mat_list.append(pred_index_mat.cpu())
    if len(pred_index_mat_list) == 0:
        return np.empty((0, task.eval_k), dtype=np.int64)
    pred = torch.cat(pred_index_mat_list, dim=0).numpy()
    return pred


state_dict = None
best_val_metric = 0
val_table = task.get_table("val")  # hoisted: get_table is uncached

for epoch in range(1, args.epochs + 1):
    train_loss = train()
    if epoch % args.eval_epochs_interval == 0:
        val_pred = test(*eval_loaders_dict["val"])
        val_metrics = task.evaluate(val_pred, val_table)
        print(
            f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
            f"Val metrics: {val_metrics}"
        )

        if val_metrics[tune_metric] >= best_val_metric:
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())


if state_dict is not None:
    model.load_state_dict(state_dict)
else:
    warnings.warn(
        "No best checkpoint was selected (state_dict is None); evaluating with current model weights."
    )
val_pred = test(*eval_loaders_dict["val"])
val_metrics = task.evaluate(val_pred, val_table)
print(f"Best Val metrics: {val_metrics}")

if "test" in eval_loaders_dict:
    test_pred = test(*eval_loaders_dict["test"])
    os.makedirs("/tmp/relbench_preds", exist_ok=True)
    pred_path = f"/tmp/relbench_preds/{args.dataset}__{args.task}.csv"
    write_prediction_table(task, test_pred, pred_path)
    test_metrics = evaluate_task(f"{args.dataset}/{args.task}", pred_path)
    print(f"Best test metrics: {test_metrics}")
else:
    print("Best test metrics: <skipped: empty test split>")
