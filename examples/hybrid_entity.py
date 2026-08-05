import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch_frame
from model import Model
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.gbdt import LightGBM
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench import load_dataset
from relbench.base import Dataset, EntityTask, TaskType
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.submit import evaluate_task, write_prediction_table

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stack")
parser.add_argument("--task", type=str, default="user-engagement")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
# If true, try to load pretrained state dict
parser.add_argument("--attempt_load_state_dict", action="store_true", default=False)
parser.add_argument(
    "--no_attempt_load_state_dict", dest="attempt_load_state_dict", action="store_false"
)
parser.add_argument(
    "--sample_size",
    type=int,
    default=50_000,
    help="Subsample the specified number of training data for training split.",
)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
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

dataset: Dataset = load_dataset(args.dataset)
task: EntityTask = dataset.load_task(args.task)

# Materialize the database once and reuse it: `get_db` is uncached, and this is
# the task\'s view of it (the columns the task must not see are already dropped).
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

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "nmae"
    higher_is_better = False
    # Get the clamp value at inference time
    clamp_min, clamp_max = np.percentile(
        task.get_table("train").df[task.target_col].to_numpy(), [2, 98]
    )
else:
    raise ValueError(f"Unsupported task type: {task.task_type}")

loader_dict: Dict[str, NeighborLoader] = {}
# Create a mapping for each split's entity table
entity_table_mapping: Dict[str, str] = {}

for split in ["train", "val", "test"]:
    table = task.get_table(split)
    # Restrict the training split to the first `sample_size` rows.
    # Note: This approach is inefficient as it loads the full dataset before discarding unused rows.
    # Improvement Suggestion: To optimize, modify the logic inside the `make_pkey_fkey_graph` function
    # to limit the dataset size during its creation, avoiding unnecessary materialization.
    if split == "train" and args.sample_size is not None:
        table.df = table.df.iloc[: args.sample_size].reset_index(drop=True)

    table_input = get_node_train_table_input(table=table, task=task)
    # Save the entity table name for this split
    entity_table_mapping[split] = table_input.nodes[0]

    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            entity_table_mapping["train"],
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table_mapping["train"]].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch,
            entity_table_mapping["test"],
        )
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


@torch.no_grad()
def embed(
    model: Model,
    loader: NeighborLoader,
    no_label: bool = False,
    stop_at: int = None,
) -> Dict[str, float]:

    # remove model.head from the model
    from torch.nn import Identity

    model_embed = copy.deepcopy(model)
    model_embed.head = Identity()  # remove the head
    model_embed.eval()

    embed_list = []
    y_list = []
    for idx, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        embed = model_embed(
            batch,
            entity_table_mapping["train"],
        )
        embed_list.append(embed.detach().cpu())

        if not no_label:
            y = batch[entity_table_mapping["train"]].y
            y_list.append(y.detach().cpu())

        if stop_at is not None and idx >= stop_at:
            break

    emb = torch.cat(embed_list, dim=0)

    if not no_label:
        y = torch.cat(y_list, dim=0)
    else:
        y = None

    return emb, y


# =====================
# Model training
# =====================
model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
).to(device)


STATE_DICT_PTH = f"results/{args.dataset}_{args.task}_state_dict.pth"

# if state dict exists, load it
if os.path.exists(STATE_DICT_PTH) and args.attempt_load_state_dict:
    # load state dict
    state_dict = torch.load(STATE_DICT_PTH)
    model.load_state_dict(state_dict)

else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state_dict = None
    best_val_metric = 0 if higher_is_better else math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        val_pred = test(loader_dict["val"])
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        print(
            f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}"
        )

        if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better and val_metrics[tune_metric] < best_val_metric
        ):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())

    # save state dict
    if args.attempt_load_state_dict:
        torch.save(state_dict, STATE_DICT_PTH)


val_pred_accum = 0
test_pred_accum = 0

print("=====================")
print("GNN model performance")
print("=====================")
model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
os.makedirs("/tmp/relbench_preds", exist_ok=True)
pred_path = f"/tmp/relbench_preds/{args.dataset}__{args.task}.csv"
write_prediction_table(task, test_pred, pred_path)
test_metrics = evaluate_task(f"{args.dataset}/{args.task}", pred_path)
print(f"Best test metrics: {test_metrics}")


print("=====================")
print("Embedding performance")
print("=====================")

emb_train, y_train = embed(
    model,
    loader_dict["train"],
    stop_at=args.sample_size // args.batch_size if args.sample_size else None,
)
emb_val, y_val = embed(model, loader_dict["val"])
emb_test, _ = embed(model, loader_dict["test"], no_label=True)


# hack to convert to torch_frame
def tensor_to_tf(data, y=None):
    tf = torch_frame.TensorFrame(
        feat_dict={torch_frame.numerical: data},
        col_names_dict={
            torch_frame.numerical: [f"feat_{i}" for i in range(data.shape[1])],
        },
    )
    if y is not None:
        tf.y = y

    return tf


tf_train = tensor_to_tf(emb_train, y_train)
tf_val = tensor_to_tf(emb_val, y_val)
tf_test = tensor_to_tf(emb_test)


from torch_frame import TaskType as TaskTypeTorchFrame

# rename tune_metric to  torch-frame Metric format
from torch_frame.typing import Metric

if tune_metric == "roc_auc":
    tune_metric = Metric.ROCAUC
elif tune_metric == "nmae":
    tune_metric = Metric.MAE


relbench2torch_frame = {
    TaskType.BINARY_CLASSIFICATION: TaskTypeTorchFrame.BINARY_CLASSIFICATION,
    TaskType.REGRESSION: TaskTypeTorchFrame.REGRESSION,
}
task_type = relbench2torch_frame[task.task_type]

lgbm_model = LightGBM(task_type=task_type, metric=tune_metric)
lgbm_model.tune(tf_train, tf_val, num_trials=10)

pred = lgbm_model.predict(tf_val).numpy()
val_metrics = task.evaluate(pred, task.get_table("val"))
print(f"LightGBM Val metrics: {val_metrics}")

test_pred = lgbm_model.predict(tf_test).numpy()
os.makedirs("/tmp/relbench_preds", exist_ok=True)
pred_path = f"/tmp/relbench_preds/{args.dataset}__{args.task}.csv"
write_prediction_table(task, test_pred, pred_path)
test_metrics = evaluate_task(f"{args.dataset}/{args.task}", pred_path)
print(f"LightGBM Best Test metrics: {test_metrics}")
