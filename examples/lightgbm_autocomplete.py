import argparse
import json
import os
from pathlib import Path
from typing import Dict

os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import torch
import torch_frame
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything

from relbench import load_task
from relbench.base import EntityTask, TaskType
from relbench.leaderboard import evaluate_task, write_prediction_table
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="results-position")
parser.add_argument("--num_trials", type=int, default=10)
parser.add_argument(
    "--sample_size",
    type=int,
    default=50_000,
    help="Subsample the specified number of training data to train lightgbm model.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--left_join_fkey", action="store_true", default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

task: EntityTask = load_task(args.dataset, args.task)
dataset = task.dataset

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")


dfs: Dict[str, pd.DataFrame] = {}
entity_table = dataset.get_db().table_dict[task.entity_table]
entity_df = entity_table.df

stypes_cache_path = Path(
    f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/stypes.json"
)
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

col_to_stype = col_to_stype_dict[task.entity_table]
remove_pkey_fkey(col_to_stype, entity_table)
for col in dataset.remove_columns:
    if col in col_to_stype:
        del col_to_stype[col]

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    col_to_stype[task.target_col] = torch_frame.categorical
elif task.task_type == TaskType.REGRESSION:
    col_to_stype[task.target_col] = torch_frame.numerical
else:
    raise ValueError(f"Unsupported task type: {task.task_type}")

# randomly subsample in case training data size is too large.
if args.sample_size > 0 and args.sample_size < len(train_table):
    sampled_idx = np.random.permutation(len(train_table))[: args.sample_size]
    train_table.df = train_table.df.iloc[sampled_idx]

for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
    entity_df = entity_df.astype({entity_table.pkey_col: table.df[left_entity].dtype})
    # Remove duplicated columns from entity_df that are already in the table df
    for col in set(entity_df.columns).intersection(set(table.df.columns)):
        if col != entity_table.pkey_col:
            entity_df.pop(col)
    dfs[split] = table.df.merge(
        entity_df,
        how="left",
        left_on=left_entity,
        right_on=entity_table.pkey_col,
    )
    if args.left_join_fkey:
        for fkey_col, pkey_table_name in entity_table.fkey_col_to_pkey_table.items():
            pkey_table = dataset.get_db().table_dict[pkey_table_name]
            dfs[split] = dfs[split].merge(
                pkey_table.df,
                how="left",
                left_on=fkey_col,
                right_on=pkey_table.pkey_col,
                suffixes=("", f"_{pkey_table_name}"),
            )
            pkey_col_to_stype = col_to_stype_dict[pkey_table_name]
            # add appropriate stypes and remove id columns
            for col, stype_str in pkey_col_to_stype.items():
                if col == pkey_table.pkey_col:
                    # remove pkey column
                    if f"{col}_{pkey_table_name}" in dfs[split].columns:
                        dfs[split].pop(f"{col}_{pkey_table_name}")
                    elif col in dfs[split].columns:
                        dfs[split].pop(col)
                elif col in pkey_table.fkey_col_to_pkey_table:
                    # remove fkey columns
                    if col in dfs[split].columns:
                        dfs[split].pop(col)
                    elif f"{col}_{pkey_table_name}" in dfs[split].columns:
                        dfs[split].pop(f"{col}_{pkey_table_name}")
                elif col not in col_to_stype:
                    # add stype for the column
                    col_to_stype[col] = stype(stype_str)
                elif (
                    f"{col}_{pkey_table_name}" not in col_to_stype
                    and f"{col}_{pkey_table_name}" in dfs[split].columns
                ):
                    # add stype for the column with suffix to avoid name collision
                    col_to_stype[f"{col}_{pkey_table_name}"] = stype(stype_str)


train_dataset = torch_frame.data.Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=task.target_col,
    col_to_text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device="cpu"),
        batch_size=256,
    ),
)
path = Path(
    f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/materialized/node_train{'_join' if args.left_join_fkey else ''}.pt"
)
path.parent.mkdir(parents=True, exist_ok=True)
train_dataset = train_dataset.materialize(path=path)

tf_train = train_dataset.tensor_frame
tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    tune_metric = Metric.ROCAUC
elif task.task_type == TaskType.REGRESSION:
    tune_metric = Metric.MAE
else:
    raise ValueError(f"Unsupported task type: {task.task_type}")

model = LightGBM(
    task_type=train_dataset.task_type,
    metric=tune_metric,
)
model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=args.num_trials)

pred = model.predict(tf_test=tf_train).cpu().numpy()
train_metrics = task.evaluate(pred, train_table)

pred = model.predict(tf_test=tf_val).cpu().numpy()
val_metrics = task.evaluate(pred, val_table)

pred = model.predict(tf_test=tf_test).cpu().numpy()
if task.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
    os.makedirs("/tmp/relbench_preds", exist_ok=True)
    pred_path = f"/tmp/relbench_preds/{args.dataset}__{args.task}.csv"
    write_prediction_table(task, pred, pred_path)
    test_metrics = evaluate_task(f"{args.dataset}/{args.task}", pred_path)
else:
    test_metrics = task.evaluate(pred)

print(f"Train: {train_metrics}")
print(f"Val: {val_metrics}")
print(f"Test: {test_metrics}")
