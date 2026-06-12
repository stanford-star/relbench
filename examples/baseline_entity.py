import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from torch_geometric.seed import seed_everything

from relbench.base import Dataset, EntityTask, Table, TaskType
from relbench import load_dataset, load_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stack")
parser.add_argument("--task", type=str, default="user-engagement")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(args.seed)

dataset: Dataset = load_dataset(args.dataset)
task: EntityTask = load_task(args.dataset, args.task)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")


def evaluate(train_table: Table, pred_table: Table, name: str) -> Dict[str, float]:
    is_test = task.target_col not in pred_table.df
    is_multiclass = False
    if name == "global_zero":
        pred = np.zeros(len(pred_table))
    elif name == "global_mean":
        mean = train_table.df[task.target_col].astype(float).values.mean()
        pred = np.ones(len(pred_table)) * mean
    elif name == "global_median":
        median = np.median(train_table.df[task.target_col].astype(float).values)
        pred = np.ones(len(pred_table)) * median
    elif name == "entity_mean":
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: "mean"})
        df.rename(columns={task.target_col: "__target__"}, inplace=True)
        df = pred_table.df.merge(df, how="left", on=fkey)
        pred = df["__target__"].fillna(0).astype(float).values
    elif name == "entity_median":
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: "median"})
        df.rename(columns={task.target_col: "__target__"}, inplace=True)
        df = pred_table.df.merge(df, how="left", on=fkey)
        pred = df["__target__"].fillna(0).astype(float).values
    elif name == "random":
        if not is_multiclass:
            pred = np.random.rand(len(pred_table))
        else:
            pred = np.random.rand(len(pred_table), num_classes)
    elif name == "majority":
        past_target = train_table.df[task.target_col].astype(int)
        majority_label = int(past_target.mode().iloc[0])
        if not is_multiclass:
            pred = torch.full((len(pred_table),), fill_value=majority_label)
        else:
            pred = torch.full((len(pred_table), num_classes), fill_value=0)
            pred[:, majority_label] = 1
    elif name == "majority_multilabel":
        past_target = train_table.df[task.target_col]
        majority = mode(np.stack(past_target.values), axis=0).mode[0]
        pred = np.stack([majority] * len(pred_table.df))
    elif name == "random_multilabel":
        num_labels = train_table.df[task.target_col].values[0].shape[0]
        pred = np.random.rand(len(pred_table), num_labels)
    elif name == "entity_majority_multiclass":
        # Output majority label in the train set if exists, otherwise output randomly
        pred = []
        past_target = train_table.df[task.target_col].astype(int)
        majority_label = int(past_target.mode().iloc[0])
        for idx, row in pred_table.df.iterrows():
            fkey = row[list(train_table.fkey_col_to_pkey_table.keys())[0]]
            if fkey in train_table.df[train_table.pkey_col]:
                train_set_rows = train_table.df.loc[
                    train_table.df[train_table.pkey_col] == fkey
                ]
                majority_labels = train_set_rows[task.target_col].mode()
                if len(majority_labels) > 0:
                    pred.append(majority_labels[0])
                else:
                    pred.append(majority_label)
            else:
                pred.append(majority_label)
        pred = np.array(pred)
    else:
        raise ValueError("Unknown eval name called {name}.")
    return task.evaluate(pred, None if is_test else pred_table)


trainval_table_df = pd.concat([train_table.df, val_table.df], axis=0)
trainval_table = Table(
    df=trainval_table_df,
    fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
    pkey_col=train_table.pkey_col,
    time_col=train_table.time_col,
)

if task.task_type == TaskType.REGRESSION:
    eval_name_list = [
        "global_zero",
        "global_mean",
        "global_median",
        "entity_mean",
        "entity_median",
    ]

    for name in eval_name_list:
        train_metrics = evaluate(train_table, train_table, name=name)
        val_metrics = evaluate(train_table, val_table, name=name)
        test_metrics = evaluate(trainval_table, test_table, name=name)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")


elif task.task_type == TaskType.BINARY_CLASSIFICATION:
    eval_name_list = ["random", "majority", "entity_mean", "entity_median"]
    for name in eval_name_list:
        train_metrics = evaluate(train_table, train_table, name=name)
        val_metrics = evaluate(train_table, val_table, name=name)
        test_metrics = evaluate(trainval_table, test_table, name=name)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")


else:
    raise ValueError(f"Unsupported task type: {task.task_type}")
