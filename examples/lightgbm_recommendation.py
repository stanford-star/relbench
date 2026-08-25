import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch_frame
from text_embedder import GloveTextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.gbdt import LightGBM
from torch_frame.typing import Metric
from torch_geometric.seed import seed_everything

from relbench import load_dataset
from relbench.base import Dataset, RecommendationTask, Table
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from relbench.submit import evaluate_task, write_prediction_table

REC_BASELINE_TARGET_COL_NAME = "rec_baseline_target_column_name"
PRED_SCORE_COL_NAME = "pred_score_col_name"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stack")
parser.add_argument("--task", type=str, default="user-post-comment")
parser.add_argument("--num_trials", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--sample_size",
    type=int,
    default=50_000,
    help="Subsample the specified number of training data to train lightgbm model.",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
parser.add_argument("--pred_dir", type=str, default="/tmp/relbench_preds")
args = parser.parse_args()

seed_everything(args.seed)

dataset: Dataset = load_dataset(args.dataset)
task: RecommendationTask = dataset.load_task(args.task)

# Features come from the task's view of the db; the stype cache is built per dataset.
db = task.get_db()
target_col_name: str = REC_BASELINE_TARGET_COL_NAME

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")
src_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
dst_entity = list(train_table.fkey_col_to_pkey_table.keys())[1]
time_col = train_table.time_col

# We plan to merge train table with entity and target table to include both
# entity and target table features during lightGBM training.
dfs: Dict[str, pd.DataFrame] = {}

src_entity_table = db.table_dict[task.src_entity_table]
src_entity_df = src_entity_table.df
dst_entity_table = db.table_dict[task.dst_entity_table]
dst_entity_df = dst_entity_table.df

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
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
for table, col in task.hidden_columns():
    col_to_stype_dict.get(table, {}).pop(col, None)


# Prepare col_to_stype dictionary mapping between column names and stypes
# for torch_frame Dataset initialization.
col_to_stype = {}
src_entity_table_col_to_stype = copy.deepcopy(col_to_stype_dict[task.src_entity_table])
dst_entity_table_col_to_stype = copy.deepcopy(col_to_stype_dict[task.dst_entity_table])

remove_pkey_fkey(src_entity_table_col_to_stype, src_entity_table)
remove_pkey_fkey(dst_entity_table_col_to_stype, dst_entity_table)

# Rename the column to stype column names appearing in both `src_entity_table`
# and `dst_entity_table` with `_x` and `_y` suffix respectively since they
# will automatically be renamed this way after train/val/test table join with
# both of them in torch frame data preparation.
src_dst_intersection_column_names = set(src_entity_table_col_to_stype.keys()) & set(
    dst_entity_table_col_to_stype.keys()
)
for column_name in src_dst_intersection_column_names:
    src_entity_table_col_to_stype[f"{column_name}_x"] = src_entity_table_col_to_stype[
        column_name
    ]
    del src_entity_table_col_to_stype[column_name]
    dst_entity_table_col_to_stype[f"{column_name}_y"] = dst_entity_table_col_to_stype[
        column_name
    ]
    del dst_entity_table_col_to_stype[column_name]
col_to_stype.update(src_entity_table_col_to_stype)
col_to_stype.update(dst_entity_table_col_to_stype)
col_to_stype["num_past_visit"] = torch_frame.numerical
col_to_stype["global_popularity_fraction"] = torch_frame.numerical
col_to_stype[target_col_name] = torch_frame.categorical

# randomly subsample in case training data size is too large.
sampled_train_table = copy.deepcopy(train_table)
if args.sample_size > 0 and args.sample_size < len(sampled_train_table):
    sampled_idx = np.random.permutation(len(sampled_train_table))[: args.sample_size]
    sampled_train_table.df = sampled_train_table.df.iloc[sampled_idx]


def dst_entities_aggr(dst_entities):
    r"concatenate and rank dst entities"
    dst_entities_concat = []
    for dst_entity_list in list(dst_entities):
        dst_entities_concat.extend(dst_entity_list)
    counter = Counter(dst_entities_concat)
    topk = [elem for elem, _ in counter.most_common(task.eval_k)]
    return topk


label_df = pd.concat([train_table.df, val_table.df], ignore_index=True)
past_pairs = label_df[[time_col, src_entity, dst_entity]].explode(dst_entity)
past_pairs[dst_entity] = past_pairs[dst_entity].astype(int)


def past_label_counts(keys):
    counts = (
        past_pairs.groupby([*keys, time_col])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(time_col, kind="stable")
    )
    counts["count"] = counts.groupby(keys)["count"].cumsum()
    return counts


pair_counts = past_label_counts([src_entity, dst_entity])
dst_counts = past_label_counts([dst_entity])
max_dst_counts = dst_counts.groupby(time_col)["count"].max().cummax().reset_index()


def add_past_label_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add, per row, the number of past visits of its src entity to its dst entity and
    the dst entity's popularity relative to the most popular dst entity, both counted
    over the label rows strictly before the row's timestamp."""
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    df[dst_entity] = df[dst_entity].astype(int)
    df = pd.merge_asof(
        df,
        pair_counts.rename(columns={"count": "num_past_visit"}),
        on=time_col,
        by=[src_entity, dst_entity],
        allow_exact_matches=False,
    )
    df = pd.merge_asof(
        df,
        dst_counts.rename(columns={"count": "dst_count"}),
        on=time_col,
        by=dst_entity,
        allow_exact_matches=False,
    )
    df = pd.merge_asof(
        df,
        max_dst_counts.rename(columns={"count": "max_count"}),
        on=time_col,
        allow_exact_matches=False,
    )
    df["num_past_visit"] = df["num_past_visit"].fillna(0).astype(int)
    df["global_popularity_fraction"] = (df["dst_count"] / df["max_count"]).fillna(0)
    return df.drop(columns=["dst_count", "max_count"])


# Prepare train/val dataset for lightGBM model training. For each src
# entity, their corresponding dst entities are used as positive label.
# The same number of random dst entities are sampled as negative label.
# lightGBM will train and eval on this binary classification task.
for split, table in [
    ("train", sampled_train_table),
    ("val", val_table),
]:
    src_entity_df = src_entity_df.astype(
        {src_entity_table.pkey_col: table.df[src_entity].dtype}
    )

    dst_entity_df = dst_entity_df.astype(
        {dst_entity_table.pkey_col: table.df[dst_entity].dtype}
    )

    # Left join train table and entity table
    df = table.df.merge(
        src_entity_df,
        how="left",
        left_on=src_entity,
        right_on=src_entity_table.pkey_col,
    )

    # Transform the mapping between one src entity with a list of dst entities
    # to src entity, dst entity pairs
    df = df.explode(dst_entity)

    # Add a target col indicating there is a link between src and dst entities
    df[target_col_name] = 1

    # Create a negative sampling df, containing src and dst entities pairs,
    # such that there are no links between them.
    negative_sample_df_columns = list(df.columns)
    negative_sample_df_columns.remove(dst_entity)
    negative_samples_df = df[negative_sample_df_columns]
    negative_samples_df[dst_entity] = np.random.choice(
        dst_entity_df[dst_entity_table.pkey_col], size=len(negative_samples_df)
    )
    negative_samples_df[target_col_name] = 0

    # Constructing a dataframe containing the same number of positive and
    # negative links and
    df = pd.concat([df, negative_samples_df], ignore_index=True)
    df = pd.merge(
        df,
        dst_entity_df,
        how="left",
        left_on=dst_entity,
        right_on=dst_entity_table.pkey_col,
    )
    dfs[split] = add_past_label_feature(df)


def prepare_for_rec_eval(
    evaluate_table_df: pd.DataFrame, past_table_df: pd.DataFrame
) -> pd.DataFrame:
    """Transform evaluation dataframe into the correct format for recommendation metric
    calculation.

    Args:
        pred_table_df (pd.DataFrame): The prediction dataframe.
        past_table_df (pd.DataFrame): The dataframe containing labels in the
            past.
    Returns:
        (pd.DataFrame): The evaluation dataframe containing past visit and
            global popularity dst entities as candidate set.
    """

    def interleave_lists(list1, list2):
        interleaved = [item for pair in zip(list1, list2) for item in pair]
        longer_list = list1 if len(list1) > len(list2) else list2
        interleaved.extend(longer_list[len(interleaved) // 2 :])
        return interleaved

    grouped_ranked_past_table_df = (
        past_table_df.groupby(src_entity)[dst_entity]
        .apply(dst_entities_aggr)
        .reset_index()
    )
    evaluate_table_df = pd.merge(
        evaluate_table_df, grouped_ranked_past_table_df, how="left", on=src_entity
    )

    # collect the most popular dst entities
    all_dst_entities = [
        entity for sublist in past_table_df[dst_entity] for entity in sublist
    ]
    dst_entity_counter = Counter(all_dst_entities)
    top_dst_entities = [
        entity for entity, _ in dst_entity_counter.most_common(task.eval_k * 2)
    ]

    evaluate_table_df[dst_entity] = evaluate_table_df[dst_entity].apply(
        lambda x: (
            interleave_lists(x, top_dst_entities)
            if isinstance(x, list)
            else top_dst_entities
        )
    )
    # For each src entity, keep at most `task.eval_k * 2` dst entity candidates
    evaluate_table_df[dst_entity] = evaluate_table_df[dst_entity].apply(
        lambda x: (
            x[: task.eval_k * 2]
            if isinstance(x, list) and len(x) > task.eval_k * 2
            else x
        )
    )

    # Include src and dst entity table features for `evaluate_table_df`
    evaluate_table_df = pd.merge(
        evaluate_table_df,
        src_entity_df,
        how="left",
        left_on=src_entity,
        right_on=src_entity_table.pkey_col,
    )

    evaluate_table_df = evaluate_table_df.explode(dst_entity)
    evaluate_table_df = pd.merge(
        evaluate_table_df,
        dst_entity_df,
        how="left",
        left_on=dst_entity,
        right_on=dst_entity_table.pkey_col,
    )

    return add_past_label_feature(evaluate_table_df)


# Prepare val dataset for lightGBM model evaluation; the candidates per src entity
# come from the train labels.
val_df_pred = prepare_for_rec_eval(
    val_table.df.drop(columns=[dst_entity]), train_table.df
)
dfs["val_pred"] = val_df_pred

# Prepare test dataset for lightGBM model evaluation. The masked test table holds
# only the time and src entity columns; the candidates come from the train and val
# labels.
test_df = prepare_for_rec_eval(test_table.df, label_df)
dfs["test"] = test_df

train_dataset = torch_frame.data.Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=target_col_name,
    col_to_text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device="cpu"),
        batch_size=256,
    ),
)
# path = Path(
#     f"{args.cache_dir}/{args.dataset}/tasks/{args.task}/materialized/link_train.pt"
# )
# path.parent.mkdir(parents=True, exist_ok=True)
# train_dataset = train_dataset.materialize(path=path)
train_dataset = train_dataset.materialize()

tf_train = train_dataset.tensor_frame
tf_val = train_dataset.convert_to_tensor_frame(dfs["val"])
tf_val_pred = train_dataset.convert_to_tensor_frame(dfs["val_pred"])
tf_test = train_dataset.convert_to_tensor_frame(dfs["test"])

# tune metric for binary classification problem
tune_metric = Metric.ROCAUC
model = LightGBM(task_type=train_dataset.task_type, metric=tune_metric)
model.tune(tf_train=tf_train, tf_val=tf_val, num_trials=args.num_trials)


def predict_link(
    lightgbm_output: pd.DataFrame,
    src_entity_name: str,
    dst_entity_name: str,
    timestamp_col_name: str,
    eval_k: int,
    pred_score: float,
    target_table: Table,
) -> np.ndarray:
    def adjust_past_dst_entities(values):
        if len(values) < eval_k:
            return values + [-1] * (eval_k - len(values))
        else:
            return values[:eval_k]

    grouped_df = (
        lightgbm_output.sort_values(pred_score, ascending=False)
        .groupby([src_entity_name, timestamp_col_name])[dst_entity_name]
        .apply(list)
        .reset_index()
    )
    grouped_df = target_table.df[[src_entity_name, timestamp_col_name]].merge(
        grouped_df, on=[src_entity_name, timestamp_col_name], how="left"
    )

    dst_entity_array = (
        grouped_df[dst_entity_name].apply(adjust_past_dst_entities).tolist()
    )
    return np.array(dst_entity_array, dtype=int)


def evaluate(
    lightgbm_output: pd.DataFrame,
    src_entity_name: str,
    dst_entity_name: str,
    timestamp_col_name: str,
    eval_k: int,
    pred_score: float,
    train_table: Table,
    task: RecommendationTask,
) -> Dict[str, float]:
    """Given the input dataframe used for lightGBM binary link classification and its
    output prediction scores and true labels, generate recommendation evaluation
    metrics.

    Args:
        lightgbm_output (pd.DataFrame): The lightGBM input dataframe merged
            with output prediction scores.
        src_entity_name (str): The src entity name.
        dst_entity_name (str): The dst entity name
        timestamp_col (str): The name of the time column.
        eval_k (int): Pre-defined eval k parameter for recommendation metric
            evaluation.
        pred_score (float): The binary classification prediction scores.
        train_table (Table): The train table.
        task (RecommendationTask): The task.

    Returns:
        Dict[str, float]: The recommendation metrics
    """

    dst_entity_array = predict_link(
        lightgbm_output,
        src_entity_name,
        dst_entity_name,
        timestamp_col_name,
        eval_k,
        pred_score,
        train_table,
    )
    metrics = task.evaluate(dst_entity_array, train_table)
    return metrics


# NOTE: the train metric is computed on the training frame, whose candidate set is
# every true link plus as many random negatives, so it is not comparable to val/test
# (top-k over past-visit and popularity candidates).
pred = model.predict(tf_test=tf_train).cpu().numpy()
lightgbm_output = dfs["train"]
lightgbm_output[PRED_SCORE_COL_NAME] = pred
train_metrics = evaluate(
    lightgbm_output,
    src_entity,
    dst_entity,
    train_table.time_col,
    task.eval_k,
    PRED_SCORE_COL_NAME,
    sampled_train_table,
    task,
)
print(f"Train: {train_metrics}")

pred = model.predict(tf_test=tf_val_pred).cpu().numpy()
lightgbm_output = val_df_pred
lightgbm_output[PRED_SCORE_COL_NAME] = pred
val_metrics = evaluate(
    lightgbm_output,
    src_entity,
    dst_entity,
    train_table.time_col,
    task.eval_k,
    PRED_SCORE_COL_NAME,
    val_table,
    task,
)
print(f"Val: {val_metrics}")


pred = model.predict(tf_test=tf_test).cpu().numpy()
lightgbm_output = dfs["test"]
lightgbm_output[PRED_SCORE_COL_NAME] = pred
test_pred = predict_link(
    lightgbm_output,
    src_entity,
    dst_entity,
    train_table.time_col,
    task.eval_k,
    PRED_SCORE_COL_NAME,
    test_table,
)
os.makedirs(args.pred_dir, exist_ok=True)
pred_path = os.path.join(args.pred_dir, f"{args.dataset}__{args.task}.csv")
write_prediction_table(task, test_pred, pred_path)
test_metrics = evaluate_task(f"{args.dataset}/{args.task}", pred_path)
print(f"Test: {test_metrics}")
