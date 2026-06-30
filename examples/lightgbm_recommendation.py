import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict

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

from relbench import load_dataset, load_task
from relbench.base import Dataset, RecommendationTask, Table
from relbench.leaderboard import write_prediction_table, evaluate_task
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey

LINK_PRED_BASELINE_TARGET_COL_NAME = "link_pred_baseline_target_column_name"
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
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = load_dataset(args.dataset)
task: RecommendationTask = load_task(args.dataset, args.task)
target_col_name: str = LINK_PRED_BASELINE_TARGET_COL_NAME

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

# We plan to merge train table with entity and target table to include both
# entity and target table features during lightGBM training.
dfs: Dict[str, pd.DataFrame] = {}
target_dfs: Dict[str, pd.DateOffset] = {}
db = dataset.get_db()
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


def add_past_label_feature(
    train_table_df: pd.DataFrame,
    past_table_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add past visit count and percentage of global popularity to train table df used
    for lightGBM training, evaluation of testing.

    Args:
        evaluate_table_df (pd.DataFrame): The dataframe used for evaluation.
        past_table_df (pd.DataFrame): The dataframe containing labels in the
            past.
    """
    # Add number of past visit for each src_entity and dst_entity pair
    # Explode the dst_entity list to get one row per (src_entity, dst_entity) pair
    exploded_past_table = past_table_df.explode(dst_entity)

    # Count occurrences of each (src_entity, dst_entity) pair
    dst_entity_count = (
        exploded_past_table.groupby([src_entity, dst_entity])
        .size()
        .reset_index(name="num_past_visit")
    )

    # Merge the count information with train_table_df
    train_table_df = train_table_df.merge(
        dst_entity_count, how="left", on=[src_entity, dst_entity]
    )

    # Fill NaN values with 0 (if there are any dst_entity in train_table_df not present in past_table_df)
    train_table_df["num_past_visit"] = (
        train_table_df["num_past_visit"].fillna(0).astype(int)
    )

    # Add percentage of global popularity for each dst_entity
    # Count occurrences of each dst_entity
    dst_entity_count = exploded_past_table[dst_entity].value_counts().reset_index()

    # Calculate the fraction
    # total_right_entities = len(exploded_past_table)
    dst_entity_count["global_popularity_fraction"] = (
        dst_entity_count["count"] / dst_entity_count["count"].max()
    )

    # Merge the fraction information with train_table_df
    train_table_df = train_table_df.merge(
        dst_entity_count[[dst_entity, "global_popularity_fraction"]],
        how="left",
        on=dst_entity,
    )

    # Fill NaN values with 0 (if there are any dst_entity in train_table_df not present in past_table_df)
    train_table_df["global_popularity_fraction"] = train_table_df[
        "global_popularity_fraction"
    ].fillna(0)

    return train_table_df


# Prepare train/val dataset for lightGBM model training. For each src
# entity, their corresponding dst entities are used as positive label.
# The same number of random dst entities are sampled as negative label.
# lightGBM will train and eval on this binary classification task.
src_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
dst_entity = list(train_table.fkey_col_to_pkey_table.keys())[1]
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
    df = add_past_label_feature(df, train_table.df)
    dfs[split] = df


def prepare_for_link_pred_eval(
    evaluate_table_df: pd.DataFrame, past_table_df: pd.DataFrame
) -> pd.DataFrame:
    """Transform evaluation dataframe into the correct format for link prediction metric
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

    evaluate_table_df = add_past_label_feature(evaluate_table_df, past_table_df)
    return evaluate_table_df


# Prepare val dataset for lightGBM model evaluation
val_df_pred_column_names = list(val_table.df.columns)
val_df_pred_column_names.remove(dst_entity)
val_df_pred = val_table.df[val_df_pred_column_names]
# Per each src entity, collect all past linked dst entities
val_past_table_df = train_table.df
val_past_table_df.drop(columns=[train_table.time_col], inplace=True)
val_df_pred = prepare_for_link_pred_eval(val_df_pred, val_past_table_df)
dfs["val_pred"] = val_df_pred

# Prepare test dataset for lightGBM model evaluation
test_df_column_names = list(test_table.df.columns)
test_df_column_names.remove(dst_entity)
test_df = test_table.df[test_df_column_names]
# Per each src entity, collect all past linked dst entities
test_past_table_df = pd.concat([train_table.df, val_table.df], axis=0)
test_past_table_df.drop(columns=[train_table.time_col], inplace=True)
test_df = prepare_for_link_pred_eval(test_df, test_past_table_df)
dfs["test"] = test_df

train_dataset = torch_frame.data.Dataset(
    df=dfs["train"],
    col_to_stype=col_to_stype,
    target_col=target_col_name,
    col_to_text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device),
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
    output prediction scores and true labels, generate link prediction evaluation
    metrics.

    Args:
        lightgbm_output (pd.DataFrame): The lightGBM input dataframe merged
            with output prediction scores.
        src_entity_name (str): The src entity name.
        dst_entity_name (str): The dst entity name
        timestamp_col (str): The name of the time column.
        eval_k (int): Pre-defined eval k parameter for link pred metric
            evaluation.
        pred_score (float): The binary classification prediction scores.
        train_table (Table): The train table.
        task (RecommendationTask): The task.

    Returns:
        Dict[str, float]: The link pred metrics
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


# NOTE: train/val metrics will be artificially high since all true links are
# included in the candidate set
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
os.makedirs("/tmp/relbench_preds", exist_ok=True)
pred_path = f"/tmp/relbench_preds/{args.dataset}__{args.task}.csv"
write_prediction_table(task, test_pred, pred_path)
test_metrics = evaluate_task(f"{args.dataset}/{args.task}", pred_path)
print(f"Test: {test_metrics}")
