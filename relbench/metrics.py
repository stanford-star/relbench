from typing import Tuple

import numpy as np
import sklearn.metrics as skm
from numpy.typing import NDArray
from scipy.stats import rankdata

###### classification metrics

### applicable to both binary and multiclass classification


def accuracy(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    if pred.ndim == 1:
        label = pred > 0.5
    else:
        label = pred.argmax(axis=1)
    return skm.accuracy_score(true, label)


def log_loss(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    if pred.ndim == 1 or pred.shape[1] == 1:
        prob = np.sigmoid(pred)
    else:
        prob = np.softmax(pred, axis=1)
    return skm.log_loss(true, prob)


### applicable to binary classification only


def f1(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    label = pred >= 0.5
    return skm.f1_score(true, label, average="binary")


def roc_auc(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    return skm.roc_auc_score(true, pred)


def average_precision(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    return skm.average_precision_score(true, pred)


def auprc(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    return skm.auc(recall, precision)


### applicable to multiclass classification only


def mrr(y: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    rankings = rankdata(-y_pred, method="min", axis=1)
    ranks = np.take_along_axis(rankings, y.astype(int).reshape(-1, 1), axis=1).flatten()
    return np.mean(1.0 / ranks).item()


def macro_f1(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim > 1
    label = pred.argmax(axis=1)
    return skm.f1_score(true, label, average="macro")


def micro_f1(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim > 1
    label = pred.argmax(axis=1)
    return skm.f1_score(true, label, average="micro")


###### regression metrics


def mae(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return skm.mean_absolute_error(true, pred)


def mse(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return skm.mean_squared_error(true, pred)


def rmse(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return skm.root_mean_squared_error(true, pred)


def r2(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return skm.r2_score(true, pred)


####### Multilabel metrics
def multilabel_auprc_micro(true: NDArray[np.int_], pred: NDArray[np.float64]) -> float:
    # Flatten true and prediction arrays for micro-average computation
    true_flat = np.ravel(np.stack(true))
    pred_flat = np.ravel(pred)
    return skm.average_precision_score(true_flat, pred_flat, average="micro")


def multilabel_auprc_macro(true: NDArray[np.int_], pred: NDArray[np.float64]) -> float:
    true = np.stack(true)
    return skm.average_precision_score(true, pred, average="macro")


def multilabel_auroc_micro(true: NDArray[np.int_], pred: NDArray[np.float64]) -> float:
    # Flatten true and prediction arrays for micro-average computation
    true_flat = np.ravel(np.stack(true))
    pred_flat = np.ravel(pred)
    return skm.roc_auc_score(true_flat, pred_flat, average="micro")


def multilabel_auroc_macro(true: NDArray[np.int_], pred: NDArray[np.float64]) -> float:
    true = np.stack(true)
    return skm.roc_auc_score(true, pred, average="macro")


def multilabel_f1_micro(true: NDArray[np.int_], pred: NDArray[np.int_]) -> float:
    return skm.f1_score(np.stack(true), (pred > 0.5).astype(int), average="micro")


def multilabel_f1_macro(true: NDArray[np.int_], pred: NDArray[np.int_]) -> float:
    return skm.f1_score(np.stack(true), (pred > 0.5).astype(int), average="macro")


def multilabel_recall_micro(true: NDArray[np.int_], pred: NDArray[np.int_]) -> float:
    return skm.recall_score(np.stack(true), (pred > 0.5).astype(int), average="micro")


def multilabel_recall_macro(true: NDArray[np.int_], pred: NDArray[np.int_]) -> float:
    return skm.recall_score(np.stack(true), (pred > 0.5).astype(int), average="macro")


def multilabel_precision_micro(true: NDArray[np.int_], pred: NDArray[np.int_]) -> float:
    return skm.precision_score(
        np.stack(true), (pred > 0.5).astype(int), average="micro"
    )


def multilabel_precision_macro(true: NDArray[np.int_], pred: NDArray[np.int_]) -> float:
    return skm.precision_score(
        np.stack(true), (pred > 0.5).astype(int), average="macro"
    )


####### Multiclass metrics
def multiclass_f1(true: NDArray[np.int_], pred: NDArray[np.int_]) -> float:
    if pred.ndim > 1:
        pred = pred.argmax(axis=1)
    return skm.f1_score(true, pred, average="micro")
####### Link prediction metrics
\
"""All link prediction metrics take two arguments.

- pred_isin: Numpy boolean array of size (num_src_nodes, eval_k)
- dst_count: Numpy integer array of size (num_src_nodes, ), storing
    the number of destination nodes attached to each source node.
"""


def _filter(
    pred_isin: NDArray[np.int_], dst_count: NDArray[np.int_]
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    is_pos = dst_count > 0
    return pred_isin[is_pos], dst_count[is_pos]


def link_prediction_recall(
    pred_isin: NDArray[np.int_],
    dst_count: NDArray[np.int_],
) -> float:
    pred_isin, dst_count = _filter(pred_isin, dst_count)
    recalls = pred_isin.sum(axis=1) / dst_count
    return recalls.mean()


def link_prediction_precision(
    pred_isin: NDArray[np.int_],
    dst_count: NDArray[np.int_],
) -> float:
    pred_isin, dst_count = _filter(pred_isin, dst_count)
    eval_k = pred_isin.shape[1]
    precisions = pred_isin.sum(axis=-1) / eval_k
    return precisions.mean()


def link_prediction_map(
    pred_isin: NDArray[np.int_],
    dst_count: NDArray[np.int_],
) -> float:
    pred_isin, dst_count = _filter(pred_isin, dst_count)
    eval_k = pred_isin.shape[1]
    clipped_dst_count = dst_count.clip(min=None, max=eval_k)
    precision_mat = np.cumsum(pred_isin, axis=1) / (np.arange(eval_k) + 1)
    maps = (precision_mat * pred_isin).sum(axis=1) / clipped_dst_count
    return maps.mean()


def link_prediction_ndcg(
    pred_isin: NDArray[np.int_],
    dst_count: NDArray[np.int_],
) -> float:
    pred_isin, dst_count = _filter(pred_isin, dst_count)
    eval_k = pred_isin.shape[1]

    # Compute the discounted multiplier (1 / log2(i + 2) for i = 0, ..., k-1)
    discounted_multiplier = np.concatenate(
        (np.zeros(1), 1 / np.log2(np.arange(1, eval_k + 1) + 1))
    )

    # Compute Discounted Cumulative Gain (DCG)
    discounted_cumulative_gain = (
        pred_isin * discounted_multiplier[1 : eval_k + 1]
    ).sum(axis=1)

    # Clip dst_count to the range [0, eval_k]
    clipped_dst_count = np.clip(dst_count, 0, eval_k)

    # Compute Ideal Discounted Cumulative Gain (IDCG)
    ideal_discounted_multiplier_cumsum = np.cumsum(discounted_multiplier)
    ideal_discounted_cumulative_gain = ideal_discounted_multiplier_cumsum[
        clipped_dst_count
    ]

    # Avoid division by zero
    ideal_discounted_cumulative_gain = np.clip(
        ideal_discounted_cumulative_gain, 1e-10, None
    )

    # Compute NDCG
    ndcg_scores = discounted_cumulative_gain / ideal_discounted_cumulative_gain
    return ndcg_scores.mean()
