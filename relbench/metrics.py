r"""The metrics RelBench evaluates on.

Only the three RelBench v1 task types are supported, with exactly one metric each
-- the user does not choose:

* binary classification -> ``roc_auc``             (AUROC)
* regression            -> NMAE, built per-task by :func:`make_nmae` (MAE
                           normalized by the train-split target std, ddof=1)
* link prediction / rec.-> ``link_prediction_map`` (MAP@k, k = task's eval_k)
"""

from typing import Callable, Tuple

import numpy as np
import sklearn.metrics as skm
from numpy.typing import NDArray

###### classification metric (AUROC)


def roc_auc(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    return skm.roc_auc_score(true, pred)


###### regression metric (NMAE)


def make_nmae(get_std: Callable[[], float]) -> Callable[[NDArray, NDArray], float]:
    r"""Build the NMAE metric for a regression task.

    NMAE = MAE / std, where ``std`` is the standard deviation (ddof=1) of the task's
    regression target on its *train* split. ``get_std`` resolves that std lazily (so
    merely loading a task does not fetch/compute it); see :func:`relbench.train_std`
    and the hosted ``relbench/core`` regression-std table.
    """

    def nmae(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
        return float(skm.mean_absolute_error(true, pred) / get_std())

    return nmae


####### Link prediction metric (MAP@k)
"""The link prediction metric takes two arguments
    - pred_isin: Numpy boolean array of size (num_src_nodes, eval_k)
    - dst_count: Numpy integer array of size (num_src_nodes, ), storing
        the number of destination nodes attached to each source node.
"""


def _filter(
    pred_isin: NDArray[np.int_], dst_count: NDArray[np.int_]
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    is_pos = dst_count > 0
    return pred_isin[is_pos], dst_count[is_pos]


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
