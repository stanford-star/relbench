r"""The metrics RelBench evaluates on.

RelBench provides an evaluator for the three core task types only, with exactly one metric
each -- the user does not choose:

* binary classification -> ``roc_auc``             (AUROC)
* regression            -> NMAE, built per-task by :func:`make_nmae` (MAE
                           normalized by the train-split target std, ddof=1)
* recommendation / rec.-> ``map`` (MAP@k, k = task's eval_k)

Multiclass and multilabel tasks are definable and loadable, but RelBench provides no
evaluator for them: ``task.metrics`` is empty -- pass your own to
``task.evaluate(pred, metrics=[...])``.
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
    regression target on its *train* split. ``get_std`` reads the value the task
    resolved when it was built (``task.nmae_std``); see :func:`relbench.train_std` and
    the hosted ``stanford-star/relbench-v1`` regression-std table.
    """

    def nmae(true: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
        return float(skm.mean_absolute_error(true, pred) / get_std())

    return nmae


####### Recommendation metric (MAP@k)
"""The recommendation metric takes two arguments
    - pred_isin: Numpy boolean array of size (num_src_nodes, eval_k)
    - dst_count: Numpy integer array of size (num_src_nodes, ), storing
        the number of destination nodes attached to each source node.
"""


def _filter(
    pred_isin: NDArray[np.int_], dst_count: NDArray[np.int_]
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    is_pos = dst_count > 0
    return pred_isin[is_pos], dst_count[is_pos]


def map(
    pred_isin: NDArray[np.int_],
    dst_count: NDArray[np.int_],
) -> float:
    pred_isin, dst_count = _filter(pred_isin, dst_count)
    eval_k = pred_isin.shape[1]
    clipped_dst_count = dst_count.clip(min=None, max=eval_k)
    precision_mat = np.cumsum(pred_isin, axis=1) / (np.arange(eval_k) + 1)
    maps = (precision_mat * pred_isin).sum(axis=1) / clipped_dst_count
    return maps.mean()
