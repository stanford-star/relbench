import numpy as np

from relbench.metrics import make_nmae, map, roc_auc


def test_roc_auc():
    true = np.array([0, 0, 1, 1])
    pred = np.array([0.1, 0.4, 0.35, 0.8])
    assert 0 <= roc_auc(true, pred) <= 1


def test_nmae():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 2.5, 2.5, 3.5])
    std = float(np.std(true, ddof=1))
    nmae = make_nmae(lambda: std)
    expected = np.mean(np.abs(true - pred)) / std
    assert np.isclose(nmae(true, pred), expected)


def test_map():
    num_src_nodes = 100
    eval_k = 10
    rng = np.random.default_rng(0)
    pred_isin = rng.integers(0, 2, size=(num_src_nodes, eval_k)).astype(bool)
    dst_count = pred_isin.sum(axis=1) + rng.integers(0, 5, size=(num_src_nodes,))
    assert 0 <= map(pred_isin, dst_count) <= 1
