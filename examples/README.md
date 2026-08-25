# Examples

Baseline scripts for RelBench tasks. Each one trains a model, prints train/val/test
metrics, and ends by writing `<pred_dir>/<dataset>__<task>.csv`, a prediction table
that `python -m relbench.submit <pred_dir>` validates and packages for the leaderboard.

| Script | Task type | Model |
| --- | --- | --- |
| `trivial_entity.py` | entity (classification / regression) | global and per-entity statistics |
| `lightgbm_entity.py` | entity | LightGBM on entity-table features |
| `gnn_entity.py` | entity | heterogeneous GraphSAGE / GAT |
| `hybrid_entity.py` | entity | GNN embeddings + LightGBM |
| `trivial_recommendation.py` | recommendation | past-visit and global-popularity |
| `lightgbm_recommendation.py` | recommendation | LightGBM link classifier |
| `gnn_recommendation.py` | recommendation | two-tower GraphSAGE / GAT (BPR loss) |
| `idgnn_recommendation.py` | recommendation | ID-GNN |
| `trivial_autocomplete.py` | autocomplete | global and per-entity statistics |
| `lightgbm_autocomplete.py` | autocomplete | LightGBM on entity-table features |
| `gnn_autocomplete.py` | autocomplete | heterogeneous GraphSAGE |

`model.py` and `text_embedder.py` are helpers shared by the GNN scripts.

## How to run

```sh
pixi run -e gpu python examples/gnn_entity.py --dataset rel-f1 --task driver-position
```

Outside pixi, install the same extras and run with plain `python`:

```bash
pip install "relbench[example]"  # + PyTorch Geometric & PyTorch Frame, for the GNN examples
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.9.0+cpu.html  # neighbor sampling; use the index matching your torch/CUDA build
```

## Common flags

- `--dataset`, `--task`: a RelBench dataset and one of its tasks; the leaderboard
  tasks are listed in `relbench/submit.py::LEADERBOARD_TASKS`.
- `--cache_dir` (default `~/.cache/relbench_examples`): stype proposals and materialized
  graphs (GNN and LightGBM scripts). Caches live under `{cache_dir}/{dataset}/` and are
  built from the dataset-level database, so every task on a dataset shares them; a
  task's hidden columns are dropped after loading. The autocomplete scripts keep the
  rows after `test_timestamp` and use `materialized_full/` next to `materialized/`.
- `--pred_dir` (default `/tmp/relbench_preds`): where the prediction CSV
  `<dataset>__<task>.csv` is written; pass it to `python -m relbench.submit`.
