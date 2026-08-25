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

Outside pixi: `pip install "relbench[example]"`, plus `pyg-lib` from
https://data.pyg.org/whl/ (matching your torch/CUDA build) for the GNN scripts, then
run with plain `python`.

## Common flags

- `--dataset`, `--task`: a RelBench dataset and one of its tasks; the leaderboard
  tasks are listed in `relbench/submit.py::LEADERBOARD_TASKS`.
- `--cache_dir` (default `~/.cache/relbench_examples`): stype proposals and materialized
  graphs, keyed per dataset (GNN and LightGBM scripts).
- `--pred_dir` (default `/tmp/relbench_preds`): where the prediction CSV is written.
