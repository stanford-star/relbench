# RelBench tutorials

Tutorials are Jupyter notebooks you can open directly in Google Colab — no local setup
required (the first cell installs RelBench).

| Notebook | What it shows | |
|---|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | Load a dataset/task from the Hub, explore the FK graph, build a task table, run a baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/quickstart.ipynb) |
| [`gnn.ipynb`](gnn.ipynb) | Train a GNN on an entity task (PyG + PyTorch Frame); needs `torch` and ideally a GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/gnn.ipynb) |

## Run locally

```bash
pip install relbench           # add [full] as well for gnn.ipynb
jupyter notebook tutorials/quickstart.ipynb
```

In Colab, run the cells top to bottom; the first cell installs RelBench. For `gnn.ipynb`,
switch the runtime to GPU first (*Runtime → Change runtime type → GPU*).
