# RelBench tutorials

Tutorials are Jupyter notebooks you can open directly in Google Colab — no local setup
required (the first cell installs RelBench).

| Notebook | What it shows | |
|---|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | Load a dataset/task from the Hub, explore the FK graph, read its label tables, run a baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stanford-star/relbench/blob/main/tutorials/quickstart.ipynb) |
| [`gnn.ipynb`](gnn.ipynb) | Train a GNN on an entity task (PyG + PyTorch Frame); needs the `[example]` extras, runs on CPU or GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stanford-star/relbench/blob/main/tutorials/gnn.ipynb) |

## Run locally

```bash
pip install relbench             # pip install "relbench[example]" for gnn.ipynb
jupyter notebook tutorials/quickstart.ipynb
```

In Colab, run the cells top to bottom; the first cell installs RelBench. For `gnn.ipynb`,
a GPU runtime (*Runtime → Change runtime type → GPU*) is faster but optional.
