# RelBench tutorials

Tutorials are [marimo](https://marimo.io) notebooks — plain, git-friendly `.py` files
(reactive; no hidden notebook JSON). They replace the old Colab notebooks.

| Notebook | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | Load a dataset/task from the Hub, explore the FK graph, build a task table, run a baseline |
| [`gnn.py`](gnn.py) | Train a GNN on an entity task (PyG + PyTorch Frame); needs `torch` and ideally a GPU |

## Run / edit locally

```bash
pip install relbench[tutorial]      # marimo; add [full] as well for gnn.py
marimo edit tutorials/quickstart.py # interactive, reactive editor
marimo run  tutorials/quickstart.py # read-only app
python      tutorials/quickstart.py # also a plain script
```

## Hosted (rendered) versions

Static, rendered versions are published on the website at
[relbench.stanford.edu/tutorials](https://relbench.stanford.edu/tutorials/): a read-only view
of each notebook (code + outputs) with a button to download the `.py` and run it locally.
Regenerate with `marimo export html <notebook>.py -o <out>/index.html`.
