# RelBench tutorials

Tutorials are [marimo](https://marimo.io) notebooks — plain, git-friendly `.py` files
(reactive; no hidden notebook JSON). They replace the old Colab notebooks.

| Notebook | What it shows | Runs in browser? |
|---|---|---|
| [`quickstart.py`](quickstart.py) | Load a dataset/task from the Hub, explore the FK graph, build a task table, run a baseline | ✅ yes (pandas + DuckDB via WebAssembly) |
| [`gnn.py`](gnn.py) | Train a GNN on an entity task (PyG + PyTorch Frame) | ❌ download-and-run (needs `torch` + a GPU) |

## Run / edit locally

```bash
pip install relbench[tutorial]      # marimo; add [full] as well for gnn.py
marimo edit tutorials/quickstart.py # interactive, reactive editor
marimo run  tutorials/quickstart.py # read-only app
python      tutorials/quickstart.py # also a plain script
```

## Publish the interactive (browser) tutorials

marimo compiles a notebook to a self-contained WebAssembly bundle — static files, no
server, click-to-run:

```bash
marimo export html-wasm tutorials/quickstart.py -o site/quickstart --mode edit
```

Serve the `site/` directory from any static host (e.g. `relbench.stanford.edu/tutorials`).
Two server notes for WASM:

- serve `.wasm` as `application/wasm`;
- optionally set `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` (self-hosting lets you set these; GitHub
  Pages cannot).

In-browser tutorials fetch their dataset from the Hub, so the dataset must be **public**.
GPU/torch tutorials (`gnn.py`) are exported as static, read-only pages plus a download
link — they are meant to be downloaded and run on your own machine.
