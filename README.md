<p align="center"><img src="https://relbench.stanford.edu/img/logo.png" alt="RelBench" width="600px" /></p>

<p align="center">
  <a href="https://pypi.org/project/relbench/"><img src="https://img.shields.io/pypi/v/relbench?color=3f9e78" alt="PyPI" /></a>
  <a href="https://pypi.org/project/relbench/"><img src="https://img.shields.io/pypi/pyversions/relbench?color=3f9e78" alt="Python" /></a>
  <a href="https://github.com/snap-stanford/relbench/actions/workflows/testing.yml"><img src="https://github.com/snap-stanford/relbench/actions/workflows/testing.yml/badge.svg" alt="Tests" /></a>
  <a href="https://arxiv.org/abs/2602.12606"><img src="https://img.shields.io/badge/arXiv-2602.12606-b31b1b.svg" alt="arXiv" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-3f9e78.svg" alt="License: MIT" /></a>
</p>

<p align="center">
  <a href="https://relbench.stanford.edu"><b>Website</b></a> ·
  <a href="#get-started"><b>Get Started</b></a> ·
  <a href="https://relbench.stanford.edu/leaderboard/"><b>Leaderboard</b></a> ·
  <a href="https://relbench.stanford.edu/papers/"><b>Papers</b></a> ·
  <a href="https://huggingface.co/relbench"><b>Hugging Face</b></a> ·
  <a href="https://relbench.stanford.edu/news/"><b>News</b></a>
</p>

**RelBench is an open benchmark for predictive machine learning on relational databases** —
end-to-end deep learning on data spread across many tables, with no manual feature
engineering. Datasets and tasks load straight from the [Hugging Face Hub](https://huggingface.co/relbench);
releases, papers, and milestones live on the [website](https://relbench.stanford.edu/news/).

## Get Started

Install from PyPI:

```bash
pip install relbench           # core data + task loading
pip install relbench[full]     # + PyTorch Geometric & PyTorch Frame, for the GNN models
```

Load a dataset and a task — both come straight from the Hub, with **no per-dataset code**:

```python
import relbench

dataset = relbench.load_dataset("rel-f1")        # a Hub repo, or a local path
db = dataset.get_db()                            # a Database: tables linked by a foreign-key graph

task = relbench.load_task("rel-f1", "driver-position")
train_table = task.get_table("train")            # train / val / test label tables
test_table  = task.get_table("test")             # target column is hidden on test

# ... train your model, predict on the test entities ...
metrics = task.evaluate(test_pred)               # standardized metrics, chosen by task type
```

`db` is a set of tables linked by a foreign-key graph; `dataset.val_timestamp` /
`dataset.test_timestamp` give the temporal splits. RelBench is framework-agnostic — bring any
modeling stack. `relbench.modeling` additionally provides a reference Graph Neural Network
implementation on [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) +
[PyTorch Frame](https://github.com/pyg-team/pytorch-frame).

### Tutorials

Jupyter notebooks you can open directly in Google Colab — no setup required:

| Tutorial | What it covers | |
|---|---|---|
| [**Quickstart**](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/quickstart.ipynb) | Load a dataset/task, explore the schema, run a baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/quickstart.ipynb) |
| [**Training a GNN**](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/gnn.ipynb) | A GNN baseline for an entity task (PyG + PyTorch Frame) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/gnn.ipynb) (needs a GPU) |

The sources live in [`tutorials/`](tutorials); to run locally, `pip install relbench`
(add `[full]` for the GNN) and open them with `jupyter notebook`.

## Contributing

Contributing a **dataset** or **task** no longer needs any Python classes — a dataset is a
self-describing folder you publish to the [Hugging Face Hub](https://huggingface.co):

```
<dataset>/
  manifest.yaml            # description, tables, primary keys, foreign-key graph, splits
  db/<table>.parquet       # one plain parquet per table (native dtypes)
  tasks/<task>/
    manifest.yaml          # task spec (+ a DuckDB query, for `forecast` tasks)
    {train,val,test}.parquet
  README.md, schema.svg    # dataset card + ER diagram (auto-generated)
```

`manifest.yaml` is the single source of truth for the relational structure (keys, the
foreign-key graph, time columns), so the parquet also load with plain pandas/DuckDB. Tasks
come in three kinds: **`forecast`** (temporal labels computed by a DuckDB query —
regenerable, and CI-checked against the shipped labels), **`autocomplete`** (predict an
existing column), and **`external`** (labels shipped as-is). Adding a task is just adding a
`tasks/<name>/` directory.

**Naming conventions.** A dataset is named `rel-<name>`, where `<name>` is a single,
lowercase, singular word (e.g. `rel-amazon`); use `rel-<name>-<qualifier>` for a variant
(e.g. `rel-amazon-fashion`). Tasks are named `<entity>-<word>` for entity tasks and
`<src-entity>-<dst-entity>-<word>` for recommendation (link) tasks — e.g. `user-churn`,
`user-item-review`.

The published [`rel-f1`](https://huggingface.co/datasets/relbench/v1) repo is a complete
worked example. Generate the card + diagram with `relbench.schema.dataset_card` /
`render_schema_svg`, and verify a `forecast` task reproduces its labels with:

```bash
python -m relbench.check_provenance <dataset-path-or-name>
```

To include your dataset in official RelBench, open a PR adding it to
[`relbench/registry.json`](relbench/registry.json) (name → Hub repo + a pinned commit
revision). For bug reports and feature requests, please open a GitHub issue or pull request.

## Cite RelBench

If you use RelBench in your work, please cite the position and benchmark papers:

```bibtex
@inproceedings{rdl,
  title={Position: Relational Deep Learning - Graph Representation Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```

```bibtex
@inproceedings{relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  author={Robinson, Joshua and Ranjan, Rishabh and Hu, Weihua and Huang, Kexin and Han, Jiaqi and Dobles, Alejandro and Fey, Matthias and Lenssen, Jan Eric and Yuan, Yiwen and Zhang, Zecheng and He, Xinwei and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

```bibtex
@misc{relbenchv2,
  title={RelBench v2: A Large-Scale Benchmark and Repository for Relational Data},
  author={Gu, Justin and Ranjan, Rishabh and Kanatsoulis, Charilaos and Tang, Haiming and Jurkovic, Martin and Hudovernik, Valter and Znidar, Mark and Chaturvedi, Pranshu and Shroff, Parth and Li, Fengyu and Leskovec, Jure},
  year={2026},
  eprint={2602.12606},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.12606}
}
```

Datasets sourced from external repositories (CTU/ReDeLEx, 4DBInfer, TGB) carry their own
citations on their [Hugging Face](https://huggingface.co/relbench) dataset cards.
