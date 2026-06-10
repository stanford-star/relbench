<p align="center"><img src="https://relbench.stanford.edu/img/logo.png" alt="logo" width="600px" /></p>

----

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relbench.stanford.edu)
[![PyPI version](https://badge.fury.io/py/relbench.svg)](https://badge.fury.io/py/relbench)
[![Testing Status](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml/badge.svg)](https://github.com/snap-stanford/relbench/actions/workflows/testing.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40RelBench)](https://twitter.com/RelBench)

<!-- **Get Started:** loading data &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1PAOktBqh_3QzgAKi53F4JbQxoOuBsUBY?usp=sharing), training model &nbsp; [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1_z0aKcs5XndEacX1eob6csDuR4DYhGQU?usp=sharing). -->


<!-- [<img align="center" src="https://relbench.stanford.edu/img/favicon.png" width="20px" />   -->
[**Website**](https://relbench.stanford.edu) | [**Position Paper**](https://proceedings.mlr.press/v235/fey24a.html) |  [**RelBench v1 Paper**](https://arxiv.org/abs/2407.20060) | [**RelBench v2 Paper**](https://arxiv.org/abs/2602.12606) | [**Mailing List**](https://groups.google.com/forum/#!forum/relbench/join)

# News

**February 13, 2026: RelBench v2 paper + Temporal Graph Benchmark integration**

The RelBench v2 paper is now accessible as a preprint! Please see the paper on [arXiv](https://arxiv.org/abs/2602.12606).

Alongside our paper, we also integrate the [Temporal Graph Benchmark](https://tgb.complexdatalab.com/) (TGB) into RelBench. TGB integration includes translating time-stamped event streams into normalized relational schemas, which enables direct comparison between temporal graph models and relational deep learning models.

**January 12, 2026: RelBench v2 is now released!**
- Introducing Autocomplete tasks: new task paradigm to predict existing columns in the database.
- 3 new databases: [SALT](https://relbench.stanford.edu/datasets/rel-salt), [RateBeer](https://relbench.stanford.edu/datasets/rel-ratebeer), and [arXiv](https://relbench.stanford.edu/datasets/rel-arxiv).
- 36 new predictive tasks, including 23 Autocomplete tasks across new and existing databases.
- CTU integration: 70+ relational datasets from the CTU repository via [ReDeLEx](https://github.com/jakubpeleska/redelex).
- Direct SQL database connectivity via [ReDeLEx](https://github.com/jakubpeleska/redelex).
- 4DBInfer integration: 7 relational datasets from the [4DBInfer](https://github.com/awslabs/multi-table-benchmark) repository in RelBench format.
- Bug fixes and performance improvements:
    - Optionally include (time-censored) labels as features in the database. ([#327](https://github.com/snap-stanford/relbench/pull/327))
    - Support NDCG metric for link prediction. ([#276](https://github.com/snap-stanford/relbench/pull/276"))
    - Optimize SentenceTransformer encoding with Torch for 10-20% faster processing than default NumPy encoding. ([#261](https://github.com/snap-stanford/relbench/pull/261"))
    - Enable configuring RelBench cache directory via environment variable. ([#336](https://github.com/snap-stanford/relbench/pull/336"))
    - ... and more (see commit history for details)

**September 26, 2024: RelBench is accepted to the NeurIPS Datasets and Benchmarks track!**

**July 3rd, 2024: RelBench v1 is now released!**


# Overview

<!-- The Relational Deep Learning Benchmark (RelBench) is a collection of realistic, large-scale, and diverse benchmark datasets for machine learning on relational databases. RelBench supports deep learning framework agnostic data loading, task specification, standardized data splitting, and transforming data into graph format. RelBench also provides standardized evaluation metric computations and a leaderboard for tracking progress. -->

<!-- <p align="center"><img src="https://relbench.stanford.edu/img/relbench-fig.png" alt="pipeline" /></p> -->

Relational Deep Learning is a new approach for end-to-end representation learning on data spread across multiple tables, such as in a _relational database_ (see our [position paper](https://relbench.stanford.edu/paper.pdf)). Relational databases are the world's most widely used data management system, and are used for industrial and scientific purposes across many domains. RelBench is a benchmark designed to facilitate efficient, robust and reproducible research on end-to-end deep learning over relational databases.

RelBench v1 contains 7 realistic, large-scale, and diverse relational databases spanning domains including medical, social networks, e-commerce and sport. RelBench v2 adds 4 more, now totaling 11 databases. Each database has multiple predictive tasks (66 in total) defined, each carefully scoped to be both challenging and of domain-specific importance. It provides full support for data downloading, task specification and standardized evaluation in an ML-framework-agnostic manner.

Additionally, RelBench provides a first open-source implementation of a Graph Neural Network based approach to relational deep learning. This implementation uses [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) to load the data as a graph and train GNN models, and [PyTorch Frame](https://github.com/pyg-team/pytorch-frame) for modeling tabular data. Finally, there is an open [leaderboard](https://huggingface.co/relbench) for tracking progress.


# Key Papers

[**RelBench: A Benchmark for Deep Learning on Relational Databases**](https://arxiv.org/abs/2407.20060)

This paper details our approach to designing the RelBench benchmark. It also includes a key user study showing that relational deep learning can produce performant models with a fraction of the manual human effort required by typical data science pipelines. This paper is useful for a detailed understanding of RelBench and our initial benchmarking results. If you just want to quickly familiarize with the data and tasks, the [**website**](https://relbench.stanford.edu) is a better place to start.
<!---Joshua Robinson*, Rishabh Ranjan*, Weihua Hu*, Kexin Huang*, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan Eric Lenssen, Yiwen Yuan, Zecheng Zhang, Xinwei He, Jure Leskovec-->

[**Position: Relational Deep Learning - Graph Representation Learning on Relational Databases (ICML 2024)**](https://proceedings.mlr.press/v235/fey24a.html)

This paper outlines our proposal for how to do end-to-end deep learning on relational databases by combining graph neural networsk with deep tabular models. We reccomend reading this paper if you want to think about new methods for end-to-end deep learning on relational databases. The paper includes a section on possible directions for future research to give a snapshot of some of the research possibilities there are in this area.

<!--- Matthias Fey*, Weihua Hu*, Kexin Huang*, Jan Eric Lenssen*, Rishabh Ranjan, Joshua Robinson*, Rex Ying, Jiaxuan You, Jure Leskovec.-->

# Design of RelBench

<p align="center"><img src="https://relbench.stanford.edu/img/relbench-fig.png" alt="logo" width="900px" /></p>

RelBench has the following main components:
1. 11 databases with a total of 66 tasks; both of these automatically downloadable for ease of use
2. Easy data loading, and graph construction from pkey-fkey links
3. Your own model, which can use any deep learning stack since RelBench is framework-agnostic. We provide a first model implementation using PyTorch Geometric and PyTorch Frame.
4. Standardized evaluators - all you need to do is produce a list of predictions for test samples, and RelBench computes metrics to ensure standardized evaluation
5. A leaderboard you can upload your results to, to track SOTA progress.


# Installation

You can install RelBench using `pip`:
```bash
pip install relbench
```

This will allow usage of the core RelBench data and task loading functionality.


To additionally use `relbench.modeling`, which requires [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) and [PyTorch Frame](https://github.com/pyg-team/pytorch-frame), install these dependencies manually or do:

```bash
pip install relbench[full]
```


For the scripts in the `examples` directory, use:
```bash
pip install relbench[example]
```

Then, to run a script:
```bash
git clone https://github.com/snap-stanford/relbench
cd relbench/examples
python gnn_entity.py --dataset rel-f1 --task driver-position
```


## Using External Integrations

**Using CTU datasets**

To use datasets from the CTU repository, use:
```bash
pip install relbench[ctu]
```

If you use the CTU datasets in your work, please cite [ReDeLEx](https://github.com/jakubpeleska/redelex) as below:

```bibtex
@misc{peleska2025redelex,
  title={REDELEX: A Framework for Relational Deep Learning Exploration},
  author={Jakub Peleška and Gustav Šír},
  year={2025},
  eprint={2506.22199},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.22199},
}
```



**Using 4DBInfer datasets**

To use datasets from the 4DBInfer repository, use:
```bash
pip install relbench[dbinfer]
```

If you use the 4DBInfer datasets in your work, please cite [4DBInfer](https://github.com/awslabs/multi-table-benchmark) as below:
```bibtex
@article{dbinfer,
  title={4DBInfer: A 4D Benchmarking Toolbox for Graph-Centric Predictive Modeling on Relational DBs},
  author={Wang, Minjie and Gan, Quan and Wipf, David and Cai, Zhenkun and Li, Ning and Tang, Jianheng and Zhang, Yanlin and Zhang, Zizhao and Mao, Zunyao and Song, Yakun and Wang, Yanbo and Li, Jiahang and Zhang, Han and Yang, Guang and Qin, Xiao and Lei, Chuan and Zhang, Muhan and Zhang, Weinan and Faloutsos, Christos and Zhang, Zheng},
  journal={arXiv preprint arXiv:2404.18209},
  year={2024}
}
```

**Using TGB datasets**

[TGB](https://tgb.complexdatalab.com/) datasets and tasks can be loaded in the RelBench format, and they cover bipartite link prediction (`tgbl-*`), heterogeneous link prediction (`thgl-*`), and node property prediction (`tgbn-*`). They use the official TGB evaluation protocols (one-vs-many MRR / Hits@10 for link prediction, NDCG@10 for node property prediction); standard RelBench-style evaluation also works as-is.

If you use the TGB datasets in your work, please cite [TGB](https://tgb.complexdatalab.com/) as below:
```bibtex
@article{huang2023temporal,
  title={Temporal graph benchmark for machine learning on temporal graphs},
  author={Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Rabbany, Reihaneh},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

```bibtex
@article{huang2024tgb2,
  title={TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs},
  author={Gastinger, Julia and Huang, Shenyang and Galkin, Mikhail and Loghmani, Erfan and Parviz, Ali and Poursafaei, Farimah and Danovitch, Jacob and Rossi, Emanuele and Koutis, Ioannis and Stuckenschmidt, Heiner and      Rabbany, Reihaneh and Rabusseau, Guillaume},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```



# Package Usage

This section provides a brief overview of using the RelBench package. For a more in-depth coverage see the [Tutorials](#tutorials) section. For detailed documentations, please see the code directly.

```python
import relbench
from relbench.base import Database, Table
```

Get a dataset, e.g., `rel-f1`:
```python
dataset = relbench.load_dataset("rel-f1")
```

Datasets live on the [Hugging Face Hub](https://huggingface.co/relbench) as self-describing
manifest folders (a `manifest.yaml` plus plain parquet), fetched and cached on first load at
a pinned revision. You can also load any Hub repo by id (`relbench.load_dataset("org/name")`)
or a local folder by path.

`dataset` consists of a `Database` object and temporal splitting times `dataset.val_timestamp` and `dataset.test_timestamp`.

To get the database:
```python
db: Database = dataset.get_db()
```

<details markdown="1"><summary>Preventing temporal leakage</summary>

By default, rows with timestamp > `dataset.test_timestamp` are excluded to prevent accidental temporal leakage. The full database can be obtained with:

```python
full_db: Database = dataset.get_db(upto_test_timestamp=False)
```

</details>

Various tasks can be defined on a dataset. For example, the `driver-position` task for `rel-f1`:
```python
task = relbench.load_task("rel-f1", "driver-position")
```

A task provides train/val/test tables:

```python
train_table: Table = task.get_table("train")
val_table: Table = task.get_table("val")
test_table: Table = task.get_table("test")
```

<details markdown="1"><summary>Preventing test leakage</summary>

By default, the target labels are hidden from the test table to prevent accidental data leakage. The full test table can be obtained with:

```python
full_test_table: Table = task.get_table("test", mask_input_cols=False)
```

</details>

You can build your model on top of the database and the task tables. After training and validation, you can make prediction from your model on the test table. Suppose your prediction `test_pred` is a NumPy array following the order of `task.test_table`, you can call the following to get the evaluation metrics:

```python
task.evaluate(test_pred)
```

Additionally, you can evaluate validation (or training) predictions as such:
```python
task.evaluate(val_pred, val_table)
```

# Tutorials

Tutorials are [marimo](https://marimo.io) notebooks (plain `.py` files) under [`tutorials/`](tutorials):

| Notebook | Description |
|---|---|
| [`quickstart.py`](tutorials/quickstart.py) | Load a dataset/task, explore the schema, run a baseline — runs in your browser via WebAssembly |
| [`gnn.py`](tutorials/gnn.py) | Train a GNN on an entity task (download and run locally) |

Run with `marimo edit tutorials/quickstart.py`. To contribute your own dataset or task, see [CONTRIBUTING.md](CONTRIBUTING.md).


# Contributing

Please check out [CONTRIBUTING.md](CONTRIBUTING.md) if you are interested in contributing datasets, tasks, bug fixes, etc. to RelBench.


# Cite RelBench

If you use RelBench in your work, please cite our position and benchmark papers:

```bibtex
@inproceedings{rdl,
  title={Position: Relational Deep Learning - Graph Representation Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

```bibtex
@misc{relbench,
      title={RelBench: A Benchmark for Deep Learning on Relational Databases},
      author={Joshua Robinson and Rishabh Ranjan and Weihua Hu and Kexin Huang and Jiaqi Han and Alejandro Dobles and Matthias Fey and Jan E. Lenssen and Yiwen Yuan and Zecheng Zhang and Xinwei He and Jure Leskovec},
      year={2024},
      eprint={2407.20060},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.20060},
}
```

If you use RelBench v2 in your work, please cite:
```bibtex
@misc{gu2026relbenchv2,
      title={{RelBench} v2: A Large-Scale Benchmark and Repository for Relational Data},
      author={Justin Gu and Rishabh Ranjan and Charilaos Kanatsoulis and Haiming Tang and Martin Jurkovic and Valter Hudovernik and Mark Znidar and Pranshu Chaturvedi and Parth Shroff and Fengyu Li and Jure Leskovec},
      year={2026},
      eprint={2602.12606},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.12606},
}
```
