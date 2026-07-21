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
  <a href="#contributing"><b>Contributing</b></a>
</p>

## What is RelBench?

Most real-world data lives in a **relational database**: many tables linked by foreign keys
— customers, orders, products, events — not one flat spreadsheet. Getting that into a
machine-learning model normally means slow, manual feature engineering to flatten it down.

**RelBench is a benchmark for skipping that step and learning directly from the raw linked
tables with deep learning.** Each dataset is a real relational database paired with
predictive **tasks** — *"will this driver finish on the podium in their next race?"*,
*"how much will this user spend next month?"* — each with temporal train/val/test splits and
a standard metric, so results are comparable. Datasets and tasks load straight from the
[Hugging Face Hub](https://huggingface.co/relbench).

## Get Started

```bash
pip install relbench           # data + task loading
pip install relbench[example]  # + PyTorch Geometric & PyTorch Frame, for the GNN examples
```

Load a dataset and a task — both come straight from the Hub, with **no per-dataset code**:

```python
import relbench

dataset = relbench.load_dataset("relbench/core/rel-f1")   # a Hub 'org/repo[/subdir]', or a local path
db = dataset.get_db()                  # a Database: tables linked by a foreign-key graph

task = relbench.load_task("relbench/core/rel-f1", "driver-position")
train_table = task.get_table("train")  # train / val / test label tables
test_table  = task.get_table("test")   # the target column is hidden on test

# ... train any model on db + train_table, predict on the test entities ...
metrics = task.evaluate(test_pred)     # standard metric for the task type
```

`dataset.val_timestamp` / `dataset.test_timestamp` give the temporal split points. RelBench
is framework-agnostic — bring any modeling stack. For a reference Graph Neural Network on
[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) +
[PyTorch Frame](https://github.com/pyg-team/pytorch-frame), see `relbench.modeling` and the
runnable scripts in [`examples/`](examples).

### Tutorials

Open these directly in Google Colab — no setup required:

| Tutorial | What it covers | |
|---|---|---|
| [**Quickstart**](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/quickstart.ipynb) | Load a dataset/task, explore the schema, run a baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/quickstart.ipynb) |
| [**Training a GNN**](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/gnn.ipynb) | A GNN baseline for an entity task (PyG + PyTorch Frame) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snap-stanford/relbench/blob/relbench-hf/tutorials/gnn.ipynb) (needs a GPU) |

## Submitting to the leaderboard

RelBench has three independent leaderboards — **classification**, **regression**, and
**recommendation** — each scored separately. A submission to one is a directory of
**prediction-table CSVs**, one per task, each named `<dataset>__<task>.csv` (a double
underscore between dataset and task).

A prediction table is a task's test table with the target column replaced by your model's
predictions:

- **binary classification** — a predicted probability in `[0, 1]`;
- **regression** — a numeric prediction on the original target scale;
- **recommendation** — the top-`eval_k` predicted destination entity ids, as a JSON list per row.

Alongside the CSVs, include a **`metadata.yaml`** describing the method — this is what the
leaderboard displays. `name` and `type` are required; `url` and `note` are
optional. The submission date is recorded automatically.

```yaml
name: RelGNN                    # required — display name
type: fine-tuned               # required — fine-tuned | in-context
url: https://arxiv.org/abs/... # optional — paper / project / code link
note: caveats / eval details, shown on hover   # optional
```

A submission is just the prediction CSVs and `metadata.yaml` — the server **rejects** a
submission that contains anything else (saved arrays, logs, subdirectories). The `--submit` /
`--package` tooling below lists and drops such files for you, so the zip it uploads is clean.

Write these from code with `relbench.leaderboard.write_prediction_table(task, pred, path)`,
and score a single task with `relbench.leaderboard.evaluate_task(task_name, csv)` (where
`task_name` is `"<dataset>/<task>"`).

Each leaderboard is graded over its **complete** task set — **classification** has 12 tasks,
**regression** 9, and **recommendation** 10. The exact lists live in
`relbench.leaderboard.LEADERBOARD_TASKS`:

- **Classification** (12): `rel-amazon/user-churn`, `rel-amazon/item-churn`, `rel-avito/user-visits`, `rel-avito/user-clicks`, `rel-event/user-repeat`, `rel-event/user-ignore`, `rel-f1/driver-dnf`, `rel-f1/driver-top3`, `rel-hm/user-churn`, `rel-stack/user-engagement`, `rel-stack/user-badge`, `rel-trial/study-outcome`
- **Regression** (9): `rel-amazon/user-ltv`, `rel-amazon/item-ltv`, `rel-avito/ad-ctr`, `rel-event/user-attendance`, `rel-f1/driver-position`, `rel-hm/item-sales`, `rel-stack/post-votes`, `rel-trial/study-adverse`, `rel-trial/site-success`
- **Recommendation** (10): `rel-amazon/user-item-purchase`, `rel-amazon/user-item-rate`, `rel-amazon/user-item-review`, `rel-avito/user-ad-visit`, `rel-f1/driver-circuit-compete`, `rel-hm/user-item-purchase`, `rel-stack/user-post-comment`, `rel-stack/post-post-related`, `rel-trial/condition-sponsor-run`, `rel-trial/site-sponsor-run`

Validate locally before sending:

```bash
python -m relbench.leaderboard <submission_dir>
```

For every task this checks coverage (a prediction for each test point), key uniqueness and
an exact match against the ground-truth test table, and value ranges; it then prints
per-task metrics, per-leaderboard aggregate metrics, and a verdict for each leaderboard. It
also parses and validates `metadata.yaml`, printing the method metadata and flagging any
missing required field or out-of-range value.

Once a leaderboard is **validated**, submit straight from the CLI:

```bash
python -m relbench.leaderboard <submission_dir> --submit
```

This re-validates, **prompts you for the `metadata.yaml` fields if it's missing** (and writes
it), builds a clean zip (only the prediction CSVs + `metadata.yaml`), and uploads it. Use
`--package` instead to only write the zip locally (e.g. to upload via the website) and print
the submit command. The submission is re-validated server-side and, on success, opens a pull
request for maintainer review; your entry appears on the leaderboard once it is merged. You
can also upload a zip on the website at
[**tabular.stanford.edu/leaderboard/submit**](https://tabular.stanford.edu/leaderboard/submit/).

## Contributing

Adding a dataset or task takes **no code**. A dataset is a self-describing folder — a
`manifest.yaml` (tables, keys, the foreign-key graph, the time splits), one plain parquet
per table, and a `tasks/` subdirectory — that you publish to the
[Hugging Face Hub](https://huggingface.co). RelBench loads it straight from its
`org/repo[/subdir]` address; there is no central registry to register with.

[**`contributing/README.md`**](contributing/README.md) is the full walkthrough, and the
published [`relbench/core/rel-f1`](https://huggingface.co/datasets/relbench/core) is a
complete worked example. For how RelBench's own databases were built, cleaned, and verified
from their original sources, see [`provenance/`](provenance). Bug reports and feature
requests: open a GitHub issue or pull request.

## Cite RelBench

If you use RelBench, please cite the position and benchmark papers:

```bibtex
@inproceedings{rdl,
  title={Position: Relational Deep Learning - Graph Representation Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}

@inproceedings{relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  author={Robinson, Joshua and Ranjan, Rishabh and Hu, Weihua and Huang, Kexin and Han, Jiaqi and Dobles, Alejandro and Fey, Matthias and Lenssen, Jan Eric and Yuan, Yiwen and Zhang, Zecheng and He, Xinwei and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

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
