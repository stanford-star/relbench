# Migration guide: RelBench < 3 → RelBench 3

RelBench 3 replaces the per-dataset Python classes and the `relbench.stanford.edu` download
server with a manifest-driven loader backed by the [Hugging Face Hub](https://huggingface.co/stanford-star).
A dataset is now just a folder — a `manifest.yaml`, one parquet per table, and a `tasks/`
subdirectory — so there is no registry, no `pooch` cache, and no code to add for a new
dataset or task.

The **data itself is unchanged** for the v1/v2 datasets and tasks: labels, splits and metrics
are the same, so numbers from earlier versions remain comparable (see
[Renames](#renames) for the one metric that changed name only).

If you are not ready to move, pin `relbench==2.1.0`. Note that the v2 download server it
depends on is deprecated in favor of the Hub.

## Leaderboard submissions

Submitting is now a supported flow in the package rather than a PR against a results file.
Write one prediction CSV per task, validate and package locally, then open a submission
issue:

```python
relbench.submit.write_prediction_table(task, test_pred, "preds/rel-f1__driver-position.csv")
```

```bash
python -m relbench.submit preds/
```

The task lists for the three boards are in `relbench.submit.LEADERBOARD_TASKS`. See the
[Leaderboard section of the README](README.md#leaderboard).

## Your own data

Custom datasets no longer subclass `Dataset` or call `register_dataset`/`register_task`.
Express the database as a `manifest.yaml` + parquet folder, express tasks as SQL in a task
manifest, and publish to the Hub (or keep it local and pass the path).
[`byod/README.md`](byod/README.md) is the full walkthrough.


## Loading datasets and tasks

```python
# RelBench 2
from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task, get_task_names

dataset = get_dataset("rel-f1", download=True)
task = get_task("rel-f1", "driver-position", download=True)
names = get_task_names("rel-f1")
```

```python
# RelBench 3
import relbench

dataset = relbench.load_dataset("rel-f1")
task = dataset.load_task("driver-position")
names = dataset.get_task_names()
```

Key points:

- `relbench.load_dataset` is the single entry point. `relbench.datasets` and
  `relbench.tasks` are gone, along with `get_dataset`, `get_task`, `get_task_names`,
  `get_dataset_names`, `register_dataset`, `register_task`, `download_dataset` and
  `download_task`.
- `load_task` / `get_task_names` live **on the dataset object**, not in a module.
- There is no `download=` argument. Data is fetched from the Hub on first use and cached
  by `huggingface_hub` (`HF_HOME` / `HF_HUB_CACHE`); `RELBENCH_CACHE_DIR` no longer does
  anything.
- `load_dataset` accepts a bare name (`"rel-f1"`, resolved across the RelBench Hub repos),
  a Hub address (`"stanford-star/relbench-v1/rel-f1"`, or any `org/repo[/subdir]`), or a local
  directory. `load_task` likewise accepts a bare task name, a Hub sub-path, or a local task
  directory — so tasks can live apart from their database.
- `revision=` pins a Hub revision: `relbench.load_dataset("rel-f1", revision="...")`.
- The datasets that used to require `pip install relbench[ctu]` (the 70+ CTU/ReDeLEx
  databases) are hosted on the Hub like everything else; `redelex` is no longer needed and
  the `[ctu]` extra is gone.

## Caching is gone

`get_db()` and `get_table()` were memoized with `lru_cache` and could persist to
`cache_dir`. In RelBench 3 both are **pure and uncached**, and `Dataset`/`BaseTask` no
longer take a `cache_dir` argument. Hold on to what you get back:

```python
db = task.get_db()          # call once
train = task.get_table("train")
```

Calling `dataset.get_db()` in a loop now re-reads the data every time. Task label tables
are downloaded as parquet and served as-is; pass `regenerate=True` to `load_task` to
recompute them from the database via the task manifest's SQL.

## Renames

| RelBench 2 | RelBench 3 |
|---|---|
| `TaskType.LINK_PREDICTION` | `TaskType.RECOMMENDATION` |
| task type string `"link_prediction"` | `"recommendation"` |
| `relbench.metrics.link_prediction_map` | `relbench.metrics.map` |

`relbench.metrics` now ships only the metrics RelBench evaluates with — `roc_auc`, `map`,
and NMAE via `make_nmae` (regression tasks carry their normalizer as `task.nmae_std`;
`relbench.train_std(task)` recomputes it). The long tail of extra metrics
(`f1`, `mae`, `rmse`, `r2`, `mrr`, the `multilabel_*` family, …) was removed — bring your
own and pass them explicitly: `task.evaluate(pred, metrics=[my_metric])`.

`task.evaluate(pred, target_table=None, metrics=None)` is otherwise unchanged.

## Modeling code (`relbench.modeling`)

`RecommendationTask.num_dst_nodes` no longer exists — it is a property of a particular
database, so it moved to the modeling layer:

```python
# RelBench 2
train_input = get_link_train_table_input(train_table, task)

# RelBench 3
from relbench.modeling.graph import get_link_train_table_input, num_dst_nodes

n = num_dst_nodes(db, task)
train_input = get_link_train_table_input(train_table, task, n)
```

Everything else in `relbench.modeling` (`make_pkey_fkey_graph`, `get_node_train_table_input`,
the loaders, `get_stype_proposal`) keeps its signature. `relbench.modeling` is now imported
lazily, so a plain `import relbench` no longer pulls in torch.

Reference scripts in [`examples/`](examples) were renamed: `baseline_*.py` → `trivial_*.py`.

