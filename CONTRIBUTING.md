# Contributing to RelBench

We welcome contributions of **datasets** and **tasks**. This guide is self-contained: by
the end you'll be able to publish a new RelBench dataset and tasks in the current format,
with no prior knowledge of RelBench internals.

For bugs and features, please open a GitHub issue or pull request.

---

## 1. The format in one minute

A RelBench dataset is a **self-describing folder** — `manifest.yaml` files plus plain
parquet. There are **no Python classes to write**:

```
<dataset>/
  manifest.yaml                       # description, tables, primary keys, fkey graph, splits
  README.md                           # dataset card (auto-generated)
  schema.svg                          # ER diagram: every column + the fkey graph (auto-generated)
  db/<table>.parquet                  # one parquet per table (plain parquet, native dtypes)
  tasks/<task>/
    manifest.yaml                     # task spec + description (+ a duckdb query, for `forecast`)
    {train,val,test}.parquet          # the task's labels
```

`manifest.yaml` is the **single source of truth** for the relational structure (primary
keys, foreign keys, time columns); the parquet files carry only data, so they load with
plain `pandas`/`duckdb` too. Publish the folder to the
[Hugging Face Hub](https://huggingface.co) and anyone can load it:

```python
import relbench
ds   = relbench.load_dataset("your-org/your-dataset")     # a Hub repo, or a local path
task = relbench.load_task("your-org/your-dataset", "your-task")
db   = ds.get_db()                  # a Database of Tables, linked by the fkey graph
train = task.get_table("train")     # train/val/test label tables
task.evaluate(pred)                 # standardized metrics (chosen automatically by task type)
```

**The published [`rel-f1`](https://huggingface.co/datasets/relbench/v1/tree/main/rel-f1)
dataset is a complete worked example — browse its `manifest.yaml`, `tasks/`, and `schema.svg`.**

### Task kinds

- **`forecast`** — temporal labels computed by a duckdb `sql` query (churn, sales, future
  links, ...). Labels are **regenerable** from the database, and CI checks they match.
- **`autocomplete`** — predict an existing column of a table (declarative; no SQL).
- **`external`** — labels produced by an external process and shipped as-is. Not regenerable.

Prefer `forecast` when labels are a function of the database over time.

---

## 2. Add a dataset

**(a) Write your tables as plain parquet + a `manifest.yaml`.** Use the manifest classes so
you get YAML + validation for free:

```python
from pathlib import Path
from relbench.manifest import DatasetManifest, TableSpec, validate_dataset_manifest

out = Path("my-dataset")
(out / "db").mkdir(parents=True, exist_ok=True)
customers.to_parquet(out / "db" / "customers.parquet", index=False)
orders.to_parquet(out / "db" / "orders.parquet", index=False)

manifest = DatasetManifest(
    name="my-dataset",
    description="One-paragraph description of the database and its domain.",
    val_timestamp="2020-01-01",     # rows up to here are inputs for the val split
    test_timestamp="2021-01-01",    # rows up to here are inputs for the test split
    tables={
        "customers": TableSpec(pkey="customer_id"),
        "orders": TableSpec(
            pkey="order_id",
            time_col="order_time",
            fkeys={"customer_id": "customers"},   # fkey column -> table it points into
        ),
    },
)
manifest.save(out / "manifest.yaml")
validate_dataset_manifest(manifest, out / "db")   # checks the named columns exist
```

Conventions: primary keys are 0..N-1 integers; foreign keys are integer indices into the
referenced table; time columns are `datetime64`.

**(b) Generate the schema diagram and dataset card** — after you've written the task
manifests (§3). `render_schema_svg` reads every column from the parquet, so point it at the
`db/` folder; it needs the `relbench[schema]` extra and the Graphviz `dot` binary on PATH:

```python
from relbench.schema import dataset_card, render_schema_svg
from relbench.manifest import TaskManifest
render_schema_svg(manifest, out / "schema.svg", db_dir=out / "db")   # ER diagram (all columns)
tasks = [TaskManifest.load(p / "manifest.yaml") for p in sorted((out / "tasks").iterdir())]
(out / "README.md").write_text(dataset_card(manifest, tasks))        # embeds schema.svg
```

**(c) Keep your processing script** for provenance (it produces the folder). Datasets are the
root of trust, but a reproducible build is encouraged. The script lives wherever you develop
the dataset — it is not shipped in the `relbench` package.

---

## 3. Add a task

Adding a task is just **adding a directory** under `tasks/` — the loader discovers tasks by
listing `tasks/*/manifest.yaml`.

### A `forecast` task

```python
from relbench.manifest import TaskManifest

TaskManifest(
    name="customer-churn",
    kind="forecast",
    task_type="binary_classification",
    description="Predict whether a customer makes no purchase in the next 90 days.",
    entity_table="customers",
    entity_col="customer_id",
    target_col="churn",
    time_col="timestamp",
    timedelta="90 days",            # the prediction-window length
    sql="""
        SELECT
            t.timestamp,
            c.customer_id,
            CAST(NOT EXISTS (
                SELECT 1 FROM orders o
                WHERE o.customer_id = c.customer_id
                  AND o.order_time >  t.timestamp
                  AND o.order_time <= t.timestamp + INTERVAL '{timedelta}'
            ) AS INTEGER) AS churn
        FROM timestamps t, customers c
        WHERE EXISTS (
            SELECT 1 FROM orders o
            WHERE o.customer_id = c.customer_id
              AND o.order_time >  t.timestamp - INTERVAL '{timedelta}'
              AND o.order_time <= t.timestamp
        )
    """,
).save("my-dataset/tasks/customer-churn/manifest.yaml")
```

**The SQL contract.** The query runs once per split and sees:
- a relation **`timestamps`** with one column `timestamp` (the anchor times RelBench
  generates for the split);
- every database table as a **view by its name**;
- the placeholder **`{timedelta}`**, substituted with the task's `timedelta`.

It must `SELECT` the declared columns — the `time_col`, entity column(s), and `target_col`.
Semantics: the label for an entity at anchor time `t` is computed from the window
`(t, t + timedelta]` (strictly future — no leakage).

For **link prediction**, set `task_type="link_prediction"`, use
`src_entity_table`/`src_entity_col`/`dst_entity_table`/`dst_entity_col` + `eval_k`, and emit
a list of destination ids (e.g. `LIST(DISTINCT o.product_id) AS product_id`). See the
[`rel-f1` tasks](https://huggingface.co/datasets/relbench/v1/tree/main/rel-f1/tasks) for
entity, binary, and link examples.

Metrics are **not** specified — they default from `task_type` (regression → R²/MAE/RMSE,
binary → AP/accuracy/F1/AUROC, link → precision/recall/MAP@k).

### An `autocomplete` task

```python
TaskManifest(
    name="orders-amount", kind="autocomplete", task_type="regression",
    description="Predict the amount of each order.",
    entity_table="orders", target_col="amount",
    remove_columns=[["orders", "discount"]],   # feature columns to drop from the graph
).save("my-dataset/tasks/orders-amount/manifest.yaml")
```

### Generate the labels

```python
import relbench
task = relbench.load_task("my-dataset", "customer-churn", regenerate=True)   # runs the SQL
for split in ["train", "val", "test"]:
    df = task.get_table(split, mask_input_cols=False).df
    df.to_parquet(f"my-dataset/tasks/customer-churn/{split}.parquet", index=False)
```

---

## 4. Publish to the Hub

```python
from huggingface_hub import create_repo, upload_folder
create_repo("your-org/my-dataset", repo_type="dataset")
upload_folder(folder_path="my-dataset", repo_id="your-org/my-dataset", repo_type="dataset")
```

The Hub renders your `README.md` card with the embedded `schema.svg`. Anyone can now
`relbench.load_dataset("your-org/my-dataset")`.

---

## 5. Submit it for inclusion in official RelBench

Open a PR adding your dataset to `relbench/registry.json`, mapping its name to the Hub repo
and a **pinned commit revision**:

```json
{ "my-dataset": {"repo_id": "your-org/my-dataset", "revision": "<commit-sha>", "path": ""} }
```

(Official datasets are grouped into family repos like `relbench/v1`; pinning the `revision`
is what makes a dataset's tasks reproducible against a fixed database version.)

For `forecast` tasks, ensure the provenance check passes — it regenerates labels from your
SQL and asserts they match the shipped parquet:

```bash
python -m relbench.check_provenance path/to/my-dataset
```

Describe the dataset/tasks in your PR and link the Hub repo. Thanks for contributing!
