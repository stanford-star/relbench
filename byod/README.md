# BYOD: bring your own data

Bringing your own data to RelBench needs **no Python classes and no central registry**. A dataset is a
self-describing folder you publish to the [Hugging Face Hub](https://huggingface.co); a task
is one subdirectory inside it. RelBench loads it straight from its `org/repo[/subdir]`
address. That's the whole contribution model — the rest of this page is just how to produce
the folder.

## The dataset folder

```
<dataset>/
  manifest.yaml             # tables, primary keys, the foreign-key graph, the time splits
  db/<table>.parquet        # one plain parquet per table (native dtypes; no RelBench metadata)
  tasks/<task>/
    manifest.yaml           # the task spec (+ a DuckDB query, for `forecast` tasks)
    {train,val,test}.parquet
  README.md                 # dataset card (generated)
  schema.svg                # entity-relationship diagram (generated)
  STATS/                    # overview tables that drive the Hub dataset viewer (generated)
    databases.parquet
    tasks.parquet
```

`manifest.yaml` is the **single source of truth** for the relational structure — primary
keys, the foreign-key graph, and the time columns. Because that lives in the manifest, the
`db/*.parquet` files stay plain: they load directly with pandas or DuckDB, no RelBench
required. The schema is defined in [`relbench/manifest.py`](../relbench/manifest.py)
(`DatasetManifest`, `TaskManifest`) — write one by hand or build it programmatically.

## Three kinds of task

A task declares how its labels are produced (`kind` in its `manifest.yaml`):

| kind | labels are… | example |
|---|---|---|
| `forecast` | computed by a SQL query over the database — **regenerable**, and checked against the shipped labels by [`provenance/check_provenance.py`](../provenance/check_provenance.py) | predict total sales of a product in the next month |
| `autocomplete` | an existing column you predict (the column is masked from the graph) | predict a product's category |
| `external` | shipped as-is (built by an upstream pipeline) | TGB / 4DBInfer tasks |

A `forecast` task's query sees a `timestamps(timestamp)` relation (the per-split seed
timestamps), every table as a view by its name, and `{timedelta}` substituted as a DuckDB
`INTERVAL`. Its `SELECT` must output the declared entity / target / time columns.

## Package it for the Hub

Once you have `manifest.yaml` + `db/*.parquet` + `tasks/`, the rest is generated. The
published [`stanford-star/relbench-v1/rel-f1`](https://huggingface.co/datasets/stanford-star/relbench-v1) is a
complete worked example.

**1. Generate the schema diagram and dataset card** from the manifest (`render_schema_svg`
needs `pip install "relbench[schema]"` plus the system `graphviz`/`dot` binary):

```python
from relbench.manifest import DatasetManifest
from relbench.schema import dataset_card, render_schema_svg

m = DatasetManifest.load("manifest.yaml")
render_schema_svg(m, "schema.svg", db_dir="db")                      # ER diagram from db/*.parquet
open("README.md", "w").write(dataset_card(m, repo="<org>/<repo>"))   # dataset card
```

**2. Verify, build the overview tables, and upload:**

```bash
python provenance/check_provenance.py .                       # forecast labels reproduce from SQL
python byod/build_databases_overview.py . --out STATS # -> STATS/databases.parquet
python byod/build_tasks_overview.py     . --out STATS # -> STATS/tasks.parquet
hf upload <org>/<repo> . --repo-type dataset                  # commit the whole folder
```

`<path>` above is `.` for a local dataset folder, but every tool also accepts a Hub repo
(`stanford-star/relbench-v1`) or a single hosted dataset (`stanford-star/relbench-v1/rel-f1`) — so you can rebuild
the tables for anything already on the Hub. The two overview builders read only manifests
and parquet *footers/labels*, never the full `db/` tables, so they stay cheap even for
repos with thousands of datasets. `check_provenance.py` and `compute_regression_stds.py`
need the full database (the first regenerates labels against it, the second loads each
dataset), so they cost as much as loading the dataset.

## The dataset viewer (`STATS/` + two configs)

The two `STATS/` tables drive the **databases** and **tasks** subsets of the
[Hub dataset viewer](https://huggingface.co/docs/hub/datasets-viewer):

- `STATS/databases.parquet` — one row per database (tables, rows, columns, per-type task
  counts, timestamps, size).
- `STATS/tasks.parquet` — one row per task (train/val/test sizes, unique entities,
  train/test entity overlap, destination links, metric, …): the statistics reported in the
  RelBench [v1](https://arxiv.org/abs/2407.20060) / [v2](https://arxiv.org/abs/2602.12606)
  papers.

Wire them into the viewer from the repo's `README.md` front matter. They have different
shapes, and the viewer unifies columns *within* a config, so they must be **two configs**
(shown as selectable "subsets"). The split name is just a label for the collection
(e.g. `eval`, `pretrain`):

```yaml
configs:
- config_name: databases
  data_files:
  - split: eval
    path: STATS/databases.parquet
- config_name: tasks
  data_files:
  - split: eval
    path: STATS/tasks.parquet
```

`build_databases_overview.py` fills the structural columns but leaves `domain`,
`description`, `license`, and `source_url` blank — fill those in by hand (`--merge`
preserves your edits across re-runs: it carries them over from the repo's existing table,
or, for a local folder, from the `databases.parquet` already under `--out`). You're
encouraged to add your own columns too.

## Tools in this directory

| script | what it builds |
|---|---|
| `build_databases_overview.py` | `STATS/databases.parquet` (one row per database) |
| `build_tasks_overview.py` | `STATS/tasks.parquet` (one row per task; `--check` cross-checks the paper tables) |
| `compute_regression_stds.py` | `regression_stds.json` — per-task NMAE normalizers for hosted regression tasks (optional; loading falls back to computing the std from the train split) |
| `sort_hosted_labels.py` | hosted task labels rewritten in canonical (sorted) order — a repair/check tool that reorders rows only, so no metric changes (dry run by default; 0 files reported means the repo is already canonical) |

Pass `--push` to any of them to upload the result to the repo. Data-provenance tooling —
how RelBench's own databases were generated, cleaned, and verified — lives in
[`../provenance/`](../provenance).
