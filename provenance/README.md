# Data provenance: source → RelBench database generators

A runnable paper trail of how each RelBench **database** is built from its original raw
source — the download, the per-table processing, and the key reindexing. Each generator is
a standalone script that downloads the raw source and writes the Hugging Face layout:

```bash
python f1.py [OUT_DIR]          # -> OUT_DIR/manifest.yaml + OUT_DIR/db/*.parquet  (default ./rel-f1)
python dbinfer.py NAME [OUT]    # collection generators take a dataset name (or --all OUT_ROOT)
```

Dependencies are deliberately minimal — the Python stdlib (`urllib`/`zipfile`), `pandas`,
and `relbench` itself (`relbench.base` for the canonical `0..n-1` key reindexing the
published data uses, `relbench.manifest` for the manifest writer). No `pooch`, no legacy
`Dataset` machinery. Shared `fetch` / `write_hf` / `clean_datetime` live in `_lib.py`; raw
downloads cache under `$RELBENCH_RAW_CACHE` (default `~/.cache/relbench-raw`).

Generators for datasets cleaned for issue
[#373](https://github.com/stanford-star/relbench/issues/373) apply the same drops at build
time, so they reproduce the *current* published data; the exact per-column list also lives
in [`clean_databases.py`](clean_databases.py).

| generator | database | raw source | status |
|---|---|---|---|
| `f1.py` | rel-f1 | Ergast F1 snapshot (`stanford-star/relbench-raw`) | ✅ reproduces published |
| `salt.py` | rel-salt | SAP SALT db.zip (`stanford-star/relbench-raw`) | ✅ reproduces published |
| `arxiv.py` | rel-arxiv | arXiv citation db.zip (`stanford-star/relbench-raw` mirror, Dropbox fallback) | ✅ reproduces published |
| `stack.py` | rel-stack | Stack Exchange dump (`stanford-star/relbench-raw`) | runnable (large; +#373 drop) |
| `trial.py` | rel-trial | ClinicalTrials/AACT (`stanford-star/relbench-raw`) | runnable (large; +#373 drops) |
| `ratebeer.py` | rel-ratebeer | RateBeer db.zip (`stanford-star/relbench-raw` mirror, Dropbox fallback) | ✅ reproduces published (large; +#373 drops) |
| `avito.py` | rel-avito | Avito ads (`stanford-star/relbench-raw`) | runnable; public source is a 100k **sample** |
| `amazon.py` | rel-amazon | Amazon reviews (McAuley `jmcauley.ucsd.edu`) | runnable; large external download |
| `event.py` | rel-event | Event Recommendation (Kaggle) | needs `$RELBENCH_RAW_CACHE/event-recommendation-engine-challenge.zip` |
| `hm.py` | rel-hm | H&M (Kaggle) | needs `$RELBENCH_RAW_CACHE/h-and-m-personalized-fashion-recommendations.zip` |
| `dbinfer.py` | `stanford-star/dbinfer` family | 4DBInfer pre-built `db.zip` artifacts | runnable (collection) |
| `tgb.py` | `stanford-star/tgb` family | TGB pre-built `db.zip` artifacts | runnable (collection) |

## Verifying and cleaning

Four non-generator scripts round out the data's paper trail:

- **`check_provenance.py`** — for a dataset's `forecast` tasks, regenerate the labels from
  their manifest DuckDB query and assert they match the shipped labels. This is the
  guarantee that hosted labels are exactly what their SQL produces.

      python provenance/check_provenance.py stanford-star/relbench-v1/rel-f1   # Hub repo, subdir, or local path

- **`clean_databases.py`** — the reproducible record of the
  [#373](https://github.com/stanford-star/relbench/issues/373) cleanup applied to the
  published databases: the canonical per-column drop list (100%-NaN / preprocessing
  artifacts / future-leakage columns), plus the streaming drop + `schema.svg` refresh. The
  generators above bake these same drops in, so this is the "what and why", not a step you
  re-run.

- **`check_dbinfer.py`** — verify a ported `dbinfer-*` dataset: referential integrity, task
  joinability, and agreement with the original 4DBInfer archive. `DBINFER_DIR` is one
  `dbinfer-<name>` folder or a root containing several; with `RAW_ROOT` (default
  `$DBINFER_RAW_ROOT`), row counts and split sizes are also compared against the archive.

      python provenance/check_dbinfer.py DBINFER_DIR [RAW_ROOT]

- **`push_dbinfer.py`** — publish a locally generated `dbinfer` collection (the output of
  `dbinfer.py --all`) to its Hugging Face repo as a single commit. Files are added or
  overwritten, never deleted: anything a regenerate dropped (a table, a diagram) stays in
  the repo until it is removed explicitly.

      python provenance/push_dbinfer.py BUILD_DIR                         # dry run: show the plan
      python provenance/push_dbinfer.py BUILD_DIR --push                  # upload
      python provenance/push_dbinfer.py BUILD_DIR --push --message "..."  # custom commit message

`dbinfer.py` / `tgb.py` mirror the legacy step for those `external` collections: the raw
temporal-graph / 4DBInfer conversion happens upstream, and RelBench ingested the resulting
pre-built `db.zip` artifacts — which these scripts download and rewrite into the HF layout.
The Kaggle-gated sources (`event`, `hm`) cannot be downloaded anonymously: place the
competition zip at `$RELBENCH_RAW_CACHE/<filename>` first (the script's docstring, and the
error it raises when the file is missing, name the exact path; a download that is not a
real archive — e.g. a login page — is discarded rather than cached).
`f1`/`salt`/`arxiv`/`ratebeer` are verified to reproduce their published databases
table-for-table (the latter exercises the #373 column drops); the rest are faithful ports
of the same logic.

## Mirroring raw sources

`arxiv.py` and `ratebeer.py` try the `stanford-star/relbench-raw` mirror first and fall
back to the original Dropbox links. A maintainer uploads the mirror files once, from the
sha256-verified `db.zip` that `fetch` cached under `$RELBENCH_RAW_CACHE`:

    hf upload stanford-star/relbench-raw <local db.zip> rel-arxiv/db.zip --repo-type dataset
    hf upload stanford-star/relbench-raw <local db.zip> rel-ratebeer/db.zip --repo-type dataset

Until then the mirror URL 404s and the Dropbox fallback is used.
