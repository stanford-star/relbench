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
[#373](https://github.com/snap-stanford/relbench/issues/373) apply the same drops at build
time, so they reproduce the *current* published data; the exact per-column list also lives
in [`../scripts/clean_databases.py`](../scripts/clean_databases.py).

| generator | database | raw source | status |
|---|---|---|---|
| `f1.py` | rel-f1 | Ergast F1 snapshot (relbench.stanford.edu) | ✅ reproduces published |
| `salt.py` | rel-salt | SAP SALT db.zip (relbench.stanford.edu) | ✅ reproduces published |
| `arxiv.py` | rel-arxiv | arXiv citation db.zip (Dropbox) | ✅ reproduces published |
| `stack.py` | rel-stack | Stack Exchange dump (relbench.stanford.edu) | runnable (large; +#373 drop) |
| `trial.py` | rel-trial | ClinicalTrials/AACT (relbench.stanford.edu) | runnable (large; +#373 drops) |
| `ratebeer.py` | rel-ratebeer | RateBeer db.zip (Dropbox) | runnable (large; +#373 drops) |
| `avito.py` | rel-avito | Avito ads (relbench.stanford.edu) | runnable; public source is a 100k **sample** |
| `amazon.py` | rel-amazon | Amazon reviews (McAuley `jmcauley.ucsd.edu`) | runnable; large external download |
| `event.py` | rel-event | Event Recommendation (Kaggle) | needs Kaggle zip in `$RELBENCH_RAW_CACHE` |
| `hm.py` | rel-hm | H&M (Kaggle) | needs Kaggle zip in `$RELBENCH_RAW_CACHE` |
| `dbinfer.py` | `relbench/dbinfer` family | 4DBInfer pre-built `db.zip` artifacts | runnable (collection) |
| `tgb.py` | `relbench/tgb` family | TGB pre-built `db.zip` artifacts | runnable (collection) |

`dbinfer.py` / `tgb.py` mirror the legacy step for those `external` collections: the raw
temporal-graph / 4DBInfer conversion happens upstream, and RelBench ingested the resulting
pre-built `db.zip` artifacts — which these scripts download and rewrite into the HF layout.
The Kaggle-gated sources (`event`, `hm`) need their competition zip placed in the cache
first (the script names the expected file). `f1`/`salt`/`arxiv` are verified to reproduce
their published databases table-for-table; the rest are faithful ports of the same logic.
