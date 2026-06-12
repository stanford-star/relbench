# Data provenance: source → RelBench database generators

A runnable paper trail of how each RelBench **database** is built from its original raw
source — the download, the per-table processing, and the key reindexing. Each generator is
a standalone script that downloads the raw source and writes the Hugging Face layout
directly:

```bash
python f1.py [OUT_DIR]      # -> OUT_DIR/manifest.yaml + OUT_DIR/db/*.parquet  (default ./rel-f1)
```

Dependencies are deliberately minimal — the Python stdlib (`urllib`/`zipfile`), `pandas`,
and `relbench` itself (`relbench.base` for the canonical `0..n-1` key reindexing the
published data uses, `relbench.manifest` for the manifest writer). No `pooch`, no legacy
`Dataset` machinery. Shared helpers (`fetch`, `write_hf`) live in `_lib.py`; raw downloads
cache under `$RELBENCH_RAW_CACHE`.

`f1.py` is verified to reproduce `relbench/core/rel-f1` exactly (every table, column, and
value). Generators for datasets that were cleaned for issue
[#373](https://github.com/snap-stanford/relbench/issues/373) apply the same drops at build
time, so they reproduce the *current* published data; the exact per-column list lives in
[`../scripts/clean_databases.py`](../scripts/clean_databases.py).

| generator | database | raw source |
|---|---|---|
| `f1.py` ✅ | rel-f1 | Ergast Formula 1 snapshot (relbench.stanford.edu) |
| `avito.py` | rel-avito | Avito ads (relbench.stanford.edu) |
| `event.py` | rel-event | Event Recommendation (Kaggle — needs credentials) |
| `hm.py` | rel-hm | H&M (Kaggle — needs credentials) |
| `amazon.py` | rel-amazon | Amazon reviews (McAuley) |
| `stack.py` | rel-stack | Stack Exchange forum dump (relbench.stanford.edu) |
| `trial.py` | rel-trial | ClinicalTrials.gov / AACT (relbench.stanford.edu) |
| `arxiv.py` | rel-arxiv | arXiv citation graph |
| `ratebeer.py` | rel-ratebeer | RateBeer scrape |
| `salt.py` | rel-salt | SAP SALT (relbench.stanford.edu) |
| `dbinfer.py` | `relbench/dbinfer` family | 4DBInfer benchmark collection |
| `tgb.py` | `relbench/tgb` family | Temporal Graph Benchmark collection |

`f1.py` + `_lib.py` are the migrated, runnable form. The remaining scripts still carry the
original `make_db` logic verbatim (legacy `relbench` + `pooch`) and are being migrated to
the same standalone form — same processing, just `pooch`→`fetch` and `Database`→`write_hf`.
`utils.py` / `hashes.json` support the not-yet-migrated ones.
