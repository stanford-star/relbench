# Data provenance: source → RelBench database generators

For transparency, this directory preserves the scripts that build each RelBench
**database** from its original raw source — download URL, hash, and the exact per-table
processing. They are the legacy (pre-3.0) `relbench.base.Dataset.make_db()` generators,
kept here **verbatim as reference**: they target the old `relbench` API and `pooch`, so
they are *not* run by the current (manifest + Hugging Face) package and are not maintained.
What they document is how the raw source became the hosted `db/*.parquet`.

| script | database(s) | raw source |
|---|---|---|
| `amazon.py` | rel-amazon | Amazon product reviews (McAuley) |
| `avito.py` | rel-avito | `relbench.stanford.edu/data/rel-avito-raw-*.zip` |
| `event.py` | rel-event | Event Recommendation (Kaggle) |
| `f1.py` | rel-f1 | `relbench.stanford.edu/data/relbench-f1-raw.zip` (Ergast) |
| `hm.py` | rel-hm | H&M Personalized Fashion (Kaggle) |
| `stack.py` | rel-stack | `relbench.stanford.edu/data/relbench-forum-raw.zip` (Stack Exchange) |
| `trial.py` | rel-trial | `relbench.stanford.edu/data/relbench-trial.zip` (ClinicalTrials.gov / AACT) |
| `arxiv.py` | rel-arxiv | arXiv citation graph |
| `ratebeer.py` | rel-ratebeer | RateBeer scrape |
| `salt.py` | rel-salt | `relbench.stanford.edu/download/rel-salt/db.zip` (SAP SALT) |
| `dbinfer.py` | the `relbench/dbinfer` family | 4DBInfer benchmark datasets |
| `tgb.py` | the `relbench/tgb` family | Temporal Graph Benchmark |

`utils.py` (`clean_datetime`, `unzip_processor`) is the shared helper they import;
`hashes.json` pins the raw-download checksums.

## Post-generation cleanup (issue [#373](https://github.com/snap-stanford/relbench/issues/373))

The generators reproduce the raw source faithfully, so some noise carried straight through.
The hosted databases additionally have those columns removed — **100%-NaN** fields,
preprocessing **artifacts** (pandas `Unnamed: 0` row indices, SQL-Server
`msrepl_tran_version`), and **future-leakage** scrape-time aggregates (mostly in
rel-ratebeer). The exact per-table list, with the reason for each column, is the single
source of truth in [`../scripts/clean_databases.py`](../scripts/clean_databases.py); it is
reproducible and leaves keys, time columns, and all task labels untouched
(`python -m relbench.check_provenance <dataset>` passes on the cleaned data).
