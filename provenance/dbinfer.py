r"""Generate the **relbench/dbinfer** database family from the 4DBInfer collection.

    python dbinfer.py DATASET [OUT_DIR]   # one dataset (default OUT_DIR: ./DATASET)
    python dbinfer.py --all  [OUT_ROOT]   # the whole collection (default OUT_ROOT: ./dbinfer)

Unlike the single-database generators in this directory, ``dbinfer`` ports a *collection*:
the 4DBInfer relational benchmark (Wang et al., "4DBInfer: A 4D Benchmarking Toolbox for
Graph-Centric Predictive Modeling on Relational DBs", NeurIPS 2024 Datasets & Benchmarks;
https://github.com/awslabs/multi-table-benchmark) exported to RelBench format. The
collection spans the seven multi-table databases listed in ``DATASETS`` below
(``dbinfer-amazon``, ``dbinfer-avs``, ``dbinfer-diginetica``, ``dbinfer-outbrain-small``,
``dbinfer-retailrocket``, ``dbinfer-seznam``, ``dbinfer-stackexchange``).

Provenance note: each 4DBInfer dataset is distributed as a *pre-built* RelBench database
(``dbinfer-<name>/db.zip``, a directory of ``db/*.parquet``) produced by the upstream
``dbinfer-relbench-adapter`` export pipeline -- which uses the 4DBInfer ``dgl``-based
loader (``load_dbinfer_data``) to read the raw multi-table data and materialize a RelBench
``Database`` (see the legacy ``DBInferDatasetBase.make_db``). The original ``make_db`` here
therefore did not rebuild from the raw 4DBInfer archives; it loaded that pre-built database
(adapter ``get_db`` -> ``Database``). This generator reproduces that step verbatim from the
hosted ``db.zip`` artifacts -- no ``dbinfer-relbench-adapter``, no ``dgl``, no
``pooch``/legacy-relbench machinery -- recovering each table's primary/foreign-key and
time-column schema from the parquet metadata, and rewriting the Hugging Face layout
(``manifest.yaml`` + ``db/*.parquet``).

4DBInfer databases are static (not time-sliced), so the legacy used placeholder
validation/test timestamps (``1970-01-01`` / ``1970-01-02``) that satisfy the RelBench API
without trimming the static tables. Those placeholders are kept verbatim here and are the
only RelBench-side processing this generator carries. (The legacy also recorded a
``default_task_name`` per dataset; that is task metadata only and does not affect the
database parquet, so it is preserved below as documentation.)
"""

import json
import sys

import pandas as pd
import pyarrow.parquet as pq
from _lib import Table, fetch, write_hf

# Pre-built RelBench databases are hosted as ``<base>/dbinfer-<name>/db.zip``.
BASE_URL = "https://relbench.stanford.edu/download"

# 4DBInfer databases are not time-sliced; the legacy ``DBInferDatasetBase`` set these
# placeholder cutoffs (kept verbatim) for every dataset.
VAL_TIMESTAMP = "1970-01-01"
TEST_TIMESTAMP = "1970-01-02"

# Per-dataset (default_task_name, db.zip sha256). ``default_task_name`` is kept verbatim
# from the legacy registry (task metadata only); the hashes are the ``dbinfer-<name>/db.zip``
# entries from ``hashes.json``.
DATASETS: dict = {
    "amazon": (
        "rating",
        "0d9291bf880b5afa232b711a462ed7522a94a0ef5c141699ee8a2687dbbf0d08",
    ),
    "avs": (
        "repeater",
        "84295768a283c3c9ee0dbb25eb1c47d6d3f2c1956b523e7361f086fcf6664a93",
    ),
    "diginetica": (
        "ctr",
        "396e2aeb88cc616672b3b0d88a02dc13829c81b855edc3e668f7f2dbad3b96f0",
    ),
    "outbrain-small": (
        "ctr",
        "d186a71fc534bcac4299c616110805a997fa56fd16c15110d02e1c8eb3975210",
    ),
    "retailrocket": (
        "cvr",
        "6dbe83488e11c0d159b4592aa4cff57f6fa22b9f4bbf4ae38d0388d536897d75",
    ),
    "seznam": (
        "charge",
        "77314bf874dc495e8a4a61f2dc5f12982bbec3c5b6b7af5555e9a2bb587154d9",
    ),
    "stackexchange": (
        "churn",
        "4c8b8ac38b56d57bc2ead3c12be1237b69cae76062ed3f176fc02ad327a84d19",
    ),
}


def _load_table(path) -> Table:
    r"""Load a pre-built RelBench table, recovering pkey/fkey/time_col from the parquet
    metadata (the verbatim equivalent of the legacy ``Table.load`` /
    ``Database.load``)."""
    table = pq.read_table(path)
    df = table.to_pandas()
    # DuckDB cannot ingest pandas StringDtype reliably; match the legacy load and coerce.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
    meta = table.schema.metadata or {}
    keys = {b"fkey_col_to_pkey_table", b"pkey_col", b"time_col"}
    m = {
        k.decode("utf-8"): json.loads(v.decode("utf-8"))
        for k, v in meta.items()
        if k in keys
    }
    return Table(
        df=df,
        fkey_col_to_pkey_table=m.get("fkey_col_to_pkey_table", {}),
        pkey_col=m.get("pkey_col"),
        time_col=m.get("time_col"),
    )


def build(raw) -> dict:
    r"""Reconstruct the table dict from an extracted ``db.zip`` (a ``db/*.parquet``
    dir)."""
    db_dir = next(raw.rglob("*.parquet")).parent
    return {p.stem: _load_table(p) for p in sorted(db_dir.glob("*.parquet"))}


def main(dataset: str, out=None) -> None:
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dbinfer dataset '{dataset}'. Known: {sorted(DATASETS)}"
        )
    _task, sha = DATASETS[dataset]
    name = f"dbinfer-{dataset}"
    raw = fetch(f"{BASE_URL}/{name}/db.zip", sha)
    write_hf(
        out or name,
        name,
        VAL_TIMESTAMP,
        TEST_TIMESTAMP,
        build(raw),
    )


def main_all(out_root="dbinfer") -> None:
    from pathlib import Path

    for dataset in DATASETS:
        print(f"=== {dataset} ===", flush=True)
        main(dataset, Path(out_root) / f"dbinfer-{dataset}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        main_all(sys.argv[2] if len(sys.argv) > 2 else "dbinfer")
    elif len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(
            f"usage: python dbinfer.py DATASET [OUT_DIR] | --all [OUT_ROOT]\n"
            f"datasets: {', '.join(DATASETS)}"
        )
