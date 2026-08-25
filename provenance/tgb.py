r"""Generate the **stanford-star/tgb** database family from the Temporal Graph Benchmark.

    python tgb.py DATASET [OUT_DIR]   # one dataset (default OUT_DIR: ./DATASET)
    python tgb.py --all  [OUT_ROOT]   # the whole collection (default OUT_ROOT: ./tgb)

Unlike the single-database generators in this directory, ``tgb`` ports a *collection*:
the Temporal Graph Benchmark (TGB, https://tgb.complexdatalab.com/) exported to RelBench
format. The collection spans three families -- ``tgbl-*`` (bipartite link), ``thgl-*``
(heterogeneous link), and ``tgbn-*`` (node property) -- 15 datasets in all (see
``DATASETS`` below).

Provenance note: each TGB dataset is distributed as a *pre-built* RelBench database
(``<name>/db.zip``, a directory of ``db/*.parquet``) produced by an upstream TGB ->
RelBench export script that uses the ``tgb`` package to read the raw temporal-graph
events. The original
``make_db`` here therefore did not rebuild from the raw events; it loaded that pre-built
database (the legacy ``Database.load``). This generator reproduces that step verbatim from the
hosted ``db.zip`` artifacts -- no ``tgb`` package, no ``pooch``/legacy-relbench machinery
-- recovering each table's primary/foreign-key and time-column schema from the parquet
metadata, and rewriting the Hugging Face layout (``manifest.yaml`` + ``db/*.parquet``).
The per-dataset validation/test cutoffs (kept verbatim as ``DATASETS``) are the only
RelBench-side processing this generator carries.
"""

import json
import sys

import pandas as pd
import pyarrow.parquet as pq
from _lib import Table, fetch, write_hf

# Pre-built RelBench databases are hosted as ``<base>/<name>/db.zip``.
BASE_URL = "https://huggingface.co/datasets/stanford-star/relbench-raw/resolve/main/tgb"

# Per-dataset (val_timestamp_s, test_timestamp_s, db.zip sha256). The cutoffs are the
# official TGB split boundaries, kept verbatim from the legacy ``_TGB_CUTOFFS``; the
# hashes are the ``<name>/db.zip`` entries from ``hashes.json``.
DATASETS: dict = {
    # tgbl-* : bipartite (link) datasets.
    "tgbl-wiki": (
        1862653,
        2218300,
        "ad7b55d1d7b7125c06588db0b8cbebe87c629a95a6c9a911369f89aca9dffdc9",
    ),
    "tgbl-wiki-v2": (
        1862653,
        2218300,
        "5eb56f8e459405e3b554ce56585903c80038d84a99ca4333b00ce32c8c7a38f1",
    ),
    "tgbl-review": (
        1464912000,
        1488844800,
        "63b405aafd8092cbda297694e649329ca806ccc3c49c8e6f53eb4a3c19075091",
    ),
    "tgbl-review-v2": (
        1464912000,
        1488844800,
        "a5f1a7a9661a700ebb1a3b43df74037853ad3bd6ed54c74342045d2dc8448bc2",
    ),
    "tgbl-coin": (
        1662096249,
        1664482319,
        "6de3b62bcfc59bb18ff8de0614ce5cbcf21179ad56cb58920d25ade32ec43e00",
    ),
    "tgbl-comment": (
        1282869285,
        1288838725,
        "4eb41776954efa2cb06cdb64e093b182335b8683e60236a6645cf4bcd83597be",
    ),
    "tgbl-flight": (
        1638162000,
        1653796800,
        "e82646893a45be6c5312e5dc1a774c08d929ecb9eb309c0ad9fdec6dae3b156a",
    ),
    # thgl-* : heterogeneous (link) datasets.
    "thgl-software": (
        1706003880,
        1706315669,
        "35816ef6a07be2291349f081814e18b498a2e46e006ed21be36fdb1a8a0eb90d",
    ),
    "thgl-forum": (
        1390426563,
        1390838358,
        "fdb20c1afc542e8026b85df850b5e8a539694c182eb4530924fad65964461256",
    ),
    "thgl-github": (
        1711075987,
        1711482874,
        "f69c1d49779a4dbead101e7325447854c8890982d254efb2b722117819ba8304",
    ),
    "thgl-myket": (
        1603724860,
        1606341312,
        "67a3af25553ceabe8a7eff8906c2213bd82e932cdaeb48ffa326b39d30e0f0cf",
    ),
    # tgbn-* : node-property datasets.
    "tgbn-trade": (
        1262304000,
        1388534400,
        "8a983ae6281ea058dc8d84f6ce339d2dacf0ae9bb98bb82cca3fed54ec3fe370",
    ),
    "tgbn-genre": (
        1216427762,
        1230448684,
        "e46aecc28315ca9872117817ce65bc1f4d00ed261ab6d35038a84bcf0ebab7bf",
    ),
    "tgbn-reddit": (
        1279485233,
        1286653871,
        "bc444f5cbaf7004ef7c6c8f98ae12ae72e8427ccc9710e96e46b0fc156cc952e",
    ),
    "tgbn-token": (
        1522889022,
        1525386888,
        "44a0f90a62642b8054ef1fbee35c862a67e5d0cec92f08b8fd44c503e0986be8",
    ),
}


def _load_table(path) -> Table:
    r"""Load a pre-built RelBench table, recovering pkey/fkey/time_col from the parquet
    metadata (the verbatim equivalent of the legacy ``Table.load`` / ``Database.load``,
    both since removed)."""
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
        raise ValueError(f"Unknown TGB dataset '{dataset}'. Known: {sorted(DATASETS)}")
    val_s, test_s, sha = DATASETS[dataset]
    val_timestamp = pd.to_datetime(int(val_s), unit="s", utc=True)
    test_timestamp = pd.to_datetime(int(test_s), unit="s", utc=True)
    raw = fetch(f"{BASE_URL}/{dataset}/db.zip", sha)
    write_hf(
        out or dataset,
        dataset,
        str(val_timestamp),
        str(test_timestamp),
        build(raw),
    )


def main_all(out_root="tgb") -> None:
    from pathlib import Path

    for dataset in DATASETS:
        print(f"=== {dataset} ===", flush=True)
        main(dataset, Path(out_root) / dataset)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        main_all(sys.argv[2] if len(sys.argv) > 2 else "tgb")
    elif len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(
            "usage: python tgb.py DATASET [OUT_DIR] | --all [OUT_ROOT]\n"
            f"datasets: {', '.join(DATASETS)}"
        )
