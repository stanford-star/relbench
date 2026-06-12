r"""Minimal helpers shared by the source -> RelBench database generators in this
directory.

Dependencies are kept deliberately small: the Python stdlib (``urllib``, ``zipfile``,
``hashlib``), ``pandas``, and ``relbench`` itself (``relbench.base`` for the canonical
primary/foreign-key reindexing the published data was built with, and ``relbench.manifest``
for the ``manifest.yaml`` writer). No ``pooch``, no legacy ``Dataset`` machinery.

Each generator: ``fetch()`` the raw source, build a few pandas DataFrames, wrap them in
``relbench`` ``Table`` objects (declaring ``pkey`` / ``fkeys`` / ``time_col``), and call
``write_hf()`` -- which reindexes keys to ``0..n-1`` (exactly as the published databases
were built) and writes the Hugging Face layout::

    <out>/manifest.yaml
    <out>/db/<table>.parquet

Run a generator with ``python <name>.py [OUT_DIR]``; raw downloads are cached under
``$RELBENCH_RAW_CACHE`` (default ``~/.cache/relbench-raw``).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from relbench.base import (  # noqa: F401  (Table re-exported for generators)
    Database,
    Table,
)
from relbench.manifest import DatasetManifest, TableSpec

CACHE = Path(
    os.environ.get("RELBENCH_RAW_CACHE", Path.home() / ".cache" / "relbench-raw")
)


def clean_datetime(df, col: str):
    r"""Coerce ``df[col]`` to datetime and drop rows that fail to parse (NaT)."""
    import pandas as pd

    df[col] = pd.to_datetime(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[col])
    print(f"  {col}: dropped {before - len(df)} rows with invalid dates", flush=True)
    return df


def fetch(url: str, sha256: Optional[str] = None) -> Path:
    r"""Download ``url`` (cached), verify ``sha256``, unzip if a ``.zip``; return the
    path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    # Key the cache by the full URL so different sources sharing a filename
    # (e.g. several "db.zip") don't collide.
    key = hashlib.sha1(url.encode()).hexdigest()[:12]
    blob = CACHE / f"{key}-{name}"
    if not blob.exists():
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, blob)
    if sha256:
        got = hashlib.sha256(blob.read_bytes()).hexdigest()
        if got != sha256:
            raise ValueError(
                f"sha256 mismatch for {url}:\n  got {got}\n  want {sha256}"
            )
    if blob.suffix == ".zip":
        out = CACHE / f"{key}-{name}.extracted"
        if not out.exists():
            with zipfile.ZipFile(blob) as z:
                z.extractall(out)
        return out
    return blob


def write_hf(
    out, name: str, val_timestamp: str, test_timestamp: str, tables: dict
) -> None:
    r"""Reindex keys and write ``tables`` ({name: ``relbench`` ``Table``}) as the HF
    layout."""
    db = Database(tables)
    db.reindex_pkeys_and_fkeys()  # pkeys -> 0..n-1 (by time order); remap fkeys
    out = Path(out)
    (out / "db").mkdir(parents=True, exist_ok=True)
    specs = {}
    for tname, t in db.table_dict.items():
        t.df.to_parquet(out / "db" / f"{tname}.parquet", index=False)
        specs[tname] = TableSpec(
            pkey=t.pkey_col,
            time_col=t.time_col,
            fkeys=dict(t.fkey_col_to_pkey_table),
        )
    DatasetManifest(
        name=name,
        val_timestamp=str(val_timestamp),
        test_timestamp=str(test_timestamp),
        tables=specs,
    ).save(out / "manifest.yaml")
    print(f"wrote {out}/manifest.yaml + {len(specs)} tables", flush=True)
