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
``$RELBENCH_RAW_CACHE`` (default ``~/.cache/relbench-raw``). A source that cannot be
downloaded anonymously (Kaggle) is read from ``$RELBENCH_RAW_CACHE/<filename>`` instead,
where ``<filename>`` is the name the generator passes to ``fetch()``.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

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


def _url_name(url: str) -> str:
    return url.split("?")[0].rstrip("/").split("/")[-1]


def _is_archive(path: Path, suffix: str) -> bool:
    if suffix == ".zip":
        return zipfile.is_zipfile(path)
    if suffix == ".tar":
        return tarfile.is_tarfile(path)
    return True


def _download(urls: list[str], filename: str | None) -> Path:
    errors = []
    for url in urls:
        # Key the cache by the full URL so different sources sharing a filename
        # (e.g. several "db.zip") don't collide.
        key = hashlib.sha1(url.encode()).hexdigest()[:12]
        blob = CACHE / f"{key}-{filename or _url_name(url)}"
        if blob.exists():
            if _is_archive(blob, blob.suffix):
                return blob
            blob.unlink()
        part = blob.with_name(blob.name + ".part")
        print(f"downloading {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, part)
        except (OSError, http.client.HTTPException) as e:
            part.unlink(missing_ok=True)
            errors.append(f"  {url}\n    {e}")
            continue
        if not _is_archive(part, blob.suffix):
            part.unlink()
            errors.append(f"  {url}\n    not a {blob.suffix} archive (a login page?)")
            continue
        part.replace(blob)
        return blob
    lines = [f"could not download {filename or _url_name(urls[0])}:", *errors]
    if filename:
        lines.append(f"place the file at {CACHE / filename}")
    raise RuntimeError("\n".join(lines))


def fetch(
    url: str | list[str], sha256: str | None = None, filename: str | None = None
) -> Path:
    r"""Download ``url`` (one URL, or candidates tried in order; cached), verify
    ``sha256``, unpack if a ``.zip`` / ``.tar``; return the path.

    A file placed by hand at
    ``CACHE/<filename>`` is used instead of downloading.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    placed = CACHE / filename if filename else None
    if placed is not None and placed.exists():
        if not _is_archive(placed, placed.suffix):
            raise RuntimeError(f"{placed} is not a {placed.suffix} archive")
        blob = placed
    else:
        blob = _download([url] if isinstance(url, str) else list(url), filename)
    if sha256:
        got = hashlib.sha256(blob.read_bytes()).hexdigest()
        if got != sha256:
            raise ValueError(
                f"sha256 mismatch for {blob}:\n  got {got}\n  want {sha256}"
            )
    if blob.suffix in (".zip", ".tar"):
        out = blob.with_name(blob.name + ".extracted")
        if not out.exists():
            tmp = blob.with_name(blob.name + ".extracting")
            if blob.suffix == ".zip":
                with zipfile.ZipFile(blob) as z:
                    z.extractall(tmp)
            else:
                with tarfile.open(blob) as t:
                    t.extractall(tmp)
            tmp.rename(out)  # atomic: a half-extracted dir never looks cached
        return out
    return blob


def write_hf(
    out,
    name: str,
    val_timestamp: str,
    test_timestamp: str,
    tables: dict,
    description: str | None = None,
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
        description=description,
        val_timestamp=str(val_timestamp),
        test_timestamp=str(test_timestamp),
        tables=specs,
    ).save(out / "manifest.yaml")
    print(f"wrote {out}/manifest.yaml + {len(specs)} tables", flush=True)
