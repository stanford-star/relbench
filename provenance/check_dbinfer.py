r"""Verify a ported ``dbinfer-*`` dataset: referential integrity, task joinability, and
agreement with the original 4DBInfer archive.

    python provenance/check_dbinfer.py DBINFER_DIR [RAW_ROOT]

``DBINFER_DIR`` is either one ``dbinfer-<name>`` folder or a root containing several.
``RAW_ROOT`` (default ``$DBINFER_RAW_ROOT``) is a directory of extracted 4DBInfer archives
(``<root>/<name>/metadata.yaml``); when given, row counts and split sizes are compared
against it.

Checks, per dataset:

  db      every declared primary key is 0..n-1 and unique; every non-null foreign key
          resolves to a row of its parent; the foreign-key null count equals the archive's
          (more nulls than the archive means values that dangle in the source).
  time    every declared time column is a real datetime with a plausible range.
  task    every entity / link endpoint column resolves into its declared table; the target
          column exists in every split; label times fall inside the split's window.
  splits  ``val_timestamp <= test_timestamp``; no val/test label strictly before its cutoff.
  source  table row counts and task split sizes match the archive.

Findings come at two levels. **FAIL** is a defect in the port -- something this generator
got wrong. **WARN** is a defect in the 4DBInfer archive that the port carries over
faithfully, most notably ``outbrain-small``, whose tables were subsampled independently so
~99.9% of its foreign keys point at rows that were not kept. Exit status is non-zero only
for FAILs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pds
import yaml

from relbench.manifest import DatasetManifest, TaskManifest

BATCH = 1 << 21
FAILURES: list = []
WARNINGS: list = []
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"    FAIL {msg}", flush=True)


def warn(msg: str) -> None:
    r"""A defect in the *source* archive, faithfully carried over.

    Reported loudly but not a port failure -- there is nothing in the archive to fix it
    with.
    """
    WARNINGS.append(msg)
    print(f"    WARN {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"    ok   {msg}", flush=True)


def _ds(path):
    return pds.dataset(path, format="parquet")


def _col(path, name) -> pd.Series:
    return _ds(path).to_table(columns=[name]).to_pandas()[name]


def check_dataset(root: Path, raw_root: Path | None) -> None:
    man = DatasetManifest.load(root / "manifest.yaml")
    print(f"\n=== {man.name}", flush=True)
    db = root / "db"

    raw_meta = None
    if raw_root is not None:
        short = man.name.removeprefix("dbinfer-")
        p = raw_root / short / "metadata.yaml"
        if p.exists():
            raw_meta = yaml.safe_load(p.read_text())

    # --- primary keys -------------------------------------------------------------------
    nrows, pk_of = {}, {}
    for tname, spec in man.tables.items():
        path = db / f"{tname}.parquet"
        if not path.exists():
            fail(f"{tname}: missing parquet")
            continue
        d = _ds(path)
        nrows[tname] = d.count_rows()
        pk_of[tname] = spec.pkey
        if spec.pkey is None:
            continue
        s = _col(path, spec.pkey)
        if s.isna().any():
            fail(f"{tname}.{spec.pkey}: primary key has nulls")
        elif not np.array_equal(
            np.sort(s.to_numpy(dtype=np.int64)), np.arange(len(s), dtype=np.int64)
        ):
            fail(f"{tname}.{spec.pkey}: primary key is not a permutation of 0..n-1")
    ok(f"{len(nrows)} tables, primary keys dense")

    # --- foreign keys -------------------------------------------------------------------
    # A published foreign key may only be null where the *archive* was null. Anything more
    # is either a dangling reference in the source (reported, with its count) or the
    # upstream adapter bug (which nulled 99.9-100% of every key).
    raw_tables = {t["name"]: t for t in (raw_meta or {}).get("tables", [])}
    short = man.name.removeprefix("dbinfer-")

    def raw_nulls(tname: str, col: str):
        t = raw_tables.get(tname)
        if t is None:
            return None
        src = raw_root / short / t["source"]
        d = _ds(src)
        if col not in d.schema.names:
            return None
        n = 0
        for b in d.to_batches(columns=[col], batch_size=BATCH):
            n += b.column(0).null_count
        return n

    for tname, spec in man.tables.items():
        path = db / f"{tname}.parquet"
        if not path.exists():
            continue
        for col, parent in spec.fkeys.items():
            if parent not in nrows:
                fail(f"{tname}.{col}: parent table '{parent}' absent")
                continue
            n, nulls, bad, hi = 0, 0, 0, -1
            for b in _ds(path).to_batches(columns=[col], batch_size=BATCH):
                a = b.column(0)
                n += len(a)
                nulls += a.null_count
                v = a.drop_null()
                if len(v) == 0:
                    continue
                mx, mn = pc.max(v).as_py(), pc.min(v).as_py()
                hi = max(hi, mx)
                if mn < 0 or mx >= nrows[parent]:
                    arr = v.to_numpy(zero_copy_only=False)
                    bad += int(((arr < 0) | (arr >= nrows[parent])).sum())
            label = f"{tname}.{col} -> {parent}"
            rate = nulls / n if n else 0.0
            want = raw_nulls(tname, col) if raw_meta else None
            if bad:
                fail(f"{label}: {bad:,} values outside 0..{nrows[parent] - 1}")
            elif want is None:
                if rate > 0.5:
                    fail(
                        f"{label}: {rate:.2%} null ({nulls:,}/{n:,}); archive not checked"
                    )
                else:
                    ok(f"{label}: {n - nulls:,} resolve, {rate:.2%} null, max={hi:,}")
            elif nulls < want:
                fail(f"{label}: {nulls:,} nulls < {want:,} in the archive")
            elif nulls == want:
                ok(
                    f"{label}: {n - nulls:,} resolve, nulls match the archive ({want:,})"
                )
            else:
                extra = nulls - want
                frac = extra / max(n - want, 1)
                msg = (
                    f"{label}: {n - nulls:,} resolve; {extra:,} of {n - want:,} non-null "
                    f"source values ({frac:.2%}) dangle in the archive"
                )
                warn(msg) if frac > 0.01 else ok(msg)

    # --- time columns -------------------------------------------------------------------
    for tname, spec in man.tables.items():
        if not spec.time_col:
            continue
        path = db / f"{tname}.parquet"
        if not path.exists():
            continue
        t = _ds(path).schema.field(spec.time_col).type
        if not (pa_is_ts := str(t).startswith("timestamp")):
            fail(f"{tname}.{spec.time_col}: dtype {t} is not a timestamp")
            continue
        del pa_is_ts
        mn = mx = None
        for b in _ds(path).to_batches(columns=[spec.time_col], batch_size=BATCH):
            v = b.column(0).drop_null()
            if not len(v):
                continue
            lo, hi = pc.min(v).as_py(), pc.max(v).as_py()
            mn = lo if mn is None else min(mn, lo)
            mx = hi if mx is None else max(mx, hi)
        if mn is None:
            fail(f"{tname}.{spec.time_col}: all null")
        elif pd.Timestamp(mn) < pd.Timestamp("1990-01-01"):
            fail(f"{tname}.{spec.time_col}: implausible range {mn} .. {mx}")
        else:
            ok(f"{tname}.{spec.time_col}: {mn} .. {mx}")

    # --- source agreement (tables) ------------------------------------------------------
    if raw_meta:
        for t in raw_meta["tables"]:
            src = raw_root / short / t["source"]
            want = _ds(src).count_rows()
            got = nrows.get(t["name"])
            if got is None:
                fail(f"{t['name']}: in archive but not published")
            elif got != want:
                fail(f"{t['name']}: {got:,} rows published vs {want:,} in archive")
        ok("table row counts match the archive")

    # --- tasks --------------------------------------------------------------------------
    vts, tts = pd.Timestamp(man.val_timestamp), pd.Timestamp(man.test_timestamp)
    # `TaskBase.__init__` rejects a task whose `timedelta` exceeds test - val, and an
    # external task's `timedelta` defaults to one day.
    if tts - vts < pd.Timedelta(days=1):
        fail(
            f"test_timestamp {tts} is less than a day after val_timestamp {vts}; "
            "tasks with the default 1-day timedelta will refuse to load"
        )
    else:
        ok(f"val_timestamp {vts} + >=1d <= test_timestamp {tts}")

    tdir = root / "tasks"
    raw_tasks = {tk["name"]: tk for tk in (raw_meta or {}).get("tasks", [])}
    for tm_path in sorted(tdir.glob("*/manifest.yaml")) if tdir.exists() else []:
        tm = TaskManifest.load(tm_path)
        print(f"  -- task {tm.name} ({tm.task_type})", flush=True)
        endpoints = []
        if tm.task_type == "recommendation":
            endpoints = [
                (tm.src_entity_col, tm.src_entity_table),
                (tm.dst_entity_col, tm.dst_entity_table),
            ]
        else:
            endpoints = [(tm.entity_col, tm.entity_table)]
        sizes = {}
        for split in ("train", "val", "test"):
            path = tm_path.parent / f"{split}.parquet"
            if not path.exists():
                fail(f"{tm.name}/{split}: missing parquet")
                continue
            d = _ds(path)
            sizes[split] = d.count_rows()
            names = set(d.schema.names)
            for col, table in endpoints:
                if col not in names:
                    fail(f"{tm.name}/{split}: endpoint column '{col}' absent")
                    continue
                if table not in nrows:
                    fail(f"{tm.name}/{split}: endpoint table '{table}' absent")
                    continue
                s = _col(path, col)
                v = s.dropna()
                if len(v) == 0:
                    fail(f"{tm.name}/{split}.{col}: all null")
                    continue
                arr = v.to_numpy(dtype=np.int64)
                bad = int(((arr < 0) | (arr >= nrows[table])).sum())
                if bad:
                    fail(
                        f"{tm.name}/{split}.{col}: {bad:,} of {len(arr):,} outside "
                        f"{table}[0..{nrows[table] - 1}]"
                    )
                elif s.isna().mean() > 0.01:
                    warn(
                        f"{tm.name}/{split}.{col} -> {table}: {s.isna().mean():.2%} of "
                        f"{len(s):,} label rows name an entity absent from the archive"
                    )
                else:
                    ok(
                        f"{tm.name}/{split}.{col} -> {table}: all {len(arr):,} resolve "
                        f"({s.isna().mean():.2%} null)"
                    )
            if tm.target_col and tm.target_col not in names:
                # retrieval train splits legitimately carry positives only
                if not (tm.task_type == "recommendation" and split == "train"):
                    fail(f"{tm.name}/{split}: target '{tm.target_col}' absent")
            if tm.time_col and tm.time_col in names:
                ts = pd.to_datetime(_col(path, tm.time_col)).dropna()
                if len(ts):
                    lo, hi = ts.min(), ts.max()
                    ok(f"{tm.name}/{split}.{tm.time_col}: {lo} .. {hi}")
                    if split == "val" and lo < vts:
                        fail(
                            f"{tm.name}/val label at {lo} precedes val_timestamp {vts}"
                        )
                    if split == "test" and lo < tts:
                        fail(
                            f"{tm.name}/test label at {lo} precedes test_timestamp {tts}"
                        )
        # remove_columns must name real db columns
        for tbl, col in tm.remove_columns or []:
            p = db / f"{tbl}.parquet"
            if not p.exists() or col not in set(_ds(p).schema.names):
                fail(f"{tm.name}: remove_columns names missing {tbl}.{col}")
            else:
                ok(
                    f"{tm.name}: remove_columns {tbl}.{col} present in db (will be dropped)"
                )
        if tm.name in raw_tasks:
            tk = raw_tasks[tm.name]
            for raw_split, split in SPLITS.items():
                src = raw_root / short / tk["source"].replace("{split}", raw_split)
                if not src.exists():
                    continue
                want = _ds(src).count_rows()
                got = sizes.get(split)
                if got != want:
                    fail(f"{tm.name}/{split}: {got:,} rows vs {want:,} in archive")
            ok(f"{tm.name}: split sizes match the archive")


def main(argv: list) -> int:
    if not argv:
        sys.exit(__doc__)
    root = Path(argv[0])
    raw = argv[1] if len(argv) > 1 else os.environ.get("DBINFER_RAW_ROOT")
    raw_root = Path(raw) if raw else None
    roots = (
        [root]
        if (root / "manifest.yaml").exists()
        else sorted(p.parent for p in root.glob("*/manifest.yaml"))
    )
    if not roots:
        sys.exit(f"no dataset manifests under {root}")
    for r in roots:
        check_dataset(r, raw_root)
    print(
        f"\n{'FAILED: ' + str(len(FAILURES)) + ' check(s)' if FAILURES else 'all checks passed'}"
        f"{f' ({len(WARNINGS)} source-defect warning(s))' if WARNINGS else ''}",
        flush=True,
    )
    for f in FAILURES:
        print(f"  - FAIL {f}", flush=True)
    for w in WARNINGS:
        print(f"  - WARN {w}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
