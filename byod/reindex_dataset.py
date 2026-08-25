r"""Normalize a dataset folder's keys the way ``relbench.load_dataset`` expects them.

Primary keys become ``0..n-1`` in row order (rows of tables with a time column sorted by
time), foreign keys are rewritten to those indices (dangling references become null), and
the key columns of ``tasks/*/{train,val,test}.parquet`` are remapped the same way, with a
stable sort so that re-running on an already normalized folder is a no-op. This is the
one key-reindexing implementation; the provenance generators use it through
``provenance/_lib.write_hf``.

    python byod/reindex_dataset.py <dataset_dir>              # in place
    python byod/reindex_dataset.py <dataset_dir> --out <dir>  # normalized copy
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from relbench.base import Database, Table
from relbench.manifest import DatasetManifest, TaskManifest, validate_dataset_manifest

SPLITS = ["train", "val", "test"]
INDEX = "__index__"


def load_db(dataset_dir: Path, manifest: DatasetManifest) -> Database:
    return Database(
        {
            name: Table(
                pd.read_parquet(dataset_dir / "db" / f"{name}.parquet"),
                dict(spec.fkeys),
                spec.pkey,
                spec.time_col,
            )
            for name, spec in manifest.tables.items()
        }
    )


def remap(ser: pd.Series, index_map: pd.Series) -> pd.Series:
    out = pd.merge(ser, index_map, how="left", left_on=ser.name, right_index=True)
    return out[INDEX]


def reindex(db: Database) -> tuple:
    index_maps = {}
    for name, t in db.table_dict.items():
        if t.pkey_col is None:
            continue
        if t.time_col is not None:
            t.df = t.df.sort_values(t.time_col, kind="stable").reset_index(drop=True)
        keys = t.df[t.pkey_col]
        if keys.isna().any() or keys.duplicated().any():
            raise RuntimeError(
                f"table '{name}': primary key '{t.pkey_col}' has nulls or duplicates"
            )
        idx = pd.RangeIndex(len(keys)).astype("Int64")
        index_maps[name] = pd.Series(idx, index=pd.Index(keys), name=INDEX)
        t.df[t.pkey_col] = idx
    dangling = {}
    for name, t in db.table_dict.items():
        dangling[name] = 0
        for fkey, pkey_table in t.fkey_col_to_pkey_table.items():
            before = int(t.df[fkey].notna().sum())
            t.df[fkey] = remap(t.df[fkey], index_maps[pkey_table])
            dangling[name] += before - int(t.df[fkey].notna().sum())
    return index_maps, dangling


def is_identity(index_map: pd.Series) -> bool:
    keys = index_map.index
    return bool(
        pd.api.types.is_integer_dtype(keys)
        and (keys.to_numpy() == np.arange(len(keys))).all()
    )


def _is_list(v) -> bool:
    return isinstance(v, (list, np.ndarray))


def remap_labels(df: pd.DataFrame, maps: dict) -> tuple:
    dropped = 0
    for col, index_map in maps.items():
        if col not in df.columns:
            continue
        if df[col].map(_is_list).any():
            exploded = df[col].explode().dropna()
            mapped = exploded.map(index_map).dropna().astype("int64")
            dropped += len(exploded) - len(mapped)
            lists = mapped.groupby(level=0).agg(list)
            df[col] = pd.Series([lists.get(i, []) for i in df.index], index=df.index)
        else:
            df[col] = remap(df[col], index_map)
            keep = df[col].notna()
            dropped += int((~keep).sum())
            df = df[keep].reset_index(drop=True)
    return df, dropped


def label_rewrites(src: Path, index_maps: dict) -> dict:
    out = {}
    changed = {t: m for t, m in index_maps.items() if not is_identity(m)}
    for mpath in sorted(src.glob("tasks/*/manifest.yaml")):
        tm = TaskManifest.load(mpath)
        pairs = [
            (tm.entity_col, tm.entity_table),
            (tm.src_entity_col, tm.src_entity_table),
            (tm.dst_entity_col, tm.dst_entity_table),
        ]
        maps = {col: changed[table] for col, table in pairs if col and table in changed}
        if not maps:
            continue
        for split in SPLITS:
            path = mpath.parent / f"{split}.parquet"
            if path.exists():
                out[path.relative_to(src)] = maps
    return out


def write_normalized(src: Path, dest: Path, db: Database, rewrites: dict) -> None:
    (dest / "db").mkdir(parents=True, exist_ok=True)
    for name, t in db.table_dict.items():
        t.df.to_parquet(dest / "db" / f"{name}.parquet", index=False)
    for rel, maps in rewrites.items():
        df = pd.read_parquet(src / rel)
        n = len(df)
        df, dropped = remap_labels(df, maps)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest / rel, index=False)
        print(
            f"{str(rel):<40} rows={n:<10} remapped {sorted(maps)}"
            f" dangling keys dropped={dropped}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: in place")
    args = parser.parse_args(argv)

    src = args.dataset_dir.resolve()
    manifest = DatasetManifest.load(src / "manifest.yaml")
    validate_dataset_manifest(manifest, src / "db")
    db = load_db(src, manifest)
    index_maps, dangling = reindex(db)
    for name, t in db.table_dict.items():
        n = len(t.df)
        pkey = f"{t.pkey_col}=0..{n - 1}" if t.pkey_col else "no pkey"
        print(
            f"{name:<28} rows={n:<10} {pkey:<36} dangling fkeys nulled={dangling[name]}",
            flush=True,
        )
    rewrites = label_rewrites(src, index_maps)

    if args.out is None or args.out.resolve() == src:
        with tempfile.TemporaryDirectory(dir=src) as tmp:
            write_normalized(src, Path(tmp), db, rewrites)
            for path in Path(tmp).rglob("*.parquet"):
                os.replace(path, src / path.relative_to(tmp))
        print(f"normalized {src} in place")
        return

    out = args.out.resolve()
    skip = {f"db/{name}.parquet" for name in manifest.tables} | {
        str(rel) for rel in rewrites
    }

    def ignore(d, names):
        rel = Path(os.path.relpath(d, src))
        return [n for n in names if str(rel / n) in skip]

    shutil.copytree(src, out, ignore=ignore, dirs_exist_ok=True)
    write_normalized(src, out, db, rewrites)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
