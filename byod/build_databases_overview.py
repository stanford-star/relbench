r"""Build a preliminary ``databases.parquet`` overview table for a RelBench dataset repo.

This produces the table behind the **databases** split of the Hugging Face dataset
viewer: one row per database, with the structural statistics reported in the RelBench
papers (`v1 <https://arxiv.org/abs/2407.20060>`_, `v2 <https://arxiv.org/abs/2602.12606>`_).

    # write databases.parquet locally (under --out, default '.') and print it
    python byod/build_databases_overview.py stanford-star/relbench-v1 --out /tmp/core

    # preserve hand-curated columns from the repo's existing table, then upload
    python byod/build_databases_overview.py stanford-star/relbench-v1 --merge --push

``<spec>`` is a Hub repo hosting one or more datasets (``stanford-star/relbench-v1``), a single hosted
dataset (``stanford-star/relbench-v1/rel-f1``), or a *local* dataset folder. Only manifests and parquet
*footers* are read -- never the table data -- so this is cheap even for large repos.

Columns. The structural ones are computed from the data:

    name, num_tables, num_rows, num_cols, num_tasks,
    tasks_binary_classification, tasks_regression, tasks_multiclass_classification,
    tasks_multilabel_classification, tasks_recommendation,
    start_timestamp, val_timestamp, test_timestamp, size_gb

A per-type task-count column that is zero for every database is dropped (a task type the
repo doesn't use gets no column).

The descriptive ones can NOT be filled in by a generic script and are left blank for a
human to complete (``--merge`` preserves any values already present in the repo's table):

    domain, description, license, source_url

``num_rows`` / ``num_cols`` are totals across all tables (every row, every column). The
papers instead count rows only up to the test cutoff -- a more expensive, data-dependent
figure -- so these are a generous upper bound, fine as a starting point to edit by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from relbench import hf
from relbench.hf import resolve_repo
from relbench.manifest import DatasetManifest, TaskManifest

COMPUTED_COLS = [
    "name",
    "num_tables",
    "num_rows",
    "num_cols",
    "num_tasks",
    "tasks_binary_classification",
    "tasks_regression",
    "tasks_multiclass_classification",
    "tasks_multilabel_classification",
    "tasks_recommendation",
    "start_timestamp",
    "val_timestamp",
    "test_timestamp",
    "size_gb",
]
# Filled in by hand -- a generic script can't infer them. Seeded from the manifest
# where possible (description), otherwise blank.
MANUAL_COLS = ["domain", "description", "license", "source_url"]
COLUMNS = [
    "name",
    "domain",
    "description",
    "num_tables",
    "num_rows",
    "num_cols",
    "num_tasks",
    "tasks_binary_classification",
    "tasks_regression",
    "tasks_multiclass_classification",
    "tasks_multilabel_classification",
    "tasks_recommendation",
    "start_timestamp",
    "val_timestamp",
    "test_timestamp",
    "size_gb",
    "license",
    "source_url",
]

_TASK_TYPE_COL = {
    "binary_classification": "tasks_binary_classification",
    "regression": "tasks_regression",
    "multiclass_classification": "tasks_multiclass_classification",
    "multilabel_classification": "tasks_multilabel_classification",
    "recommendation": "tasks_recommendation",
}


def _col_min(md: "pq.FileMetaData", colname: Optional[str]):
    r"""Smallest value of a column, read from parquet footer statistics (no data
    read)."""
    if not colname:
        return None
    names = md.schema.to_arrow_schema().names
    if colname not in names:
        return None
    idx = names.index(colname)
    mins = []
    for rg in range(md.num_row_groups):
        st = md.row_group(rg).column(idx).statistics
        if st is not None and st.has_min_max and st.min is not None:
            mins.append(st.min)
    return min(mins) if mins else None


def _fmt_ts(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(pd.Timestamp(value))
    except Exception:
        return str(value)


def database_row(
    name: str,
    dm: DatasetManifest,
    open_pq,
    folder_bytes,
    task_dir_names,
    load_task_manifest,
) -> dict:
    r"""One overview row.

    ``open_pq`` reads a ``db/<table>.parquet`` footer by relative path; the loaders
    enumerate and read the task manifests. ``folder_bytes`` is the on-disk size of the
    whole dataset folder (db + task labels + card/diagram) -- ``size_gb`` is the actual
    size a user downloads, which for datasets with large recommendation-task labels is
    well above the ``db/`` size alone.
    """
    num_rows = num_cols = 0
    start = None
    for tname, spec in dm.tables.items():
        rel = f"db/{tname}.parquet"
        md = open_pq(rel)
        if md is None:
            continue
        num_rows += md.num_rows
        num_cols += md.num_columns
        tmin = _col_min(md, spec.time_col)
        if tmin is not None:
            start = tmin if start is None else min(start, tmin)

    counts = {c: 0 for c in _TASK_TYPE_COL.values()}
    n_tasks = 0
    for task in task_dir_names():
        tm = load_task_manifest(task)
        if tm is None:
            continue
        n_tasks += 1
        col = _TASK_TYPE_COL.get(tm.task_type)
        if col:
            counts[col] += 1

    return {
        "name": name,
        "domain": None,
        "description": (dm.description or "").strip() or None,
        "num_tables": len(dm.tables),
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_tasks": n_tasks,
        **counts,
        "start_timestamp": _fmt_ts(start),
        "val_timestamp": _fmt_ts(dm.val_timestamp),
        "test_timestamp": _fmt_ts(dm.test_timestamp),
        "size_gb": round(folder_bytes / 1e9, 4) if folder_bytes else None,
        "license": None,
        "source_url": None,
    }


# --------------------------------------------------------------------------- #
# Local-folder and Hub backends (read manifests + parquet footers only)
# --------------------------------------------------------------------------- #


def _rows_from_local(root: Path) -> list[dict]:
    def one(name: str, dsdir: Path) -> dict:
        dm = DatasetManifest.load(dsdir / "manifest.yaml")

        def open_pq(rel):
            p = dsdir / rel
            return pq.read_metadata(p) if p.exists() else None

        folder_bytes = sum(f.stat().st_size for f in dsdir.rglob("*") if f.is_file())

        tasks_dir = dsdir / "tasks"

        def task_names():
            local = (
                {p.name for p in tasks_dir.iterdir() if (p / "manifest.yaml").exists()}
                if tasks_dir.exists()
                else set()
            )
            # Tasks for this database can be hosted in a different RelBench repo (the
            # v2-only tasks on the v1 databases), and they count towards it all the same.
            # Only for the RelBench repos: a third-party repo owns its own task list.
            if repo_id not in hf.RELBENCH_REPOS:
                return sorted(local)
            return sorted(local | set(hf.list_task_names(name)))

        def load_tm(task):
            local = tasks_dir / task / "manifest.yaml"
            if local.exists():
                return TaskManifest.load(local)
            path = hf.download_task_manifest(name, task)
            if path is None:
                raise FileNotFoundError(f"no manifest for task {name}/{task}")
            return TaskManifest.load(path)

        return database_row(name, dm, open_pq, folder_bytes, task_names, load_tm)

    if (root / "manifest.yaml").exists():
        return [one(DatasetManifest.load(root / "manifest.yaml").name, root)]
    out = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "manifest.yaml").exists():
            out.append(one(child.name, child))
    return out


def _rows_from_hub(repo_id: str, subdir: str) -> list[dict]:
    from huggingface_hub import HfApi, HfFileSystem, snapshot_download

    prefix = f"{subdir}/" if subdir else ""
    # Whole-folder sizes from a single metadata call (db + task labels + card/diagram).
    siblings = (
        HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True).siblings
    )

    def folder_bytes_for(repo_folder: str) -> int:
        return sum(
            (s.size or 0) for s in siblings if s.rfilename.startswith(repo_folder)
        )

    # Tiny: only the manifests (no db/, no task labels).
    manifest_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[
                f"{prefix}manifest.yaml",
                f"{prefix}tasks/*/manifest.yaml",
                f"{prefix}*/manifest.yaml",
                f"{prefix}*/tasks/*/manifest.yaml",
            ],
        )
    )
    base = manifest_root / subdir if subdir else manifest_root
    fs = HfFileSystem()
    hub_base = f"datasets/{repo_id}/{prefix}"

    def one(name: str, mdir: Path, hub_prefix: str) -> dict:
        dm = DatasetManifest.load(mdir / "manifest.yaml")

        def open_pq(rel):
            path = f"{hub_base}{hub_prefix}{rel}"
            try:
                with fs.open(path) as f:  # footer-only range reads
                    return pq.read_metadata(f)
            except FileNotFoundError:
                return None

        folder_bytes = folder_bytes_for(f"{prefix}{hub_prefix}")

        tasks_dir = mdir / "tasks"

        def task_names():
            local = (
                {p.name for p in tasks_dir.iterdir() if (p / "manifest.yaml").exists()}
                if tasks_dir.exists()
                else set()
            )
            # Tasks for this database can be hosted in a different RelBench repo (the
            # v2-only tasks on the v1 databases), and they count towards it all the same.
            # Only for the RelBench repos: a third-party repo owns its own task list.
            if repo_id not in hf.RELBENCH_REPOS:
                return sorted(local)
            return sorted(local | set(hf.list_task_names(name)))

        def load_tm(task):
            local = tasks_dir / task / "manifest.yaml"
            if local.exists():
                return TaskManifest.load(local)
            path = hf.download_task_manifest(name, task)
            if path is None:
                raise FileNotFoundError(f"no manifest for task {name}/{task}")
            return TaskManifest.load(path)

        return database_row(name, dm, open_pq, folder_bytes, task_names, load_tm)

    if (base / "manifest.yaml").exists():
        return [one(DatasetManifest.load(base / "manifest.yaml").name, base, "")]
    out = []
    for child in sorted(p for p in base.iterdir() if p.is_dir()):
        if (child / "manifest.yaml").exists():
            out.append(one(child.name, child, f"{child.name}/"))
    return out


# Explicit dtypes so a whole-column-null field (e.g. unfilled ``license``) still gets a
# concrete Arrow type rather than the ``null`` type the HF dataset viewer can't read.
_INT_COLS = {
    "num_tables",
    "num_rows",
    "num_cols",
    "num_tasks",
    "tasks_binary_classification",
    "tasks_regression",
    "tasks_multiclass_classification",
    "tasks_multilabel_classification",
    "tasks_recommendation",
}
_FLOAT_COLS = {"size_gb"}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS:
        if col in _INT_COLS:
            df[col] = df[col].astype("Int64")
        elif col in _FLOAT_COLS:
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype("string")
    return df


def build(spec: str) -> pd.DataFrame:
    if Path(spec).exists():
        rows = _rows_from_local(Path(spec))
    else:
        repo_id, subdir = resolve_repo(spec)
        rows = _rows_from_hub(repo_id, subdir)
    if not rows:
        raise FileNotFoundError(f"no dataset manifest found under {spec!r}")
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values("name").reset_index(drop=True)


def merge_existing(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    r"""Carry over hand-curated columns from a repo's existing databases.parquet (or the
    legacy overview.parquet) so re-running doesn't wipe manual edits."""
    if Path(spec).exists():
        return df
    repo_id, _ = resolve_repo(spec)
    from huggingface_hub import hf_hub_download

    prev = None
    for fname in ("STATS/databases.parquet", "databases.parquet", "overview.parquet"):
        try:
            prev = pd.read_parquet(hf_hub_download(repo_id, fname, repo_type="dataset"))
            print(f"merging curated columns from existing {fname}", flush=True)
            break
        except Exception:
            continue
    if prev is None:
        return df
    prev = prev.set_index("name")
    for col in MANUAL_COLS:
        if col in prev.columns:
            df[col] = df.apply(
                lambda r: (
                    prev[col].get(r["name"])
                    if r["name"] in prev.index and pd.notna(prev[col].get(r["name"]))
                    else r[col]
                ),
                axis=1,
            )
    return df


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    spec = args[0]
    do_push = "--push" in args
    do_merge = "--merge" in args
    out = Path(args[args.index("--out") + 1]) if "--out" in args else Path(".")

    print(f"building databases overview for {spec!r}", flush=True)
    df = build(spec)
    if do_merge:
        df = merge_existing(df, spec)
    df = _finalize(df)
    # Drop per-type task-count columns that are zero for every database -- a task type the
    # repo doesn't use shouldn't get a column (e.g. no multiclass tasks -> no
    # tasks_multiclass_classification). Structural and (blank) manual columns are kept.
    for col in _TASK_TYPE_COL.values():
        if col in df.columns and (df[col].fillna(0) == 0).all():
            df = df.drop(columns=[col])
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "databases.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path} ({len(df)} databases)", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[[c for c in COMPUTED_COLS if c in df.columns]].to_string(index=False))
    blank = [c for c in MANUAL_COLS if df[c].isna().all()]
    if blank:
        print(f"\nFill in by hand (left blank): {', '.join(blank)}", flush=True)

    if do_push:
        if Path(spec).exists():
            sys.exit(
                "--push needs a Hub repo spec (e.g. 'stanford-star/relbench-v1'), not a local path"
            )
        repo_id, _ = resolve_repo(spec)
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="STATS/databases.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add STATS/databases.parquet (databases subset of the viewer)",
        )
        print(f"pushed STATS/databases.parquet to {repo_id}", flush=True)


if __name__ == "__main__":
    main()
