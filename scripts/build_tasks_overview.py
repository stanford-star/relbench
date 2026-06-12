r"""Build the ``tasks.parquet`` overview table for a RelBench dataset repository.

This produces the table behind the **tasks** split of the Hugging Face dataset viewer:
one row per task, with the per-task statistics reported in the RelBench papers
(`v1 <https://arxiv.org/abs/2407.20060>`_, `v2 <https://arxiv.org/abs/2602.12606>`_).

    # write tasks.parquet locally (under --out, default '.') and print it
    python scripts/build_tasks_overview.py relbench/core --out /tmp/core

    # also upload it to STATS/tasks.parquet, next to STATS/databases.parquet
    python scripts/build_tasks_overview.py relbench/core --push

    # cross-check the computed numbers against the published paper tables
    python scripts/build_tasks_overview.py relbench/core --check

``<spec>`` is a Hub repo that hosts one or more datasets (``relbench/core``), a single
hosted dataset (``relbench/core/rel-f1``), or a *local* dataset folder (a directory with a
``manifest.yaml`` and a ``tasks/`` subfolder). Only the manifests and the task label
tables are downloaded -- never the (much larger) ``db/`` tables -- so this stays cheap
even for repos with thousands of datasets.

Columns (the headline statistics mirror the paper "task statistics" tables; the rest is
metadata pulled straight from each task manifest, handy for filtering the viewer):

    database, task, task_type, description, kind, num_rows_train, num_rows_val,
    num_rows_test, num_unique_entities, pct_train_test_entity_overlap, num_dst_entities,
    metric, entity_table, entity_col, src_entity_table, src_entity_col, dst_entity_table,
    dst_entity_col, target_col, time_col, timedelta, num_eval_timestamps, eval_k

Columns that never apply to the repo being built -- always null, e.g. the link-only
``src/dst_entity_*``, ``eval_k`` and ``num_dst_entities`` for a repo with no link tasks --
are dropped, so each repo's table carries only the columns that mean something for it.

Statistic definitions (computed from the data, never copied from the papers):

* num_rows_{train,val,test} -- rows of each split's label table.
* num_unique_entities       -- distinct entities over train+val+test. The "entity" is
                               the entity column for node/auto-complete tasks and the
                               *source* entity column for link/recommendation tasks.
* pct_train_test_entity_overlap -- 100 * |(train u val) entities ^ test entities|
                                   / |test entities|.
* num_dst_entities          -- link tasks only: total number of (src -> dst) links in
                               the train and val tables (the destination candidates seen
                               at fit time; the test links are the prediction target and
                               are excluded). ``None`` for non-link tasks.

``--check`` compares the freshly computed numbers against the values published in the
papers (embedded below purely as a sanity check -- they are *not* written to the
parquet). Released datasets that were revised after a paper will show expected
differences; these are reported, not silenced.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from relbench.hf import resolve_repo
from relbench.manifest import DatasetManifest, TaskManifest

# Column order: identity, the task spec (task_type, description, kind), the headline
# statistics (mirroring the paper task-stats tables), then the rest of the manifest
# metadata. Columns that never apply to a given repo (always null -- e.g. the link-only
# destination columns for a repo with no link tasks) are dropped per repo in ``build``.
COLUMNS = [
    "database",
    "task",
    "task_type",
    "description",
    "kind",
    "num_rows_train",
    "num_rows_val",
    "num_rows_test",
    "num_unique_entities",
    "pct_train_test_entity_overlap",
    "num_dst_entities",
    "metric",
    "entity_table",
    "entity_col",
    "src_entity_table",
    "src_entity_col",
    "dst_entity_table",
    "dst_entity_col",
    "target_col",
    "time_col",
    "timedelta",
    "num_eval_timestamps",
    "eval_k",
]
# Columns shown in the printed summary and compared by the paper sanity check.
HEADLINE_COLS = [
    "database",
    "task",
    "task_type",
    "num_rows_train",
    "num_rows_val",
    "num_rows_test",
    "num_unique_entities",
    "pct_train_test_entity_overlap",
    "num_dst_entities",
]
_METRIC = {
    "binary_classification": "roc_auc",
    "regression": "nmae",
    "link_prediction": "link_prediction_map",
}


def _metric(tm: TaskManifest) -> Optional[str]:
    if tm.evaluator:
        return tm.evaluator  # external custom evaluator (e.g. 'tgb')
    return _METRIC.get(tm.task_type)


def _entity_col(dm: DatasetManifest, tm: TaskManifest, task_dir: Path) -> Optional[str]:
    r"""The column whose distinct values are the task's (source) entities."""
    if tm.task_type == "link_prediction":
        return tm.src_entity_col
    if tm.entity_col:
        return tm.entity_col
    if tm.entity_table:
        # Auto-complete-style: the entity is the entity table's primary key; fall back to
        # inferring it from the label schema if the manifest omits the pkey.
        pkey = dm.tables[tm.entity_table].pkey if tm.entity_table in dm.tables else None
        if pkey:
            return pkey
        train = task_dir / "train.parquet"
        if train.exists():
            names = [f.name for f in pq.read_schema(train)]
            rest = [c for c in names if c not in (tm.time_col, tm.target_col)]
            return rest[0] if rest else None
    return None


def _distinct(path: Path, col: Optional[str]) -> Optional[set]:
    if col is None or not path.exists():
        return None
    table = pq.read_table(path, columns=[col])
    return set(pc.unique(table.column(col)).to_pylist())


def _num_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows if path.exists() else 0


def _num_links(path: Path, col: str) -> Optional[int]:
    r"""Total number of links (sum of destination-list lengths) in a link label
    table."""
    if not path.exists():
        return None
    table = pq.read_table(path, columns=[col])
    arr = table.column(col)
    if not (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)):
        return None  # not a list-valued destination column (e.g. external tgb labels)
    return int(pc.sum(pc.list_value_length(arr.combine_chunks())).as_py() or 0)


def task_row(dm: DatasetManifest, name: str, task: str, task_dir: Path) -> dict:
    tm = TaskManifest.load(task_dir / "manifest.yaml")
    ent_col = _entity_col(dm, tm, task_dir)
    splits = {s: task_dir / f"{s}.parquet" for s in ("train", "val", "test")}

    rows = {s: _num_rows(p) for s, p in splits.items()}
    ents = {s: _distinct(p, ent_col) for s, p in splits.items()}

    # Unique entities over whatever splits exist (some repos ship a train table only).
    uniq = overlap = None
    present = [e for e in ents.values() if e is not None]
    if present:
        uniq = len(set().union(*present))
    # Train/test overlap needs a test split (and at least one of train/val to compare to).
    test = ents["test"]
    if test and (ents["train"] is not None or ents["val"] is not None):
        seen = (ents["train"] or set()) | (ents["val"] or set())
        overlap = round(100 * len(seen & test) / len(test), 2)

    num_dst = None
    if tm.task_type == "link_prediction" and tm.dst_entity_col:
        # Destination candidates seen at fit time = links in train + val (not test).
        parts = [_num_links(splits[s], tm.dst_entity_col) for s in ("train", "val")]
        if any(p is not None for p in parts):
            num_dst = sum(p or 0 for p in parts)

    return {
        "database": name,
        "task": task,
        "task_type": tm.task_type,
        "description": (tm.description or "").strip() or None,
        "kind": tm.kind,
        "num_rows_train": rows["train"],
        "num_rows_val": rows["val"],
        "num_rows_test": rows["test"],
        "num_unique_entities": uniq,
        "pct_train_test_entity_overlap": overlap,
        "num_dst_entities": num_dst,
        "metric": _metric(tm),
        "entity_table": tm.entity_table,
        "entity_col": tm.entity_col,
        "src_entity_table": tm.src_entity_table,
        "src_entity_col": tm.src_entity_col,
        "dst_entity_table": tm.dst_entity_table,
        "dst_entity_col": tm.dst_entity_col,
        "target_col": tm.target_col,
        "time_col": tm.time_col,
        "timedelta": tm.timedelta,
        "num_eval_timestamps": tm.num_eval_timestamps,
        "eval_k": tm.eval_k,
    }


# --------------------------------------------------------------------------- #
# Dataset discovery (local folder, single hosted dataset, or multi-dataset repo)
# --------------------------------------------------------------------------- #


def _snapshot(repo_id: str, patterns: list[str], force: bool = False) -> str:
    r"""``snapshot_download`` retried with backoff on the Hub rate limiter (HTTP 429).

    Already-cached files are skipped, so retries resume rather than restart. ``force``
    re-fetches even cached files (used to recover from a corrupt/partial cache entry).
    """
    import time

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    for attempt in range(8):
        try:
            return snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=patterns,
                max_workers=4,
                force_download=force,
            )
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) == 429 and attempt < 7:
                wait = min(60, 2**attempt)
                print(f"  rate-limited (429); retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise


def _download_one(repo_id: str, name: str) -> Path:
    r"""Download a single dataset's manifests + task labels (never the big ``db/``
    tables).

    Scoped to one dataset's subtree, so the Hub file listing stays small and reliable --
    a repo-wide snapshot of a thousands-of-files repo can come back silently truncated.
    Verifies the manifest actually landed, re-fetching (forced) once if a flaky download
    left it missing; raises if it still can't be obtained.
    """
    prefix = f"{name}/" if name else ""
    patterns = [
        f"{prefix}manifest.yaml",
        f"{prefix}tasks/*/manifest.yaml",
        f"{prefix}tasks/*/*.parquet",
    ]
    for attempt in range(3):
        local = _snapshot(repo_id, patterns, force=attempt > 0)
        dsdir = Path(local) / name if name else Path(local)
        if (dsdir / "manifest.yaml").exists():
            return dsdir
    raise FileNotFoundError(f"{prefix}manifest.yaml missing after download")


def _hub_dataset_names(repo_id: str, subdir: str) -> list[str]:
    r"""Names of the datasets a Hub repo addresses (the single ``subdir``, or every top-
    level ``<name>/manifest.yaml``).

    ``list_repo_files`` paginates fully, so this is
    reliable even for repos with tens of thousands of files.
    """
    from huggingface_hub import HfApi

    if subdir:
        return [subdir]
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    return sorted(
        {
            f.split("/")[0]
            for f in files
            if f.endswith("/manifest.yaml") and f.count("/") == 1
        }
    )


def iter_datasets(spec: str):
    r"""Yield ``(dataset_name, local_dir)`` for every dataset addressed by ``spec``.

    For a multi-dataset Hub repo each dataset is fetched on demand (scoped to its own
    subtree), so a repo with thousands of datasets streams through reliably instead of
    relying on one giant -- and easily truncated -- repo-wide snapshot.
    """
    local = Path(spec)
    if local.exists():
        if (local / "manifest.yaml").exists():
            yield DatasetManifest.load(local / "manifest.yaml").name, local
        else:
            children = [
                p
                for p in sorted(local.iterdir())
                if p.is_dir() and (p / "manifest.yaml").exists()
            ]
            if not children:
                raise FileNotFoundError(f"no dataset manifest found under {spec!r}")
            for child in children:
                yield child.name, child
        return

    repo_id, subdir = resolve_repo(spec)
    names = _hub_dataset_names(repo_id, subdir)
    if not names:
        raise FileNotFoundError(f"no dataset manifest found under {spec!r}")
    for i, name in enumerate(names, 1):
        if len(names) > 1:
            print(f"[{i}/{len(names)}] {name}", flush=True)
        try:
            dsdir = _download_one(repo_id, name)
            canonical = DatasetManifest.load(dsdir / "manifest.yaml").name
        except Exception as e:  # one flaky dataset must not abort a 1000-dataset run
            print(f"  WARNING: skipping {name}: {type(e).__name__}: {e}", flush=True)
            yield name, None
            continue
        yield canonical, dsdir


# Explicit dtypes so every column has a concrete Arrow type even when a whole column is
# empty/all-null (e.g. no link tasks -> num_dst_entities all null). An all-null *object*
# column serializes to the Arrow ``null`` type, which the HF dataset viewer can't read.
_INT_COLS = {
    "num_rows_train",
    "num_rows_val",
    "num_rows_test",
    "num_unique_entities",
    "num_dst_entities",
    "num_eval_timestamps",
    "eval_k",
}
_FLOAT_COLS = {"pct_train_test_entity_overlap"}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS:
        if col in _INT_COLS:
            df[col] = df[col].astype("Int64")  # pandas nullable int -> Arrow int64
        elif col in _FLOAT_COLS:
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype("string")  # -> Arrow string, never null-typed
    return df


def build(spec: str) -> pd.DataFrame:
    records = []
    skipped = []
    for name, dsdir in iter_datasets(spec):
        if dsdir is None:  # download failed for this dataset (already warned)
            skipped.append(name)
            continue
        tasks_dir = dsdir / "tasks"
        if not tasks_dir.exists():
            continue
        dm = DatasetManifest.load(dsdir / "manifest.yaml")
        for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
            if not (task_dir / "manifest.yaml").exists():
                continue
            print(f"  {name}/{task_dir.name}", flush=True)
            records.append(task_row(dm, name, task_dir.name, task_dir))
    if skipped:
        print(
            f"\nWARNING: {len(skipped)} datasets could not be downloaded and were "
            f"left out: {skipped}",
            flush=True,
        )
    df = pd.DataFrame(records, columns=COLUMNS)
    df = df.sort_values(["database", "task"]).reset_index(drop=True)
    df = _finalize(df)
    if len(df):
        # Drop columns that never apply to this repo (always null) -- e.g. the link-only
        # destination/eval_k columns for a repo whose tasks are all node tasks.
        df = df.dropna(axis=1, how="all")
    return df


# --------------------------------------------------------------------------- #
# Sanity check against the published paper tables (NOT a data source)
# --------------------------------------------------------------------------- #

# Per-task (num_rows_train, num_rows_val, num_rows_test, num_unique_entities,
# pct_train_test_entity_overlap, num_dst_entities) exactly as printed in the papers.
# v1 = RelBench (NeurIPS 2024, arXiv:2407.20060), Table "Full list of predictive tasks".
# v2 = RelBench v2 (arXiv:2602.12606), Table "Full list of tasks" / per-kind tables.
# Used only by --check; released data revised after a paper is expected to differ.
PAPER_REFERENCE: dict[str, dict[str, tuple]] = {
    # ---- v1 (published) ----
    "rel-amazon": {
        "user-churn": (4732555, 409792, 351885, 1585983, 88.0, None),
        "item-churn": (2559264, 177689, 166842, 416352, 93.1, None),
        "user-ltv": (4732555, 409792, 351885, 1585983, 88.0, None),
        "item-ltv": (2707679, 166978, 178334, 427537, 93.5, None),
        "user-item-purchase": (5112803, 351876, 393985, 1632909, 87.4, 12562384),
        "user-item-rate": (3667157, 257939, 292609, 1481360, 81.0, 7665611),
        "user-item-review": (2324177, 116970, 127021, 894136, 74.1, 5406835),
    },
    "rel-avito": {
        "ad-ctr": (5100, 1766, 1816, 4997, 59.8, None),
        "user-clicks": (59454, 21183, 47996, 66449, 45.3, None),
        "user-visits": (86619, 29979, 36129, 63405, 64.6, None),
        "user-ad-visit": (86616, 29979, 36129, 63402, 64.6, 3616174),
    },
    "rel-event": {
        "user-attendance": (19261, 2014, 2006, 9694, 14.6, None),
        "user-repeat": (3842, 268, 246, 1514, 11.5, None),
        "user-ignore": (19239, 4185, 4010, 9799, 21.1, None),
    },
    "rel-f1": {
        "driver-dnf": (11411, 566, 702, 821, 50.0, None),
        "driver-top3": (1353, 588, 726, 134, 50.0, None),
        "driver-position": (7453, 499, 760, 826, 44.6, None),
    },
    "rel-hm": {
        "user-churn": (3871410, 76556, 74575, 1002984, 89.7, None),
        "item-sales": (5488184, 105542, 105542, 105542, 100.0, None),
        "user-item-purchase": (3878451, 74575, 67144, 1004046, 89.2, 13428473),
    },
    "rel-stack": {
        "user-engagement": (1360850, 85838, 88137, 88137, 97.4, None),
        "user-badge": (3386276, 247398, 255360, 255360, 96.9, None),
        "post-votes": (2453921, 156216, 160903, 160903, 97.1, None),
        "user-post-comment": (21239, 825, 758, 11453, 59.9, 44940),
        "post-post-related": (5855, 226, 258, 5924, 8.5, 7456),
    },
    "rel-trial": {
        "study-outcome": (11994, 960, 825, 13779, 0.0, None),
        "study-adverse": (43335, 3596, 3098, 50029, 0.0, None),
        "site-success": (151407, 19740, 22617, 129542, 42.0, None),
        "condition-sponsor-run": (36934, 2081, 2057, 3956, 98.4, 533624),
        "site-sponsor-run": (669310, 37003, 27428, 445513, 48.3, 1565463),
    },
    # ---- v2 (preprint) ----
    "rel-arxiv": {
        "paper-citation": (534233, 155845, 193696, 136183, 70.31, None),
        "author-category": (210769, 39015, 39655, 126219, 62.15, None),
        "author-publication": (210769, 39015, 39655, 101886, 62.15, None),
        "paper-paper-cocitation": (246341, 71257, 82033, 94289, 60.57, 138688),
    },
    "rel-ratebeer": {
        "beer-churn": (2470686, 92367, 79927, 516368, 45.46, None),
        "user-churn": (373709, 19908, 9392, 154071, 35.21, None),
        "brewer-dormant": (98697, 15840, 16366, 28333, 64.07, None),
        "user-count": (373709, 19908, 9392, 154071, 35.21, None),
        "user-beer-favorite": (1099, 1043, 499, 2296, 10.82, 7745),
        "user-beer-liked": (150322, 5681, 2783, 35010, 58.53, 170964),
        "user-place-liked": (38444, 547, 351, 7425, 81.77, 46814),
        "beer_ratings-total_score": (10620177, 1227702, 2495360, 14343239, 0.0, None),
    },
    "rel-salt": {
        "item-plant": (1622787, 293823, 400206, 2316816, 0.0, None),
        "item-shippoint": (1622787, 293780, 398536, 2315103, 0.0, None),
        "item-incoterms": (1622787, 293891, 402835, 2319513, 0.0, None),
        "sales-office": (340491, 71474, 88942, 500907, 0.0, None),
        "sales-group": (340491, 70224, 83193, 493908, 0.0, None),
        "sales-payterms": (340491, 71472, 88831, 500794, 0.0, None),
        "sales-shipcond": (340491, 71398, 88422, 500311, 0.0, None),
        "sales-incoterms": (340491, 71470, 88925, 500886, 0.0, None),
    },
}
_CHECK_FIELDS = [
    "num_rows_train",
    "num_rows_val",
    "num_rows_test",
    "num_unique_entities",
    "pct_train_test_entity_overlap",
    "num_dst_entities",
]


def check(df: pd.DataFrame) -> None:
    r"""Print a per-task agreement report of computed vs.

    published numbers.
    """
    print(
        "\nSanity check vs. published paper tables "
        "(v1 arXiv:2407.20060, v2 arXiv:2602.12606):",
        flush=True,
    )
    n_ok = n_diff = n_tasks = 0
    for name, ref in PAPER_REFERENCE.items():
        sub = df[df["database"] == name]
        if sub.empty:
            continue
        print(f"\n  {name}:")
        for task, vals in ref.items():
            hit = sub[sub["task"] == task]
            if hit.empty:
                print(f"    {task:28s} (not present in this build)")
                continue
            n_tasks += 1
            r = hit.iloc[0]
            diffs = []
            for field, want in zip(_CHECK_FIELDS, vals):
                if want is None:
                    continue
                got = (
                    r[field] if field in r.index else None
                )  # may be dropped (all-null)
                if got is None or pd.isna(got):
                    diffs.append(f"{field}: got None want {want}")
                elif field == "pct_train_test_entity_overlap":
                    if abs(float(got) - float(want)) > 0.6:  # rounding tolerance
                        diffs.append(f"{field}: {got} vs {want}")
                elif int(got) != int(want):
                    diffs.append(f"{field}: {got} vs {want}")
            if diffs:
                n_diff += 1
                print(f"    {task:28s} DIFF  " + "; ".join(diffs))
            else:
                n_ok += 1
                print(f"    {task:28s} ok")
    print(
        f"\n  {n_ok}/{n_tasks} checked tasks match the paper exactly "
        f"({n_diff} differ -- expected where the released data was revised "
        f"post-paper).",
        flush=True,
    )


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    spec = args[0]
    do_push = "--push" in args
    do_check = "--check" in args
    out = Path(args[args.index("--out") + 1]) if "--out" in args else Path(".")

    print(f"building tasks overview for {spec!r}", flush=True)
    df = build(spec)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "tasks.parquet"
    df.to_parquet(out_path, index=False)
    print(
        f"\nwrote {out_path} ({len(df)} tasks, {df['database'].nunique()} databases)",
        flush=True,
    )
    headline = [c for c in HEADLINE_COLS if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[headline].to_string(index=False))

    if do_check:
        check(df)

    if do_push:
        if Path(spec).exists():
            sys.exit(
                "--push needs a Hub repo spec (e.g. 'relbench/core'), not a local path"
            )
        repo_id, _ = resolve_repo(spec)
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="STATS/tasks.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add STATS/tasks.parquet (tasks subset of the dataset viewer)",
        )
        print(f"pushed STATS/tasks.parquet to {repo_id}", flush=True)


if __name__ == "__main__":
    main()
