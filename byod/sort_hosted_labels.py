r"""Rewrite hosted task labels in canonical (sorted) order.

Label tables have a canonical order -- ``relbench.base.task_base.sort_labels``, by time
column then key columns -- applied when labels are written. Loading does not re-sort, so
the hosted ``train/val/test.parquet`` must already be in that order.

Generators sort before writing, so this is a repair tool, not part of the normal flow:
run it to migrate labels published before the ordering existed, or to check that a repo
is still canonical (a dry run reporting 0 files is the check). It **reorders rows
only**: no value is recomputed, no row added or dropped, so every metric is unchanged
(evaluation joins on keys). Files already in order are left alone and not re-uploaded.

Usage::

    python byod/sort_hosted_labels.py                  # dry run, whole repo
    python byod/sort_hosted_labels.py --dataset rel-f1 # dry run, one dataset
    python byod/sort_hosted_labels.py --push           # upload the changes
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from relbench.base.task_base import label_sort_cols, sort_labels
from relbench.hf import RELBENCH_HF
from relbench.manifest import TaskManifest

SPLITS = ["train", "val", "test"]


def _sort_cols(tm: TaskManifest) -> list:
    r"""The key columns of a task's label table, in sort order.

    The manifest carries the same field names the task does, so the pinned order in
    :func:`relbench.base.task_base.label_sort_cols` applies directly.
    """
    return [c for c in label_sort_cols(tm) if c]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=RELBENCH_HF)
    parser.add_argument("--dataset", default=None, help="limit to one sub-dataset")
    parser.add_argument("--push", action="store_true", help="upload (default: dry run)")
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    files = api.list_repo_files(args.repo, repo_type="dataset")
    task_manifests = sorted(
        f for f in files if f.endswith("/manifest.yaml") and "/tasks/" in f
    )
    if args.dataset:
        task_manifests = [f for f in task_manifests if f.split("/")[0] == args.dataset]
    print(f"{args.repo}: {len(task_manifests)} tasks", flush=True)

    changed, checked = [], 0
    for mpath in task_manifests:
        tdir = mpath.rsplit("/", 1)[0]  # "<dataset>/tasks/<task>"
        local = Path(
            snapshot_download(
                repo_id=args.repo,
                repo_type="dataset",
                allow_patterns=[f"{tdir}/*"],
            )
        )
        tm = TaskManifest.load(local / mpath)
        cols = _sort_cols(tm)
        for split in SPLITS:
            rel = f"{tdir}/{split}.parquet"
            path = local / rel
            if not path.exists():
                continue
            checked += 1
            df = pd.read_parquet(path)
            out = sort_labels(df.copy(), cols)
            if out.equals(df):
                continue
            # Reorder only: identical rows, identical columns, identical dtypes.
            assert len(out) == len(df), (rel, len(out), len(df))
            assert list(out.columns) == list(df.columns), rel
            assert (out.dtypes == df.dtypes).all(), rel
            key = [
                c
                for c in cols
                if c in df.columns
                and not df[c].map(lambda v: isinstance(v, (list, np.ndarray))).any()
            ]
            assert (
                df.sort_values(key, kind="stable")
                .reset_index(drop=True)[key]
                .equals(out[key])
            ), rel
            changed.append((rel, len(df), out))
            print(f"  REORDER {rel:<60} rows={len(df)}", flush=True)

    print(f"\n{len(changed)} of {checked} label files need reordering")
    if not changed:
        return
    if not args.push:
        print("dry run -- pass --push to upload")
        return

    with tempfile.TemporaryDirectory() as tmp:
        for rel, _, out in changed:
            dst = Path(tmp) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(dst, index=False)
        api.upload_folder(
            repo_id=args.repo,
            repo_type="dataset",
            folder_path=tmp,
            allow_patterns=[rel for rel, *_ in changed],
            commit_message=(
                "Write task labels in canonical sorted order (row reorder only)"
            ),
        )
    print(f"pushed {len(changed)} files to {args.repo}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
