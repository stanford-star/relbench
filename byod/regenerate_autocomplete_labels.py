r"""Regenerate the hosted ``autocomplete`` task labels from their databases.

Every ``kind: autocomplete`` task in the repo is rebuilt with the library's current split
boundaries (train: time <= val_timestamp; val: (val_timestamp, test_timestamp];
test: > test_timestamp), written to a temp mirror and compared with the hosted labels.
Dry run by default; ``--push`` uploads the mirror in one commit.

    python byod/regenerate_autocomplete_labels.py --list           # the tasks, no data
    python byod/regenerate_autocomplete_labels.py                  # dry run, all tasks
    python byod/regenerate_autocomplete_labels.py --dataset rel-f1 rel-salt
    python byod/regenerate_autocomplete_labels.py --push
"""

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

import relbench
from relbench.hf import RELBENCH_EXTRA_HF
from relbench.manifest import KIND_AUTOCOMPLETE, TaskManifest

SPLITS = ["train", "val", "test"]
COMMIT_MESSAGE = "Regenerate autocomplete labels with inclusive split boundaries"


def autocomplete_tasks(api: HfApi, repo: str, datasets) -> list:
    out = []
    for f in sorted(api.list_repo_files(repo, repo_type="dataset")):
        parts = f.split("/")
        if len(parts) != 4 or parts[1] != "tasks" or parts[3] != "manifest.yaml":
            continue
        if datasets and parts[0] not in datasets:
            continue
        tm = TaskManifest.load(hf_hub_download(repo, f, repo_type="dataset"))
        if tm.kind == KIND_AUTOCOMPLETE:
            out.append((parts[0], parts[2], tm))
    return out


def diff(old: pd.DataFrame, new: pd.DataFrame, key: str, target: str) -> tuple:
    o = old.set_index(key)[target]
    n = new.set_index(key)[target]
    common = o.index.intersection(n.index)
    changed = int((o.loc[common].astype(str) != n.loc[common].astype(str)).sum())
    return len(n.index.difference(o.index)), len(o.index.difference(n.index)), changed


def regenerate(repo: str, tasks: list, mirror: Path) -> None:
    for dataset in sorted({d for d, _, _ in tasks}):
        ds = relbench.load_dataset(dataset)
        for _, task_name, tm in [t for t in tasks if t[0] == dataset]:
            task = ds.load_task(task_name, regenerate=True)
            db = task.get_db(upto_test_timestamp=False)
            for split in SPLITS:
                rel = f"{dataset}/tasks/{task_name}/{split}.parquet"
                new = task.get_table(split, mask_input_cols=False, db=db).df
                old = pd.read_parquet(hf_hub_download(repo, rel, repo_type="dataset"))
                added, removed, changed = diff(old, new, task.entity_col, tm.target_col)
                dst = mirror / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                new.to_parquet(dst, index=False)
                print(
                    f"  {rel:<56} hosted={len(old):>9} new={len(new):>9} "
                    f"+{added} -{removed} changed={changed}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=RELBENCH_EXTRA_HF)
    parser.add_argument("--dataset", nargs="+", default=None, help="limit to these")
    parser.add_argument("--list", action="store_true", help="list the tasks and exit")
    parser.add_argument("--push", action="store_true", help="upload (default: dry run)")
    args = parser.parse_args()

    api = HfApi()
    tasks = autocomplete_tasks(api, args.repo, args.dataset)
    print(f"{args.repo}: {len(tasks)} autocomplete tasks", flush=True)
    if args.list:
        for dataset, task_name, tm in tasks:
            print(
                f"  {dataset}/{task_name}: {tm.entity_table}.{tm.target_col} ({tm.task_type})"
            )
        return
    if not tasks:
        return

    with tempfile.TemporaryDirectory() as tmp:
        regenerate(args.repo, tasks, Path(tmp))
        if not args.push:
            print("dry run -- pass --push to upload")
            return
        api.upload_folder(
            repo_id=args.repo,
            repo_type="dataset",
            folder_path=tmp,
            commit_message=COMMIT_MESSAGE,
        )
        print(f"pushed {len(tasks)} tasks to {args.repo}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
