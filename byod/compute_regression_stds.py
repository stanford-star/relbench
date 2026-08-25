r"""Compute and publish the RelBench regression-target stds.

For every regression task of the datasets hosted in ``stanford-star/relbench-v1`` and
``stanford-star/relbench-v2-extra``, compute the standard deviation (ddof=1) of the target
on the *train* split and store them all together in one file at the root of the
``stanford-star/relbench-v1`` Hub repo. These stds normalize MAE into NMAE (the regression
metric); see ``relbench.metrics.make_nmae`` / ``relbench.hf.load_core_regression_stds``.

    # write regression_stds.json locally (under --out, default '.') and print it
    python byod/compute_regression_stds.py

    # also upload it to the stanford-star/relbench-v1 dataset repo
    python byod/compute_regression_stds.py --push

Keyed by "<dataset>/<task>" (e.g. "rel-f1/driver-position").
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

import relbench
from relbench.hf import (
    REGRESSION_STDS_FILE,
    RELBENCH_HF,
    RELBENCH_REPOS,
    download_task_manifest,
)
from relbench.manifest import TaskManifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--push", action="store_true", help="upload to RELBENCH_HF")
    args = parser.parse_args(argv)

    api = HfApi()

    datasets = set()
    for repo in RELBENCH_REPOS:
        files = api.list_repo_files(repo, repo_type="dataset")
        names = {
            f.split("/")[0]
            for f in files
            if f.endswith("/manifest.yaml") and f.count("/") == 1
        }
        print(f"{repo}: {len(names)} datasets", flush=True)
        datasets |= names

    stds: dict[str, float] = {}
    for name in sorted(datasets):
        ds = relbench.load_dataset(name)
        for task_name in ds.get_task_names():
            tm = TaskManifest.load(download_task_manifest(name, task_name))
            if tm.task_type != "regression":
                continue
            std = relbench.train_std(ds.load_task(task_name))
            stds[f"{name}/{task_name}"] = std
            print(f"  {name}/{task_name}: {std:.6g}", flush=True)

    payload = {
        "_comment": (
            "Per-task standard deviation (ddof=1) of the regression target on the train "
            "split, for the RelBench datasets hosted in stanford-star/relbench-v1 and "
            "stanford-star/relbench-v2-extra. Used to normalize MAE into NMAE "
            "(nmae = mae / std). Keyed by '<dataset>/<task>'."
        ),
        "stds": stds,
    }

    out_path = args.out / REGRESSION_STDS_FILE
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path} ({len(stds)} tasks)", flush=True)

    if args.push:
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=REGRESSION_STDS_FILE,
            repo_id=RELBENCH_HF,
            repo_type="dataset",
            commit_message="Add regression-target stds (normalize MAE -> NMAE)",
        )
        print(f"pushed {REGRESSION_STDS_FILE} to {RELBENCH_HF}", flush=True)


if __name__ == "__main__":
    main()
