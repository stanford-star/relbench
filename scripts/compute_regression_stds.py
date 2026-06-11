r"""Compute and publish the RelBench v1 regression-target stds.

For every regression task in the ``relbench/v1`` datasets, compute the standard deviation
(ddof=1) of the target on the *train* split and store them all together in one file at the
root of the ``relbench/v1`` Hub repo. These stds normalize MAE into NMAE (the regression
metric); see ``relbench.metrics.make_nmae`` / ``relbench.hf.load_v1_regression_stds``.

    # write regression_stds.json locally (under OUT) and print it
    python scripts/compute_regression_stds.py

    # also upload it to the relbench/v1 dataset repo
    python scripts/compute_regression_stds.py --push

Keyed by "<dataset>/<task>" (e.g. "rel-f1/driver-position").
"""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

import relbench
from relbench.base import TaskType
from relbench.hf import REGRESSION_STDS_FILE, V1_REPO

do_push = "--push" in sys.argv
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(".")

api = HfApi()

# Enumerate the sub-datasets of relbench/v1 (each has a top-level <name>/manifest.yaml).
files = api.list_repo_files(V1_REPO, repo_type="dataset")
datasets = sorted(
    {f.split("/")[0] for f in files if f.endswith("/manifest.yaml") and f.count("/") == 1}
)
print(f"{V1_REPO}: {len(datasets)} sub-datasets", flush=True)

stds: dict[str, float] = {}
for name in datasets:
    spec = f"{V1_REPO}/{name}"
    ds = relbench.load_dataset(spec)
    for task_name in relbench.get_task_names(spec):
        task = relbench.load_task(ds, task_name)
        if task.task_type != TaskType.REGRESSION:
            continue
        std = relbench.train_std(task)
        stds[f"{name}/{task_name}"] = std
        print(f"  {name}/{task_name}: {std:.6g}", flush=True)

payload = {
    "_comment": (
        "Per-task standard deviation (ddof=1) of the regression target on the train "
        "split, for the RelBench v1 datasets. Used to normalize MAE into NMAE "
        "(nmae = mae / std). Keyed by '<dataset>/<task>'."
    ),
    "stds": stds,
}

out_path = OUT / REGRESSION_STDS_FILE
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"wrote {out_path} ({len(stds)} tasks)", flush=True)

if do_push:
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=REGRESSION_STDS_FILE,
        repo_id=V1_REPO,
        repo_type="dataset",
        commit_message="Add regression-target stds (normalize MAE -> NMAE)",
    )
    print(f"pushed {REGRESSION_STDS_FILE} to {V1_REPO}", flush=True)
