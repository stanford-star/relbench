r"""Compute and publish the RelBench core regression-target stds.

For every regression task in the ``stanford-star/relbench-v1`` datasets, compute the standard deviation
(ddof=1) of the target on the *train* split and store them all together in one file at the
root of the ``stanford-star/relbench-v1`` Hub repo. These stds normalize MAE into NMAE (the regression
metric); see ``relbench.metrics.make_nmae`` / ``relbench.hf.load_core_regression_stds``.

    # write regression_stds.json locally (under OUT) and print it
    python byod/compute_regression_stds.py

    # also upload it to the stanford-star/relbench-v1 dataset repo
    python byod/compute_regression_stds.py --push

Keyed by "<dataset>/<task>" (e.g. "rel-f1/driver-position").
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

import relbench
from relbench.base import TaskType
from relbench.hf import REGRESSION_STDS_FILE, RELBENCH_HF


def main() -> None:
    do_push = "--push" in sys.argv
    OUT = (
        Path(sys.argv[sys.argv.index("--out") + 1])
        if "--out" in sys.argv
        else Path(".")
    )

    api = HfApi()

    # Enumerate stanford-star/relbench-v1's sub-datasets (each has a top-level <name>/manifest.yaml).
    files = api.list_repo_files(RELBENCH_HF, repo_type="dataset")
    datasets = sorted(
        {
            f.split("/")[0]
            for f in files
            if f.endswith("/manifest.yaml") and f.count("/") == 1
        }
    )
    print(f"{RELBENCH_HF}: {len(datasets)} sub-datasets", flush=True)

    stds: dict[str, float] = {}
    for name in datasets:
        spec = f"{RELBENCH_HF}/{name}"
        ds = relbench.load_dataset(spec)
        for task_name in ds.get_task_names():
            try:
                task = ds.load_task(task_name)
            except NotImplementedError:
                # Unsupported task type (e.g. a multiclass autocomplete task); not a
                # regression task, so it has no std to store.
                continue
            if task.task_type != TaskType.REGRESSION:
                continue
            std = relbench.train_std(task)
            stds[f"{name}/{task_name}"] = std
            print(f"  {name}/{task_name}: {std:.6g}", flush=True)

    payload = {
        "_comment": (
            "Per-task standard deviation (ddof=1) of the regression target on the train "
            "split, for the RelBench core datasets. Used to normalize MAE into NMAE "
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
            repo_id=RELBENCH_HF,
            repo_type="dataset",
            commit_message="Add regression-target stds (normalize MAE -> NMAE)",
        )
        print(f"pushed {REGRESSION_STDS_FILE} to {RELBENCH_HF}", flush=True)


if __name__ == "__main__":
    main()
