r"""Repair the hosted task manifests for RelBench 3.

Renames ``task_type: link_prediction`` to ``recommendation`` and fills the fields a few
external tasks are missing, re-serializing every manifest in canonical YAML. Dry run by
default: prints the plan; ``--push`` uploads the changed manifests, one commit per repo.

    python byod/fix_task_manifests.py
    python byod/fix_task_manifests.py --repo stanford-star/tgb
    python byod/fix_task_manifests.py --push
"""

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from relbench.manifest import KIND_EXTERNAL, DatasetManifest, TaskManifest

REPOS = [
    "stanford-star/relbench-v2-extra",
    "stanford-star/dbinfer",
    "stanford-star/tgb",
]
COMMIT_MESSAGE = "Rename task_type link_prediction -> recommendation (RelBench 3)"
EVALUATORS = {
    "stanford-star/dbinfer/dbinfer-amazon/tasks/purchase": "dbinfer",
    "stanford-star/dbinfer/dbinfer-diginetica/tasks/purchase": "dbinfer",
}
ENTITY_COL_FROM_PKEY = {
    "stanford-star/relbench-v2-extra/rel-ratebeer/tasks/beer_ratings-total_score",
}
KNOWN_FIELDS = {f.name for f in dataclasses.fields(TaskManifest)}


def _download(repo: str, path: str) -> Path:
    return Path(hf_hub_download(repo, path, repo_type="dataset"))


def _task_manifest_paths(api: HfApi, repo: str) -> list:
    files = api.list_repo_files(repo, repo_type="dataset")
    return sorted(
        f
        for f in files
        if f.count("/") == 3
        and f.split("/")[1] == "tasks"
        and f.endswith("/manifest.yaml")
    )


def _fix(repo: str, path: str, d: dict) -> list:
    changes = []
    task_dir = f"{repo}/{path.rsplit('/', 1)[0]}"
    if d.get("task_type") == "link_prediction":
        d["task_type"] = "recommendation"
        changes.append("link_prediction -> recommendation")
    evaluator = EVALUATORS.get(task_dir)
    if evaluator and d.get("evaluator") != evaluator:
        d["evaluator"] = evaluator
        changes.append(f"evaluator: {evaluator}")
    if task_dir in ENTITY_COL_FROM_PKEY and not d.get("entity_col"):
        dataset = path.split("/")[0]
        dm = DatasetManifest.load(_download(repo, f"{dataset}/manifest.yaml"))
        d["entity_col"] = dm.tables[d["entity_table"]].pkey
        changes.append(f"entity_col: {d['entity_col']}")
    unknown = sorted(set(d) - KNOWN_FIELDS)
    if unknown:
        changes.append(f"drop unknown keys {unknown}")
    return changes


def _validate(tm: TaskManifest, where: str) -> None:
    try:
        tm.validate()
    except ValueError as e:
        if (
            tm.kind == KIND_EXTERNAL
            and tm.evaluator
            and str(e).endswith("missing ['eval_k']")
        ):
            print(
                f"{where}: validate: no eval_k (external {tm.task_type} task with "
                f"evaluator={tm.evaluator}); continuing",
                flush=True,
            )
            return
        raise ValueError(f"{where}: {e}") from e


def plan(api: HfApi, repo: str, tmp: Path) -> list:
    ops = []
    paths = _task_manifest_paths(api, repo)
    for path in paths:
        where = f"{repo}/{path}"
        src = _download(repo, path)
        with open(src) as f:
            d = yaml.safe_load(f)
        changes = _fix(repo, path, d)
        dst = tmp / repo / path
        TaskManifest.from_dict(d).save(dst)
        _validate(TaskManifest.load(dst), where)
        if dst.read_bytes() == src.read_bytes():
            continue
        print(f"{where}: {', '.join(changes) or 'reformat only'}", flush=True)
        ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(dst)))
    print(f"{repo}: {len(ops)} of {len(paths)} task manifests change\n", flush=True)
    return ops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", nargs="+", default=REPOS, help="limit to these repos")
    parser.add_argument("--push", action="store_true", help="upload (default: dry run)")
    args = parser.parse_args()

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        plans = {repo: plan(api, repo, Path(tmp)) for repo in args.repo}
        total = sum(len(ops) for ops in plans.values())
        print(f"{total} task manifests change across {len(plans)} repos")
        if not total:
            return
        if not args.push:
            print("dry run -- pass --push to upload")
            return
        for repo, ops in plans.items():
            if not ops:
                continue
            api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                operations=ops,
                commit_message=COMMIT_MESSAGE,
            )
            print(f"pushed {len(ops)} manifests to {repo}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
