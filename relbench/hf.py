r"""Hugging Face Hub download layer.

A RelBench dataset is any local directory or Hugging Face Hub location that follows the
manifest layout (a ``manifest.yaml`` next to ``db/<table>.parquet`` and ``tasks/``).
Address it as:

* a local path, e.g. ``/data/rel-f1``;
* a Hub repo, e.g. ``relbench/rel-f1`` (manifest at the repo root); or
* a Hub sub-path, e.g. ``stanford-star/relbench/rel-f1`` (manifest under ``rel-f1/`` in ``stanford-star/relbench``).

There is no central registry of names and no pinned revisions: the latest ``main`` is used
unless you pass ``revision=`` explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Where the per-task regression-target stds for the RelBench core datasets live. One JSON
# file at the root of the ``stanford-star/relbench`` Hub dataset repo, keyed by "<dataset>/<task>".
# These normalize MAE into NMAE (nmae = mae / std); see relbench.metrics.make_nmae.
RELBENCH_HF = "stanford-star/relbench"

# The v2-only datasets and tasks live in a second repo. Bare names are resolved against
# both, in order, so `load_dataset("rel-arxiv")` and `ds.load_task("results-position")`
# work regardless of which repo currently hosts the artifact.
RELBENCH_EXTRA_HF = "stanford-star/relbench-v2-extra"
RELBENCH_REPOS = (RELBENCH_HF, RELBENCH_EXTRA_HF)

REGRESSION_STDS_FILE = "regression_stds.json"


def load_core_regression_stds(revision: Optional[str] = None) -> dict[str, float]:
    r"""Fetch the hosted ``stanford-star/relbench`` regression-std table as
    ``{"<dataset>/<task>": std}``.

    Uses ``hf_hub_download``, so if the file is already in the local HF cache (e.g. the user
    has it from a prior call) it is read from disk without a network round-trip.

    Not cached: hold the returned dict if you need it more than once. RelBench's own
    caller -- the per-task NMAE normalizer -- resolves it once per task.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=RELBENCH_HF,
        filename=REGRESSION_STDS_FILE,
        repo_type="dataset",
        revision=revision,
    )
    with open(path) as f:
        return dict(json.load(f).get("stds", {}))


def _repo_has(repo_id: str, path: str, revision: Optional[str] = None) -> bool:
    from huggingface_hub import file_exists

    try:
        return file_exists(repo_id, path, repo_type="dataset", revision=revision)
    except Exception:
        return False


def resolve_repo(spec: str, revision: Optional[str] = None) -> tuple[str, str]:
    r"""Split a Hub spec into ``(repo_id, subdir)``.

    ``"org/name"`` -> ``("org/name", "")``; ``"org/name/a/b"`` -> ``("org/name", "a/b")``.
    A bare ``"name"`` (no ``org/``) is looked up in each repo of :data:`RELBENCH_REPOS`
    in order and resolves to the first one holding ``<name>/manifest.yaml``:
    ``"rel-f1"`` -> ``(RELBENCH_HF, "rel-f1")``, ``"rel-arxiv"`` ->
    ``(RELBENCH_EXTRA_HF, "rel-arxiv")``. This lets datasets be addressed by their short
    name without spelling out the hosting repo, and without the caller having to know
    which RelBench repo a dataset lives in. An unknown name falls back to
    :data:`RELBENCH_HF`, so the error the caller sees names a concrete repo.
    """
    parts = [p for p in spec.strip("/").split("/") if p]
    if len(parts) == 0:
        raise ValueError(
            f"'{spec}' is not a Hub 'org/name' repo id (optionally with a '/subdir'). "
            f"Pass a Hub 'org/name[/subdir]', a bare dataset name, or a local path."
        )
    if len(parts) == 1:
        name = parts[0]
        for repo_id in RELBENCH_REPOS:
            if _repo_has(repo_id, f"{name}/manifest.yaml", revision):
                return repo_id, name
        return RELBENCH_HF, name
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def download_subdir(
    repo_id: str, subdir: str, revision: Optional[str] = None
) -> Path:
    r"""Download one sub-path of a Hub dataset repo and return its local directory."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns=[f"{subdir}/*"] if subdir else None,
    )
    return Path(local) / subdir if subdir else Path(local)


def find_task_dir(
    dataset_name: str, task_name: str, revision: Optional[str] = None
) -> Optional[Path]:
    r"""Locate ``<dataset>/tasks/<task>`` across :data:`RELBENCH_REPOS` and download it.

    This is what makes a task loadable by bare name even when it is hosted apart from its
    database -- the v2-only tasks on the v1 datasets live in :data:`RELBENCH_EXTRA_HF`
    while the databases stay in :data:`RELBENCH_HF`. ``None`` if no repo has it.
    """
    subdir = f"{dataset_name}/tasks/{task_name}"
    for repo_id in RELBENCH_REPOS:
        if _repo_has(repo_id, f"{subdir}/manifest.yaml", revision):
            return download_subdir(repo_id, subdir, revision=revision)
    return None


def list_task_names(dataset_name: str, revision: Optional[str] = None) -> list[str]:
    r"""Every task hosted for ``dataset_name``, across :data:`RELBENCH_REPOS`."""
    from huggingface_hub import HfApi

    api = HfApi()
    out: set[str] = set()
    for repo_id in RELBENCH_REPOS:
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        except Exception:
            continue
        for f in files:
            parts = f.split("/")
            if (
                len(parts) == 4
                and parts[0] == dataset_name
                and parts[1] == "tasks"
                and parts[3] == "manifest.yaml"
            ):
                out.add(parts[2])
    return sorted(out)


def download_dataset_dir(spec: str, revision: Optional[str] = None) -> Path:
    r"""Download a dataset from the Hub and return its local directory.

    Only the addressed sub-path is fetched, so loading ``stanford-star/relbench/rel-f1`` does not pull
    every dataset in ``stanford-star/relbench``.
    """
    repo_id, subdir = resolve_repo(spec, revision=revision)
    return download_subdir(repo_id, subdir, revision=revision)
