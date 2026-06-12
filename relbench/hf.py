r"""Hugging Face Hub download layer.

A RelBench dataset is any local directory or Hugging Face Hub location that follows the
manifest layout (a ``manifest.yaml`` next to ``db/<table>.parquet`` and ``tasks/``).
Address it as:

* a local path, e.g. ``/data/rel-f1``;
* a Hub repo, e.g. ``relbench/rel-f1`` (manifest at the repo root); or
* a Hub sub-path, e.g. ``relbench/core/rel-f1`` (manifest under ``rel-f1/`` in ``relbench/core``).

There is no central registry of names and no pinned revisions: the latest ``main`` is used
unless you pass ``revision=`` explicitly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Where the per-task regression-target stds for the RelBench v1 datasets live. One JSON
# file at the root of the ``relbench/core`` Hub dataset repo, keyed by "<dataset>/<task>".
# These normalize MAE into NMAE (nmae = mae / std); see relbench.metrics.make_nmae.
V1_REPO = "relbench/core"
REGRESSION_STDS_FILE = "regression_stds.json"


@lru_cache(maxsize=None)
def load_v1_regression_stds(revision: Optional[str] = None) -> dict[str, float]:
    r"""Fetch the hosted ``relbench/core`` regression-std table as ``{"<dataset>/<task>":
    std}``.

    Uses ``hf_hub_download``, so if the file is already in the local HF cache (e.g. the user
    has it from a prior call) it is read from disk without a network round-trip.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=V1_REPO,
        filename=REGRESSION_STDS_FILE,
        repo_type="dataset",
        revision=revision,
    )
    with open(path) as f:
        return dict(json.load(f).get("stds", {}))


def resolve_repo(spec: str) -> tuple[str, str]:
    r"""Split a Hub spec into ``(repo_id, subdir)``.

    ``"org/name"`` -> ``("org/name", "")``; ``"org/name/a/b"`` -> ``("org/name", "a/b")``.
    """
    parts = spec.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"'{spec}' is not a Hub 'org/name' repo id (optionally with a '/subdir'). "
            f"Pass a Hub 'org/name[/subdir]' or a local path."
        )
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def download_dataset_dir(spec: str, revision: Optional[str] = None) -> Path:
    r"""Download a dataset from the Hub and return its local directory.

    Only the addressed sub-path is fetched, so loading ``relbench/core/rel-f1`` does not pull
    every dataset in ``relbench/core``.
    """
    from huggingface_hub import snapshot_download

    repo_id, subdir = resolve_repo(spec)
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns=[f"{subdir}/*"] if subdir else None,
    )
    return Path(local) / subdir if subdir else Path(local)
