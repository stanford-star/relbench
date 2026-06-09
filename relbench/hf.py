r"""Hugging Face Hub download layer (replaces pooch).

Official datasets are listed in ``registry.json`` mapping a RelBench name to a
Hub ``repo_id`` and a pinned ``revision``. Pinning the revision is what makes task
labels reproducible: a task's provenance is its manifest SQL run against the database
at a *fixed* dataset revision. Integrity comes from the pinned commit, not a hash file.

Third-party datasets need no registry entry -- ``relbench.load_dataset("org/name")``
downloads any Hub dataset repo that follows the manifest layout, and a local path works
with no Hub access at all.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_REGISTRY_PATH = Path(__file__).parent / "registry.json"


@lru_cache(maxsize=None)
def _registry() -> dict:
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def registered_names() -> list[str]:
    r"""Names of the official, curated RelBench datasets."""
    return list(_registry())


def is_registered(name: str) -> bool:
    return name in _registry()


def resolve_repo(name: str) -> tuple[str, Optional[str]]:
    r"""Return ``(repo_id, revision)`` for a registered name, or treat ``name`` as a
    Hub repo id (``"org/dataset"``) with no pinned revision."""
    reg = _registry()
    if name in reg:
        return reg[name]["repo_id"], reg[name].get("revision")
    if "/" in name:  # looks like a Hub repo id
        return name, None
    raise KeyError(
        f"'{name}' is not a registered RelBench dataset (known: {sorted(reg)}) "
        f"and is not a 'org/name' Hub repo id. Pass a local path instead."
    )


def download_dataset_dir(name: str, revision: Optional[str] = None) -> Path:
    r"""Download a dataset repo from the Hub and return the local snapshot dir."""
    from huggingface_hub import snapshot_download

    repo_id, pinned = resolve_repo(name)
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision or pinned,
        repo_type="dataset",
    )
    return Path(local)
