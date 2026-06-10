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


def resolve_repo(name: str) -> tuple[str, Optional[str], str]:
    r"""Return ``(repo_id, revision, subdir)`` for ``name``.

    Official datasets are grouped into family repos (e.g. ``relbench/v1`` holds
    ``rel-f1/``, ``rel-amazon/`` ...; likewise ``relbench/dbinfer``, ``relbench/tgb``),
    so a registry entry carries the family ``repo_id`` plus the in-repo ``path`` (the
    dataset subdir, defaulting to the name). A bare ``"org/dataset"`` Hub repo id is
    treated as a single-dataset repo with the manifest at its root (``subdir=""``).
    """
    reg = _registry()
    if name in reg:
        entry = reg[name]
        return entry["repo_id"], entry.get("revision"), entry.get("path", name)
    if "/" in name:  # a Hub repo id whose root is the dataset
        return name, None, ""
    raise KeyError(
        f"'{name}' is not a registered RelBench dataset (known: {sorted(reg)}) "
        f"and is not a 'org/name' Hub repo id. Pass a local path instead."
    )


def download_dataset_dir(name: str, revision: Optional[str] = None) -> Path:
    r"""Download a dataset from its (possibly shared) Hub repo and return its local dir.

    Only the dataset's subdir is fetched from a family repo, so loading ``rel-f1`` does
    not pull every dataset in ``relbench/v1``.
    """
    from huggingface_hub import snapshot_download

    repo_id, pinned, subdir = resolve_repo(name)
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision or pinned,
        repo_type="dataset",
        allow_patterns=[f"{subdir}/*"] if subdir else None,
    )
    return Path(local) / subdir if subdir else Path(local)
