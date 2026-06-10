r"""Dataset access.

RelBench 3.0 removed the per-dataset ``Dataset`` classes and the pooch download registry.
Datasets are now self-describing manifest folders on the Hugging Face Hub, loaded by
:mod:`relbench.load`. The functions here are thin, backward-compatible aliases over
:func:`relbench.load_dataset`.
"""

from functools import lru_cache
from typing import List

from relbench import hf
from relbench.base import Dataset
from relbench.load import load_dataset


@lru_cache(maxsize=None)
def get_dataset(name: str, download: bool = True) -> Dataset:
    r"""Deprecated alias for :func:`relbench.load_dataset`.

    ``download`` is ignored; the pinned Hugging Face revision is always used.
    """
    return load_dataset(name)


def download_dataset(name: str) -> None:
    r"""Deprecated: pre-fetch the dataset into the local Hugging Face cache."""
    hf.download_dataset_dir(name)


def get_dataset_names() -> List[str]:
    r"""Names of the registered official datasets."""
    return hf.registered_names()
