r"""Task access.

RelBench 3.0 removed the per-task classes. Tasks are now manifest directories within a
dataset's Hugging Face folder, discovered and built by :mod:`relbench.load`. The functions
here are thin, backward-compatible aliases over :func:`relbench.load_task`.
"""

from functools import lru_cache
from typing import List

from relbench.base import BaseTask
from relbench.load import get_task_names as _get_task_names
from relbench.load import load_task


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str, download: bool = False) -> BaseTask:
    r"""Deprecated alias for :func:`relbench.load_task`. ``download`` is ignored."""
    return load_task(dataset_name, task_name)


def download_task(dataset_name: str, task_name: str) -> None:
    r"""Deprecated: pre-fetch the task's labels into the local Hugging Face cache."""
    load_task(dataset_name, task_name)


def get_task_names(dataset_name: str) -> List[str]:
    r"""List tasks available for a dataset (enumerates ``tasks/*/manifest.json``)."""
    return _get_task_names(dataset_name)
