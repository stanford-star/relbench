r"""Provenance check for a RelBench dataset (Hub ``org/repo[/subdir]`` or a local path).

For every ``forecast`` task, regenerate labels from the database via the manifest SQL and
assert they match the shipped labels. This is the guarantee that a task's hosted labels are
exactly what its SQL produces against the database; it fails on drift (SQL edited without
refreshing labels, or vice versa). Autocomplete and external tasks have no regenerating SQL
and are skipped.

    python provenance/check_provenance.py relbench/core/rel-f1
    python provenance/check_provenance.py your-org/your-dataset
    python provenance/check_provenance.py ./path/to/local/dataset
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from relbench.base import RecommendationTask
from relbench.load import get_task_names, load_task

SPLITS = ["train", "val", "test"]


def _keys_target(task):
    if isinstance(task, RecommendationTask):
        return [task.time_col, task.src_entity_col], task.dst_entity_col, True
    return [task.time_col, task.entity_col], task.target_col, False


def _match(a: pd.DataFrame, b: pd.DataFrame, keys, target, is_list) -> bool:
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b):
        return False
    if set(map(tuple, a[keys].to_numpy())) != set(map(tuple, b[keys].to_numpy())):
        return False
    if is_list:
        return all(
            set(map(int, x)) == set(map(int, y)) for x, y in zip(a[target], b[target])
        )
    x, y = a[target].to_numpy(), b[target].to_numpy()
    if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
        return bool(np.allclose(x, y, equal_nan=True))
    return bool(
        (
            pd.Series(x).fillna("∅").astype(str) == pd.Series(y).fillna("∅").astype(str)
        ).all()
    )


def check_provenance(dataset: str) -> bool:
    r"""Return True iff every ``forecast`` task's regenerated labels match its hosted
    labels."""
    ok_all, checked = True, 0
    for name in get_task_names(dataset):
        regen = load_task(dataset, name, regenerate=True)
        if not getattr(regen, "_sql", None):
            continue  # only forecast tasks carry regenerating SQL; nothing to check
        cached = load_task(dataset, name, regenerate=False)
        keys, target, is_list = _keys_target(regen)
        for split in SPLITS:
            r = regen.get_table(split, mask_input_cols=False).df
            c = cached.get_table(split, mask_input_cols=False).df
            ok = _match(r, c, keys, target, is_list)
            ok_all &= ok
            checked += 1
            print(
                f"{name:<24} {split:<5} regen={len(r):>7} cached={len(c):>7} "
                f"{'PASS' if ok else 'FAIL'}",
                flush=True,
            )
    print(f"\n{'PROVENANCE OK' if ok_all else 'PROVENANCE DRIFT'} ({checked} checks)")
    return ok_all


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if check_provenance(sys.argv[1]) else 1)


if __name__ == "__main__":
    main()
