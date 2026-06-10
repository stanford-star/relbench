r"""CI provenance check for a RelBench dataset folder (local path or registered name).

For every ``forecast`` task, regenerate labels from the database via the manifest SQL and
assert they match the shipped cached labels. This is the guarantee that a task's hosted
labels are exactly what its SQL produces against the pinned database revision; it fails
on drift (SQL edited without refreshing labels, or vice versa). Autocomplete/external
tasks are skipped (autocomplete is checked separately; external labels have no SQL).

    pixi run --frozen python build/check_provenance.py [DATASET]   # default rel-f1 artifact
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from relbench.base import RecommendationTask
from relbench.load import get_task_names, load_task
from relbench.manifest import KIND_FORECAST, TaskManifest

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
        return all(set(map(int, x)) == set(map(int, y)) for x, y in zip(a[target], b[target]))
    x, y = a[target].to_numpy(), b[target].to_numpy()
    if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
        return bool(np.allclose(x, y, equal_nan=True))
    return bool((pd.Series(x).fillna("∅").astype(str) == pd.Series(y).fillna("∅").astype(str)).all())


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "/tmp/relbench-build/rel-f1"
    names = get_task_names(dataset)
    ok_all, checked = True, 0
    for name in names:
        tm = TaskManifest.load(Path(dataset) / "tasks" / name / "manifest.yaml") \
            if Path(dataset).exists() else None
        # When loading by registered name, kind is read off the loaded task instead.
        regen = load_task(dataset, name, regenerate=True)
        if tm is not None and tm.kind != KIND_FORECAST:
            continue
        cached = load_task(dataset, name, regenerate=False)
        keys, target, is_list = _keys_target(regen)
        for split in SPLITS:
            r = regen.get_table(split, mask_input_cols=False).df
            c = cached.get_table(split, mask_input_cols=False).df
            ok = _match(r, c, keys, target, is_list)
            ok_all &= ok
            checked += 1
            print(f"{name:<24} {split:<5} regen={len(r):>6} cached={len(c):>6} "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
    print(f"\n{'PROVENANCE OK' if ok_all else 'PROVENANCE DRIFT'} ({checked} checks)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
