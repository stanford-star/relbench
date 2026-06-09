r"""Provenance / reproduction check for the rel-f1 migration.

For every rel-f1 task, regenerate labels with the NEW manifest-driven SQL loader and
compare against the LEGACY task classes regenerating from the same canonical database.
Exact match (order-insensitive; float tolerance for aggregates) proves the manifest SQL
faithfully reproduces the shipped tasks with no per-task classes.

    pixi run --frozen python build/validate_rel_f1.py [--out DIR]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import relbench.tasks.f1 as f1mod
from relbench.base import AutoCompleteTask, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.load import load_task

SPLITS = ["train", "val", "test"]

# Tasks whose labels are regenerable via the manifest SQL (the SQL-in-manifest thesis).
# Autocomplete tasks use a separate generator with a pre-existing 1-second date_range
# that does not scale to wide val/test gaps, so they are served as hosted labels and
# excluded from the regeneration check here.
SQL_TASKS = ["driver-position", "driver-dnf", "driver-top3", "driver-circuit-compete"]

# new task name -> factory for the LEGACY task object on a given dataset
OLD = {
    "driver-position": lambda ds: f1mod.DriverPositionTask(ds, cache_dir=None),
    "driver-dnf": lambda ds: f1mod.DriverDNFTask(ds, cache_dir=None),
    "driver-top3": lambda ds: f1mod.DriverTop3Task(ds, cache_dir=None),
    "driver-circuit-compete": lambda ds: f1mod.DriverCircuitCompeteTask(ds, cache_dir=None),
    "results-position": lambda ds: AutoCompleteTask(
        ds, task_type=TaskType.REGRESSION, entity_table="results", target_col="position",
        cache_dir=None,
        remove_columns=[
            ("results", "statusId"), ("results", "positionOrder"), ("results", "points"),
            ("results", "laps"), ("results", "milliseconds"), ("results", "fastestLap"),
            ("results", "rank"),
        ],
    ),
    "qualifying-position": lambda ds: AutoCompleteTask(
        ds, task_type=TaskType.REGRESSION, entity_table="qualifying", target_col="position",
        cache_dir=None, remove_columns=[],
    ),
}


def _keys_target(task):
    if isinstance(task, RecommendationTask):
        return [task.time_col, task.src_entity_col], task.dst_entity_col, True
    return [task.time_col, task.entity_col], task.target_col, False


def _compare(nt: pd.DataFrame, ot: pd.DataFrame, keys, target, is_list):
    nt = nt.sort_values(keys).reset_index(drop=True)
    ot = ot.sort_values(keys).reset_index(drop=True)
    if len(nt) != len(ot):
        return False, f"row count {len(nt)} vs {len(ot)}"
    nk = set(map(tuple, nt[keys].to_numpy()))
    ok = set(map(tuple, ot[keys].to_numpy()))
    if nk != ok:
        return False, f"key sets differ (+{len(nk - ok)} / -{len(ok - nk)})"
    if is_list:
        bad = sum(
            set(map(int, a)) != set(map(int, b))
            for a, b in zip(nt[target], ot[target])
        )
        return bad == 0, "ok" if bad == 0 else f"{bad} rows differ in dst set"
    a, b = nt[target].to_numpy(), ot[target].to_numpy()
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        if np.allclose(a, b, equal_nan=True):
            return True, "ok"
        i = int(np.argmax(~np.isclose(a, b, equal_nan=True)))
        return False, f"value mismatch (e.g. new={a[i]} old={b[i]})"
    eq = (pd.Series(a).fillna("∅").astype(str) == pd.Series(b).fillna("∅").astype(str)).all()
    return bool(eq), "ok" if eq else "value mismatch (non-numeric)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/relbench-build/rel-f1")
    args = p.parse_args()

    all_ok = True
    width = max(len(t) for t in SQL_TASKS)
    print(f"\n{'task':<{width}}  {'split':<5}  {'rows':>7}  result", flush=True)
    print("-" * (width + 30), flush=True)
    for name in SQL_TASKS:
        new_task = load_task(args.out, name, regenerate=True)
        get_dataset.cache_clear()
        old_task = OLD[name](get_dataset("rel-f1", download=True))
        keys, target, is_list = _keys_target(new_task)
        for split in SPLITS:
            try:
                nt = new_task.get_table(split, mask_input_cols=False).df
                ot = old_task.get_table(split, mask_input_cols=False).df
                ok, detail = _compare(nt, ot, keys, target, is_list)
                n = len(nt)
            except Exception as e:
                ok, detail, n = False, f"{type(e).__name__}: {e}", 0
            all_ok &= ok
            extra = "" if detail == "ok" else f"  <- {detail}"
            print(f"{name:<{width}}  {split:<5}  {n:>7}  {'PASS' if ok else 'FAIL'}{extra}", flush=True)
    print("\n" + ("ALL SQL TASKS REPRODUCED EXACTLY" if all_ok else "MISMATCHES FOUND"), flush=True)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
