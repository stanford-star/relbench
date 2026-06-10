r"""Finalize + (optionally) push a ported dataset.

For each task: validate the ported forecast SQL in an isolated subprocess. If regen ==
canonical hosted labels -> keep as `forecast` (ship regenerated labels). If it crashes or
mismatches (a handful of tasks have SQL that can't yet be faithfully ported, e.g.
split-dependent or pathological joins) -> fall back to `external` (ship the canonical
hosted labels, no regeneration). Either way the dataset ships complete; `external` tasks
can later be promoted to `forecast` by fixing their SQL.

    pixi run --frozen python build/finalize.py rel-event [rel-hm ...] [--push]
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT_ROOT,
    assemble,
    fetch_canonical_db,
    fetch_canonical_labels,
    push,
)

from relbench.manifest import KIND_EXTERNAL, TaskManifest  # noqa: E402

from port_all import CONFIG, copy_manifests, load_tasks  # noqa: E402

PORT_TASK = str(Path(__file__).resolve().parent / "port_task.py")
ENV = {**os.environ, "RELBENCH_DUCKDB_MEMORY_LIMIT": "300GB"}


def classify(out: Path, ds: str, tasks: list[TaskManifest]) -> set:
    ok = set()
    for tm in tasks:
        try:
            p = subprocess.run([sys.executable, PORT_TASK, str(out), ds, tm.name],
                               capture_output=True, text=True, timeout=900, env=ENV)
            passed = p.returncode == 0
        except subprocess.TimeoutExpired:
            passed = False
        if passed:
            ok.add(tm.name)
        else:
            detail = (p.stdout + p.stderr).strip().splitlines()
            print(f"      reason: {(detail[-1] if detail else '(no output -> hard crash)')[:200]}",
                  flush=True)
        print(f"    {tm.name:<30} {tm.kind if passed else 'external (fallback)'}", flush=True)
    return ok


def finalize(ds: str, do_push: bool) -> None:
    family, val_ts, test_ts = CONFIG[ds]
    tasks = load_tasks(ds)
    db = fetch_canonical_db(ds)
    out = OUT_ROOT / ds
    assemble(ds, val_ts, test_ts, db, tasks, out)
    del db
    # classify() runs each task in its own subprocess; passing tasks have their
    # regenerated labels written there (isolating duckdb), so we only need to handle
    # the external fallback for the failures here.
    ok = classify(out, ds, tasks)

    for tm in tasks:
        if tm.name in ok:
            continue
        tdir = out / "tasks" / tm.name
        dataclasses.replace(tm, kind=KIND_EXTERNAL).save(tdir / "manifest.yaml")
        for split, df in fetch_canonical_labels(ds, tm.name).items():
            df.to_parquet(tdir / f"{split}.parquet", index=False)

    copy_manifests(ds, out)
    print(f"[{ds}] {len(ok)}/{len(tasks)} regenerable, {len(tasks) - len(ok)} external", flush=True)
    if do_push:
        push(ds, family, out)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_push = "--push" in sys.argv
    for ds in args:
        print(f"\n===== {ds} =====", flush=True)
        finalize(ds, do_push)


if __name__ == "__main__":
    main()
