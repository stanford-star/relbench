r"""Per-task triage: for each dataset, assemble db + all task manifests, then validate
each task in its own subprocess (so a duckdb crash isolates to one task). Prints a
per-task PASS/FAIL map so we know exactly which ported SQL works.

    pixi run --frozen python build/triage.py [rel-event rel-stack ...]   # default: all
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT_ROOT, assemble, fetch_canonical_db  # noqa: E402

from port_all import CONFIG, load_tasks  # noqa: E402

PORT_TASK = str(Path(__file__).resolve().parent / "port_task.py")


def triage(ds: str) -> dict:
    _, val_ts, test_ts = CONFIG[ds]
    tasks = load_tasks(ds)
    db = fetch_canonical_db(ds)
    out = OUT_ROOT / ds
    assemble(ds, val_ts, test_ts, db, tasks, out)
    del db
    results = {}
    for tm in tasks:
        print(f"  [{ds}] {tm.name}", flush=True)
        try:
            p = subprocess.run(
                [sys.executable, PORT_TASK, str(out), ds, tm.name],
                capture_output=True, text=True, timeout=900,
                env={**os.environ, "RELBENCH_DUCKDB_MEMORY_LIMIT": "300GB"},
            )
            ok = p.returncode == 0
            sys.stdout.write(p.stdout)
            if not ok:
                tail = (p.stderr.strip().splitlines() or
                        ["(no stderr -> hard crash / segfault)"])[-1]
                print(f"    -> FAIL rc={p.returncode}: {tail[:200]}", flush=True)
        except subprocess.TimeoutExpired:
            ok = False
            print("    -> FAIL: timeout", flush=True)
        results[tm.name] = ok
    return results


def main() -> None:
    names = sys.argv[1:] or list(CONFIG)
    allres = {}
    for ds in names:
        print(f"\n===== {ds} =====", flush=True)
        allres[ds] = triage(ds)
    print("\n===== TASK MAP =====", flush=True)
    for ds in names:
        res = allres.get(ds, {})
        n_ok, n = sum(res.values()), len(res)
        fails = ",".join(k for k, v in res.items() if not v)
        print(f"  {ds:<14} {n_ok}/{n}" + (f"  FAIL: {fails}" if fails else ""), flush=True)


if __name__ == "__main__":
    main()
