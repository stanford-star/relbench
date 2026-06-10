r"""Port the remaining RelBench datasets from per-dataset JSON task specs (in /tmp/specs)
into the manifest layout, validating each against canonical hosted labels.

    pixi run --frozen python build/port_all.py [rel-amazon rel-stack ...]   # default: all

For each validated dataset, the authored manifests are copied to datasets/<name>/ (the
reviewed source). Pushing to HF is done separately (build/push_all.py) once validated.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import port  # noqa: E402

from relbench.manifest import TaskManifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = Path("/tmp/specs")
DATASETS_DIR = REPO_ROOT / "datasets"

# name -> (family, val_timestamp, test_timestamp)
CONFIG = {
    "rel-amazon": ("v1", "2015-10-01", "2016-01-01"),
    "rel-avito": ("v1", "2015-05-08", "2015-05-14"),
    "rel-event": ("v1", "2012-11-21", "2012-11-29"),
    "rel-hm": ("v1", "2020-09-07", "2020-09-14"),
    "rel-stack": ("v1", "2020-10-01", "2021-01-01"),
    "rel-trial": ("v1", "2020-01-01", "2021-01-01"),
    "rel-salt": ("v2", "2020-02-01", "2020-07-01"),
    "rel-ratebeer": ("v2", "2018-09-01", "2020-01-01"),
    "rel-arxiv": ("v2", "2022-01-01", "2023-01-01"),
}


def load_tasks(name: str) -> list[TaskManifest]:
    spec = json.loads((SPEC_DIR / f"{name}.json").read_text())
    tasks = []
    for t in spec["tasks"]:
        kw = {k: v for k, v in t.items() if v is not None}
        kw.pop("metrics", None)  # metrics now default from task_type; not stored in manifests
        kw.setdefault("num_eval_timestamps", 1)
        kw.setdefault("remove_columns", [])
        if kw.get("sql"):
            # `{timedelta}` already expands to a pandas string like "91 days";
            # normalize any redundant trailing unit some specs emitted.
            kw["sql"] = kw["sql"].replace("{timedelta} days", "{timedelta}")
        tasks.append(TaskManifest(**kw))
    return tasks


def copy_manifests(name: str, out: Path) -> None:
    dst = DATASETS_DIR / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "tasks").mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "manifest.yaml", dst / "manifest.yaml")
    for tdir in sorted((out / "tasks").iterdir()):
        (dst / "tasks" / tdir.name).mkdir(parents=True, exist_ok=True)
        shutil.copy(tdir / "manifest.yaml", dst / "tasks" / tdir.name / "manifest.yaml")


def main() -> None:
    names = sys.argv[1:] or list(CONFIG)
    results = {}
    for name in names:
        _, val_ts, test_ts = CONFIG[name]
        print(f"\n===== {name} =====", flush=True)
        try:
            tasks = load_tasks(name)
            out, ok, _rows = port(name, val_ts, test_ts, tasks)
            results[name] = ok
            if ok:
                copy_manifests(name, out)
        except Exception:
            traceback.print_exc()
            results[name] = False
    print("\n===== SUMMARY =====", flush=True)
    for n in names:
        print(f"  {n:<14} {'VALIDATED' if results.get(n) else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
