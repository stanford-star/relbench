r"""Shared tooling to port a legacy RelBench dataset into the HF manifest layout.

For a dataset, this:
  1. fetches the canonical processed database (and task labels) from relbench.stanford.edu,
  2. repackages the db as plain parquet + a `manifest.yaml` (keys/fkeys/time from the
     db's own metadata), writes the authored task manifests, and regenerates labels
     through the new loader (forecast SQL / autocomplete),
  3. validates that regenerated labels match the canonical hosted labels (the provenance
     safety net -- catches any SQL porting error),
  4. (separately) uploads the folder into its family repo `relbench/<family>/<name>/` and
     pins the commit in registry.json.

pooch-free: canonical artifacts are fetched with urllib.
"""

from __future__ import annotations

import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from relbench.base import Database, RecommendationTask
from relbench.manifest import (
    DatasetManifest,
    TableSpec,
    TaskManifest,
    validate_dataset_manifest,
)

BASE_URL = "https://relbench.stanford.edu/download"
OUT_ROOT = Path("/tmp/relbench-build")
REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "relbench" / "registry.json"
SPLITS = ["train", "val", "test"]


def _download_unzip(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    zip_path = dest / "_download.zip"
    # Stream to disk (datasets can be multi-GB; don't buffer the whole zip in memory).
    with urllib.request.urlopen(url) as resp, open(zip_path, "wb") as f:
        shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    zip_path.unlink()
    return dest


def fetch_canonical_db(name: str) -> Database:
    d = OUT_ROOT / "_canonical" / name / "db"
    if not (d / ".done").exists():
        _download_unzip(f"{BASE_URL}/{name}/db.zip", d)
        (d / ".done").touch()
    db_dir = next(iter(d.rglob("*.parquet"))).parent
    return Database.load(db_dir)


def fetch_canonical_labels(name: str, task: str) -> dict:
    d = OUT_ROOT / "_canonical" / name / "tasks" / task
    if not (d / ".done").exists():
        _download_unzip(f"{BASE_URL}/{name}/tasks/{task}.zip", d)
        (d / ".done").touch()
    out = {}
    for split in SPLITS:
        matches = list(d.rglob(f"{split}.parquet"))
        if matches:
            out[split] = pd.read_parquet(matches[0])
    return out


def assemble(name: str, val_ts: str, test_ts: str, db: Database,
             tasks: list[TaskManifest], out: Path) -> DatasetManifest:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "db").mkdir(parents=True)
    for tname, table in db.table_dict.items():
        table.df.to_parquet(out / "db" / f"{tname}.parquet", index=False)
    tables = {
        tname: TableSpec(pkey=t.pkey_col, time_col=t.time_col,
                         fkeys=dict(t.fkey_col_to_pkey_table))
        for tname, t in db.table_dict.items()
    }
    manifest = DatasetManifest(name=name, val_timestamp=val_ts,
                               test_timestamp=test_ts, tables=tables)
    manifest.save(out / "manifest.yaml")
    for tm in tasks:
        tm.validate()
        tm.save(out / "tasks" / tm.name / "manifest.yaml")
    validate_dataset_manifest(manifest, out / "db")
    return manifest


def generate_labels(out: Path, tasks: list[TaskManifest]) -> None:
    from relbench.load import load_task

    for tm in tasks:
        task = load_task(str(out), tm.name, regenerate=True)
        for split in SPLITS:
            df = task.get_table(split, mask_input_cols=False).df
            df.to_parquet(out / "tasks" / tm.name / f"{split}.parquet", index=False)


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


def validate(out: Path, name: str, tasks: list[TaskManifest]) -> list[tuple]:
    r"""Return per-(task, split) (task, split, n_regen, n_hosted, ok) rows."""
    from relbench.load import load_task

    rows = []
    for tm in tasks:
        regen = load_task(str(out), tm.name, regenerate=True)
        keys, target, is_list = _keys_target(regen)
        hosted = fetch_canonical_labels(name, tm.name)
        for split in SPLITS:
            r = regen.get_table(split, mask_input_cols=False).df
            ok = split in hosted and _match(r, hosted[split], keys, target, is_list)
            rows.append((tm.name, split, len(r), len(hosted.get(split, [])), ok))
    return rows


def port(name: str, val_ts: str, test_ts: str, tasks: list[TaskManifest]):
    r"""Build the artifact + validate against canonical labels. Returns (out_dir, ok, rows)."""
    out = OUT_ROOT / name
    db = fetch_canonical_db(name)
    assemble(name, val_ts, test_ts, db, tasks, out)
    generate_labels(out, tasks)
    rows = validate(out, name, tasks)
    ok = all(r[4] for r in rows)
    width = max((len(r[0]) for r in rows), default=10)
    for tn, split, nr, nh, row_ok in rows:
        print(f"  {tn:<{width}} {split:<5} regen={nr:>7} hosted={nh:>7} "
              f"{'ok' if row_ok else 'MISMATCH'}", flush=True)
    print(f"[{name}] {'VALIDATED' if ok else 'FAILED'} "
          f"({sum(r[4] for r in rows)}/{len(rows)} checks)", flush=True)
    return out, ok, rows


def push(name: str, family: str, out: Path) -> str:
    r"""Upload out/ into relbench/<family>/<name>/ and pin the revision in registry.json."""
    from huggingface_hub import create_repo, upload_folder

    repo_id = f"relbench/{family}"
    create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = upload_folder(
        folder_path=str(out), repo_id=repo_id, path_in_repo=name,
        repo_type="dataset", commit_message=f"Add {name} (manifest layout)",
    )
    sha = getattr(info, "oid", None)
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    registry[name] = {"repo_id": repo_id, "revision": sha, "path": name}
    REGISTRY.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"[{name}] pushed -> {repo_id}/{name} @ {sha}", flush=True)
    return sha
