r"""Refresh already-pushed datasets to the current format, in place on the Hub:
inject descriptions, convert manifests to YAML, regenerate the schema card (README.md),
and push -- WITHOUT re-uploading the parquet (only manifests + README change).

    pixi run --frozen python build/refresh.py [rel-f1 ...] [--push]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from descriptions import (  # noqa: E402
    DATASET_DESCRIPTIONS,
    fallback_description,
    harvest_task_descriptions,
)

from relbench.manifest import DatasetManifest, TaskManifest  # noqa: E402
from relbench.schema import dataset_card, render_schema_svg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = REPO_ROOT / "datasets"
REGISTRY = REPO_ROOT / "relbench" / "registry.json"


def _load(d: Path, cls):
    p = d / "manifest.yaml"
    return cls.load(p if p.exists() else d / "manifest.json")


def refresh(name: str, do_push: bool) -> None:
    reg = json.loads(REGISTRY.read_text())
    repo_id = reg[name]["repo_id"]
    d = DATASETS / name

    m = _load(d, DatasetManifest)
    m.description = DATASET_DESCRIPTIONS.get(name, m.description)
    tdesc = harvest_task_descriptions(name)

    tasks = []
    for tdir in sorted((d / "tasks").iterdir()):
        if not tdir.is_dir():
            continue
        t = _load(tdir, TaskManifest)
        t.description = tdesc.get(t.name) or fallback_description(t)
        t.save(tdir / "manifest.yaml")
        (tdir / "manifest.json").unlink(missing_ok=True)
        tasks.append(t)

    m.save(d / "manifest.yaml")
    (d / "manifest.json").unlink(missing_ok=True)
    render_schema_svg(m, d / "schema.svg")
    (d / "README.md").write_text(dataset_card(m, tasks))
    n_desc = sum(1 for t in tasks if t.description)
    print(f"[{name}] yaml + card ready ({len(tasks)} tasks, {n_desc} described, "
          f"harvested {len(tdesc)})", flush=True)

    if do_push:
        from huggingface_hub import upload_folder

        info = upload_folder(
            folder_path=str(d), repo_id=repo_id, path_in_repo=name, repo_type="dataset",
            delete_patterns=["*.json", "tasks/*/*.json"],  # drop old JSON manifests; keep parquet
            commit_message=f"{name}: YAML manifests + descriptions + schema card",
        )
        reg[name]["revision"] = getattr(info, "oid", None)
        REGISTRY.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
        print(f"[{name}] pushed -> {repo_id}/{name} @ {reg[name]['revision']}", flush=True)


def main() -> None:
    do_push = "--push" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(
        json.loads(REGISTRY.read_text())
    )
    for name in names:
        refresh(name, do_push)


if __name__ == "__main__":
    main()
