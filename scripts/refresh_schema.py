r"""Regenerate every registered dataset's schema.svg (+ README.md) on the Hub, in place.

Reads the manifest and the parquet *footers* straight from the Hub (no full download, so
even hundred-GB datasets are cheap), rerenders the ER diagram and dataset card with the
current ``relbench.schema`` styling, and uploads only ``schema.svg`` + ``README.md`` --
parquet and manifests are untouched. Registry revisions are bumped to the new commits.

    pixi run --frozen python scripts/refresh_schema.py [name ...] [--push]

Without ``--push`` it renders locally to ``/tmp/relbench_schema/<name>.svg`` for inspection.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, snapshot_download

from relbench.manifest import DatasetManifest, TaskManifest
from relbench.schema import _short_type, dataset_card, render_schema_svg

REGISTRY = Path(__file__).resolve().parent.parent / "relbench" / "registry.json"


def _hub_reader(fs: HfFileSystem, repo_id: str, revision: str, subdir: str):
    base = f"datasets/{repo_id}/{(subdir + '/') if subdir else ''}db"

    def reader(tname: str):
        path = f"{base}/{tname}.parquet"
        try:
            with fs.open(path, "rb", revision=revision) as f:
                pf = pq.ParquetFile(f)
                cols = [(x.name, _short_type(x.type)) for x in pf.schema_arrow]
                return cols, pf.metadata.num_rows
        except FileNotFoundError:
            return None, None

    return reader


def _load_manifests(repo_id: str, revision: str, subdir: str):
    r"""Download just the (small) YAML manifests via snapshot_download and load them."""
    prefix = f"{subdir}/" if subdir else ""
    root = Path(snapshot_download(
        repo_id, repo_type="dataset", revision=revision,
        allow_patterns=[f"{prefix}manifest.yaml", f"{prefix}tasks/*/manifest.yaml"],
    ))
    ddir = root / subdir if subdir else root
    manifest = DatasetManifest.load(ddir / "manifest.yaml")
    tasks_dir = ddir / "tasks"
    tasks = []
    if tasks_dir.exists():
        for td in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
            mf = td / "manifest.yaml"
            if mf.exists():
                tasks.append(TaskManifest.load(mf))
    return manifest, tasks


def refresh(name: str, entry: dict, api: HfApi, fs: HfFileSystem, do_push: bool) -> dict:
    repo_id, revision, subdir = entry["repo_id"], entry.get("revision"), entry.get("path", name)
    manifest, tasks = _load_manifests(repo_id, revision, subdir)
    reader = _hub_reader(fs, repo_id, revision, subdir)

    out_dir = Path(tempfile.mkdtemp(prefix="relbench_schema_"))
    svg = out_dir / "schema.svg"
    render_schema_svg(manifest, svg, reader=reader)
    (out_dir / "README.md").write_text(dataset_card(manifest, tasks))

    print(f"[{name}] rendered {len(manifest.tables)} tables, {len(tasks)} tasks "
          f"-> {svg} ({svg.stat().st_size // 1024} KB)", flush=True)

    if not do_push:
        keep = Path("/tmp/relbench_schema"); keep.mkdir(exist_ok=True)
        (keep / f"{name}.svg").write_bytes(svg.read_bytes())
        return entry

    prefix = f"{subdir}/" if subdir else ""
    info = api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        path_in_repo=subdir or ".",
        repo_type="dataset",
        allow_patterns=["schema.svg", "README.md"],
        commit_message=f"{name}: refresh ER schema diagram + dataset card",
    )
    oid = getattr(info, "oid", None)
    entry = {**entry, "revision": oid}
    print(f"[{name}] pushed -> {repo_id}/{prefix} @ {oid}", flush=True)
    return entry


def main() -> None:
    do_push = "--push" in sys.argv
    reg = json.loads(REGISTRY.read_text())
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(reg)

    api = HfApi()
    fs = HfFileSystem()
    for name in names:
        try:
            reg[name] = refresh(name, reg[name], api, fs, do_push)
        except Exception as exc:  # keep going; report at the end
            print(f"[{name}] FAILED: {exc!r}", flush=True)

    if do_push:
        REGISTRY.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
        print(f"registry written: {REGISTRY}", flush=True)


if __name__ == "__main__":
    main()
