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


def _retry(fn, on_missing=None):
    import time

    from huggingface_hub.errors import HfHubHTTPError
    for attempt in range(9):
        try:
            return fn()
        except FileNotFoundError:
            return on_missing
        except HfHubHTTPError as e:
            if "429" in str(e):
                time.sleep(30 + 10 * attempt)
                continue
            raise
        except Exception:
            if attempt == 8:
                raise
            time.sleep(5)
    return on_missing


def _hub_reader(fs: HfFileSystem, repo_id: str, revision: str, subdir: str):
    base = f"datasets/{repo_id}/{(subdir + '/') if subdir else ''}db"

    def reader(tname: str):
        path = f"{base}/{tname}.parquet"

        def go():
            with fs.open(path, "rb", revision=revision) as f:
                pf = pq.ParquetFile(f)
                cols = [(x.name, _short_type(x.type)) for x in pf.schema_arrow]
                return cols, pf.metadata.num_rows

        return _retry(go, on_missing=(None, None))

    return reader


def _load_manifests(repo_id: str, revision: str, subdir: str):
    r"""Download just the (small) YAML manifests via snapshot_download and load them."""
    prefix = f"{subdir}/" if subdir else ""
    root = Path(_retry(lambda: snapshot_download(
        repo_id, repo_type="dataset", revision=revision,
        allow_patterns=[f"{prefix}manifest.yaml", f"{prefix}tasks/*/manifest.yaml"],
    )))
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


def render_into(name: str, entry: dict, fs: HfFileSystem, out_root: Path) -> str:
    r"""Render schema.svg + README.md for one dataset into ``out_root/<subdir>/``.

    Returns the subdir (so the caller can build allow_patterns for a single per-repo commit).
    """
    repo_id, revision, subdir = entry["repo_id"], entry.get("revision"), entry.get("path", name)
    manifest, tasks = _load_manifests(repo_id, revision, subdir)
    reader = _hub_reader(fs, repo_id, revision, subdir)
    d = out_root / subdir if subdir else out_root
    d.mkdir(parents=True, exist_ok=True)
    render_schema_svg(manifest, d / "schema.svg", reader=reader)
    (d / "README.md").write_text(dataset_card(manifest, tasks))
    print(f"[{name}] rendered {len(manifest.tables)} tables, {len(tasks)} tasks", flush=True)
    return subdir


def main() -> None:
    r"""Render every requested dataset and push each *repo* in a SINGLE commit.

    HF rate-limits commits to 128/hour/repo, so committing per-dataset blows the budget on
    big family repos (redelex, dbinfer, ...). Grouping by repo means one commit per repo.
    """
    do_push = "--push" in sys.argv
    reg = json.loads(REGISTRY.read_text())
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(reg)

    by_repo: dict[str, list] = {}
    for name in names:
        by_repo.setdefault(reg[name]["repo_id"], []).append(name)

    api = HfApi()
    fs = HfFileSystem()
    for repo_id, repo_names in by_repo.items():
        out_root = Path(tempfile.mkdtemp(prefix="relbench_schema_"))
        subdirs = []
        for name in repo_names:
            try:
                subdirs.append((name, render_into(name, reg[name], fs, out_root)))
            except Exception as exc:
                print(f"[{name}] FAILED: {exc!r}", flush=True)
        if not do_push or not subdirs:
            continue
        patterns = []
        for _, sub in subdirs:
            pre = f"{sub}/" if sub else ""
            patterns += [f"{pre}schema.svg", f"{pre}README.md"]
        info = _retry(lambda: api.upload_folder(
            folder_path=str(out_root), repo_id=repo_id, repo_type="dataset",
            allow_patterns=patterns,
            commit_message=f"Refresh ER schema diagrams ({len(subdirs)} datasets)",
        ))
        oid = getattr(info, "oid", None)
        for name, _ in subdirs:
            reg[name] = {**reg[name], "revision": oid}
        print(f"[{repo_id}] pushed {len(subdirs)} datasets @ {oid}", flush=True)

    if do_push:
        REGISTRY.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
        print(f"registry written: {REGISTRY}", flush=True)


if __name__ == "__main__":
    main()
