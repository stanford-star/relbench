r"""Mirror the externally hosted raw sources into ``stanford-star/relbench-raw``.

    python provenance/mirror_raw.py            # download + verify, print the plan
    python provenance/mirror_raw.py --push     # also upload to the Hub

Each generator that fetches from a third-party host lists its mirror URL first and the
original link second (``URLS``); this downloads from the original, checks the pinned
sha256, and uploads the archive under ``<dataset>/db.zip`` so the mirror URL resolves.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arxiv  # noqa: E402
import ratebeer  # noqa: E402
from _lib import CACHE  # noqa: E402

REPO = "stanford-star/relbench-raw"
SOURCES = {
    "rel-arxiv": (arxiv.URLS[-1], arxiv.SHA),
    "rel-ratebeer": (ratebeer.URLS[-1], ratebeer.SHA),
}


def download(url: str, dest: Path, sha256: str) -> Path:
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tmp)
        if not zipfile.is_zipfile(tmp):
            tmp.unlink()
            raise RuntimeError(f"{url} did not return a zip archive")
        tmp.rename(dest)
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    if got != sha256:
        raise ValueError(f"sha256 mismatch for {dest}: got {got}, want {sha256}")
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="upload (default: dry run)")
    parser.add_argument("--dataset", nargs="+", default=sorted(SOURCES))
    args = parser.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in args.dataset:
        url, sha = SOURCES[name]
        blob = download(url, CACHE / f"{name}-db.zip", sha)
        size = blob.stat().st_size / 2**20
        print(f"{name}: {blob} ({size:.0f} MiB) -> {REPO}/{name}/db.zip", flush=True)
        if args.push:
            from huggingface_hub import HfApi

            HfApi().upload_file(
                path_or_fileobj=str(blob),
                path_in_repo=f"{name}/db.zip",
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"Mirror the {name} raw source",
            )
            print(f"  pushed {name}/db.zip", flush=True)
    if not args.push:
        print("dry run -- pass --push to upload", flush=True)


if __name__ == "__main__":
    main()
