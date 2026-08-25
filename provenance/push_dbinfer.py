r"""Publish a locally generated ``dbinfer`` collection to its Hugging Face repo.

    python provenance/push_dbinfer.py BUILD_DIR                 # dry run: show the plan
    python provenance/push_dbinfer.py BUILD_DIR --push          # upload
    python provenance/push_dbinfer.py BUILD_DIR --push --repo stanford-star/dbinfer
    python provenance/push_dbinfer.py BUILD_DIR --push --cards-only  # just the READMEs
    python provenance/push_dbinfer.py BUILD_DIR --push --message "Refresh dbinfer-mag"

``BUILD_DIR`` is the output of ``provenance/dbinfer.py --all`` (a directory of
``dbinfer-<name>/`` folders), optionally also holding the repo-level ``README.md`` and a
``STATS/`` folder.

The upload is a single commit (``--message``, default "Update dbinfer datasets") that adds
or overwrites the files present in ``BUILD_DIR``. Nothing is deleted: a table or diagram
that a regenerate dropped stays in the repo until it is removed explicitly. ``STATS/`` and
the repo ``README.md`` are overwritten when present in ``BUILD_DIR`` and left untouched
otherwise.

``--cards-only`` uploads just the dataset cards and the repo card -- for refreshing prose
without re-pushing gigabytes of unchanged parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_REPO = "stanford-star/dbinfer"
DEFAULT_MESSAGE = "Update dbinfer datasets"


def plan(build: Path) -> tuple[list, list]:
    datasets = sorted(p.parent for p in build.glob("dbinfer-*/manifest.yaml"))
    if not datasets:
        sys.exit(f"no dbinfer-*/manifest.yaml under {build}")
    extras = [p for p in (build / "README.md", build / "STATS") if p.exists()]
    return datasets, extras


def main(argv: list) -> int:
    if not argv:
        sys.exit(__doc__)
    build = Path(argv[0])
    do_push = "--push" in argv
    cards_only = "--cards-only" in argv
    repo = DEFAULT_REPO
    if "--repo" in argv:
        repo = argv[argv.index("--repo") + 1]
    message = DEFAULT_MESSAGE
    if "--message" in argv:
        message = argv[argv.index("--message") + 1]

    datasets, extras = plan(build)
    total = 0
    print(f"repo: {repo}\nbuild: {build}\nmessage: {message}\n")
    if cards_only:
        cards = [d / "README.md" for d in datasets if (d / "README.md").exists()]
        if (build / "README.md").exists():
            cards.append(build / "README.md")
        for p in cards:
            print(f"  {p.relative_to(build)}")
        print(f"\n  {len(cards)} card(s)")
    else:
        for d in datasets:
            files = [p for p in d.rglob("*") if p.is_file()]
            size = sum(p.stat().st_size for p in files)
            total += size
            print(f"  {d.name:26} {len(files):>4} files  {size / 2**30:>8.3f} GiB")
        for p in extras:
            print(f"  {p.name:26} {'(dir)' if p.is_dir() else '(file)'}")
        print(f"\n  total {total / 2**30:.3f} GiB")

    if not do_push:
        print("\ndry run -- pass --push to upload")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    if cards_only:
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(build),
            allow_patterns=["dbinfer-*/README.md", "README.md"],
            commit_message=message,
        )
        print(f"\npushed cards to https://huggingface.co/datasets/{repo}")
        return 0
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(build),
        allow_patterns=["dbinfer-*/**", "STATS/**", "README.md"],
        commit_message=message,
    )
    print(f"\npushed to https://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
