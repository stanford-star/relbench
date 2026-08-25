r"""Declare ``papers.Primary_Category_ID`` as a foreign key to ``categories`` in the
hosted **rel-arxiv** manifest (https://github.com/stanford-star/relbench/issues/374).

    python provenance/fix_arxiv_fkey.py           # patch locally under --out, print plan
    python provenance/fix_arxiv_fkey.py --push    # also upload manifest.yaml + schema.svg

Metadata only: in the source, ``2Category.csv`` already keys categories 0..52 in file
order, so ``write_hf``'s reindexing is the identity on that column and the published
``papers.parquet`` values already point at the right rows. ``arxiv.py`` declares the same
key, so a regenerated database matches what this pushes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from clean_databases import _reader
from relbench.manifest import DatasetManifest
from relbench.schema import render_schema_svg

REPO, SUB = "stanford-star/relbench-v2-extra", "rel-arxiv"
FKEY, PKEY_TABLE = "Primary_Category_ID", "categories"


def main() -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    out = (
        Path(sys.argv[sys.argv.index("--out") + 1])
        if "--out" in sys.argv
        else Path("fix_out")
    ) / SUB
    out.mkdir(parents=True, exist_ok=True)

    manifest = DatasetManifest.load(
        hf_hub_download(REPO, f"{SUB}/manifest.yaml", repo_type="dataset")
    )
    manifest.tables["papers"].fkeys[FKEY] = PKEY_TABLE
    manifest.save(out / "manifest.yaml")
    print(f"  papers.{FKEY} -> {PKEY_TABLE}", flush=True)

    render_schema_svg(manifest, out / "schema.svg", reader=_reader(REPO, SUB, {}))
    print(f"  schema.svg refreshed -> {out / 'schema.svg'}", flush=True)

    if "--push" in sys.argv:
        HfApi().create_commit(
            repo_id=REPO,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(f"{SUB}/manifest.yaml", str(out / "manifest.yaml")),
                CommitOperationAdd(f"{SUB}/schema.svg", str(out / "schema.svg")),
            ],
            commit_message=f"rel-arxiv: declare papers.{FKEY} as a foreign key to {PKEY_TABLE} (#374)",
        )
        print(f"  pushed to {REPO}", flush=True)


if __name__ == "__main__":
    main()
