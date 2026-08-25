r"""Sort the hosted tables that have a time column but no primary key by time.

    python byod/sort_hosted_tables.py            # scan both RelBench repos, print the plan
    python byod/sort_hosted_tables.py --push     # upload the sorted tables

A table without a primary key is keyed by row position, which only survives the time
cuts if rows after ``test_timestamp`` come last. Tables with a primary key already
satisfy that (``get_db`` checks their keys stay consecutive); this brings the rest in line.
"""

import argparse
import tempfile
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from relbench.base import is_time_sorted
from relbench.hf import RELBENCH_REPOS
from relbench.manifest import DatasetManifest


def candidates(api: HfApi, repo: str) -> list:
    files = api.list_repo_files(repo, repo_type="dataset")
    out = []
    for f in files:
        if f.count("/") != 1 or not f.endswith("/manifest.yaml"):
            continue
        name = f.split("/")[0]
        dm = DatasetManifest.load(hf_hub_download(repo, f, repo_type="dataset"))
        for table, spec in dm.tables.items():
            if spec.pkey is None and spec.time_col is not None:
                out.append((name, table, spec.time_col))
    return out


def sort_table(repo: str, name: str, table: str, time_col: str, mirror: Path):
    path = hf_hub_download(repo, f"{name}/db/{table}.parquet", repo_type="dataset")
    times = pq.read_table(path, columns=[time_col]).column(0).to_pandas()
    if is_time_sorted(times):
        return None
    data = pq.read_table(path)
    order = pc.sort_indices(
        data.select([time_col]),
        sort_keys=[(time_col, "ascending")],
        null_placement="at_end",
    )
    out = mirror / name / "db" / f"{table}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data.take(order), out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", nargs="+", default=list(RELBENCH_REPOS))
    parser.add_argument("--push", action="store_true", help="upload (default: dry run)")
    args = parser.parse_args(argv)
    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp)
        for repo in args.repo:
            ops = []
            for name, table, time_col in candidates(api, repo):
                out = sort_table(repo, name, table, time_col, mirror)
                status = "sorted -> upload" if out else "already sorted"
                print(
                    f"{repo}/{name}/db/{table}.parquet ({time_col}): {status}",
                    flush=True,
                )
                if out is not None:
                    ops.append(
                        CommitOperationAdd(f"{name}/db/{table}.parquet", str(out))
                    )
            if ops and args.push:
                api.create_commit(
                    repo_id=repo,
                    repo_type="dataset",
                    operations=ops,
                    commit_message="Sort tables without a primary key by their time column",
                )
                print(f"pushed {len(ops)} tables to {repo}", flush=True)
            elif ops:
                print(f"{len(ops)} tables to upload to {repo} (dry run)", flush=True)


if __name__ == "__main__":
    main()
