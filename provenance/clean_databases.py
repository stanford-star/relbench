r"""Clean published RelBench databases: drop noise / leakage columns and refresh schemas.

This is the reproducible record (provenance) of a one-off cleanup applied to the hosted
databases in response to https://github.com/snap-stanford/relbench/issues/373. It removes,
per the categories below, only columns that are *not* used by any task and are *not*
keys/time columns -- so labels and the foreign-key graph are untouched (verify with
``python provenance/check_provenance.py <dataset>``). It then re-renders each affected
dataset's ``schema.svg`` so the diagram matches the data.

    python provenance/clean_databases.py                 # clean locally under --out, print plan
    python provenance/clean_databases.py --push          # also upload cleaned db/ + schema.svg
    python provenance/clean_databases.py rel-trial --push # restrict to one dataset

Only manifests + parquet footers are read for the schema diagram; affected table parquets
are streamed (constant memory) to drop columns while preserving row order.

Why each column goes (see DROP below):
  * 100%-NaN      -- entirely null, pure schema noise.
  * artifact      -- preprocessing leftovers (pandas ``Unnamed: 0`` row index; SQL-Server
                     replication ``msrepl_tran_version``); carry no information.
  * leakage       -- rel-ratebeer scrape-time, full-history aggregates (counts / averages /
                     min-max / percentiles / first-last timestamps / ``updated_at``) that
                     summarise an entity's entire rating history and so can encode the
                     future relative to a row's timestamp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from relbench.manifest import DatasetManifest
from relbench.schema import _short_type, render_schema_svg

# repo -> dataset -> {table: {column: reason}}
DROP: dict[str, dict[str, dict[str, dict[str, str]]]] = {
    "stanford-star/relbench": {
        "rel-stack": {"users": {"ProfileImageUrl": "100%-NaN"}},
        "rel-trial": {
            "outcome_analyses": {
                "p_value_raw": "100%-NaN",
                "ci_lower_limit_raw": "100%-NaN",
                "ci_upper_limit_raw": "100%-NaN",
            },
            "studies": {"limitations_and_caveats": "100%-NaN"},
        },
        "rel-event": {
            "event_attendees": {"Unnamed: 0": "artifact"},
            "user_friends": {"Unnamed: 0": "artifact"},
        },
    },
    "stanford-star/relbench-v2-extra": {
        "rel-ratebeer": {
            "users": {
                c: "leakage"
                for c in [
                    "beer_first_rating",
                    "beer_last_rating",
                    "beer_rating_count",
                    "avg_beer_rating",
                    "max_beer_rating",
                    "min_beer_rating",
                    "place_first_rating",
                    "place_last_rating",
                    "place_rating_count",
                    "avg_place_rating",
                    "max_place_rating",
                    "min_place_rating",
                    "favorite_count",
                    "total_activity_count",
                    "updated_at",
                ]
            },
            "places": {
                c: "leakage"
                for c in [
                    "avg_rating",
                    "weighted_avg",
                    "bay_mean",
                    "percentile",
                    "rating_count",
                    "valid_rating_count",
                    "score",
                    "rating_text",
                    "updated_at",
                ]
            },
            "beers": {
                c: "leakage"
                for c in [
                    "view_count",
                    "avg_rating",
                    "rating_count",
                    "real_avg_rating",
                    "rating_std_dev",
                    "overall_percentile",
                    "style_percentile",
                    "last_9m_avg",
                    "last_9m_count",
                    "straight_avg_rating",
                    "straight_rating_count",
                    "year4_avg",
                    "year4_overall",
                    "year4_style",
                    "year4_count",
                    "updated_at",
                ]
            },
            "brewers": {
                "view_count": "leakage",
                "score": "leakage",
                "updated_at": "leakage",
                "msrepl_tran_version": "artifact",
            },
            "beer_ratings": {"updated_at": "leakage"},
            "place_ratings": {"updated_at": "leakage"},
            "states": {"msrepl_tran_version": "artifact"},
        },
    },
}


def _hf_meta(repo: str, rel: str):
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{repo}/{rel}") as f:
        return pq.read_metadata(f)


def clean_parquet(src: Path, dst: Path, drop: set[str]) -> tuple[int, int]:
    r"""Stream ``src`` to ``dst`` without ``drop`` columns (preserves row order)."""
    pf = pq.ParquetFile(src)
    names = pf.schema_arrow.names
    keep = [n for n in names if n not in drop]
    schema = pa.schema([pf.schema_arrow.field(n) for n in keep])
    with pq.ParquetWriter(dst, schema) as w:
        for batch in pf.iter_batches(columns=keep, batch_size=200_000):
            w.write_batch(batch)
    return len(names), len(keep)


def _reader(repo: str, sub: str, drops: dict[str, dict]):
    r"""schema.svg reader: columns + row count from the Hub footer, with drops
    applied."""

    def reader(tname: str):
        try:
            md = _hf_meta(repo, f"{sub}/db/{tname}.parquet")
        except FileNotFoundError:
            return None, None
        drop = set(drops.get(tname, {}))
        cols = [
            (f.name, _short_type(f.type))
            for f in md.schema.to_arrow_schema()
            if f.name not in drop
        ]
        return cols, md.num_rows

    return reader


def main() -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    do_push = "--push" in sys.argv
    out = (
        Path(sys.argv[sys.argv.index("--out") + 1])
        if "--out" in sys.argv
        else Path("clean_out")
    )
    api = HfApi()

    for repo, datasets in DROP.items():
        for sub, tables in datasets.items():
            if args and sub not in args:
                continue
            print(f"\n=== {repo}/{sub} ===", flush=True)
            dsout = out / sub
            (dsout / "db").mkdir(parents=True, exist_ok=True)
            ops = []
            for table, cols in tables.items():
                src = Path(
                    hf_hub_download(
                        repo, f"{sub}/db/{table}.parquet", repo_type="dataset"
                    )
                )
                dst = dsout / "db" / f"{table}.parquet"
                before, after = clean_parquet(src, dst, set(cols))
                reasons = sorted(set(cols.values()))
                print(
                    f"  {table}: {before} -> {after} cols (dropped {len(cols)}: {reasons})",
                    flush=True,
                )
                ops.append(CommitOperationAdd(f"{sub}/db/{table}.parquet", str(dst)))
            # refresh schema.svg from Hub footers (with drops applied)
            mpath = hf_hub_download(repo, f"{sub}/manifest.yaml", repo_type="dataset")
            svg = dsout / "schema.svg"
            render_schema_svg(
                DatasetManifest.load(mpath), svg, reader=_reader(repo, sub, tables)
            )
            ops.append(CommitOperationAdd(f"{sub}/schema.svg", str(svg)))
            print(f"  schema.svg refreshed -> {svg}", flush=True)
            if do_push:
                api.create_commit(
                    repo_id=repo,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"{sub}: drop NaN/leakage/artifact columns (#373); refresh schema",
                )
                print(f"  pushed {len(ops)} files to {repo}", flush=True)


if __name__ == "__main__":
    main()
