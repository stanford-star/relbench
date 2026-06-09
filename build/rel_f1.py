r"""Build the ``rel-f1`` RelBench dataset folder in the Hugging Face manifest layout.

Outputs (default ``/tmp/relbench-build/rel-f1``)::

    manifest.json
    db/<table>.parquet           # plain parquet, no RelBench metadata
    tasks/<task>/manifest.json   # authored task specs (incl. SQL for kind="sql")

The database is the canonical processed ``rel-f1`` (currently hosted as ``db.zip``),
repackaged as plain parquet -- datasets are the root of trust, so we don't re-derive
from raw here (a separate raw->db provenance script can replace the source step).

The authored task manifests are also written back to ``datasets/rel-f1/`` in the repo,
which is the reviewed source for the manifests.

    pixi run --frozen python build/rel_f1.py [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from relbench.manifest import DatasetManifest, TableSpec, TaskManifest

NAME = "rel-f1"
VAL_TS = "2005-01-01"
TEST_TS = "2010-01-01"

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "datasets" / NAME  # reviewed source for manifests

# --------------------------------------------------------------------------- #
# Task manifests (SQL ported verbatim from relbench/tasks/f1.py:
#   timestamp_df -> timestamps, {self.timedelta} -> {timedelta}; top3 cast folded in)
# --------------------------------------------------------------------------- #

TASKS: list[TaskManifest] = [
    TaskManifest(
        name="driver-position",
        kind="forecast",
        task_type="regression",
        entity_table="drivers",
        entity_col="driverId",
        target_col="position",
        time_col="date",
        timedelta="60 days",
        num_eval_timestamps=40,
        metrics=["r2", "mae", "rmse"],
        sql="""
SELECT
    t.timestamp as date,
    re.driverId as driverId,
    mean(re.positionOrder) as position,
FROM
    timestamps t
LEFT JOIN
    results re
ON
    re.date <= t.timestamp + INTERVAL '{timedelta}'
    and re.date  > t.timestamp
WHERE
    re.driverId IN (
        SELECT DISTINCT driverId
        FROM results
        WHERE date > t.timestamp - INTERVAL '1 year'
    )
GROUP BY t.timestamp, re.driverId
""".strip(),
    ),
    TaskManifest(
        name="driver-dnf",
        kind="forecast",
        task_type="binary_classification",
        entity_table="drivers",
        entity_col="driverId",
        target_col="did_not_finish",
        time_col="date",
        timedelta="30 days",
        num_eval_timestamps=40,
        metrics=["average_precision", "accuracy", "f1", "roc_auc"],
        sql="""
SELECT
    t.timestamp as date,
    re.driverId as driverId,
    MAX(CASE WHEN re.statusId != 1 THEN 1 ELSE 0 END) AS did_not_finish
FROM
    timestamps t
LEFT JOIN
    results re
ON
    re.date <= t.timestamp + INTERVAL '{timedelta}'
    and re.date  > t.timestamp
WHERE
    re.driverId IN (
        SELECT DISTINCT driverId
        FROM results
        WHERE date > t.timestamp - INTERVAL '1 year'
    )
GROUP BY t.timestamp, re.driverId
""".strip(),
    ),
    TaskManifest(
        name="driver-top3",
        kind="forecast",
        task_type="binary_classification",
        entity_table="drivers",
        entity_col="driverId",
        target_col="qualifying",
        time_col="date",
        timedelta="30 days",
        num_eval_timestamps=40,
        metrics=["average_precision", "accuracy", "f1", "roc_auc"],
        sql="""
SELECT
    t.timestamp as date,
    qu.driverId as driverId,
    CAST(
        CASE WHEN MIN(qu.position) <= 3 THEN 1 ELSE 0 END AS BIGINT
    ) AS qualifying
FROM
    timestamps t
LEFT JOIN
    qualifying qu
ON
    qu.date <= t.timestamp + INTERVAL '{timedelta}'
    and qu.date > t.timestamp
WHERE
    qu.driverId IN (
        SELECT DISTINCT driverId
        FROM qualifying
        WHERE date > t.timestamp - INTERVAL '1 year'
    )
GROUP BY t.timestamp, qu.driverId
""".strip(),
    ),
    TaskManifest(
        name="driver-circuit-compete",
        kind="forecast",
        task_type="link_prediction",
        src_entity_table="drivers",
        src_entity_col="driverId",
        dst_entity_table="circuits",
        dst_entity_col="circuitId",
        target_col="circuitId",
        time_col="date",
        timedelta="365 days",
        eval_k=10,
        metrics=[
            "link_prediction_precision",
            "link_prediction_recall",
            "link_prediction_map",
        ],
        sql="""
SELECT
    t.timestamp as date,
    re.driverId as driverId,
    LIST(DISTINCT race.circuitId) as circuitId
FROM
    timestamps t
LEFT JOIN
    races race
ON
    race.date <= t.timestamp + INTERVAL '{timedelta}'
    and race.date > t.timestamp
LEFT JOIN
    results re
ON
    re.raceId = race.raceId
GROUP BY t.timestamp, re.driverId
""".strip(),
    ),
    TaskManifest(
        name="results-position",
        kind="autocomplete",
        task_type="regression",
        entity_table="results",
        target_col="position",
        remove_columns=[
            ["results", "statusId"],
            ["results", "positionOrder"],
            ["results", "points"],
            ["results", "laps"],
            ["results", "milliseconds"],
            ["results", "fastestLap"],
            ["results", "rank"],
        ],
    ),
    TaskManifest(
        name="qualifying-position",
        kind="autocomplete",
        task_type="regression",
        entity_table="qualifying",
        target_col="position",
        remove_columns=[],
    ),
]


def author_task_manifests() -> None:
    for tm in TASKS:
        tm.validate()
        tm.save(SRC_DIR / "tasks" / tm.name / "manifest.json")
    print(f"Authored {len(TASKS)} task manifests -> {SRC_DIR / 'tasks'}")


def load_canonical_db():
    r"""Load the canonical processed rel-f1 database via the legacy download path."""
    from relbench.datasets import get_dataset

    ds = get_dataset(NAME, download=True)
    return ds.get_db(upto_test_timestamp=False)  # full db, not time-sliced


def build_dataset_manifest(db) -> DatasetManifest:
    tables = {
        name: TableSpec(
            pkey=t.pkey_col,
            time_col=t.time_col,
            fkeys=dict(t.fkey_col_to_pkey_table),
        )
        for name, t in db.table_dict.items()
    }
    manifest = DatasetManifest(
        name=NAME, val_timestamp=VAL_TS, test_timestamp=TEST_TS, tables=tables
    )
    manifest.save(SRC_DIR / "manifest.json")
    print(f"Wrote dataset manifest ({len(tables)} tables) -> {SRC_DIR / 'manifest.json'}")
    return manifest


def assemble(out: Path, db, manifest: DatasetManifest) -> None:
    import shutil

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "db").mkdir(parents=True, exist_ok=True)

    for name, t in db.table_dict.items():
        # Plain parquet: native column dtypes only, no RelBench metadata.
        t.df.to_parquet(out / "db" / f"{name}.parquet", index=False)

    manifest.save(out / "manifest.json")
    for tm in TASKS:
        dst = out / "tasks" / tm.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC_DIR / "tasks" / tm.name / "manifest.json", dst / "manifest.json")

    from relbench.manifest import validate_dataset_manifest

    validate_dataset_manifest(manifest, out / "db")
    print(f"Assembled HF-ready folder -> {out}")
    print(f"  db tables: {len(db.table_dict)} | tasks: {len(TASKS)}")


def generate_labels(out: Path) -> None:
    r"""Regenerate + cache task labels through the new loader (forecast via SQL,
    autocomplete via the efficient window generator) into the artifact."""
    from relbench.load import load_task

    for tm in TASKS:
        task = load_task(out, tm.name, regenerate=True)
        for split in ["train", "val", "test"]:
            tbl = task.get_table(split, mask_input_cols=False)
            tbl.df.to_parquet(out / "tasks" / tm.name / f"{split}.parquet", index=False)
        print(f"  labels cached: {tm.name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/relbench-build/rel-f1")
    args = p.parse_args()

    author_task_manifests()
    db = load_canonical_db()
    manifest = build_dataset_manifest(db)
    out = Path(args.out)
    assemble(out, db, manifest)
    generate_labels(out)


if __name__ == "__main__":
    main()
