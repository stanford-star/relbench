from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from relbench.manifest import (
    KIND_AUTOCOMPLETE,
    KIND_EXTERNAL,
    KIND_FORECAST,
    DatasetManifest,
    TaskManifest,
)

ERROR = "error"
WARNING = "warning"
SPLITS = ("train", "val", "test")
REINDEX_HINT = "run `python byod/reindex_dataset.py <dataset_dir>`"


@dataclass(frozen=True)
class Issue:
    level: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():<7} {self.where}: {self.message}"


def _id(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def _scan(path: Path, row_numbers: bool = False) -> str:
    arg = ", file_row_number = true" if row_numbers else ""
    return "read_parquet('" + str(path).replace("'", "''") + "'" + arg + ")"


def _is_str(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def _compatible(a: pa.DataType, b: pa.DataType) -> bool:
    return (
        a == b
        or (pa.types.is_integer(a) and pa.types.is_integer(b))
        or (_is_str(a) and _is_str(b))
    )


def _is_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t)


def _fmt(values) -> str:
    return ", ".join(repr(v) for v in values)


class _Validator:
    def __init__(self, root: Path, key_contract: bool, max_examples: int):
        self.root = root
        self.key_contract = key_contract
        self.k = max_examples
        self.issues: list[Issue] = []
        self.con = duckdb.connect()
        self.manifest: Optional[DatasetManifest] = None
        self.schemas: dict[str, pa.Schema] = {}
        self.rows: dict[str, int] = {}
        self.time_ranges: dict[str, tuple] = {}
        self.val_ts = self.test_ts = self.data_max = None

    def error(self, where: str, message: str) -> None:
        self.issues.append(Issue(ERROR, where, message))

    def warn(self, where: str, message: str) -> None:
        self.issues.append(Issue(WARNING, where, message))

    def q(self, sql: str) -> list:
        return self.con.execute(sql).fetchall()

    def path(self, table: str) -> Path:
        return self.root / "db" / f"{table}.parquet"

    def has(self, table: str, col: Optional[str]) -> bool:
        return (
            col is not None
            and table in self.schemas
            and col in self.schemas[table].names
        )

    def run(self, tasks: bool) -> list[Issue]:
        if self.load_manifest():
            tables = self.manifest.tables
            for name, spec in tables.items():
                self.check_schema(name, spec)
            self.check_stray_tables()
            for name, spec in tables.items():
                self.check_pkey(name, spec)
            for name, spec in tables.items():
                self.check_fkeys(name, spec)
            for name, spec in tables.items():
                self.check_time(name, spec)
            self.check_split_timestamps()
            if tasks:
                self.check_tasks()
        return self.issues

    def load_manifest(self) -> bool:
        path = self.root / "manifest.yaml"
        if not path.exists():
            self.error("manifest.yaml", f"not found in {self.root}")
            return False
        try:
            self.manifest = DatasetManifest.load(path)
        except (yaml.YAMLError, KeyError, TypeError, AttributeError) as e:
            self.error("manifest.yaml", f"cannot be parsed: {e!r}")
            return False
        self.val_ts = self.manifest_timestamp("val_timestamp")
        self.test_ts = self.manifest_timestamp("test_timestamp")
        if (
            self.val_ts is not None
            and self.test_ts is not None
            and self.val_ts >= self.test_ts
        ):
            self.error(
                "manifest.yaml",
                f"val_timestamp {self.val_ts} is not before test_timestamp "
                f"{self.test_ts}",
            )
        return True

    def manifest_timestamp(self, key: str) -> Optional[pd.Timestamp]:
        value = getattr(self.manifest, key)
        try:
            ts = pd.Timestamp(value)
        except (ValueError, TypeError):
            self.error("manifest.yaml", f"{key} {value!r} is not a parseable timestamp")
            return None
        if ts.tzinfo is not None:
            self.error("manifest.yaml", f"{key} {value!r} must be timezone-naive")
            return None
        return ts

    def check_schema(self, name: str, spec) -> None:
        where = f"table '{name}'"
        path = self.path(name)
        if not path.exists():
            self.error(where, f"missing parquet db/{name}.parquet")
            return
        try:
            meta = pq.ParquetFile(path)
        except (pa.ArrowInvalid, OSError) as e:
            self.error(where, f"unreadable parquet db/{name}.parquet: {e}")
            return
        self.schemas[name] = meta.schema_arrow
        self.rows[name] = meta.metadata.num_rows
        named = [("pkey", spec.pkey), ("time_col", spec.time_col)]
        named += [("fkey", c) for c in spec.fkeys]
        for role, col in named:
            if col is not None and not self.has(name, col):
                self.error(
                    where,
                    f"{role} column '{col}' not in parquet "
                    f"(columns: {self.schemas[name].names})",
                )

    def check_stray_tables(self) -> None:
        db = self.root / "db"
        if not db.is_dir():
            return
        for path in sorted(db.glob("*.parquet")):
            if path.stem not in self.manifest.tables:
                self.warn(
                    f"table '{path.stem}'",
                    f"db/{path.name} is not listed in manifest.yaml and will be ignored",
                )

    def check_pkey(self, name: str, spec) -> None:
        if not self.has(name, spec.pkey):
            return
        where = f"{name}.{spec.pkey}"
        scan, k = _scan(self.path(name)), _id(spec.pkey)
        [(nulls, dups)] = self.q(
            f"SELECT count(*) - count({k}), count({k}) - count(DISTINCT {k}) "
            f"FROM {scan}"
        )
        if nulls:
            self.error(where, f"primary key has {nulls:,} null values")
        if dups:
            ex = self.q(
                f"SELECT {k}, count(*) AS n FROM {scan} WHERE {k} IS NOT NULL "
                f"GROUP BY {k} HAVING count(*) > 1 ORDER BY n DESC LIMIT {self.k}"
            )
            self.error(
                where,
                f"primary key has {dups:,} duplicate values, e.g. "
                + ", ".join(f"{v!r} (x{n})" for v, n in ex),
            )
        if not self.key_contract or nulls or dups:
            return
        typ = self.schemas[name].field(spec.pkey).type
        if not pa.types.is_integer(typ):
            self.error(
                where,
                f"primary key has type {typ}; RelBench needs integers 0..n-1 in row "
                f"order, {REINDEX_HINT}",
            )
            return
        [(bad, row)] = self.q(
            f"SELECT count(*), min(file_row_number) FROM "
            f"{_scan(self.path(name), row_numbers=True)} "
            f"WHERE {k} <> file_row_number"
        )
        if bad:
            self.error(
                where,
                f"primary key is not 0..n-1 in row order ({bad:,} rows differ, first "
                f"at row {row}), {REINDEX_HINT}",
            )

    def check_fkeys(self, name: str, spec) -> None:
        tables = self.manifest.tables
        for fkey, target in spec.fkeys.items():
            where = f"{name}.{fkey}"
            if target not in tables:
                self.error(where, f"foreign key references unknown table '{target}'")
                continue
            pkey = tables[target].pkey
            if pkey is None:
                self.error(
                    where,
                    f"foreign key references table '{target}', which has no pkey",
                )
                continue
            if not self.has(name, fkey) or not self.has(target, pkey):
                continue
            ft = self.schemas[name].field(fkey).type
            pt = self.schemas[target].field(pkey).type
            if not _compatible(ft, pt):
                self.error(
                    where,
                    f"foreign key type {ft} does not match {target}.{pkey} type {pt}",
                )
                continue
            if self.key_contract and not pa.types.is_integer(ft):
                self.error(
                    where,
                    f"foreign key has type {ft}; RelBench needs integer row indices "
                    f"of '{target}', {REINDEX_HINT}",
                )
                continue
            self.check_refs(
                where,
                f"SELECT {_id(fkey)} AS v FROM {_scan(self.path(name))}",
                target,
                f"{target}.{pkey}",
            )

    def ref_set(self, table: str) -> Optional[str]:
        pkey = self.manifest.tables[table].pkey
        if pkey is None:
            if table not in self.rows:
                return None
            return f"(SELECT range AS k FROM range({self.rows[table]}))"
        if not self.has(table, pkey):
            return None
        return f"(SELECT DISTINCT {_id(pkey)} AS k FROM {_scan(self.path(table))})"

    def check_refs(
        self, where: str, values: str, table: str, target: str, nulls_ok=True
    ) -> None:
        ref = self.ref_set(table)
        if ref is None:
            return
        [(total, nulls, missing)] = self.q(
            f"SELECT count(*), count(*) - count(l.v), "
            f"count(*) FILTER (WHERE l.v IS NOT NULL AND r.k IS NULL) "
            f"FROM ({values}) l LEFT JOIN {ref} r ON l.v = r.k"
        )
        if nulls and not nulls_ok:
            self.error(where, f"{nulls:,} of {total:,} values are null")
        if missing:
            ex = self.q(
                f"SELECT DISTINCT l.v FROM ({values}) l ANTI JOIN {ref} r "
                f"ON l.v = r.k WHERE l.v IS NOT NULL LIMIT {self.k}"
            )
            self.error(
                where,
                f"{missing:,} of {total - nulls:,} non-null values do not resolve to "
                f"{target}, e.g. {_fmt(v for (v,) in ex)}",
            )

    def check_time(self, name: str, spec) -> None:
        col = spec.time_col
        if not self.has(name, col):
            return
        where = f"{name}.{col}"
        typ = self.schemas[name].field(col).type
        if not self.check_time_type(where, self.path(name), col, typ):
            return
        c = _id(col)
        [(lo, hi)] = self.q(f"SELECT min({c}), max({c}) FROM {_scan(self.path(name))}")
        if lo is not None:
            self.time_ranges[name] = (pd.Timestamp(lo), pd.Timestamp(hi))
        if self.key_contract:
            self.check_sorted(where, name, col)

    def check_time_type(self, where: str, path: Path, col: str, typ) -> bool:
        if pa.types.is_timestamp(typ):
            if typ.tz is not None:
                self.error(
                    where,
                    f"time column is timezone-aware ({typ}); store naive timestamps",
                )
                return False
            return True
        message = f"time column has type {typ}, expected a timestamp"
        if _is_str(typ):
            c = _id(col)
            unparsed = f"{c} IS NOT NULL AND TRY_CAST({c} AS TIMESTAMP) IS NULL"
            [(bad,)] = self.q(f"SELECT count(*) FROM {_scan(path)} WHERE {unparsed}")
            if bad:
                ex = self.q(
                    f"SELECT DISTINCT {c} FROM {_scan(path)} WHERE {unparsed} "
                    f"LIMIT {self.k}"
                )
                message += (
                    f"; {bad:,} values do not parse as timestamps, e.g. "
                    f"{_fmt(v for (v,) in ex)}"
                )
            else:
                message += "; all values parse, convert it with pd.to_datetime"
        self.error(where, message)
        return False

    def check_sorted(self, where: str, name: str, col: str) -> None:
        arr = pq.read_table(self.path(name), columns=[col]).column(col)
        arr = arr.combine_chunks()
        valid = len(arr) - arr.null_count
        if arr.null_count and pc.any(pc.is_null(arr.slice(0, valid))).as_py():
            self.error(
                where,
                f"missing times must come last (the table is not time-sorted), "
                f"{REINDEX_HINT}",
            )
            return
        if valid > 1:
            head = arr.slice(0, valid)
            bad = pc.sum(pc.less(head.slice(1), head.slice(0, valid - 1))).as_py()
            if bad:
                self.error(
                    where,
                    f"rows are not sorted by time ({bad:,} out of order), "
                    f"{REINDEX_HINT}",
                )

    def check_split_timestamps(self) -> None:
        if not self.time_ranges:
            return
        data_min = min(lo for lo, _ in self.time_ranges.values())
        self.data_max = max(hi for _, hi in self.time_ranges.values())
        span = f"data spans {data_min} .. {self.data_max}"
        for key, ts in [
            ("val_timestamp", self.val_ts),
            ("test_timestamp", self.test_ts),
        ]:
            if ts is not None and not data_min < ts < self.data_max:
                self.error("manifest.yaml", f"{key} {ts} is outside the data ({span})")

    def check_tasks(self) -> None:
        tasks_dir = self.root / "tasks"
        if not tasks_dir.is_dir():
            return
        for d in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
            if not (d / "manifest.yaml").exists():
                self.warn(f"task '{d.name}'", "no manifest.yaml, skipped")
                continue
            self.check_task(d)

    def check_task(self, d: Path) -> None:
        where = f"task '{d.name}'"
        try:
            tm = TaskManifest.load(d / "manifest.yaml")
            tm.validate()
        except (yaml.YAMLError, TypeError, AttributeError, ValueError) as e:
            self.error(where, f"invalid manifest.yaml: {e}")
            return
        tables = self.manifest.tables
        if tm.task_type == "recommendation":
            keys = [
                (tm.src_entity_table, tm.src_entity_col),
                (tm.dst_entity_table, tm.dst_entity_col),
            ]
        else:
            keys = [(tm.entity_table, tm.entity_col)]
        unknown = [t for t, _ in keys if t not in tables]
        if unknown:
            for t in unknown:
                self.error(where, f"entity table '{t}' is not in the dataset manifest")
            return
        time_col = tm.time_col
        if tm.kind == KIND_AUTOCOMPLETE and tm.task_type != "recommendation":
            spec = tables[tm.entity_table]
            keys = [(tm.entity_table, spec.pkey or "primary_key")]
            time_col = spec.time_col
            target = tm.target_col
            if tm.entity_table in self.schemas and not self.has(
                tm.entity_table, target
            ):
                self.error(
                    where,
                    f"target column '{target}' is not in table '{tm.entity_table}'",
                )
            elif target == spec.pkey or target in spec.fkeys:
                self.error(
                    where,
                    f"target column '{target}' is a key of table '{tm.entity_table}'",
                )
        for pair in tm.remove_columns:
            table, col = pair
            if table not in tables or (
                table in self.schemas and not self.has(table, col)
            ):
                self.warn(
                    where, f"remove_columns entry [{table}, {col}] does not exist"
                )
        delta = None
        if tm.timedelta:
            try:
                delta = pd.Timedelta(tm.timedelta)
            except ValueError:
                self.error(where, f"timedelta {tm.timedelta!r} is not parseable")
        for split in SPLITS:
            path = d / f"{split}.parquet"
            if not path.exists():
                if tm.kind == KIND_EXTERNAL:
                    self.error(where, f"missing {split}.parquet")
                else:
                    self.warn(
                        where,
                        f"missing {split}.parquet; it is regenerated on load",
                    )
                continue
            self.check_labels(
                f"{where} {split}", split, path, tm, keys, time_col, delta
            )

    def check_labels(self, where, split, path, tm, keys, time_col, delta) -> None:
        try:
            schema = pq.read_schema(path)
        except (pa.ArrowInvalid, OSError) as e:
            self.error(where, f"unreadable parquet: {e}")
            return
        key_cols = [c for _, c in keys if c]
        required = list(dict.fromkeys([*key_cols, time_col, tm.target_col]))
        missing = [c for c in required if c and c not in schema.names]
        if missing:
            self.error(where, f"missing columns {missing} (columns: {schema.names})")
        scan = _scan(path)
        for table, col in keys:
            if col and col in schema.names:
                self.check_entity_col(where, scan, schema.field(col).type, table, col)
        if time_col and time_col in schema.names:
            self.check_label_times(where, split, path, tm, time_col, delta)
        target = tm.target_col
        if target and target in schema.names and target not in key_cols:
            c = _id(target)
            [(nulls,)] = self.q(f"SELECT count(*) - count({c}) FROM {scan}")
            if nulls:
                self.warn(where, f"target column '{target}' has {nulls:,} nulls")

    def check_entity_col(self, where, scan, typ, table, col) -> None:
        pkey = self.manifest.tables[table].pkey
        target = f"{table}.{pkey}" if pkey else f"row indices of '{table}'"
        elem = typ.value_type if _is_list(typ) else typ
        if pkey and self.has(table, pkey):
            ptype = self.schemas[table].field(pkey).type
            if not _compatible(elem, ptype):
                self.error(
                    f"{where}.{col}",
                    f"entity column type {elem} does not match {target} type {ptype}",
                )
                return
        elif not pa.types.is_integer(elem):
            self.error(f"{where}.{col}", f"entity column type {elem} is not integer")
            return
        expr = f"unnest({_id(col)})" if _is_list(typ) else _id(col)
        self.check_refs(
            f"{where}.{col}",
            f"SELECT {expr} AS v FROM {scan}",
            table,
            target,
            nulls_ok=False,
        )

    def check_label_times(self, where, split, path, tm, col, delta) -> None:
        typ = pq.read_schema(path).field(col).type
        if not self.check_time_type(f"{where}.{col}", path, col, typ):
            return
        c = _id(col)
        [(lo, hi, nulls)] = self.q(
            f"SELECT min({c}), max({c}), count(*) - count({c}) FROM {_scan(path)}"
        )
        if nulls:
            self.error(f"{where}.{col}", f"{nulls:,} null timestamps")
        if lo is None:
            return
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        val, test = self.val_ts, self.test_ts
        delta = delta if tm.kind == KIND_FORECAST else None
        window = f"timestamps span {lo} .. {hi}"

        def leak(bound, name):
            if delta is not None and bound is not None and hi + delta > bound:
                self.error(
                    where,
                    f"latest timestamp {hi} + timedelta {tm.timedelta} passes {name} "
                    f"{bound}, so its labels look past it",
                )

        if split == "train":
            if val is not None and hi > val:
                self.error(where, f"{window}, past val_timestamp {val}")
            else:
                leak(val, "val_timestamp")
        elif split == "val":
            if val is not None and lo < val:
                self.error(where, f"{window}, before val_timestamp {val}")
            if test is not None and hi > test:
                self.error(where, f"{window}, past test_timestamp {test}")
            else:
                leak(test, "test_timestamp")
        else:
            if test is not None and lo < test:
                self.error(where, f"{window}, before test_timestamp {test}")
            if self.data_max is not None and hi > self.data_max:
                self.error(
                    where, f"{window}, past the last data timestamp {self.data_max}"
                )
            else:
                leak(self.data_max, "the last data timestamp")


def validate_dataset(
    dataset_dir: Union[str, Path],
    *,
    tasks: bool = True,
    key_contract: bool = True,
    max_examples: int = 5,
) -> list[Issue]:
    root = Path(dataset_dir).resolve()
    return _Validator(root, key_contract, max_examples).run(tasks)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m relbench.validate",
        description="Check a local RelBench dataset folder against its manifests.",
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--no-tasks", action="store_true", help="skip tasks/*/ label tables"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="keys are not normalized yet: skip the 0..n-1 / time-sorted checks",
    )
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args(argv)
    issues = validate_dataset(
        args.dataset_dir,
        tasks=not args.no_tasks,
        key_contract=not args.raw,
        max_examples=args.max_examples,
    )
    for issue in issues:
        print(issue)
    errors = sum(i.level == ERROR for i in issues)
    print(
        f"{args.dataset_dir}: {errors} error(s), {len(issues) - errors} warning(s)",
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
