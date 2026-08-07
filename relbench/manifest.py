r"""Manifest schema for RelBench datasets and tasks.

A RelBench dataset on disk / on the Hugging Face Hub is a self-describing folder::

    <dataset>/
        manifest.yaml                 # DatasetManifest: description + tables + fkey graph + splits
        README.md                     # dataset card (links the schema.svg ER diagram)
        schema.svg                    # entity-relationship diagram (see relbench.schema)
        db/<table>.parquet            # plain data, native column dtypes only
        tasks/<task>/
            manifest.yaml             # TaskManifest: description + spec (+ SQL for kind="forecast")
            {train,val,test}.parquet  # labels (regenerable for kind="forecast")

The manifest is the *sole source of truth* for relational semantics (primary keys, the
foreign-key graph, time columns). Parquet files carry only their native column schema, so
they load directly with pandas/duckdb. Manifests are YAML for readability (the SQL of a
``forecast`` task is a block scalar).
"""

from __future__ import annotations

import dataclasses
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml

MANIFEST_VERSION = 1

# Task ``kind`` -- how labels are produced.
KIND_FORECAST = (
    "forecast"  # temporal-aggregation labels regenerated via a duckdb SQL query
)
KIND_AUTOCOMPLETE = "autocomplete"  # generic column-prediction generator
KIND_EXTERNAL = (
    "external"  # labels sourced/built externally, served as-is (TGB / dbinfer)
)
KINDS = (KIND_FORECAST, KIND_AUTOCOMPLETE, KIND_EXTERNAL)

# Task ``task_type`` -- what is predicted, and therefore which evaluator applies.
# Spelled out here rather than imported from relbench.base so a manifest can be
# written and validated without pulling in pandas/numpy; kept in step with
# :class:`relbench.base.TaskType`, which the round-trip test pins.
TASK_TYPES = (
    "regression",
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "recommendation",
)


class _Dumper(yaml.SafeDumper):
    pass


def _represent_str(dumper, data):
    # Render multi-line strings (e.g. SQL) as readable block scalars (`|`).
    #
    # YAML cannot express a block scalar whose first line is blank or whose
    # lines carry trailing spaces, and pyyaml silently falls back to a
    # double-quoted one-liner full of \n escapes when asked to -- which is how
    # every hosted manifest ended up with its SQL unreadable. The strings come
    # from triple-quoted Python literals, so that leading newline and that
    # trailing indentation are an artefact of how the source was written, not
    # content: normalise them away and the block scalar applies.
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", _block(data), style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _block(s: str) -> str:
    """A multi-line string as it should read in a block scalar.

    ``textwrap.dedent`` alone is not enough: a triple-quoted literal that opens
    on the same line as its first word (``sql = \"\"\"SELECT ...``) has no indent
    on line 1, which makes the common prefix empty and leaves every other line
    stranded at the source file's indentation. Dedent the continuation lines on
    their own, and line 1 goes back where it belongs.
    """
    lines = [line.rstrip() for line in s.strip("\n").splitlines()]
    if not lines:
        return ""
    # The first line carries no meaningful indentation (it is whatever the
    # source literal happened to open with), and leaving any on it forces yaml
    # to write an explicit indentation indicator -- `|2-` instead of `|`.
    head, rest = lines[0].lstrip(), lines[1:]
    if rest:
        rest = textwrap.dedent("\n".join(rest)).splitlines()
    return "\n".join([head, *rest]).strip("\n")


_Dumper.add_representer(str, _represent_str)


def _dump_yaml(d: dict, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            d,
            f,
            Dumper=_Dumper,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=4096,
        )


def _load_yaml(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class TableSpec:
    r"""Relational semantics for one table (the part parquet can't express)."""

    pkey: Optional[str] = None
    time_col: Optional[str] = None
    fkeys: dict[str, str] = field(default_factory=dict)  # fkey_col -> pkey_table

    @classmethod
    def from_dict(cls, d: dict) -> "TableSpec":
        return cls(
            pkey=d.get("pkey"),
            time_col=d.get("time_col"),
            fkeys=dict(d.get("fkeys", {})),
        )

    def to_dict(self) -> dict:
        return {"pkey": self.pkey, "time_col": self.time_col, "fkeys": self.fkeys}


@dataclass
class DatasetManifest:
    r"""Dataset-level manifest: a description, the table set, the fkey graph, and splits."""

    name: str
    val_timestamp: str  # ISO date/datetime, e.g. "2005-01-01"
    test_timestamp: str
    description: Optional[str] = None
    tables: dict[str, TableSpec] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetManifest":
        return cls(
            name=d["name"],
            val_timestamp=d["val_timestamp"],
            test_timestamp=d["test_timestamp"],
            description=d.get("description"),
            tables={k: TableSpec.from_dict(v) for k, v in d.get("tables", {}).items()},
            manifest_version=d.get("manifest_version", MANIFEST_VERSION),
        )

    def to_dict(self) -> dict:
        out = {"name": self.name, "manifest_version": self.manifest_version}
        if self.description:
            out["description"] = self.description
        out["val_timestamp"] = self.val_timestamp
        out["test_timestamp"] = self.test_timestamp
        out["tables"] = {k: v.to_dict() for k, v in self.tables.items()}
        return out

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DatasetManifest":
        return cls.from_dict(_load_yaml(path))

    def save(self, path: Union[str, Path]) -> None:
        _dump_yaml(self.to_dict(), path)


@dataclass
class TaskManifest:
    r"""Task-level manifest. One self-contained directory per task.

    ``kind="forecast"`` tasks carry a duckdb ``sql`` query that regenerates labels from the
    database; the SQL sees a ``timestamps(timestamp)`` relation (the per-split seed
    timestamps), every db table as a view by name, and ``{timedelta}`` substituted as a
    duckdb INTERVAL string. Its SELECT must output the declared entity/target/time cols.

    ``remove_columns`` names the database columns the task must not see. It is honored for
    every ``kind`` -- an autocomplete task's label *is* a database column, but a forecast or
    external task's label is often *derived from* one, and that leaks identically.
    ``Dataset.get_db`` drops the named columns from the graph.

    Metrics are not stored here -- they default from ``task_type`` (see relbench.load).
    """

    name: str
    kind: str
    task_type: str  # one of relbench.base.TaskType values
    description: Optional[str] = None

    # Entity (node) tasks.
    entity_table: Optional[str] = None
    entity_col: Optional[str] = None
    target_col: Optional[str] = None
    time_col: Optional[str] = None

    # Recommendation tasks.
    src_entity_table: Optional[str] = None
    src_entity_col: Optional[str] = None
    dst_entity_table: Optional[str] = None
    dst_entity_col: Optional[str] = None
    eval_k: Optional[int] = None

    # Temporal label window.
    timedelta: Optional[str] = None  # pandas-parseable, e.g. "60 days"
    num_eval_timestamps: int = 1

    # Database columns this task must not see, as [table, col] pairs -- the columns its
    # label is derived from or trivially implied by. Applies to *every* kind, not just
    # autocomplete: a forecast or external task whose label comes from a database column
    # (dbinfer's `cvr` <- `View.added_to_cart`) leaks exactly the same way. `Dataset.get_db`
    # drops each pair from the graph.
    remove_columns: list = field(default_factory=list)  # list of [table, col]

    # Label generation (kind="forecast").
    sql: Optional[str] = None

    # External tasks (labels shipped as-is).
    evaluator: Optional[str] = None  # named evaluator for custom eval (e.g. tgb)
    extra_files: list = field(default_factory=list)

    manifest_version: int = MANIFEST_VERSION

    @classmethod
    def from_dict(cls, d: dict) -> "TaskManifest":
        # Tolerant: ignore unknown/legacy keys (e.g. a removed "metrics" field).
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        out = {"name": self.name, "kind": self.kind, "task_type": self.task_type}
        if self.description:
            out["description"] = self.description
        for k in [
            "entity_table",
            "entity_col",
            "target_col",
            "time_col",
            "src_entity_table",
            "src_entity_col",
            "dst_entity_table",
            "dst_entity_col",
            "eval_k",
            "timedelta",
            "evaluator",
        ]:
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        if self.num_eval_timestamps != 1:
            out["num_eval_timestamps"] = self.num_eval_timestamps
        if self.remove_columns:
            out["remove_columns"] = self.remove_columns
        if self.extra_files:
            out["extra_files"] = self.extra_files
        if self.sql is not None:
            out["sql"] = self.sql
        out["manifest_version"] = self.manifest_version
        return out

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TaskManifest":
        return cls.from_dict(_load_yaml(path))

    def save(self, path: Union[str, Path]) -> None:
        _dump_yaml(self.to_dict(), path)

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"task '{self.name}': kind must be one of {KINDS}")
        if self.task_type not in TASK_TYPES:
            raise ValueError(
                f"task '{self.name}': task_type must be one of {TASK_TYPES}, "
                f"got {self.task_type!r}"
            )
        if self.kind == KIND_FORECAST and not self.sql:
            raise ValueError(
                f"task '{self.name}': kind='forecast' requires a 'sql' field"
            )
        if self.task_type == "recommendation" and self.kind == KIND_FORECAST:
            missing = [
                k
                for k in (
                    "src_entity_table",
                    "src_entity_col",
                    "dst_entity_table",
                    "dst_entity_col",
                    "eval_k",
                )
                if getattr(self, k) is None
            ]
            if missing:
                raise ValueError(f"task '{self.name}': link task missing {missing}")


def validate_dataset_manifest(
    manifest: DatasetManifest, db_dir: Union[str, Path]
) -> None:
    r"""Check the manifest is consistent with the parquet files in ``db_dir`` (schema
    only)."""
    import pyarrow.parquet as pq

    db_dir = Path(db_dir)
    for tname, spec in manifest.tables.items():
        path = db_dir / f"{tname}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"table '{tname}': missing parquet {path}")
        cols = set(pq.read_schema(path).names)
        named = [c for c in [spec.pkey, spec.time_col, *spec.fkeys.keys()] if c]
        missing = [c for c in named if c not in cols]
        if missing:
            raise ValueError(f"table '{tname}': columns {missing} not in parquet")
        for fkey_col, pkey_table in spec.fkeys.items():
            if pkey_table not in manifest.tables:
                raise ValueError(
                    f"table '{tname}': fkey '{fkey_col}' -> unknown table '{pkey_table}'"
                )
