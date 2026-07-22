r"""Manifest-driven, Hugging Face-backed loader for RelBench datasets and tasks.

This is the whole consumption path: no per-dataset or per-task classes. A dataset is a
folder of plain parquet + a ``manifest.yaml``; a task is a subdirectory with its own
``manifest.yaml`` (and, for ``kind="sql"`` tasks, the duckdb query that regenerates its
labels). Adding a task later is just adding a directory.

Public API::

    ds   = relbench.load_dataset("relbench/core/rel-f1")   # Hub 'org/repo[/subdir]' or a local path
    task = relbench.load_task("relbench/core/rel-f1", "driver-position")
    db   = ds.get_db()
    train = task.get_table("train")
    task.evaluate(pred)

The generic ``Task`` is built by parametrizing the existing
``EntityTask`` / ``RecommendationTask`` / ``AutoCompleteTask`` scaffolding with the
manifest -- so split logic, dangling-entity filtering, masking, evaluation and metrics
are exactly the shipped behavior; only ``make_table`` changes to run the manifest SQL.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from relbench import hf, metrics
from relbench.base import (
    AutoCompleteTask,
    Database,
    Dataset,
    EntityTask,
    RecommendationTask,
    Table,
    TaskType,
)
from relbench.manifest import (
    KIND_AUTOCOMPLETE,
    KIND_EXTERNAL,
    KIND_FORECAST,
    DatasetManifest,
    TaskManifest,
    validate_dataset_manifest,
)

# The default metric per task type. RelBench provides evaluators for the three core task
# types only (the user does not choose the metric); multiclass/multilabel tasks are
# definable but carry no provided evaluator. Regression's NMAE is built per-task (it needs
# the train-split target std), so it is handled specially in `_resolve_metrics`.
DEFAULT_METRICS: dict[TaskType, list[str]] = {
    TaskType.BINARY_CLASSIFICATION: ["roc_auc"],
    TaskType.REGRESSION: ["nmae"],
    TaskType.LINK_PREDICTION: ["link_prediction_map"],
}


def train_std(task) -> float:
    r"""Standard deviation (ddof=1) of a regression task's target on its train split.

    This is the normalizer that turns MAE into NMAE. For the hosted relbench/core tasks the
    same values are precomputed and stored at ``relbench/core`` (see
    :func:`relbench.hf.load_core_regression_stds`); this utility recomputes one from scratch.
    """
    df = task.get_table("train").df
    return float(df[task.target_col].std(ddof=1))


def _make_std_getter(task, dataset_name: Optional[str], task_name: Optional[str]):
    r"""Lazily resolve the NMAE normalizer for ``task``: hosted core std if available,
    else compute it from the train split.

    Lazy so merely loading a task triggers no I/O.
    """
    from functools import lru_cache as _lru_cache

    @_lru_cache(maxsize=1)
    def get_std() -> float:
        if dataset_name is not None and task_name is not None:
            try:
                stds = hf.load_core_regression_stds()
            except Exception:
                stds = {}
            std = stds.get(f"{dataset_name}/{task_name}")
            if std is not None:
                return float(std)
        return train_std(task)

    return get_std


def _resolve_metrics(
    tm: TaskManifest, task=None, dataset_name: Optional[str] = None
) -> list:
    # Metrics are not stored in the manifest; they default from the task type. RelBench
    # provides evaluators for the core task types only; multiclass/multilabel tasks are
    # loadable but carry no evaluator -- bring your own via task.evaluate(pred, metrics=).
    task_type = TaskType(tm.task_type)
    if task_type not in DEFAULT_METRICS:
        return []
    out = []
    for name in DEFAULT_METRICS[task_type]:
        if name == "nmae":
            out.append(metrics.make_nmae(_make_std_getter(task, dataset_name, tm.name)))
        else:
            out.append(getattr(metrics, name))
    return out


def _dataset_name(dataset) -> Optional[str]:
    r"""The dataset's name if it carries a manifest (a hosted ``RelBenchDataset``), else
    ``None``.

    Tasks can be built on a plain in-memory :class:`~relbench.base.Dataset` (e.g. the test
    fixtures) that has no manifest and no hosted regression std; ``None`` makes the NMAE
    normalizer fall back to computing the std from the train split.
    """
    return getattr(getattr(dataset, "manifest", None), "name", None)


def _coerce_string_dtype(df: pd.DataFrame) -> pd.DataFrame:
    # DuckDB cannot reliably ingest pandas StringDtype columns; match base.Table.load.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
    return df


def _load_database(db_dir: Path, manifest: DatasetManifest) -> Database:
    r"""Build a :class:`Database` from plain parquet + manifest relational metadata.

    The parquet files carry no RelBench metadata; all keys/time columns come from the
    manifest, which is the sole source of truth.
    """
    table_dict: dict[str, Table] = {}
    for name, spec in manifest.tables.items():
        df = _coerce_string_dtype(pd.read_parquet(db_dir / f"{name}.parquet"))
        table_dict[name] = Table(
            df=df,
            fkey_col_to_pkey_table=dict(spec.fkeys),
            pkey_col=spec.pkey,
            time_col=spec.time_col,
        )
    return Database(table_dict)


class RelBenchDataset(Dataset):
    r"""A dataset loaded from a manifest + parquet folder.

    Mirrors the *download* path of the legacy ``Dataset`` (load -> upto ->
    validate/correct -> optional autocomplete modification) but reads relational
    metadata from the manifest instead of parquet, and does **not** reindex: hosted
    artifacts are already reindexed at build time, so load is pure I/O.
    """

    def __init__(
        self, dataset_dir: Union[str, Path], manifest: DatasetManifest
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.db_dir = self.dataset_dir / "db"
        self.manifest = manifest
        self.val_timestamp = pd.Timestamp(manifest.val_timestamp)
        self.test_timestamp = pd.Timestamp(manifest.test_timestamp)
        super().__init__(cache_dir=None)

    def __repr__(self) -> str:
        return f"RelBenchDataset(name={self.manifest.name!r})"

    def make_db(self) -> Database:
        return _load_database(self.db_dir, self.manifest)

    @lru_cache(maxsize=None)
    def get_db(self, upto_test_timestamp: bool = True) -> Database:
        db = self.make_db()
        if upto_test_timestamp:
            db = db.upto(self.test_timestamp)
        self.validate_and_correct_db(db)
        if self.target_col:
            db = self.get_modified_db(db)
        return db


def _parse_timedelta(value: Optional[str]) -> pd.Timedelta:
    if value is None:
        raise ValueError("task manifest is missing 'timedelta'")
    return pd.Timedelta(value)


def _run_task_sql(
    sql: str,
    db: Database,
    timestamps: "pd.DatetimeIndex",
    timedelta: pd.Timedelta,
) -> pd.DataFrame:
    r"""Run a task's manifest SQL.

    Exposes a ``timestamps(timestamp)`` relation (the seed timestamps for the split) and
    every db table as a view by name, and substitutes ``{timedelta}`` with the duckdb
    INTERVAL string -- the same environment the legacy ``make_table`` queries assumed.
    """
    import duckdb  # lazy: keeps `import relbench` importable where duckdb isn't (e.g. Pyodide)

    con = duckdb.connect()
    try:
        # A memory limit makes duckdb raise a catchable OOM (and spill to disk) rather
        # than letting the OS kill the process on a runaway query. Opt-in via env.
        limit = os.getenv("RELBENCH_DUCKDB_MEMORY_LIMIT")
        if limit:
            con.execute(f"SET memory_limit='{limit}'")
        con.register("timestamps", pd.DataFrame({"timestamp": timestamps}))
        for name, table in db.table_dict.items():
            con.register(name, table.df)
        return con.sql(sql.replace("{timedelta}", str(timedelta))).df()
    finally:
        con.close()


class _HostedLabelsMixin:
    r"""Use hosted plain-parquet labels when present, else regenerate from the database.

    ``_get_table`` (regeneration via ``make_table``) and all masking/eval/filtering are
    inherited unchanged from the base task class.
    """

    _task_dir: Optional[Path] = None
    _regenerate: bool = False

    def _label_fkeys(self) -> dict[str, str]:
        if isinstance(self, RecommendationTask):
            return {
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            }
        return {self.entity_col: self.entity_table}

    @lru_cache(maxsize=None)
    def get_table(self, split: str, mask_input_cols: Optional[bool] = None) -> Table:
        if mask_input_cols is None:
            mask_input_cols = split == "test"

        path = None
        if self._task_dir is not None and not self._regenerate:
            candidate = Path(self._task_dir) / f"{split}.parquet"
            if candidate.exists():
                path = candidate

        if path is not None:
            df = _coerce_string_dtype(pd.read_parquet(path))
            table = Table(
                df=df,
                fkey_col_to_pkey_table=self._label_fkeys(),
                pkey_col=None,
                time_col=self.time_col,
            )
        else:
            table = self._get_table(split)  # regenerate (already filters dangling)

        if mask_input_cols:
            table = self._mask_input_cols(table)
        return table


class _ForecastEntityTask(_HostedLabelsMixin, EntityTask):
    def __init__(
        self, dataset: Dataset, tm: TaskManifest, task_dir=None, regenerate=False
    ):
        self.task_type = TaskType(tm.task_type)
        self.entity_table = tm.entity_table
        self.entity_col = tm.entity_col
        self.target_col = tm.target_col
        self.time_col = tm.time_col
        self.timedelta = _parse_timedelta(tm.timedelta)
        self.num_eval_timestamps = tm.num_eval_timestamps
        self.metrics = _resolve_metrics(
            tm, task=self, dataset_name=_dataset_name(dataset)
        )
        self._sql = tm.sql
        self._task_dir = Path(task_dir) if task_dir is not None else None
        self._regenerate = regenerate
        super().__init__(dataset, cache_dir=None)

    def make_table(self, db: Database, timestamps) -> Table:
        df = _run_task_sql(self._sql, db, timestamps, self.timedelta)
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class _ForecastRecommendationTask(_HostedLabelsMixin, RecommendationTask):
    def __init__(
        self, dataset: Dataset, tm: TaskManifest, task_dir=None, regenerate=False
    ):
        self.task_type = TaskType(tm.task_type)
        self.src_entity_table = tm.src_entity_table
        self.src_entity_col = tm.src_entity_col
        self.dst_entity_table = tm.dst_entity_table
        self.dst_entity_col = tm.dst_entity_col
        self.target_col = tm.target_col
        self.time_col = tm.time_col
        self.timedelta = _parse_timedelta(tm.timedelta)
        self.num_eval_timestamps = tm.num_eval_timestamps
        self.eval_k = tm.eval_k
        self.metrics = _resolve_metrics(
            tm, task=self, dataset_name=_dataset_name(dataset)
        )
        self._sql = tm.sql
        self._task_dir = Path(task_dir) if task_dir is not None else None
        self._regenerate = regenerate
        super().__init__(dataset, cache_dir=None)

    def make_table(self, db: Database, timestamps) -> Table:
        df = _run_task_sql(self._sql, db, timestamps, self.timedelta)
        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


class _AutoCompleteTask(_HostedLabelsMixin, AutoCompleteTask):
    def __init__(
        self, dataset: Dataset, tm: TaskManifest, task_dir=None, regenerate=False
    ):
        self._task_dir = Path(task_dir) if task_dir is not None else None
        self._regenerate = regenerate
        super().__init__(
            dataset,
            task_type=TaskType(tm.task_type),
            entity_table=tm.entity_table,
            target_col=tm.target_col,
            cache_dir=None,
            remove_columns=[tuple(pair) for pair in tm.remove_columns],
        )

    def _get_table(self, split: str) -> Table:
        # Efficient autocomplete window. The legacy AutoCompleteTask materializes
        # pd.date_range at 1-second freq across the whole split span (~1.7B entries /
        # OOM on wide val/test gaps like rel-f1's 60 years). make_table only uses the
        # min/max of that range, so we pass just the two bounds -- identical labels,
        # no giant allocation.
        db = self.dataset.get_db(upto_test_timestamp=split != "test")
        if split == "train":
            start, end = self.dataset.val_timestamp - self.timedelta, db.min_timestamp
        elif split == "val":
            if self.dataset.val_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError("val timestamp + timedelta exceeds db max timestamp")
            start, end = (
                self.dataset.test_timestamp - self.timedelta,
                self.dataset.val_timestamp,
            )
        elif split == "test":
            if self.dataset.test_timestamp + self.timedelta > db.max_timestamp:
                raise RuntimeError(
                    "test timestamp + timedelta exceeds db max timestamp"
                )
            start, end = db.max_timestamp, self.dataset.test_timestamp
        else:
            raise ValueError(f"unknown split: {split!r}")
        table = self.make_table(db, pd.DatetimeIndex([start, end]))
        return self.filter_dangling_entities(table)


class _ExternalEntityTask(_HostedLabelsMixin, EntityTask):
    r"""Entity task whose labels are built externally (e.g. dbinfer) and served as-
    is."""

    def __init__(
        self, dataset: Dataset, tm: TaskManifest, task_dir=None, regenerate=False
    ):
        self.task_type = TaskType(tm.task_type)
        self.entity_table = tm.entity_table
        self.entity_col = tm.entity_col
        self.target_col = tm.target_col
        self.time_col = tm.time_col
        self.timedelta = (
            _parse_timedelta(tm.timedelta) if tm.timedelta else pd.Timedelta(days=1)
        )
        self.num_eval_timestamps = tm.num_eval_timestamps
        self.metrics = _resolve_metrics(
            tm, task=self, dataset_name=_dataset_name(dataset)
        )
        self._task_dir = Path(task_dir) if task_dir is not None else None
        self._regenerate = False  # external labels are not regenerable
        super().__init__(dataset, cache_dir=None)

    def make_table(self, db: Database, timestamps) -> Table:
        raise NotImplementedError("external task labels are hosted, not regenerable")


class _ExternalRecommendationTask(_HostedLabelsMixin, RecommendationTask):
    r"""Link task whose labels/eval are built externally (e.g. TGB) and served as-is."""

    def __init__(
        self, dataset: Dataset, tm: TaskManifest, task_dir=None, regenerate=False
    ):
        self.task_type = TaskType(tm.task_type)
        self.src_entity_table = tm.src_entity_table
        self.src_entity_col = tm.src_entity_col
        self.dst_entity_table = tm.dst_entity_table
        self.dst_entity_col = tm.dst_entity_col
        self.target_col = tm.target_col
        self.time_col = tm.time_col
        self.timedelta = (
            _parse_timedelta(tm.timedelta) if tm.timedelta else pd.Timedelta(days=1)
        )
        self.num_eval_timestamps = tm.num_eval_timestamps
        self.eval_k = tm.eval_k
        self.metrics = _resolve_metrics(
            tm, task=self, dataset_name=_dataset_name(dataset)
        )
        self._task_dir = Path(task_dir) if task_dir is not None else None
        self._regenerate = False
        super().__init__(dataset, cache_dir=None)

    def make_table(self, db: Database, timestamps) -> Table:
        raise NotImplementedError("external task labels are hosted, not regenerable")


def build_task(
    dataset: Dataset,
    tm: TaskManifest,
    *,
    task_dir: Optional[Path] = None,
    regenerate: bool = False,
):
    r"""Instantiate the generic task object for a task manifest."""
    tm.validate()
    is_link = TaskType(tm.task_type) == TaskType.LINK_PREDICTION
    if tm.kind == KIND_AUTOCOMPLETE:
        return _AutoCompleteTask(dataset, tm, task_dir=task_dir, regenerate=regenerate)
    if tm.kind == KIND_FORECAST:
        cls = _ForecastRecommendationTask if is_link else _ForecastEntityTask
        return cls(dataset, tm, task_dir=task_dir, regenerate=regenerate)
    if tm.kind == KIND_EXTERNAL:
        cls = _ExternalRecommendationTask if is_link else _ExternalEntityTask
        return cls(dataset, tm, task_dir=task_dir, regenerate=regenerate)
    raise ValueError(f"unknown task kind: {tm.kind!r}")


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def _resolve_dataset_dir(
    name_or_path: Union[str, Path], revision: Optional[str]
) -> Path:
    p = Path(name_or_path)
    if p.exists() and (p / "manifest.yaml").exists():
        return p
    return hf.download_dataset_dir(str(name_or_path), revision=revision)


def load_dataset(
    name_or_path: Union[str, Path], *, revision: Optional[str] = None
) -> RelBenchDataset:
    r"""Load a RelBench dataset from a Hub ``org/repo[/subdir]`` or a local path."""
    dataset_dir = _resolve_dataset_dir(name_or_path, revision)
    manifest = DatasetManifest.load(dataset_dir / "manifest.yaml")
    return RelBenchDataset(dataset_dir, manifest)


def get_task_names(
    name_or_path: Union[str, Path], *, revision: Optional[str] = None
) -> list[str]:
    r"""List tasks available for a dataset by enumerating ``tasks/*/manifest.yaml``."""
    dataset_dir = _resolve_dataset_dir(name_or_path, revision)
    tasks_dir = dataset_dir / "tasks"
    if not tasks_dir.exists():
        return []
    return sorted(d.name for d in tasks_dir.iterdir() if (d / "manifest.yaml").exists())


def load_task(
    dataset: Union[str, Path, RelBenchDataset],
    task_name: str,
    *,
    revision: Optional[str] = None,
    regenerate: bool = False,
):
    r"""Load a task. Uses hosted labels when present; ``regenerate=True`` forces the SQL.

    For ``kind="sql"`` tasks, ``regenerate`` recomputes labels from the database via the
    manifest SQL -- this is also the provenance check used in CI.
    """
    if isinstance(dataset, RelBenchDataset):
        ds = dataset
        dataset_dir = ds.dataset_dir
    else:
        dataset_dir = _resolve_dataset_dir(dataset, revision)
        ds = RelBenchDataset(
            dataset_dir, DatasetManifest.load(dataset_dir / "manifest.yaml")
        )

    task_dir = dataset_dir / "tasks" / task_name
    tm = TaskManifest.load(task_dir / "manifest.yaml")
    return build_task(ds, tm, task_dir=task_dir, regenerate=regenerate)
