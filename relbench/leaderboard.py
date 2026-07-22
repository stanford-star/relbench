r"""Prediction-table submission and leaderboard evaluation for RelBench.

This module is the *consumption-free* evaluation path: it depends only on the standard
library, ``pandas``/``numpy`` and the rest of RelBench's data layer (no torch /
torch_geometric / pyg), so it imports cleanly in a minimal environment and can score a
submission produced by any modelling stack.

Concept -- the prediction table
================================
The test split of a task is a *task table*. A *prediction table* is the same table with
the target column replaced by predictions:

* ``binary_classification`` (EntityTask) -- target column holds the predicted probability
  (a float in ``[0, 1]``);
* ``regression`` (EntityTask) -- target column holds the numeric prediction, on the
  original target scale;
* ``link_prediction`` (RecommendationTask) -- the destination-entity column holds, per
  source row, the top-``eval_k`` predicted destination ids (same id space as the
  ground-truth destination lists), encoded as a JSON list string in the CSV cell.

Prediction tables are stored as CSV. Their *key columns* (the non-prediction columns that
uniquely identify a test row) are ``[entity_col, time_col]`` for an EntityTask and
``[src_entity_col, time_col]`` for a RecommendationTask; they must form a 1:1 bijection
with the ground-truth test table (every test row covered exactly once, no extras, no
duplicate keys).

Public API
==========
* :func:`write_prediction_table` -- write a prediction-table CSV from a predictions array.
* :func:`evaluate_task` -- score one prediction-table CSV against a task, returning the
  same metric dict RelBench's own ``task.evaluate`` produces.
* :func:`evaluate_submission` -- discover prediction CSVs in a directory, score every task
  (in parallel), aggregate per leaderboard family, and report per-family suitability.

CLI
===
``python -m relbench.leaderboard <pred_dir> [--num-workers N] [--quiet]`` runs
:func:`evaluate_submission` and prints the report. It exits ``0`` if *at least one*
leaderboard family is validated (complete and fully valid), and non-zero otherwise.
``--package`` additionally writes a clean submission zip to attach to a leaderboard
submission issue on GitHub (method metadata is entered in the issue form itself).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from relbench.base import TaskType
from relbench.hf import RELBENCH_HF
from relbench.load import load_task

__all__ = [
    "LEADERBOARD_TASKS",
    "write_prediction_table",
    "evaluate_task",
    "evaluate_submission",
    "main",
]


# --------------------------------------------------------------------------- #
# Canonical leaderboard task lists. A submission is graded per family; the three
# families are independent (a submission can qualify for one without the others).
# --------------------------------------------------------------------------- #
LEADERBOARD_TASKS: Dict[str, List[str]] = {
    "classification": [
        "rel-amazon/user-churn",
        "rel-amazon/item-churn",
        "rel-avito/user-visits",
        "rel-avito/user-clicks",
        "rel-event/user-repeat",
        "rel-event/user-ignore",
        "rel-f1/driver-dnf",
        "rel-f1/driver-top3",
        "rel-hm/user-churn",
        "rel-stack/user-engagement",
        "rel-stack/user-badge",
        "rel-trial/study-outcome",
    ],
    "regression": [
        "rel-amazon/user-ltv",
        "rel-amazon/item-ltv",
        "rel-avito/ad-ctr",
        "rel-event/user-attendance",
        "rel-f1/driver-position",
        "rel-hm/item-sales",
        "rel-stack/post-votes",
        "rel-trial/study-adverse",
        "rel-trial/site-success",
    ],
    "recommendation": [
        "rel-amazon/user-item-purchase",
        "rel-amazon/user-item-rate",
        "rel-amazon/user-item-review",
        "rel-avito/user-ad-visit",
        "rel-f1/driver-circuit-compete",
        "rel-hm/user-item-purchase",
        "rel-stack/user-post-comment",
        "rel-stack/post-post-related",
        "rel-trial/condition-sponsor-run",
        "rel-trial/site-sponsor-run",
    ],
}

# Reverse lookup: fully-qualified task name -> leaderboard family.
_TASK_TO_FAMILY: Dict[str, str] = {
    task: family for family, tasks in LEADERBOARD_TASKS.items() for task in tasks
}

# Display fallback when no task in a family was scored (the metric is otherwise read off
# the loaded task's evaluation result).
_FAMILY_METRIC: Dict[str, str] = {
    "classification": "roc_auc",
    "regression": "nmae",
    "recommendation": "link_prediction_map",
}

# Padding id for link-prediction rows shorter than eval_k. Destination ids are
# non-negative reindexed integers, so -1 can never match a ground-truth id.
_LINK_PAD = -1


# --------------------------------------------------------------------------- #
# Submission directory contents
# --------------------------------------------------------------------------- #
# A submission is just the prediction tables: one ``<dataset>__<task>.csv`` per task. Method metadata (name, url, note, in-context flag) is entered in the GitHub
# submission-issue form, not carried in the directory.
def _is_prediction_file(p: Path) -> bool:
    return p.suffix == ".csv"


def _extra_files(pred_dir: Union[str, os.PathLike]) -> List[str]:
    r"""Names of entries in ``pred_dir`` that are not prediction tables.

    A submission must contain **only** the prediction-table CSVs — anything else (other
    file types, stray artifacts, subdirectories) is reported so it can be removed.
    Hidden dotfiles are ignored.
    """
    extra: List[str] = []
    for p in sorted(Path(pred_dir).iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            extra.append(p.name + "/")
        elif not _is_prediction_file(p):
            extra.append(p.name)
    return extra


# --------------------------------------------------------------------------- #
# Task introspection helpers
# --------------------------------------------------------------------------- #
def _is_link(task: Any) -> bool:
    return task.task_type == TaskType.LINK_PREDICTION


def _key_cols(task: Any) -> List[str]:
    r"""The non-prediction columns that uniquely identify a test row."""
    if _is_link(task):
        return [task.src_entity_col, task.time_col]
    return [task.entity_col, task.time_col]


def _pred_col(task: Any) -> str:
    r"""The column that holds predictions in a prediction table."""
    if _is_link(task):
        return task.dst_entity_col
    return task.target_col


def _supported(task: Any) -> None:
    if task.task_type not in (
        TaskType.BINARY_CLASSIFICATION,
        TaskType.REGRESSION,
        TaskType.LINK_PREDICTION,
    ):
        raise NotImplementedError(
            f"leaderboard evaluation supports binary_classification, regression and "
            f"link_prediction only; got task_type={task.task_type}"
        )


# --------------------------------------------------------------------------- #
# write_prediction_table
# --------------------------------------------------------------------------- #
def write_prediction_table(
    task: Any,
    pred: Any,
    path: Union[str, os.PathLike],
    split: str = "test",
) -> Path:
    r"""Write a prediction-table CSV for ``task`` from a predictions array.

    Args:
        task: A loaded RelBench task (EntityTask or RecommendationTask).
        pred: Predictions aligned to ``task.get_table(split, mask_input_cols=True)`` row
            order. For an EntityTask: a 1-D array of shape ``(N,)`` (probabilities for
            binary classification, numeric values for regression). For a
            RecommendationTask: an array of shape ``(N, eval_k)`` of destination ids.
        path: Output CSV path. Parent directories are created as needed.
        split: Which split to write predictions for (default ``"test"``).

    Returns:
        The path the CSV was written to.

    The CSV columns are the task's key columns followed by the prediction column
    (``target_col`` for entity tasks, ``dst_entity_col`` for link tasks). For link
    prediction each cell is a JSON-encoded list of the predicted destination ids.
    """
    _supported(task)
    masked = task.get_table(split, mask_input_cols=True)
    key_cols = _key_cols(task)
    pred_col = _pred_col(task)
    n = len(masked)

    out = masked.df[key_cols].copy()

    if _is_link(task):
        rows = list(pred)
        if len(rows) != n:
            raise ValueError(
                f"pred has {len(rows)} rows but the masked '{split}' table has {n}"
            )
        out[pred_col] = [json.dumps([int(x) for x in row]) for row in rows]
    else:
        arr = np.asarray(pred).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(
                f"pred has {arr.shape[0]} values but the masked '{split}' table has {n}"
            )
        out[pred_col] = arr

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Validation + alignment
# --------------------------------------------------------------------------- #
def _coerce_keys(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, key_cols: Sequence[str]
) -> pd.DataFrame:
    r"""Coerce ``pred_df`` key columns to the ground-truth dtypes so joins/compares line
    up (CSV reads timestamps as strings, etc.)."""
    pred_df = pred_df.copy()
    for col in key_cols:
        ref_dtype = gt_df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(ref_dtype):
            pred_df[col] = pd.to_datetime(pred_df[col])
        else:
            try:
                pred_df[col] = pred_df[col].astype(ref_dtype)
            except (TypeError, ValueError):
                pass  # leave as-is; the key-set check below surfaces any mismatch
    return pred_df


def _key_set(df: pd.DataFrame, key_cols: Sequence[str]) -> set:
    r"""A dtype-robust set of key tuples (compared as strings)."""
    cols = []
    for col in key_cols:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s)
        cols.append(s.astype(str).tolist())
    return set(zip(*cols)) if cols else set()


def _decode_id_list(raw: Any) -> List[int]:
    r"""Decode one link-prediction cell into a list of integer destination ids."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        seq = list(raw)
    else:
        try:
            seq = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"could not decode link-prediction cell as a JSON list: {raw!r} ({exc})"
            )
    if not isinstance(seq, list):
        raise ValueError(
            f"link-prediction cell must decode to a JSON list, got "
            f"{type(seq).__name__}: {raw!r}"
        )
    out: List[int] = []
    for x in seq:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            raise ValueError(f"link-prediction id {x!r} is not an integer")
    return out


def _validate_keys(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, key_cols: Sequence[str], pred_col: str
) -> None:
    r"""Check column presence, key uniqueness and the exact GT key bijection."""
    missing_cols = [c for c in key_cols if c not in pred_df.columns]
    if missing_cols:
        raise ValueError(
            f"prediction CSV is missing key column(s) {missing_cols}; "
            f"found columns {list(pred_df.columns)}"
        )
    if pred_col not in pred_df.columns:
        raise ValueError(
            f"prediction CSV is missing the prediction column '{pred_col}'; "
            f"found columns {list(pred_df.columns)}"
        )

    dup = pred_df.duplicated(subset=list(key_cols))
    if dup.any():
        raise ValueError(
            f"prediction CSV has {int(dup.sum())} duplicate key row(s) over {list(key_cols)}"
        )

    gt_keys = _key_set(gt_df, key_cols)
    pred_keys = _key_set(pred_df, key_cols)
    missing = gt_keys - pred_keys
    extra = pred_keys - gt_keys
    if missing or extra:
        raise ValueError(
            f"prediction keys do not match the ground-truth test set: "
            f"{len(missing)} missing, {len(extra)} extra "
            f"(GT has {len(gt_keys)} rows, CSV has {len(pred_keys)} unique keys)"
        )


def _aligned_to_gt(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, key_cols: Sequence[str], value_col: str
) -> pd.Series:
    r"""Left-join ``pred_df`` onto the GT key frame, returning ``value_col`` in GT
    order."""
    order = gt_df[list(key_cols)].copy()
    order["__order__"] = np.arange(len(order))
    merged = order.merge(
        pred_df[list(key_cols) + [value_col]], on=list(key_cols), how="left"
    ).sort_values("__order__")
    return merged[value_col]


def _build_pred_array(
    task: Any, gt_df: pd.DataFrame, pred_df: pd.DataFrame
) -> np.ndarray:
    r"""Validate ``pred_df`` against ``gt_df`` and return predictions aligned to GT order
    in the exact shape ``task.evaluate`` expects.

    Raises ``ValueError`` with a clear message on any validation failure.
    """
    key_cols = _key_cols(task)
    pred_col = _pred_col(task)

    _validate_keys(pred_df, gt_df, key_cols, pred_col)
    pred_df = _coerce_keys(pred_df[list(key_cols) + [pred_col]], gt_df, key_cols)

    if _is_link(task):
        eval_k = task.eval_k
        decoded = []
        for raw in pred_df[pred_col].tolist():
            ids = _decode_id_list(raw)
            if len(ids) > eval_k:
                raise ValueError(
                    f"link-prediction row has {len(ids)} ids, exceeding eval_k={eval_k}"
                )
            decoded.append(ids)
        pred_df = pred_df.assign(**{pred_col: decoded})
        ordered = _aligned_to_gt(pred_df, gt_df, key_cols, pred_col).tolist()
        arr = np.full((len(ordered), eval_k), _LINK_PAD, dtype=np.int64)
        for i, ids in enumerate(ordered):
            for j, v in enumerate(ids[:eval_k]):
                arr[i, j] = int(v)
        return arr

    # entity task (binary classification / regression)
    vals = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError(
            f"prediction column '{pred_col}' contains non-finite (NaN/inf) values"
        )
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        if vals.min() < 0.0 or vals.max() > 1.0:
            raise ValueError(
                f"binary-classification predictions in '{pred_col}' must lie in [0, 1] "
                f"(got min={vals.min()}, max={vals.max()})"
            )
    pred_df = pred_df.assign(__val__=vals)
    return _aligned_to_gt(pred_df, gt_df, key_cols, "__val__").to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# evaluate_task
# --------------------------------------------------------------------------- #
def _split_task_name(task_name: str) -> Tuple[str, str]:
    if "/" not in task_name:
        raise ValueError(
            f"task name {task_name!r} must be fully-qualified as '<dataset>/<task>'"
        )
    dataset_name, name = task_name.split("/", 1)
    return dataset_name, name


def evaluate_task(
    task_name: str,
    csv_path: Union[str, os.PathLike],
    dataset: Optional[Any] = None,
) -> Dict[str, float]:
    r"""Score a prediction-table CSV for a single task.

    Args:
        task_name: Fully-qualified ``"<dataset>/<task>"`` (e.g. ``"rel-amazon/user-churn"``).
        csv_path: Path to the prediction-table CSV (see :func:`write_prediction_table`).
        dataset: Optional dataset override passed straight to
            :func:`relbench.load.load_task` -- a local dataset directory, a Hub
            ``org/repo[/subdir]`` spec, or a loaded ``RelBenchDataset``. When ``None`` the
            dataset is resolved from the ``task_name`` prefix as
            ``"<RELBENCH_HF>/<dataset>"`` (the hosted RelBench core location).

    Returns:
        The metric dict produced by the task's own ``task.evaluate`` (identical to the
        in-memory modelling path), e.g. ``{"roc_auc": 0.83}``.

    Raises:
        ValueError: if the CSV fails validation (missing/extra/duplicate keys, out-of-range
            probabilities, undecodable link lists, ...).
    """
    dataset_name, name = _split_task_name(task_name)
    dataset_arg = dataset if dataset is not None else f"{RELBENCH_HF}/{dataset_name}"
    task = load_task(dataset_arg, name)
    _supported(task)

    gt_table = task.get_table("test", mask_input_cols=False)
    pred_df = pd.read_csv(csv_path)
    pred_array = _build_pred_array(task, gt_table.df, pred_df)
    return task.evaluate(pred_array, target_table=gt_table)


# --------------------------------------------------------------------------- #
# evaluate_submission
# --------------------------------------------------------------------------- #
def _task_name_from_path(path: Path) -> str:
    r"""``<dataset>__<task>.csv`` -> ``<dataset>/<task>``.

    The first double underscore separates dataset from task; task names may contain
    single hyphens (and are otherwise left untouched).
    """
    stem = path.stem
    if "__" not in stem:
        return stem
    dataset_name, name = stem.split("__", 1)
    return f"{dataset_name}/{name}"


def _quiet_hf_progress() -> None:
    r"""Disable huggingface_hub progress bars.

    Every task triggers a (usually fully cached) ``snapshot_download`` of its dataset, so
    a multi-task submission would otherwise print dozens of no-op "Fetching N files" /
    "Download complete: 0.00B" bars. Called in the parent and as the process-pool
    initializer, since the setting is per-process.
    """
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001 -- cosmetic only; never fail evaluation over it
        pass


def _evaluate_one(job: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    r"""Process-pool worker: score one task, capturing success or a failure reason.

    Module-level (picklable) so it can run under :class:`ProcessPoolExecutor`.
    """
    task_name, csv_path = job
    try:
        metrics = evaluate_task(task_name, csv_path)
        return task_name, {
            "status": "ok",
            "metrics": {k: float(v) for k, v in metrics.items()},
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 -- report any failure as a per-task status
        return task_name, {
            "status": "error",
            "metrics": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def evaluate_submission(
    pred_dir: Union[str, os.PathLike],
    *,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    r"""Evaluate a directory of prediction CSVs as a leaderboard submission.

    Discovers ``*.csv`` files in ``pred_dir`` (filename convention
    ``<dataset>__<task>.csv`` -> task ``<dataset>/<task>``), scores every task in parallel,
    groups results by leaderboard family, and reports per-family suitability.

    Args:
        pred_dir: Directory containing the prediction CSVs.
        num_workers: Process-pool size. ``None`` uses
            ``min(num_tasks, os.cpu_count())``. ``1`` (or fewer) runs synchronously
            in-process (no pool).
        verbose: If True, print a per-task table and the per-family verdicts.

    Returns:
        A structured dict::

            {
              "tasks": {
                "<dataset>/<task>": {
                    "family": str | None,
                    "status": "ok" | "error",
                    "metric_name": str | None,
                    "metric": float | None,
                    "metrics": dict | None,
                    "error": str | None,
                },
                ...
              },
              "families": {
                "<family>": {
                    "metric_name": str,
                    "aggregate": float | None,   # mean metric over valid tasks
                    "num_valid": int,
                    "num_total": int,
                    "present": [task, ...],
                    "valid": [task, ...],
                    "invalid": [task, ...],      # present but failed validation/eval
                    "missing": [task, ...],      # canonical but absent
                    "complete": bool,            # all canonical tasks present and valid
                    "verdict": str,
                },
                ...
              },
              "validated": [family, ...],        # families that passed (all tasks valid)
              "extra_files": [str],              # non-prediction-table entries (ignored)
            }
    """
    pred_dir = Path(pred_dir)
    csv_paths = sorted(pred_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no prediction CSVs (*.csv) found in {pred_dir}")

    jobs: List[Tuple[str, str]] = [(_task_name_from_path(p), str(p)) for p in csv_paths]

    if num_workers is None:
        num_workers = min(len(jobs), os.cpu_count() or 1)

    if verbose:
        print(
            f"Evaluating {len(jobs)} prediction tables with {num_workers} workers.\n"
            f"(Downloads test sets from HuggingFace if not cached)",
            flush=True,
        )

    _quiet_hf_progress()
    if num_workers <= 1:
        raw = [_evaluate_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=_quiet_hf_progress
        ) as executor:
            raw = list(executor.map(_evaluate_one, jobs))

    tasks_out: Dict[str, Dict[str, Any]] = {}
    for task_name, res in raw:
        entry: Dict[str, Any] = {
            "family": _TASK_TO_FAMILY.get(task_name),
            "status": res["status"],
            "metric_name": None,
            "metric": None,
            "metrics": res["metrics"],
            "error": res["error"],
        }
        if res["status"] == "ok" and res["metrics"]:
            metric_name = next(iter(res["metrics"]))
            entry["metric_name"] = metric_name
            entry["metric"] = float(res["metrics"][metric_name])
        tasks_out[task_name] = entry

    families_out: Dict[str, Dict[str, Any]] = {}
    validated: List[str] = []
    for family, canonical in LEADERBOARD_TASKS.items():
        present = [t for t in canonical if t in tasks_out]
        valid = [t for t in present if tasks_out[t]["status"] == "ok"]
        invalid = [t for t in present if tasks_out[t]["status"] != "ok"]
        missing = [t for t in canonical if t not in tasks_out]
        complete = len(valid) == len(canonical)

        metric_vals = [
            tasks_out[t]["metric"] for t in valid if tasks_out[t]["metric"] is not None
        ]
        aggregate = float(np.mean(metric_vals)) if metric_vals else None
        metric_name = _FAMILY_METRIC.get(family, "metric")
        if valid:
            metric_name = tasks_out[valid[0]]["metric_name"] or metric_name

        if complete:
            verdict = (
                f"validated for the {family} leaderboard "
                f"({len(valid)}/{len(canonical)} tasks valid)"
            )
            validated.append(family)
        else:
            verdict = (
                f"rejected for the {family} leaderboard "
                f"({len(valid)}/{len(canonical)} tasks valid) "
                f"-- missing: {missing}; invalid: {invalid}"
            )

        families_out[family] = {
            "metric_name": metric_name,
            "aggregate": aggregate,
            "num_valid": len(valid),
            "num_total": len(canonical),
            "present": present,
            "valid": valid,
            "invalid": invalid,
            "missing": missing,
            "complete": complete,
            "verdict": verdict,
        }

    result = {
        "tasks": tasks_out,
        "families": families_out,
        "validated": validated,
        "extra_files": _extra_files(pred_dir),
    }
    if verbose:
        _print_report(result)
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_report(result: Dict[str, Any]) -> None:
    tasks = result["tasks"]
    families = result["families"]

    print("=" * 80)
    print("RelBench leaderboard submission report")
    print("=" * 80)

    extra = result.get("extra_files") or []
    if extra:
        print("Not part of a submission (--package drops these when building the zip):")
        for name in extra:
            print(f"  - {name}")
        print()

    name_w = max([len("task")] + [len(t) for t in tasks]) if tasks else len("task")
    metric_w = max(
        [len("metric")] + [len(e["metric_name"] or "") for e in tasks.values()]
    )
    header = (
        f"  {'task'.ljust(name_w)}  {'metric'.ljust(metric_w)}  {'value':>12}  status"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for task_name in sorted(tasks):
        entry = tasks[task_name]
        metric_name = entry["metric_name"] or "-"
        if entry["metric"] is not None:
            value = f"{entry['metric']:.6f}"
        else:
            value = "-"
        status = entry["status"]
        if status != "ok" and entry["error"]:
            status = f"error ({entry['error']})"
        print(
            f"  {task_name.ljust(name_w)}  {metric_name.ljust(metric_w)}  "
            f"{value:>12}  {status}"
        )

    print()
    print("Per-family verdicts:")
    print("  " + "-" * 78)
    for family in LEADERBOARD_TASKS:
        fam = families[family]
        mark = "[validated] " if fam["complete"] else "[rejected]  "
        agg = "n/a" if fam["aggregate"] is None else f"{fam['aggregate']:.6f}"
        print(
            f"  {mark}{family:<16} valid={fam['num_valid']}/{fam['num_total']}  "
            f"metric={fam['metric_name']}  aggregate={agg}"
        )
        print(f"        {fam['verdict']}")
    print()
    if result["validated"]:
        print(f"Validated leaderboard(s): {', '.join(result['validated'])}")
    else:
        print("Validated leaderboard(s): none")
    print("=" * 80)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
# Where validated submissions are turned into leaderboard entries: a submission is a GitHub
# issue on the RelBench repository, created from the submission form with the prediction
# tables attached. CI validates it and a maintainer approval publishes the entry.
SUBMISSION_ISSUE_URL = (
    "https://github.com/rishabh-ranjan/relbench/issues/new?template=submission.yml"
)


def _zip_submission(pred_dir: Union[str, os.PathLike], extra: Sequence[str]) -> bytes:
    r"""Zip the submission (prediction tables only) and return the bytes.

    Anything else in the directory (``extra``) is reported and left out of the zip.
    """
    import io
    import zipfile

    pred_dir = Path(pred_dir)
    if extra:
        print("\nExcluding files that aren't part of a submission:")
        for name in extra:
            print(f"  - {name}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(pred_dir.iterdir()):
            if p.is_file() and _is_prediction_file(p):
                zf.write(p, p.name)
    return buf.getvalue()


def _package(
    pred_dir: Union[str, os.PathLike], extra: Sequence[str], out: Path
) -> None:
    r"""Write the submission zip and print how to submit it."""
    zip_bytes = _zip_submission(pred_dir, extra)
    out.write_bytes(zip_bytes)
    print(f"\nCreated submission package: {out} ({len(zip_bytes) / 1e6:.1f} MB)")
    print("\nSubmit by opening a submission issue and dragging the zip into it:")
    print(f"  {SUBMISSION_ISSUE_URL}")
    print(
        "  (method name, links and the in-context flag are entered in the issue form)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    r"""CLI entry point.

    Returns the process exit code (0 if any family is validated).
    """
    parser = argparse.ArgumentParser(
        prog="python -m relbench.leaderboard",
        description=(
            "Evaluate a directory of RelBench prediction CSVs as a leaderboard "
            "submission. Exit code 0 if at least one leaderboard family is validated "
            "(all of its tasks present and valid), non-zero otherwise."
        ),
    )
    parser.add_argument(
        "pred_dir", help="directory of <dataset>__<task>.csv prediction files"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="process-pool size (default: min(num_tasks, cpu_count); 1 runs in-process)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the printed report"
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="after validating, write a clean submission zip (prediction tables only) and "
        "print how to submit it as a GitHub issue",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output zip path for --package (default: <dir-name>.zip in the cwd)",
    )
    args = parser.parse_args(argv)

    result = evaluate_submission(
        args.pred_dir, num_workers=args.num_workers, verbose=not args.quiet
    )

    if args.package:
        if not result["validated"]:
            print(
                "\nCannot package — no leaderboard was validated; fix the prediction "
                "tables first."
            )
            return 1
        out = Path(args.out or f"{Path(args.pred_dir).resolve().name}.zip")
        _package(args.pred_dir, result.get("extra_files") or [], out)
        return 0

    return 0 if result["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
