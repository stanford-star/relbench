r"""Generate a schema diagram + dataset card for a RelBench dataset.

The diagram is rendered with `sqlalchemy_schemadisplay
<https://pypi.org/project/sqlalchemy-schemadisplay/>`_: we reconstruct a SQLAlchemy
``MetaData`` from the parquet column schemas (every column, with its type) plus the
manifest's primary-/foreign-key graph, and let the library draw a standard ER diagram via
graphviz. The result is a standalone ``schema.svg`` that the Hugging Face file viewer
renders as a zoomable image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from relbench.manifest import DatasetManifest, TaskManifest


def _arrow_to_sa(arrow_type):
    r"""Map a pyarrow column type to a generic SQLAlchemy type (for display only)."""
    import pyarrow as pa
    import sqlalchemy as sa

    if pa.types.is_boolean(arrow_type):
        return sa.Boolean()
    if pa.types.is_integer(arrow_type):
        return sa.BigInteger() if pa.types.is_int64(arrow_type) else sa.Integer()
    if pa.types.is_floating(arrow_type) or pa.types.is_decimal(arrow_type):
        return sa.Float()
    if pa.types.is_date(arrow_type):
        return sa.Date()
    if pa.types.is_timestamp(arrow_type):
        return sa.DateTime()
    return sa.Text()


def _metadata_from_db(manifest: DatasetManifest, db_dir: Union[str, Path]):
    r"""Build a SQLAlchemy ``MetaData`` (all columns + types + PK/FK graph) for the dataset.

    Columns and types come from the parquet schemas in ``db_dir``; primary keys, foreign
    keys, and time columns come from the manifest. Tables whose parquet is missing fall back
    to the columns named in the manifest (pkey / fkeys / time_col) so rendering never fails.
    """
    import pyarrow.parquet as pq
    import sqlalchemy as sa

    db_dir = Path(db_dir)
    md = sa.MetaData()
    parent_pkey = {t: s.pkey for t, s in manifest.tables.items()}

    for tname, spec in manifest.tables.items():
        path = db_dir / f"{tname}.parquet"
        if path.exists():
            schema = pq.read_schema(path)
            cols = [(f.name, _arrow_to_sa(f.type)) for f in schema]
        else:  # fall back to the manifest-named columns
            named = [c for c in [spec.pkey, spec.time_col, *spec.fkeys] if c]
            cols = [(c, sa.Text()) for c in dict.fromkeys(named)]

        columns = []
        for cname, ctype in cols:
            args = []
            parent = spec.fkeys.get(cname)
            if parent and parent_pkey.get(parent):
                args.append(sa.ForeignKey(f"{parent}.{parent_pkey[parent]}"))
            columns.append(
                sa.Column(cname, ctype, *args, primary_key=(cname == spec.pkey))
            )
        sa.Table(tname, md, *columns)

    return md


def render_schema_svg(
    manifest: DatasetManifest, path, db_dir: Optional[Union[str, Path]] = None
) -> None:
    r"""Render the ER diagram to a standalone SVG at ``path`` (zoomable on the HF Hub).

    ``db_dir`` is the dataset's ``db/`` folder; when given, every column and its type is
    shown. It defaults to ``<path>/../db`` (the standard dataset layout).
    """
    import sqlalchemy as sa
    from sqlalchemy_schemadisplay import create_schema_graph

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if db_dir is None:
        db_dir = path.parent / "db"

    md = _metadata_from_db(manifest, db_dir)
    graph = create_schema_graph(
        engine=sa.create_engine("sqlite://"),
        metadata=md,
        show_datatypes=True,
        show_indexes=False,
        show_column_keys=True,
        rankdir="LR",
        concentrate=False,
        font="Helvetica",
    )
    graph.write_svg(str(path), prog="dot")


def dataset_card(
    manifest: DatasetManifest, tasks: Optional[Iterable[TaskManifest]] = None
) -> str:
    r"""Return README.md content (dataset card) with the schema diagram and task table."""
    parts = [
        "---",
        "tags:",
        "- relbench",
        "- relational-deep-learning",
        f"pretty_name: {manifest.name}",
        "---",
        "",
        f"# {manifest.name}",
        "",
    ]
    if manifest.description:
        parts += [manifest.description.strip(), ""]
    parts += [
        "## Schema",
        "",
        "![schema diagram](schema.svg)",
        "",
        "Open [`schema.svg`](schema.svg) for a zoomable view: each table lists all of its "
        "columns and types, with primary keys, foreign keys, and the foreign-key edges "
        "between tables.",
        "",
        f"Splits: validation `{manifest.val_timestamp}`, test `{manifest.test_timestamp}` "
        "(rows up to a split's timestamp are the inputs for that split).",
        "",
    ]
    tasks = list(tasks or [])
    if tasks:
        parts += ["## Tasks", "", "| task | kind | type | description |", "|---|---|---|---|"]
        for t in tasks:
            parts.append(
                f"| `{t.name}` | {t.kind} | {t.task_type} | {(t.description or '').strip()} |"
            )
        parts.append("")
    parts += [
        "## Loading",
        "",
        "```python",
        "import relbench",
        f'ds = relbench.load_dataset("{manifest.name}")',
        f'task = relbench.load_task("{manifest.name}", "<task>")',
        "```",
        "",
        "Manifest layout (`manifest.yaml` + plain parquet); see the RelBench "
        "[CONTRIBUTING guide](https://github.com/snap-stanford/relbench/blob/main/CONTRIBUTING.md).",
    ]
    return "\n".join(parts) + "\n"
