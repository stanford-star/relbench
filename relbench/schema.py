r"""Generate a schema diagram + dataset card for a RelBench dataset.

The diagram is a `Mermaid <https://mermaid.js.org>`_ ``erDiagram`` built from the
manifest's foreign-key graph. Hugging Face dataset cards (README.md) render Mermaid code
fences natively, so embedding it there gives a zoomable schema view on the Hub with no
image dependencies (and it degrades to readable text everywhere else).
"""

from __future__ import annotations

from typing import Iterable, Optional

from relbench.manifest import DatasetManifest, TaskManifest


def mermaid_er(manifest: DatasetManifest) -> str:
    r"""Return a Mermaid ``erDiagram`` of the dataset's tables and foreign-key graph."""
    lines = ["erDiagram"]
    for tname, spec in manifest.tables.items():
        lines.append(f"    {tname} {{")
        if spec.pkey:
            lines.append(f"        key {spec.pkey} PK")
        for fkey in spec.fkeys:
            lines.append(f"        key {fkey} FK")
        if spec.time_col:
            lines.append(f"        datetime {spec.time_col}")
        lines.append("    }")
    for tname, spec in manifest.tables.items():
        for fkey, parent in spec.fkeys.items():
            # child many-to-one parent, labeled by the foreign-key column.
            lines.append(f"    {tname} }}o--|| {parent} : {fkey}")
    return "\n".join(lines)


def graphviz_er(manifest: DatasetManifest):
    r"""Build a graphviz ER diagram (tables with PK/FK/time, edges for the fkey graph)."""
    import graphviz

    g = graphviz.Digraph("schema")
    g.attr(rankdir="LR", bgcolor="white", pad="0.3")
    g.attr("node", shape="plaintext", fontname="Helvetica")
    g.attr("edge", color="#555555", fontname="Helvetica", fontsize="10", arrowhead="crow")
    for tname, spec in manifest.tables.items():
        rows = [f'<tr><td bgcolor="#4C78A8" align="center">'
                f'<font color="white"><b>{tname}</b></font></td></tr>']
        if spec.pkey:
            rows.append(f'<tr><td align="left"><b>PK</b>  {spec.pkey}</td></tr>')
        for fkey in spec.fkeys:
            rows.append(f'<tr><td align="left"><b>FK</b>  {fkey}</td></tr>')
        if spec.time_col:
            rows.append(f'<tr><td align="left"><i>time</i>  {spec.time_col}</td></tr>')
        label = ('<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">'
                 + "".join(rows) + "</table>>")
        g.node(tname, label=label)
    for tname, spec in manifest.tables.items():
        for fkey, parent in spec.fkeys.items():
            g.edge(tname, parent, label=fkey)
    return g


def render_schema_svg(manifest: DatasetManifest, path) -> None:
    r"""Render the ER diagram to a standalone SVG at ``path`` (zoomable on the HF Hub)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graphviz_er(manifest).render(outfile=str(path), cleanup=True)


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
        "Open [`schema.svg`](schema.svg) for a zoomable view of the foreign-key graph "
        "(PK = primary key, FK = foreign key).",
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
