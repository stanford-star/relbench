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


def dataset_card(
    manifest: DatasetManifest, tasks: Optional[Iterable[TaskManifest]] = None
) -> str:
    r"""Return README.md content (dataset card) with the schema diagram and task table."""
    parts = [f"# {manifest.name}", ""]
    if manifest.description:
        parts += [manifest.description.strip(), ""]
    parts += [
        "## Schema",
        "",
        "```mermaid",
        mermaid_er(manifest),
        "```",
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
