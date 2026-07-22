r"""Generate a schema diagram + dataset card for a RelBench dataset.

The diagram is a hand-styled entity-relationship diagram rendered with `graphviz
<https://pypi.org/project/graphviz/>`_ (the ``dot`` engine lays out the tables; the
foreign-key connectors and a couple of cosmetic touch-ups are applied in a small SVG
post-process). Each table is a card with a header (table name + row count) and its columns
ordered ``pk, fk, time, float, int, str`` and colour-coded; the second column (badges,
dtypes, row count) is right-justified. Foreign keys are drawn in standard ER crow's-foot
notation (crow's foot = "many" at the FK, single bar = "one" at the referenced primary
key). The result is a transparent ``schema.svg`` (works on light and dark Hub themes) that
the Hugging Face file viewer renders as a zoomable image.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple, Union

from relbench.manifest import DatasetManifest, TaskManifest

# Palette (chosen to read on both light and dark Hub themes).
_HEADER = "#5273A6"  # medium blue header, white text
_HTXT = "white"
_COUNT = "#FFFFFF"
_PKBG = "#FCEFC7"  # primary key row (warm yellow)
_TIMEBG = "#DFF1E1"  # time column row (green)
_FKBG = "#DEE8F6"  # foreign key row (light blue)
_FEATBG = "#EDEFF2"  # feature column row (light gray)
_BORDER = "#D3D9E2"
_EDGE = "#9AA7B8"
_PKTAG = "#B0791A"
_FKTAG = "#5273A6"
_TTAG = "#3B9B5B"
_TYPE = "#6B7686"  # dtype text

_FS = 11  # base font size
_BADGE = 10  # PK/FK/TIME badge size
_RH = 22  # uniform non-header row height
_HH = int(1.2 * _RH)  # header height = 1.2x a normal row
_NAME_CAP = 20  # cap longest name used for the (uniform) table width
_DTYPE_RANK = {"float": 3, "int": 4, "str": 5}

# A reader returns ``(columns, num_rows)`` for a table, or ``(None, None)`` if unavailable.
# ``columns`` is a list of ``(name, short_dtype)``.
Reader = Callable[[str], Tuple[Optional[list], Optional[int]]]


def _short_type(arrow_type) -> str:
    r"""Map a pyarrow column type to a short display string."""
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_integer(arrow_type):
        return "int"
    if pa.types.is_floating(arrow_type) or pa.types.is_decimal(arrow_type):
        return "float"
    if pa.types.is_date(arrow_type):
        return "date"
    if pa.types.is_timestamp(arrow_type):
        return "datetime"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "str"
    return str(arrow_type)


def _fmt_count(n: Optional[int]) -> str:
    r"""Row count in K/M/B shorthand, rounded to no decimal places."""
    if n is None:
        return ""
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{round(n / 1e3)}K"
    if n < 1_000_000_000:
        return f"{round(n / 1e6)}M"
    return f"{round(n / 1e9)}B"


def _rows_word(count: str) -> str:
    return f"{count} rows" if count else ""


def _esc(s) -> str:
    return html.escape(str(s))


def _badge(text: str, color: str) -> str:
    return f'<FONT COLOR="{color}" POINT-SIZE="{_BADGE}"><B><I>{text}</I></B></FONT>'


def _parquet_reader(db_dir: Path) -> Reader:
    import pyarrow.parquet as pq

    def reader(tname: str):
        p = db_dir / f"{tname}.parquet"
        if not p.exists():
            return None, None
        cols = [(f.name, _short_type(f.type)) for f in pq.read_schema(p)]
        return cols, pq.read_metadata(p).num_rows

    return reader


def render_schema_svg(
    manifest: DatasetManifest,
    path,
    db_dir: Optional[Union[str, Path]] = None,
    reader: Optional[Reader] = None,
) -> None:
    r"""Render the ER diagram to a standalone transparent SVG at ``path``.

    Column names/types and row counts come from the parquet files: by default they are read
    from ``db_dir`` (the dataset's ``db/`` folder, defaulting to ``<path>/../db``). Pass a
    custom ``reader`` to source them elsewhere (e.g. parquet footers read straight from the
    Hub, avoiding a full download). Tables whose data is unavailable fall back to the
    manifest-named columns so rendering never fails.
    """
    import graphviz

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reader is None:
        reader = _parquet_reader(
            Path(db_dir) if db_dir is not None else path.parent / "db"
        )

    tables = manifest.tables
    pkey_of = {t: s.pkey for t, s in tables.items()}

    # pass 1: collect ordered columns + row counts; size left/right columns.
    collected = {}
    lmax = rmax = 0
    for tname, spec in tables.items():
        pkey, tcol, fkeys = spec.pkey, spec.time_col, spec.fkeys
        cols, nrows = reader(tname)
        if cols is None:
            named = [c for c in [pkey, tcol, *fkeys] if c]
            cols = [(c, "") for c in dict.fromkeys(named)]

        def _rank(item, pkey=pkey, tcol=tcol, fkeys=fkeys):
            cname, ctype = item
            if cname == pkey:
                return 0
            if cname in fkeys:
                return 1
            if cname == tcol:
                return 2
            return _DTYPE_RANK.get(ctype, 6)

        cols = sorted(cols, key=_rank)
        count = _fmt_count(nrows)
        collected[tname] = (cols, count)
        lmax = max(lmax, len(tname))
        rmax = max(rmax, len(_rows_word(count)))
        for cname, ctype in cols:
            right = (
                "PK"
                if cname == pkey
                else "FK" if cname in fkeys else "TIME" if cname == tcol else ctype
            )
            lmax = max(lmax, len(cname))
            rmax = max(rmax, len(right))

    wl = int(min(lmax, _NAME_CAP) * 6.6) + 12  # left column (names), capped
    wr = int(min(rmax, 11) * 6.5) + 12  # right column (badges/dtypes/count)

    # pass 2: emit nodes (one HTML-table card per table) + base edges (FK -> PK).
    g = graphviz.Digraph("schema", engine="dot")
    g.attr(
        rankdir="LR",
        bgcolor="transparent",
        splines="line",
        nodesep="0.45",
        ranksep="1.1",
        pad="0.3",
        fontname="Helvetica",
    )
    g.attr("node", shape="plaintext", fontname="Helvetica")
    g.attr("edge", color=_EDGE, penwidth="1.3", dir="none")  # ER symbols added in post

    for tname, spec in tables.items():
        cols, count = collected[tname]
        pkey, tcol, fkeys = spec.pkey, spec.time_col, spec.fkeys
        cnt_txt = _esc(_rows_word(count))
        cnt_html = (
            f'<FONT COLOR="{_COUNT}" POINT-SIZE="{_FS}"><I>{cnt_txt}</I></FONT>'
            if cnt_txt
            else ""
        )
        rows = [
            f"<TR>"
            f'<TD WIDTH="{wl}" HEIGHT="{_HH}" BGCOLOR="{_HEADER}" PORT="__t" ALIGN="LEFT">'
            f'<FONT COLOR="{_HTXT}" POINT-SIZE="{_FS}"><B>{_esc(tname)}</B></FONT></TD>'
            f'<TD WIDTH="{wr}" HEIGHT="{_HH}" BGCOLOR="{_HEADER}" ALIGN="RIGHT">{cnt_html}</TD>'
            f"</TR>"
        ]
        for cname, ctype in cols:
            is_pk = cname == pkey
            is_fk = cname in fkeys
            is_t = cname == tcol
            bg = (
                _PKBG if is_pk else (_TIMEBG if is_t else (_FKBG if is_fk else _FEATBG))
            )
            nm = f"<I>{_esc(cname)}</I>" if (is_pk or is_fk) else _esc(cname)
            if is_pk:
                right = _badge("PK", _PKTAG)
            elif is_fk:
                right = _badge("FK", _FKTAG)
            elif is_t:
                right = _badge("TIME", _TTAG)
            elif ctype:
                right = f'<FONT COLOR="{_TYPE}" POINT-SIZE="{_FS}"><I>{_esc(ctype)}</I></FONT>'
            else:
                right = ""
            # left cell port = entry (PK, west edge); right cell port = exit (FK, east edge)
            rows.append(
                f'<TR><TD WIDTH="{wl}" HEIGHT="{_RH}" BGCOLOR="{bg}" PORT="{_esc(cname)}" ALIGN="LEFT">'
                f'<FONT POINT-SIZE="{_FS}">{nm}</FONT></TD>'
                f'<TD WIDTH="{wr}" HEIGHT="{_RH}" BGCOLOR="{bg}" PORT="{_esc(cname)}_r" ALIGN="RIGHT">{right}</TD></TR>'
            )
        g.node(
            tname,
            color=_BORDER,
            label=(
                f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" '
                f'STYLE="rounded">{"".join(rows)}</TABLE>>'
            ),
        )

    for tname, spec in tables.items():
        for fcol, parent in spec.fkeys.items():
            if parent in tables and pkey_of.get(parent):
                g.edge(f"{tname}:{fcol}_r:e", f"{parent}:{pkey_of[parent]}:w")

    g.render(str(path.with_suffix("")), format="svg", cleanup=True)
    _postprocess_svg(path)


def _postprocess_svg(svgpath: Path) -> None:
    r"""Two SVG touch-ups graphviz can't do directly:

    1. Rewrite each straight edge into ER crow's-foot notation -- a short horizontal stub at
       the FK end carrying a crow's foot ("many"), a straight middle, and a stub at the PK
       end carrying a single bar ("one"). Ports are fixed (``:e`` tail, ``:w`` head) so stub
       directions are constant (FK exits east, PK exits west).
    2. Widen the header-coloured background cells slightly so the two header cells overlap,
       hiding the 1px seam between the table name and the row count.
    """
    import re
    import xml.etree.ElementTree as ET

    NS = "http://www.w3.org/2000/svg"
    ET.register_namespace("", NS)
    tree = ET.parse(svgpath)
    graph = tree.getroot().find(f"{{{NS}}}g")
    if graph is None:
        return

    L, FH, TICK, TGAP = 16.0, 5.0, 5.0, 7.0
    for e in [c for c in graph if c.get("class") == "edge"]:
        path = e.find(f"{{{NS}}}path")
        if path is None:
            continue
        pts = re.findall(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", path.get("d", ""))
        if len(pts) < 2:
            continue
        x0, y0 = float(pts[0][0]), float(pts[0][1])
        x1, y1 = float(pts[-1][0]), float(pts[-1][1])
        color = path.get("stroke") or _EDGE
        for p in list(e.findall(f"{{{NS}}}path")):
            e.remove(p)

        def seg(xa, ya, xb, yb):
            el = ET.SubElement(e, f"{{{NS}}}path")
            el.set("d", f"M{xa},{ya} L{xb},{yb}")
            el.set("stroke", color)
            el.set("stroke-width", "1.3")
            el.set("fill", "none")

        ax, bx = x0 + L, x1 - L
        seg(x0, y0, ax, y0)  # FK horizontal stub
        seg(ax, y0, bx, y1)  # straight middle
        seg(bx, y1, x1, y1)  # PK horizontal stub
        seg(ax, y0, x0, y0 - FH)  # crow's foot (many) at FK
        seg(ax, y0, x0, y0 + FH)
        seg(x1 - TGAP, y1 - TICK, x1 - TGAP, y1 + TICK)  # single bar (one) at PK

    # Hide the header seam by overlapping the header-coloured background polygons.
    ex = 1.5
    hdr = _HEADER.lower()
    for poly in graph.iter(f"{{{NS}}}polygon"):
        if (poly.get("fill") or "").lower() != hdr:
            continue
        pts = re.findall(r"(-?\d+\.?\d*),(-?\d+\.?\d*)", poly.get("points", ""))
        if len(pts) < 4:
            continue
        xs = [float(a) for a, _ in pts]
        ys = [float(b) for _, b in pts]
        x0p, x1p, y0p, y1p = min(xs), max(xs), min(ys), max(ys)
        poly.set(
            "points",
            f"{x0p - ex},{y0p} {x1p + ex},{y0p} {x1p + ex},{y1p} {x0p - ex},{y1p}",
        )

    # paint edges behind the table boxes: SVG paints in document order, so move edge groups
    # ahead of node groups (after the background) -> connectors pass behind the cards.
    children = list(graph)
    edges = [c for c in children if c.get("class") == "edge"]
    nodes = [c for c in children if c.get("class") == "node"]
    others = [c for c in children if c.get("class") not in ("edge", "node")]
    for c in children:
        graph.remove(c)
    for c in others + edges + nodes:
        graph.append(c)

    tree.write(svgpath)


# RelBench (NeurIPS 2024 Datasets & Benchmarks) -- always cited for datasets hosted here.
RELBENCH_PAPER_URL = (
    "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
    "25cd345233c65fac1fec0ce61d0f7836-Abstract-Datasets_and_Benchmarks_Track.html"
)
RELBENCH_BIBTEX = """@inproceedings{robinson2024relbench,
  title     = {{RelBench}: A Benchmark for Deep Learning on Relational Databases},
  author    = {Robinson, Joshua and Ranjan, Rishabh and Hu, Weihua and Huang, Kexin and Han, Jiaqi and Dobles, Alejandro and Fey, Matthias and Lenssen, Jan E. and Yuan, Yiwen and Zhang, Zecheng and He, Xinwei and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024) Datasets and Benchmarks Track},
  year      = {2024}
}"""


def dataset_card(
    manifest: DatasetManifest,
    tasks: Optional[Iterable[TaskManifest]] = None,
    repo: Optional[str] = None,
    source: Optional[dict] = None,
) -> str:
    r"""Return README.md content (dataset card) with the schema diagram and task table.

    ``repo`` is the dataset's address for the loading example (a Hub ``org/repo`` or
    ``org/repo/subdir``, or a local path); defaults to the dataset name. ``source``, if
    given, describes the original dataset paper with keys ``label``, ``url``, ``bibtex``;
    the card cites it and adds a "please also cite RelBench" note.
    """
    addr = repo or manifest.name
    parts = [f"# {manifest.name}", ""]
    if manifest.description:
        parts += [manifest.description.strip(), ""]
    parts += ["## Schema", "", "![schema diagram](schema.svg)", ""]
    tasks = list(tasks or [])
    if tasks:
        parts += [
            "## Tasks",
            "",
            "| task | kind | type | description |",
            "|---|---|---|---|",
        ]
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
        f'ds = relbench.load_dataset("{addr}")',
        f'task = relbench.load_task("{addr}", "<task>")',
        "```",
        "",
        "## Citation",
        "",
    ]
    if source:
        parts += [
            f"Original dataset: [{source['label']}]({source['url']}).",
            "",
            "```bibtex",
            source["bibtex"].strip(),
            "```",
            "",
            "If you use this dataset as hosted by RelBench, please also cite "
            f"[RelBench]({RELBENCH_PAPER_URL}):",
            "",
        ]
    else:
        parts += [f"Please cite [RelBench]({RELBENCH_PAPER_URL}):", ""]
    parts += ["```bibtex", RELBENCH_BIBTEX, "```"]
    return "\n".join(parts) + "\n"
