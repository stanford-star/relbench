r"""CI driver for GitHub-issue leaderboard submissions.

A submission is an issue created from ``.github/ISSUE_TEMPLATE/submit.yml``: the form
fields carry the method metadata (name, type, url, note) and the prediction tables are
attached to the issue body as a zip.

This script is run by the leaderboard workflows with the issue body in a file:

* validate mode (default): parse the form, download the attachments, score them with
  ``relbench.submit.evaluate_submission``, and write a markdown report (posted back
  to the issue as a comment). Exit 0 iff at least one leaderboard family is validated.
* publish mode (``--entry``): additionally write the leaderboard entry JSON for the issue
  and regenerate the aggregate ``leaderboard.json`` from all entry files.

Only the issue *body* is consumed — it is data (form text + attachment URLs), never code,
and the attachments are only ever parsed as CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from relbench.submit import (  # noqa: E402
    _FAMILY_METRIC,
    LEADERBOARD_TASKS,
    _format_value,
    _markdown_report,
    _metric_display,
    evaluate_submission,
)

# Issue-form section headings (as rendered by GitHub) -> entry fields. The "In-context"
# dropdown section is handled separately (rendered as a "Yes"/"No" line).
FORM_FIELDS = {
    "Name": "name",
    "URL": "url",
    "Note": "note",
}
IN_CONTEXT_HEADING = "In-context"
NO_RESPONSE = "_No response_"
MAX_UNZIPPED_BYTES = 4 << 30
LEADERBOARD = Path(__file__).resolve().parents[2] / "leaderboard" / "leaderboard.json"

# GitHub handle pinged in the validation report to review the submission. Set via the
# LEADERBOARD_MAINTAINER repository variable; the workflow falls back to the repo owner.
MAINTAINER = os.environ.get("LEADERBOARD_MAINTAINER") or "maintainer"

# Leaderboard family (relbench.submit) -> board key used by the website.
FAMILY_TO_BOARD = {
    "classification": "binary_classification",
    "regression": "regression",
    "recommendation": "recommendation",
}

# Attachment URLs GitHub produces for files dragged into an issue.
ATTACHMENT_RE = re.compile(
    r"https://(?:github\.com/user-attachments/files/\d+/[^\s()\[\]]+"
    r"|github\.com/[\w.-]+/[\w.-]+/files/\d+/[^\s()\[\]]+"
    r"|huggingface\.co/datasets/[\w.-]+/[\w.-]+/resolve/[0-9a-f]{40}/[^\s()\[\]]+)"
)


def parse_form(body: str) -> dict:
    r"""Split an issue-form body into ``{heading: text}`` sections."""
    sections: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("### "):
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            # issue-form labels may carry a parenthetical hint, e.g. "Name (shown on ...)"
            # or a trailing "?" ("In-context?")
            current = line[4:].split(" (")[0].strip().rstrip("?")
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def form_metadata(sections: dict) -> tuple[dict, list[str]]:
    r"""Extract and check the method-metadata fields from the parsed form."""
    fields = {}
    for heading, key in FORM_FIELDS.items():
        val = sections.get(heading, "").strip()
        if val and val != NO_RESPONSE:
            fields[key] = val
    fields["in_context"] = sections.get(IN_CONTEXT_HEADING, "").strip().lower() == "yes"
    errors = []
    if not fields.get("name"):
        errors.append("the form is missing the method name")
    return fields, errors


def download_attachments(body: str, dest: Path) -> list[str]:
    r"""Download every issue attachment into ``dest``, extracting zips (flat, CSVs only).

    Returns a list of problems (empty on success).
    """
    urls = ATTACHMENT_RE.findall(body)
    if not urls:
        return [
            "no attachments found — drag the submission zip into the issue body, or link it "
            "from the Hugging Face Hub at a pinned commit (see leaderboard/README.md)"
        ]
    problems = []
    for url in urls:
        name = url.rstrip("/").rsplit("/", 1)[-1]
        target = dest / name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "relbench-ci"})
            with urllib.request.urlopen(req, timeout=300) as r, open(target, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception as exc:  # noqa: BLE001 -- surfaced in the report
            problems.append(f"could not download attachment {name}: {exc}")
            continue
        if target.suffix == ".zip":
            try:
                with zipfile.ZipFile(target) as zf:
                    total = sum(info.file_size for info in zf.infolist())
                    if total > MAX_UNZIPPED_BYTES:
                        problems.append(
                            f"attachment {name} unpacks to {total / 2**30:.1f} GiB, "
                            f"over the {MAX_UNZIPPED_BYTES >> 30} GiB limit"
                        )
                        target.unlink()
                        continue
                    for info in zf.infolist():
                        base = Path(info.filename).name  # flatten; ignore any dirs
                        if not base.endswith(".csv"):
                            continue
                        with zf.open(info) as src, open(dest / base, "wb") as out:
                            while chunk := src.read(1 << 20):
                                out.write(chunk)
            except zipfile.BadZipFile:
                problems.append(f"attachment {name} is not a valid zip")
            target.unlink()
    return problems


BOARD_TITLES = {
    "binary_classification": "Classification",
    "regression": "Regression",
    "recommendation": "Recommendation",
}
LOWER_IS_BETTER = {"nmae"}


def _rank_key(entry: dict, board_key: str, sign: float) -> tuple:
    board = entry["boards"][board_key]
    mean = board.get("mean")
    return (mean is None, sign * mean if mean is not None else 0.0, -board["cov"])


def leaderboard_preview(entries: list, new: dict) -> str:
    entries = [
        e for e in entries if not new.get("issue") or e.get("issue") != new["issue"]
    ]
    entries.append(new)
    lines = ["## Leaderboard preview (if accepted)", ""]
    for family, board_key in FAMILY_TO_BOARD.items():
        if board_key not in new["boards"]:
            continue
        metric = _FAMILY_METRIC[family]
        sign = 1.0 if metric in LOWER_IS_BETTER else -1.0
        rows = sorted(
            (e for e in entries if board_key in e.get("boards", {})),
            key=lambda e: _rank_key(e, board_key, sign),
        )
        tasks = LEADERBOARD_TASKS[family]
        arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
        lines += [
            f"<details open><summary><b>{BOARD_TITLES[board_key]}</b> "
            f"({_metric_display(metric)} {arrow})</summary>",
            "",
            "| # | method | in-context | mean | "
            + " | ".join(f"`{t.removeprefix('rel-')}`" for t in tasks)
            + " |",
            "|---:|---|:---:|---:|" + "---:|" * len(tasks),
        ]
        values = [
            [e["boards"][board_key].get("mean")]
            + [e["boards"][board_key]["results"].get(t) for t in tasks]
            for e in rows
        ]
        best = [
            min((v for v in col if v is not None), key=lambda v: sign * v, default=None)
            for col in zip(*values)
        ]
        for rank, (e, vals) in enumerate(zip(rows, values), 1):
            name = (e.get("name") or "?").replace("|", "\\|")
            cells = [
                str(rank) if vals[0] is not None else "-",
                f"🆕 {name}" if e is new else name,
                "✓" if e.get("in_context") else "",
            ] + [
                (
                    f"**{_format_value(metric, v)}**"
                    if v is not None
                    and _format_value(metric, v) == _format_value(metric, b)
                    else _format_value(metric, v)
                )
                for v, b in zip(vals, best)
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "</details>", ""]
    return "\n".join(lines)


def write_report(
    path: Path, problems: list, result: dict | None, entry: dict | None
) -> None:
    lines = ["## RelBench leaderboard validation report", ""]
    for p in problems:
        lines.append(f"- :x: {p}")
    if problems:
        lines.append("")
    if result is not None:
        lines += [_markdown_report(result)]
        if result["validated"]:
            lines.append(
                f"@{MAINTAINER} please review and either add the `accept` label "
                "or close this issue."
            )
            current = json.loads(LEADERBOARD.read_text())
            lines += ["", leaderboard_preview(current, entry)]
        else:
            lines.append(
                "No leaderboard was validated. Edit the issue (fix the "
                "attachments or form) to re-run validation."
            )
    path.write_text("\n".join(lines) + "\n")


def build_entry(
    fields: dict, result: dict, issue: int, author: str, created_at: str = ""
) -> dict:
    boards = {}
    for family, board_key in FAMILY_TO_BOARD.items():
        fam = result["families"][family]
        if fam["num_valid"] == 0:
            continue
        results = {t: result["tasks"][t]["metric"] for t in fam["valid"]}
        boards[board_key] = {
            "results": results,
            "mean": fam["aggregate"] if fam["complete"] else None,
            "cov": fam["num_valid"] / fam["num_total"],
        }
    return {
        "name": fields.get("name"),
        "in_context": bool(fields.get("in_context")),
        "url": fields.get("url"),
        "note": fields.get("note"),
        "date": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "author": author,
        "issue": issue,
        "boards": boards,
    }


def rebuild_aggregate(entries_dir: Path, out: Path) -> None:
    entries = []
    for p in sorted(entries_dir.glob("*.json")):
        entries.append(json.loads(p.read_text()))
    entries.sort(key=lambda e: e.get("issue") or 0)
    out.write_text(json.dumps(entries, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", help="file holding the issue body")
    ap.add_argument(
        "--rebuild",
        help="regenerate --aggregate from this entries dir and exit; no validation",
    )
    ap.add_argument("--report", help="output markdown report path")
    ap.add_argument("--entry", help="publish mode: write entries/<issue>.json here")
    ap.add_argument(
        "--aggregate", help="publish mode: regenerate this leaderboard.json"
    )
    ap.add_argument("--issue", type=int, default=0, help="issue number (publish mode)")
    ap.add_argument("--author", default="", help="issue author login (publish mode)")
    ap.add_argument(
        "--created-at",
        default="",
        help="issue creation time, ISO 8601 UTC (publish mode)",
    )
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    # Rebuild-only: regenerate the aggregate from whatever entry files are on disk. The
    # publish workflow uses this to rebase its result onto a moved main without ever
    # merging the generated leaderboard.json, which cannot be merged.
    if args.rebuild:
        rebuild_aggregate(Path(args.rebuild), Path(args.aggregate))
        return 0

    if not args.body or not args.report:
        ap.error("--body and --report are required unless --rebuild is given")

    body = Path(args.body).read_text()
    fields, problems = form_metadata(parse_form(body))

    result = None
    with tempfile.TemporaryDirectory() as tmp:
        pred_dir = Path(tmp)
        problems += download_attachments(body, pred_dir)
        if not problems or any(pred_dir.glob("*.csv")):
            try:
                result = evaluate_submission(
                    pred_dir, num_workers=args.num_workers, verbose=False
                )
            except Exception as exc:  # noqa: BLE001 -- surfaced in the report
                problems.append(f"could not evaluate the submission: {exc}")

    ok = bool(result and result["validated"]) and not any(
        p.startswith("the form") for p in problems
    )
    entry = result and build_entry(
        fields, result, args.issue, args.author, args.created_at
    )
    write_report(Path(args.report), problems, result, entry)
    if not ok:
        return 1

    if args.entry:
        entry_path = Path(args.entry)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(json.dumps(entry, indent=1) + "\n")
        if args.aggregate:
            rebuild_aggregate(entry_path.parent, Path(args.aggregate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
