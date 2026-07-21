r"""CI driver for GitHub-issue leaderboard submissions.

A submission is an issue created from ``.github/ISSUE_TEMPLATE/submission.yml``: the form
fields carry the method metadata (name, type, url, note) and the prediction tables are
attached to the issue body (one zip, or per-task ``.csv.gz`` files).

This script is run by the leaderboard workflows with the issue body in a file:

* validate mode (default): parse the form, download the attachments, score them with
  ``relbench.leaderboard.evaluate_submission``, and write a markdown report (posted back
  to the issue as a comment). Exit 0 iff at least one leaderboard family is validated.
* publish mode (``--entry``): additionally write the leaderboard entry JSON for the issue
  and regenerate the aggregate ``leaderboard.json`` from all entry files.

Only the issue *body* is consumed — it is data (form text + attachment URLs), never code,
and the attachments are only ever parsed as CSV.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from relbench.leaderboard import evaluate_submission

# Issue-form section headings (as rendered by GitHub) -> entry fields.
FORM_FIELDS = {
    "Method name": "name",
    "Type": "type",
    "URL": "url",
    "Note": "note",
}
NO_RESPONSE = "_No response_"

# Leaderboard family (relbench.leaderboard) -> board key used by the website.
FAMILY_TO_BOARD = {
    "classification": "binary_classification",
    "regression": "regression",
    "recommendation": "link_prediction",
}

# Attachment URLs GitHub produces for files dragged into an issue.
ATTACHMENT_RE = re.compile(
    r"https://(?:github\.com/user-attachments/files/\d+/[^\s()\[\]]+"
    r"|github\.com/[\w.-]+/[\w.-]+/files/\d+/[^\s()\[\]]+)"
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
            current = line[4:].strip()
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
    errors = []
    if not fields.get("name"):
        errors.append("the form is missing the method name")
    if fields.get("type") not in ("fine-tuned", "in-context"):
        errors.append("the form 'Type' must be 'fine-tuned' or 'in-context'")
    return fields, errors


def download_attachments(body: str, dest: Path) -> list[str]:
    r"""Download every issue attachment into ``dest``, extracting zips (flat, CSVs only).

    Returns a list of problems (empty on success).
    """
    urls = ATTACHMENT_RE.findall(body)
    if not urls:
        return ["no attachments found — drag the submission zip (or the per-task "
                ".csv.gz files) into the issue body"]
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
                    for info in zf.infolist():
                        base = Path(info.filename).name  # flatten; ignore any dirs
                        if not (base.endswith(".csv") or base.endswith(".csv.gz")):
                            continue
                        with zf.open(info) as src, open(dest / base, "wb") as out:
                            while chunk := src.read(1 << 20):
                                out.write(chunk)
            except zipfile.BadZipFile:
                problems.append(f"attachment {name} is not a valid zip")
            target.unlink()
    return problems


def write_report(path: Path, fields: dict, problems: list, result: dict | None) -> None:
    lines = ["## RelBench leaderboard validation report", ""]
    if fields:
        lines += [f"**{fields.get('name', '?')}** ({fields.get('type', '?')})", ""]
    for p in problems:
        lines.append(f"- :x: {p}")
    if problems:
        lines.append("")
    if result is not None:
        lines += ["| Task | Metric | Value | Status |", "|---|---|---:|---|"]
        for task in sorted(result["tasks"]):
            e = result["tasks"][task]
            val = "–" if e["metric"] is None else f"{e['metric']:.6f}"
            status = "ok" if e["status"] == "ok" else f"error: {e['error']}"
            lines.append(f"| {task} | {e['metric_name'] or '–'} | {val} | {status} |")
        lines.append("")
        for family, fam in result["families"].items():
            mark = ":white_check_mark: validated" if fam["complete"] else ":x: rejected"
            agg = "n/a" if fam["aggregate"] is None else f"{fam['aggregate']:.6f}"
            lines.append(
                f"- **{family}** — {mark} ({fam['num_valid']}/{fam['num_total']} tasks, "
                f"mean {fam['metric_name']} = {agg})"
            )
        lines.append("")
        if result["validated"]:
            lines.append("Validated leaderboard(s): " + ", ".join(result["validated"])
                         + ". A maintainer will review this submission.")
        else:
            lines.append("No leaderboard was validated. Edit the issue (fix the "
                         "attachments or form) to re-run validation.")
    path.write_text("\n".join(lines) + "\n")


def build_entry(fields: dict, result: dict, issue: int, author: str) -> dict:
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
        "type": fields.get("type"),
        "url": fields.get("url"),
        "note": fields.get("note"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m"),
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
    ap.add_argument("--body", required=True, help="file holding the issue body")
    ap.add_argument("--report", required=True, help="output markdown report path")
    ap.add_argument("--entry", help="publish mode: write entries/<issue>.json here")
    ap.add_argument("--aggregate", help="publish mode: regenerate this leaderboard.json")
    ap.add_argument("--issue", type=int, default=0, help="issue number (publish mode)")
    ap.add_argument("--author", default="", help="issue author login (publish mode)")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    body = Path(args.body).read_text()
    fields, problems = form_metadata(parse_form(body))

    result = None
    with tempfile.TemporaryDirectory() as tmp:
        pred_dir = Path(tmp)
        problems += download_attachments(body, pred_dir)
        if not problems or any(pred_dir.glob("*.csv*")):
            try:
                result = evaluate_submission(
                    pred_dir, num_workers=args.num_workers, verbose=False
                )
            except Exception as exc:  # noqa: BLE001 -- surfaced in the report
                problems.append(f"could not evaluate the submission: {exc}")

    write_report(Path(args.report), fields, problems, result)

    ok = bool(result and result["validated"]) and not any(
        p.startswith("the form") for p in problems
    )
    if not ok:
        return 1

    if args.entry:
        entry_path = Path(args.entry)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry = build_entry(fields, result, args.issue, args.author)
        entry_path.write_text(json.dumps(entry, indent=1) + "\n")
        if args.aggregate:
            rebuild_aggregate(entry_path.parent, Path(args.aggregate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
