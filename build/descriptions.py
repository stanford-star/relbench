r"""Descriptions for dataset cards/manifests.

Dataset descriptions are curated here; task descriptions are harvested from the legacy
task-class docstrings on ``origin/main`` (mapped to task names via the register_task calls),
with a generated fallback for autocomplete tasks (which have no per-task class docstring).
"""

from __future__ import annotations

import re
import subprocess

DATASET_DESCRIPTIONS = {
    "rel-amazon": "Amazon product reviews: customers, products, and time-stamped reviews and ratings across the Amazon catalog.",
    "rel-avito": "Avito online classifieds: users, ads, search queries, and impression / click / visit streams.",
    "rel-event": "Event recommendation: users, events, attendance records, and social and interest signals.",
    "rel-f1": "Formula 1 motorsport database: races, drivers, constructors, circuits, race results, qualifying, and championship standings.",
    "rel-hm": "H&M e-commerce: customers, articles, and time-stamped purchase transactions.",
    "rel-stack": "Stack Exchange Q&A: users, posts, comments, votes, badges, and post links.",
    "rel-trial": "ClinicalTrials.gov clinical trials: studies, outcomes, adverse events, eligibilities, sponsors, conditions, and facilities.",
    "rel-salt": "SAP SALT sales documents: sales orders, line items, customers, and addresses.",
    "rel-ratebeer": "RateBeer reviews: users, beers, brewers, places, and time-stamped ratings.",
    "rel-arxiv": "arXiv scholarly papers: papers, authors, citations, and subject categories.",
}


def _git_show(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"origin/main:{path}"], capture_output=True, text=True
    ).stdout


def harvest_task_descriptions(name: str) -> dict:
    r"""task-name -> description, from legacy class docstrings for dataset ``name`` (rel-<x>)."""
    short = name[len("rel-"):] if name.startswith("rel-") else name
    init = _git_show("relbench/tasks/__init__.py")
    # task-name -> ClassName (only class-backed tasks; autocomplete uses no module.Class)
    name_to_cls = dict(
        re.findall(rf'register_task\(\s*"{re.escape(name)}",\s*"([^"]+)",\s*\w+\.(\w+)', init)
    )
    src = _git_show(f"relbench/tasks/{short}.py")
    cls_to_doc = {}
    for m in re.finditer(r'class (\w+)\([^)]*\):\s*\n\s*r?"""(.*?)"""', src, re.S):
        cls_to_doc[m.group(1)] = " ".join(m.group(2).split())
    return {task: cls_to_doc[cls] for task, cls in name_to_cls.items() if cls in cls_to_doc}


def fallback_description(tm) -> str:
    if tm.kind == "autocomplete":
        return f"Predict the `{tm.target_col}` column of the `{tm.entity_table}` table."
    if tm.task_type == "link_prediction":
        return f"Link-prediction task between `{tm.src_entity_table}` and `{tm.dst_entity_table}`."
    return f"{tm.task_type.replace('_', ' ')} task on `{tm.entity_table}`."
