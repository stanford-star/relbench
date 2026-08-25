import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "leaderboard_submission.py"
)


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("leaderboard_submission", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BODY = """### Name

My method

### Note

_No response_

### In-context?

Yes

### URL

https://example.org/paper

### File(s)

[preds-classification.zip](https://github.com/user-attachments/files/123/preds-classification.zip)
"""


def test_parse_form_and_metadata(script):
    sections = script.parse_form(BODY)
    assert sections["Name"] == "My method"
    assert sections["In-context"] == "Yes"
    assert sections["File(s)"].startswith("[preds-classification.zip]")
    fields, errors = script.form_metadata(sections)
    assert errors == []
    assert fields == {
        "name": "My method",
        "url": "https://example.org/paper",
        "in_context": True,
    }
    _, errors = script.form_metadata(script.parse_form("### Name\n\n\n"))
    assert errors and "name" in errors[0]


def test_attachment_regex(script):
    urls = script.ATTACHMENT_RE.findall(BODY)
    assert urls == [
        "https://github.com/user-attachments/files/123/preds-classification.zip"
    ]
    assert script.ATTACHMENT_RE.findall("https://evil.example/files/1/x.zip") == []


def _result():
    tasks = {
        "rel-f1/driver-dnf": {"status": "ok", "metric": 0.7, "metric_name": "roc_auc"},
        "rel-f1/driver-position": {
            "status": "ok",
            "metric": 0.5,
            "metric_name": "nmae",
        },
    }
    families = {
        "classification": {
            "num_valid": 1,
            "num_total": 12,
            "valid": ["rel-f1/driver-dnf"],
            "complete": False,
            "aggregate": 0.7,
        },
        "regression": {
            "num_valid": 1,
            "num_total": 1,
            "valid": ["rel-f1/driver-position"],
            "complete": True,
            "aggregate": 0.5,
        },
        "recommendation": {
            "num_valid": 0,
            "num_total": 10,
            "valid": [],
            "complete": False,
            "aggregate": None,
        },
    }
    return {"tasks": tasks, "families": families, "validated": ["regression"]}


def test_build_entry_and_aggregate(script, tmp_path):
    fields = {"name": "M", "url": "u", "in_context": True}
    entry = script.build_entry(
        fields, _result(), issue=7, author="me", created_at="2026-01-01T00:00:00Z"
    )
    assert entry["name"] == "M" and entry["issue"] == 7 and entry["in_context"] is True
    assert entry["date"] == "2026-01-01T00:00:00Z"
    assert set(entry["boards"]) == {"binary_classification", "regression"}
    assert entry["boards"]["binary_classification"]["mean"] is None
    assert entry["boards"]["binary_classification"]["cov"] == pytest.approx(1 / 12)
    assert entry["boards"]["regression"] == {
        "results": {"rel-f1/driver-position": 0.5},
        "mean": 0.5,
        "cov": 1.0,
    }
    entries = tmp_path / "entries"
    entries.mkdir()
    (entries / "10.json").write_text(json.dumps({**entry, "issue": 10}))
    (entries / "7.json").write_text(json.dumps(entry))
    out = tmp_path / "leaderboard.json"
    script.rebuild_aggregate(entries, out)
    assert [e["issue"] for e in json.loads(out.read_text())] == [7, 10]
