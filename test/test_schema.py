import shutil

import pytest

from relbench.manifest import DatasetManifest, TaskManifest
from relbench.schema import dataset_card, render_schema_svg


def test_dataset_card(dataset_dir):
    manifest = DatasetManifest.load(dataset_dir / "manifest.yaml")
    tasks = [
        TaskManifest.load(p) for p in sorted(dataset_dir.glob("tasks/*/manifest.yaml"))
    ]
    card = dataset_card(manifest, tasks, repo="org/repo/fakeds")
    assert card.startswith("# fakeds\n")
    assert "![schema diagram](schema.svg)" in card
    assert "| `user-churn` | forecast | binary_classification |" in card
    assert 'relbench.load_dataset("org/repo/fakeds")' in card
    assert "@inproceedings{robinson2024relbench" in card
    sourced = dataset_card(
        manifest, source={"label": "Orig", "url": "https://x", "bibtex": "@misc{x}"}
    )
    assert "Original dataset: [Orig](https://x)." in sourced and "@misc{x}" in sourced


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz dot binary missing")
def test_render_schema_svg(dataset_dir, tmp_path):
    pytest.importorskip("graphviz")
    manifest = DatasetManifest.load(dataset_dir / "manifest.yaml")
    out = tmp_path / "schema.svg"
    render_schema_svg(manifest, out, db_dir=dataset_dir / "db")
    svg = out.read_text()
    assert svg.lstrip().startswith("<?xml") or "<svg" in svg
    for table in manifest.tables:
        assert table in svg
