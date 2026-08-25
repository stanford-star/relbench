import dataclasses

import pytest

from relbench.base import TaskType
from relbench.manifest import (
    TASK_TYPES,
    DatasetManifest,
    TableSpec,
    TaskManifest,
    validate_dataset_manifest,
)


def test_task_types_match_enum():
    assert set(TASK_TYPES) == {t.value for t in TaskType}


def test_task_manifest_roundtrip(tmp_path, churn_manifest):
    path = tmp_path / "manifest.yaml"
    churn_manifest.save(path)
    assert "sql: |" in path.read_text()
    assert TaskManifest.load(path) == churn_manifest


def test_dataset_manifest_roundtrip(tmp_path):
    manifest = DatasetManifest(
        name="x",
        val_timestamp="2020-01-01",
        test_timestamp="2021-01-01",
        description="a database",
        tables={
            "a": TableSpec(pkey="id", time_col="t", fkeys={"b_id": "b"}),
            "b": TableSpec(pkey="id"),
        },
    )
    manifest.save(tmp_path / "manifest.yaml")
    assert DatasetManifest.load(tmp_path / "manifest.yaml") == manifest


def test_from_dict_ignores_unknown_keys(churn_manifest):
    d = {**churn_manifest.to_dict(), "metrics": ["roc_auc"]}
    assert TaskManifest.from_dict(d) == churn_manifest


@pytest.mark.parametrize(
    "change, match",
    [
        ({"kind": "magic"}, "kind"),
        ({"task_type": "link_prediction"}, "task_type"),
        ({"sql": None}, "sql"),
    ],
)
def test_validate_rejects(churn_manifest, change, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(churn_manifest, **change).validate()


def test_validate_link_task_requires_fields(purchase_manifest):
    purchase_manifest.validate()
    with pytest.raises(ValueError, match="eval_k"):
        dataclasses.replace(purchase_manifest, eval_k=None).validate()
    dataclasses.replace(
        purchase_manifest, kind="external", sql=None, eval_k=None, evaluator="dbinfer"
    ).validate()


@pytest.mark.parametrize(
    "field", ["entity_table", "entity_col", "target_col", "time_col", "timedelta"]
)
def test_validate_forecast_entity_task_requires_fields(churn_manifest, field):
    churn_manifest.validate()
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(churn_manifest, **{field: None}).validate()


def test_validate_external_and_autocomplete_minimums(churn_manifest, rating_manifest):
    dataclasses.replace(
        churn_manifest, kind="external", sql=None, target_col=None, time_col=None
    ).validate()
    with pytest.raises(ValueError, match="entity_table"):
        dataclasses.replace(
            churn_manifest, kind="external", sql=None, entity_table=None
        ).validate()
    rating_manifest.validate()
    with pytest.raises(ValueError, match="target_col"):
        dataclasses.replace(rating_manifest, target_col=None).validate()


def test_validate_dataset_manifest(dataset_dir):
    manifest = DatasetManifest.load(dataset_dir / "manifest.yaml")
    validate_dataset_manifest(manifest, dataset_dir / "db")
    manifest.tables["review"].fkeys["nope"] = "customer"
    with pytest.raises(ValueError, match="nope"):
        validate_dataset_manifest(manifest, dataset_dir / "db")
