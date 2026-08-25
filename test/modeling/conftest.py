import pytest
from torch_frame.config import TextEmbedderConfig
from torch_frame.testing.text_embedder import HashTextEmbedder

from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal


def _make_graph(db, cache_dir=None):
    return make_pkey_fkey_graph(
        db,
        get_stype_proposal(db),
        text_embedder_cfg=TextEmbedderConfig(HashTextEmbedder(8), batch_size=None),
        cache_dir=cache_dir,
    )


@pytest.fixture
def make_graph():
    return _make_graph


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("materialized")


@pytest.fixture(scope="session")
def graph(dataset, cache_dir):
    return _make_graph(dataset.get_db(), cache_dir)
