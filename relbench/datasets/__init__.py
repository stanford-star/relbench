import json
import pkgutil
from functools import lru_cache
from typing import List

import pooch

from relbench.base import Dataset
from relbench.datasets import (
    amazon,
    arxiv,
    avito,
    dbinfer,
    event,
    f1,
    hm,
    ratebeer,
    salt,
    stack,
    tgb,
    trial,
)
from relbench.utils import get_relbench_cache_dir

dataset_registry = {}

hashes_str = pkgutil.get_data(__name__, "hashes.json")
hashes = json.loads(hashes_str)

DOWNLOAD_REGISTRY = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/download/",
    registry=hashes,
    env="RELBENCH_CACHE_DIR",
)


def register_dataset(
    name: str,
    cls: Dataset,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`Dataset` subclass with the given name.

    Args:
        name: The name of the dataset.
        cls: The class of the dataset.
        args: The arguments to instantiate the dataset.
        kwargs: The keyword arguments to instantiate the dataset.

    The name is used to enable caching and downloading functionalities.
    `cache_dir` is added to kwargs by default. If you want to override it, you
    can pass `cache_dir` as a keyword argument in `kwargs`.
    """

    cache_dir = f"{get_relbench_cache_dir()}/{name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    dataset_registry[name] = (cls, args, kwargs)


def get_dataset_names() -> List[str]:
    r"""Return a list of names of the registered datasets."""
    return list(dataset_registry.keys())


def download_dataset(name: str) -> None:
    r"""Download dataset from RelBench server into its cache directory.

    The downloaded database will be automatically picked up by the dataset object, when
    `dataset.get_db()` is called.
    """

    DOWNLOAD_REGISTRY.fetch(
        f"{name}/db.zip",
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True,
    )


@lru_cache(maxsize=None)
def get_dataset(name: str, download=True) -> Dataset:
    r"""Return a dataset object by name.

    Args:
        name: The name of the dataset.
        download: If True, download the dataset from the RelBench server.

    Returns:
        Dataset: The dataset object.

    If `download` is True, the database comprising the dataset will be
    downloaded into the cache from the RelBench server. If you use
    `download=False` the first time, the database will be processed from the
    raw files of the original source.

    Once the database is cached, either because of download or processing from
    raw files, the cache will be used. `download=True` will verify that the
    cached database matches the RelBench version even in this case.
    """

    from relbench import hf

    if hf.is_registered(name):
        # Migrated datasets are served by the new manifest/HF loader.
        from relbench.load import load_dataset as _load_dataset

        return _load_dataset(name)

    if download:
        download_dataset(name)

    if name.startswith("ctu-"):
        try:
            import redelex
        except ImportError:
            raise ImportError(
                "Redelex is not installed. Please install it with `pip install redelex`."
            )

    cls, args, kwargs = dataset_registry[name]

    dataset = cls(*args, **kwargs)
    return dataset


register_dataset("rel-amazon", amazon.AmazonDataset)
register_dataset("rel-avito", avito.AvitoDataset)
register_dataset("rel-event", event.EventDataset)
register_dataset("rel-f1", f1.F1Dataset)
register_dataset("rel-hm", hm.HMDataset)
register_dataset("rel-stack", stack.StackDataset)
register_dataset("rel-trial", trial.TrialDataset)
register_dataset("rel-arxiv", arxiv.ArxivDataset)
register_dataset("rel-salt", salt.SALTDataset)
register_dataset("rel-ratebeer", ratebeer.RateBeerDataset)
register_dataset("dbinfer-avs", dbinfer.DBInferAVSDataset)
# register_dataset("dbinfer-mag", dbinfer.DBInferMAGDataset)    # Experimental dataset: dbinfer-mag
register_dataset("dbinfer-diginetica", dbinfer.DBInferDigineticaDataset)
register_dataset("dbinfer-retailrocket", dbinfer.DBInferRetailRocketDataset)
register_dataset("dbinfer-seznam", dbinfer.DBInferSeznamDataset)
register_dataset("dbinfer-amazon", dbinfer.DBInferAmazonDataset)
register_dataset("dbinfer-stackexchange", dbinfer.DBInferStackExchangeDataset)
register_dataset("dbinfer-outbrain-small", dbinfer.DBInferOutbrainSmallDataset)

# Temporal Graph Benchmark (TGB)
register_dataset("tgbl-wiki", tgb.TGBDataset, tgb_name="tgbl-wiki")
register_dataset("tgbl-wiki-v2", tgb.TGBDataset, tgb_name="tgbl-wiki-v2")
register_dataset("tgbl-review", tgb.TGBDataset, tgb_name="tgbl-review")
register_dataset("tgbl-review-v2", tgb.TGBDataset, tgb_name="tgbl-review-v2")
register_dataset("tgbl-coin", tgb.TGBDataset, tgb_name="tgbl-coin")
register_dataset("tgbl-comment", tgb.TGBDataset, tgb_name="tgbl-comment")
register_dataset("tgbl-flight", tgb.TGBDataset, tgb_name="tgbl-flight")

register_dataset("thgl-software", tgb.TGBDataset, tgb_name="thgl-software")
register_dataset("thgl-forum", tgb.TGBDataset, tgb_name="thgl-forum")
register_dataset("thgl-github", tgb.TGBDataset, tgb_name="thgl-github")
register_dataset("thgl-myket", tgb.TGBDataset, tgb_name="thgl-myket")

register_dataset("tgbn-trade", tgb.TGBDataset, tgb_name="tgbn-trade")
register_dataset("tgbn-genre", tgb.TGBDataset, tgb_name="tgbn-genre")
register_dataset("tgbn-reddit", tgb.TGBDataset, tgb_name="tgbn-reddit")
register_dataset("tgbn-token", tgb.TGBDataset, tgb_name="tgbn-token")
