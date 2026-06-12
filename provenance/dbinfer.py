import os
from typing import Dict

import pandas as pd
import pooch

from relbench.base import Database, Dataset, Table

DEFAULT_DBINFER_ADAPTER_CACHE = os.path.join(
    pooch.os_cache("relbench"), "dbinfer-adapters"
)


class DBInferDatasetBase(Dataset):
    """Materialize a 4DBInfer dataset as a RelBench Database."""

    dbinfer_name: str | None = None
    default_task_name: str | None = None

    # DBInfer datasets are not time-sliced, so we set placeholder timestamps that
    # satisfy the Dataset API without affecting static tables.
    val_timestamp = pd.Timestamp("1970-01-01")
    test_timestamp = pd.Timestamp("1970-01-02")

    def __init__(
        self,
        cache_dir: str | None = None,
        adapter_cache_dir: str | None = None,
    ):
        super().__init__(cache_dir=cache_dir)
        if not self.dbinfer_name or not self.default_task_name:
            raise ValueError(
                "DBInferDatasetBase subclasses must define 'dbinfer_name' and "
                "'default_task_name'."
            )
        if adapter_cache_dir is None:
            adapter_cache_dir = DEFAULT_DBINFER_ADAPTER_CACHE
        self._adapter_cache_dir = adapter_cache_dir

    def _load_dataset_adapter(self):
        try:
            from dbinfer_relbench_adapter.loader import load_dbinfer_data
        except ImportError:
            raise ImportError(
                "dbinfer-relbench-adapter is not installed. "
                "Please install it with:\n"
                "  pip install dbinfer-relbench-adapter\n"
                "  pip install dgl --force-reinstall -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html\n"
                "Note: dgl may conflict with other dependencies. We recommend building "
                "dbinfer datasets in a separate virtual environment and using the cached "
                "datasets in your main working environment."
            ) from None

        dataset_adapter, _ = load_dbinfer_data(
            dataset_name=self.dbinfer_name,
            task_name=self.default_task_name,
            use_cache=True,
            cache_dir=self._adapter_cache_dir,
        )
        return dataset_adapter

    def _build_table_dict(self) -> Dict[str, Table]:
        dataset_adapter = self._load_dataset_adapter()
        mock_db = dataset_adapter.get_db()

        table_dict: Dict[str, Table] = {}
        for name, mock_table in mock_db.table_dict.items():
            table_dict[name] = Table(
                df=mock_table.df.copy(),
                fkey_col_to_pkey_table=getattr(
                    mock_table, "fkey_col_to_pkey_table", {}
                ),
                pkey_col=getattr(mock_table, "pkey_col", None),
                time_col=getattr(mock_table, "time_col", None),
            )
        return table_dict

    def make_db(self) -> Database:
        return Database(table_dict=self._build_table_dict())

    def get_db(self, upto_test_timestamp: bool = True) -> Database:
        """DBInfer datasets are static, so never trim by timestamp."""

        return super().get_db(upto_test_timestamp=False)


class DBInferAVSDataset(DBInferDatasetBase):
    dbinfer_name = "avs"
    default_task_name = "repeater"


class DBInferMAGDataset(DBInferDatasetBase):
    dbinfer_name = "mag"
    default_task_name = "cite"


class DBInferDigineticaDataset(DBInferDatasetBase):
    dbinfer_name = "diginetica"
    default_task_name = "ctr"


class DBInferRetailRocketDataset(DBInferDatasetBase):
    dbinfer_name = "retailrocket"
    default_task_name = "cvr"


class DBInferSeznamDataset(DBInferDatasetBase):
    dbinfer_name = "seznam"
    default_task_name = "charge"


class DBInferAmazonDataset(DBInferDatasetBase):
    dbinfer_name = "amazon"
    default_task_name = "rating"


class DBInferStackExchangeDataset(DBInferDatasetBase):
    dbinfer_name = "stackexchange"
    default_task_name = "churn"


class DBInferOutbrainSmallDataset(DBInferDatasetBase):
    dbinfer_name = "outbrain-small"
    default_task_name = "ctr"
