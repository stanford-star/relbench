import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import Self


class Table:
    r"""A table in a database.

    Args:
        df: The underlying data frame of the table.
        fkey_col_to_pkey_table: A dictionary mapping
            foreign key names to table names that contain the foreign keys as
            primary keys.
        pkey_col: The primary key column if it exists.
        time_col: The time column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fkey_col_to_pkey_table: Dict[str, str],
        pkey_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ):
        self.df = df
        self.fkey_col_to_pkey_table = fkey_col_to_pkey_table
        self.pkey_col = pkey_col
        self.time_col = time_col
        self.removed_cols = None

    def __repr__(self) -> str:
        return (
            f"Table(df=\n{self.df},\n"
            f"  fkey_col_to_pkey_table={self.fkey_col_to_pkey_table},\n"
            f"  pkey_col={self.pkey_col},\n"
            f"  time_col={self.time_col}"
            f")"
        )

    def __len__(self) -> int:
        r"""Return the number of rows in the table."""
        return len(self.df)

    def save(self, path: Union[str, os.PathLike]) -> None:
        r"""Save the table to a parquet file.

        Stores other attributes as parquet metadata.
        """
        assert str(path).endswith(".parquet")
        metadata = {
            "fkey_col_to_pkey_table": self.fkey_col_to_pkey_table,
            "pkey_col": self.pkey_col,
            "time_col": self.time_col,
        }

        # Convert DataFrame to a PyArrow Table
        table = pa.Table.from_pandas(self.df, preserve_index=False)

        # Add metadata to the PyArrow Table
        metadata_bytes = {
            key: json.dumps(value).encode("utf-8") for key, value in metadata.items()
        }

        table = table.replace_schema_metadata(
            {**table.schema.metadata, **metadata_bytes}
        )

        # Write the PyArrow Table to a Parquet file using pyarrow.parquet
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> Self:
        r"""Load a table from a parquet file."""
        assert str(path).endswith(".parquet")

        # Read the Parquet file using pyarrow
        table = pa.parquet.read_table(path)
        df = table.to_pandas()
        # DuckDB currently cannot ingest pandas StringDtype columns reliably.
        # Convert them to plain object dtype on load.
        for col in df.columns:
            if isinstance(df[col].dtype, pd.StringDtype):
                df[col] = df[col].astype(object)

        # Extract metadata
        metadata_bytes = table.schema.metadata
        metadata = {
            key.decode("utf-8"): json.loads(value.decode("utf-8"))
            for key, value in metadata_bytes.items()
            if key in [b"fkey_col_to_pkey_table", b"pkey_col", b"time_col"]
        }
        return cls(
            df=df,
            fkey_col_to_pkey_table=metadata["fkey_col_to_pkey_table"],
            pkey_col=metadata["pkey_col"],
            time_col=metadata["time_col"],
        )

    def upto(self, timestamp: pd.Timestamp) -> Self:
        r"""Return a table with all rows upto timestamp (inclusive).

        Table without time_col are returned as is.
        """
        if self.time_col is None:
            return self

        return Table(
            df=self.df.query(f"{self.time_col} <= @timestamp"),
            fkey_col_to_pkey_table=self.fkey_col_to_pkey_table,
            pkey_col=self.pkey_col,
            time_col=self.time_col,
        )

    def from_(self, timestamp: pd.Timestamp) -> Self:
        r"""Return a table with all rows from timestamp onwards (inclusive).

        Table without time_col are returned as is.
        """
        if self.time_col is None:
            return self

        return Table(
            df=self.df.query(f"{self.time_col} >= @timestamp"),
            fkey_col_to_pkey_table=self.fkey_col_to_pkey_table,
            pkey_col=self.pkey_col,
            time_col=self.time_col,
        )

    @property
    @lru_cache(maxsize=None)
    def min_timestamp(self) -> pd.Timestamp:
        r"""Return the earliest time in the table."""
        if self.time_col is None:
            raise ValueError("Table has no time column.")

        return self.df[self.time_col].min()

    @property
    @lru_cache(maxsize=None)
    def max_timestamp(self) -> pd.Timestamp:
        r"""Return the latest time in the table."""
        if self.time_col is None:
            raise ValueError("Table has no time column.")

        return self.df[self.time_col].max()
