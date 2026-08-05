from functools import cached_property
from typing import Dict, Optional

import pandas as pd
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

    @cached_property
    def min_timestamp(self) -> pd.Timestamp:
        r"""Return the earliest time in the table."""

        if self.time_col is None:
            raise ValueError("Table has no time column.")

        return self.df[self.time_col].min()

    @cached_property
    def max_timestamp(self) -> pd.Timestamp:
        r"""Return the latest time in the table."""

        if self.time_col is None:
            raise ValueError("Table has no time column.")

        return self.df[self.time_col].max()
