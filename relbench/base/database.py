from typing import Dict

import pandas as pd
from typing_extensions import Self

from .table import Table


class Database:
    r"""A database is a collection of named tables linked by foreign key - primary key
    connections."""

    def __init__(self, table_dict: Dict[str, Table]) -> None:
        r"""Creates a database from a dictionary of tables."""

        self.table_dict = table_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def min_timestamp(self) -> pd.Timestamp:
        r"""Return the earliest timestamp in the database."""

        return min(
            table.min_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    @property
    def max_timestamp(self) -> pd.Timestamp:
        r"""Return the latest timestamp in the database."""

        return max(
            table.max_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    def upto(self, timestamp: pd.Timestamp) -> Self:
        r"""Return a database with all rows upto timestamp."""

        return Database(
            table_dict={
                name: table.upto(timestamp) for name, table in self.table_dict.items()
            }
        )

    def reindex_pkeys_and_fkeys(self) -> None:
        r"""Map primary and foreign keys into indices according to the ordering in the
        primary key tables."""
        # Get pkey to idx mapping:
        index_map_dict: Dict[str, pd.Series] = {}
        for table_name, table in self.table_dict.items():
            if table.pkey_col is not None:
                if table.time_col is not None:
                    table.df = table.df.sort_values(table.time_col).reset_index(
                        drop=True
                    )

                ser = table.df[table.pkey_col]

                if ser.nunique() != len(ser):
                    raise RuntimeError(
                        f"The primary key '{table.pkey_col}' "
                        f"of table '{table_name}' contains "
                        "duplicated elements"
                    )
                arange_ser = pd.RangeIndex(len(ser)).astype("Int64")
                index_map_dict[table_name] = pd.Series(
                    index=ser,
                    data=arange_ser,
                    name="index",
                )
                table.df[table.pkey_col] = arange_ser

        # Replace fkey_col_to_pkey_table with indices.
        for table in self.table_dict.values():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                out = pd.merge(
                    table.df[fkey_col],
                    index_map_dict[pkey_table_name],
                    how="left",
                    left_on=fkey_col,
                    right_index=True,
                )
                table.df[fkey_col] = out["index"]
