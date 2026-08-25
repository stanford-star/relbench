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
        tables = ", ".join(f"{name}={len(t)}" for name, t in self.table_dict.items())
        return f"{self.__class__.__name__}({tables})"

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
