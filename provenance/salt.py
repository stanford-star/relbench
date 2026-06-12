import os

import pandas as pd
import pooch
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

from relbench.base import Database, Dataset, Table


class SALTDataset(Dataset):

    val_timestamp = pd.Timestamp("2020-02-01")
    test_timestamp = pd.Timestamp("2020-07-01")

    header_target_cols = [
        "SALESOFFICE",
        "SALESGROUP",
        "CUSTOMERPAYMENTTERMS",
        "SHIPPINGCONDITION",
        "HEADERINCOTERMSCLASSIFICATION",
    ]
    item_target_cols = ["PLANT", "SHIPPINGPOINT", "ITEMINCOTERMSCLASSIFICATION"]
    all_target_cols = header_target_cols + item_target_cols

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""

        try:
            _split = "train+test"
            salesdocument = load_dataset(
                "sap-ai-research/SALT", "salesdocuments", split=_split
            ).to_pandas()
            salesdocumentitem = load_dataset(
                "sap-ai-research/SALT", "salesdocument_items", split=_split
            ).to_pandas()
            customer = load_dataset(
                "sap-ai-research/SALT", "customers", split=_split
            ).to_pandas()
            address = load_dataset(
                "sap-ai-research/SALT", "addresses", split=_split
            ).to_pandas()

            # Rename the incoterm columns for clarity
            salesdocument = salesdocument.rename(
                columns={"INCOTERMSCLASSIFICATION": "HEADERINCOTERMSCLASSIFICATION"}
            )
            salesdocumentitem = salesdocumentitem.rename(
                columns={"INCOTERMSCLASSIFICATION": "ITEMINCOTERMSCLASSIFICATION"}
            )

            # Ensure we only look at sales order items for which we have the header information as well
            salesdocumentitem = salesdocumentitem[
                salesdocumentitem.SALESDOCUMENT.isin(salesdocument.SALESDOCUMENT)
            ]

            # Use creation date and time to build the timestamp
            salesdocument["CREATIONTIMESTAMP"] = pd.to_datetime(
                salesdocument.CREATIONDATE.dt.strftime("%Y%m%d")
                + salesdocument.CREATIONTIME.astype(str),
                format="%Y%m%d%H:%M:%S",
            )
            salesdocument = salesdocument.drop(["CREATIONDATE", "CREATIONTIME"], axis=1)
            salesdocument = salesdocument[
                (salesdocument.CREATIONTIMESTAMP >= pd.Timestamp("2018-01-01"))
                & (salesdocument.CREATIONTIMESTAMP < pd.Timestamp("2021-01-01"))
            ]

            # Add date column to item df as well. The other dfs don't have a timestamp
            salesdocumentitem = salesdocumentitem.merge(
                salesdocument[["SALESDOCUMENT", "CREATIONTIMESTAMP"]],
                on="SALESDOCUMENT",
                how="left",
            )
            salesdocumentitem = salesdocumentitem[
                (salesdocumentitem.CREATIONTIMESTAMP >= pd.Timestamp("2018-01-01"))
                & (salesdocumentitem.CREATIONTIMESTAMP < pd.Timestamp("2021-01-01"))
            ]

            # Join the sales document and item IDS to for the primary key of the sales document item table
            # This is not necessary to be able to join the tables but it will be necessary for the prediction task
            salesdocumentitem["ID"] = (
                salesdocumentitem["SALESDOCUMENT"]
                + salesdocumentitem["SALESDOCUMENTITEM"]
            )

            # Also convert the columns which will be used as targets to integers, to avoid modeling issues

            for col in self.header_target_cols:
                salesdocument[col], _ = pd.factorize(salesdocument[col])
            for col in self.item_target_cols:
                salesdocumentitem[col], _ = pd.factorize(salesdocumentitem[col])

            # Note: there are currently two customers registered with more than one address in the customer table
            # For lack of a better solution atm, remove the customers with duplicate entries
            customer = customer[(~customer.CUSTOMER.isin(["1025999907", "5508069095"]))]

            # Make sure to remove additional targets from the sales order and sales order item tables to avoid leakers

            # Drop duplicate columns in the customer and address tables
            customer = customer.drop_duplicates()
            address = address.drop_duplicates()

            # Drop the ADDRESSREPRESENTATIONCODE column as it is always NaN
            address = address.drop(columns=["ADDRESSREPRESENTATIONCODE"])

            # Drop the "__index_level_0__" column artifacts
            customer = customer.drop(columns=["__index_level_0__"])
            salesdocument = salesdocument.drop(columns=["__index_level_0__"])
            salesdocumentitem = salesdocumentitem.drop(columns=["__index_level_0__"])
            address = address.drop(columns=["__index_level_0__"])
        except DatasetNotFoundError:
            # Fallback to download from relbench server
            paths = pooch.retrieve(
                "https://relbench.stanford.edu/download/rel-salt/db.zip",
                known_hash="fca91ab7d9e37646dcf1cb0007cc4229e9b23ef3c85f3c9e578d0f3fcb167001",
                progressbar=True,
                processor=pooch.Unzip(),
            )

            path = os.path.dirname(paths[0])
            salesdocumentitem = pd.read_parquet(
                os.path.join(path, "salesdocumentitem.parquet")
            )
            salesdocument = pd.read_parquet(os.path.join(path, "salesdocument.parquet"))
            customer = pd.read_parquet(os.path.join(path, "customer.parquet"))
            address = pd.read_parquet(os.path.join(path, "address.parquet"))

        # Collect all tables in the database as relbench.base.Table objects.
        tables = {}
        tables["salesdocumentitem"] = Table(
            df=salesdocumentitem,
            fkey_col_to_pkey_table={
                "SALESDOCUMENT": "salesdocument",
                "SOLDTOPARTY": "customer",
                "SHIPTOPARTY": "customer",
                "BILLTOPARTY": "customer",
                "PAYERPARTY": "customer",
            },
            pkey_col="ID",  # + SALESDOCUMENTITEM
            time_col="CREATIONTIMESTAMP",
        )
        tables["salesdocument"] = Table(
            df=salesdocument,
            fkey_col_to_pkey_table={},
            pkey_col="SALESDOCUMENT",
            time_col="CREATIONTIMESTAMP",
        )
        tables["customer"] = Table(
            df=customer,
            fkey_col_to_pkey_table={"ADDRESSID": "address"},
            pkey_col="CUSTOMER",
            time_col=None,
        )
        tables["address"] = Table(
            df=address, fkey_col_to_pkey_table={}, pkey_col="ADDRESSID", time_col=None
        )

        return Database(tables)
