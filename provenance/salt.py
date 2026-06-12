r"""Generate the **rel-salt** database from its raw source (SAP SALT sales data).

    python salt.py [OUT_DIR]      # default OUT_DIR: ./rel-salt

Source: a prebuilt snapshot of the SAP ``sap-ai-research/SALT`` sales-document data
(salesdocument / salesdocumentitem / customer / address parquet dumps), hosted by
RelBench. Produces the Hugging Face layout (manifest.yaml + db/*.parquet) reproducing
relbench/core/rel-salt.
"""

import sys

import pandas as pd
from _lib import Table, fetch, write_hf

URL = "https://relbench.stanford.edu/download/rel-salt/db.zip"
SHA = "fca91ab7d9e37646dcf1cb0007cc4229e9b23ef3c85f3c9e578d0f3fcb167001"
VAL_TIMESTAMP, TEST_TIMESTAMP = "2020-02-01", "2020-07-01"


def build(raw) -> dict:
    def read(name):
        # Locate the parquet robustly regardless of whether it sits at the extract
        # root or inside a subdirectory (e.g. db/).
        matches = sorted(raw.rglob(f"{name}.parquet"))
        if not matches:
            raise FileNotFoundError(f"{name}.parquet not found under {raw}")
        return pd.read_parquet(matches[0])

    salesdocumentitem = read("salesdocumentitem")
    salesdocument = read("salesdocument")
    customer = read("customer")
    address = read("address")

    # Collect all tables in the database as relbench Table objects.
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

    return tables


def main(out="rel-salt") -> None:
    raw = fetch(URL, SHA)
    write_hf(out, "rel-salt", VAL_TIMESTAMP, TEST_TIMESTAMP, build(raw))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-salt")
