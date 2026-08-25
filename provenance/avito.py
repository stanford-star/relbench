r"""Generate the **rel-avito** database from its raw source (a 100k-row subsample of the
Avito context-ad-clicks Kaggle competition data).

    python avito.py [OUT_DIR]      # default OUT_DIR: ./rel-avito

Source: a subsampled snapshot of the Avito competition data, hosted by RelBench.
Produces the Hugging Face layout (manifest.yaml + db/*.parquet) reproducing
stanford-star/relbench-v1/rel-avito. The search stream ranges from 2015-04-25 to 2015-05-20; rows
before 2015-04-25 are dropped (the legacy ``Database.from_`` cutoff).
"""

import sys

import pandas as pd
from _lib import Table, clean_datetime, fetch, write_hf

URL = "https://huggingface.co/datasets/stanford-star/relbench-raw/resolve/main/rel-avito/rel-avito-raw-100k.zip"
SHA = "ad4fc1789d8a5073ea449049888c671899525c9a8a42359ca75d1f17d04d7929"
VAL_TIMESTAMP, TEST_TIMESTAMP = "2015-05-08", "2015-05-14"
FROM_TIMESTAMP = pd.Timestamp("2015-04-25")


def build(raw) -> dict:
    # Define table names
    ads_info = raw / "AdsInfo"
    category = raw / "Category"
    location = raw / "Location"
    phone_requests_stream = raw / "PhoneRequestsStream"
    search_info = raw / "SearchInfo"
    search_stream = raw / "SearchStream"
    user_info = raw / "UserInfo"
    visit_stream = raw / "VisitStream"

    # Load table as pandas dataframes
    ads_info_df = pd.read_parquet(ads_info)
    ads_info_df.dropna(subset=["AdID"], inplace=True)
    # Params column contains a dictionary of type Dict[int, str].
    # Drop it for now since we can not handle this column type yet.
    ads_info_df.drop(columns=["Params"], inplace=True)
    ads_info_df["Title"].fillna("", inplace=True)
    category_df = pd.read_parquet(category)
    location_df = pd.read_parquet(location)
    location_df.dropna(subset=["LocationID"], inplace=True)
    phone_requests_stream_df = pd.read_parquet(phone_requests_stream)
    search_info_df = pd.read_parquet(search_info)
    # SearchParams column contains a dictionary of type Dict[int, str].
    # Drop it for now since we can not handle this column type yet.
    search_info_df.drop(columns=["SearchParams"], inplace=True)
    search_stream_df = pd.read_parquet(search_stream)
    user_info_df = pd.read_parquet(user_info)
    visit_stream_df = pd.read_parquet(visit_stream)
    search_info_df = clean_datetime(search_info_df, "SearchDate")
    search_stream_df = clean_datetime(search_stream_df, "SearchDate")
    phone_requests_stream_df = clean_datetime(
        phone_requests_stream_df, "PhoneRequestDate"
    )
    visit_stream_df = clean_datetime(visit_stream_df, "ViewDate")

    category_df.drop(columns=["__index_level_0__"], inplace=True)

    # Drop rows before the cutoff (legacy ``Database.from_``); only the streams have a
    # time column, so the pkey-only tables are left unchanged.
    search_info_df = search_info_df.query("SearchDate >= @FROM_TIMESTAMP")
    search_stream_df = search_stream_df.query("SearchDate >= @FROM_TIMESTAMP")
    phone_requests_stream_df = phone_requests_stream_df.query(
        "PhoneRequestDate >= @FROM_TIMESTAMP"
    )
    visit_stream_df = visit_stream_df.query("ViewDate >= @FROM_TIMESTAMP")

    tables = {}
    tables["AdsInfo"] = Table(
        df=ads_info_df,
        fkey_col_to_pkey_table={
            "LocationID": "Location",
            "CategoryID": "Category",
        },
        pkey_col="AdID",
    )
    tables["Category"] = Table(
        df=category_df,
        fkey_col_to_pkey_table={},
        pkey_col="CategoryID",
    )
    tables["Location"] = Table(
        df=location_df,
        fkey_col_to_pkey_table={},
        pkey_col="LocationID",
    )
    tables["PhoneRequestsStream"] = Table(
        df=phone_requests_stream_df,
        fkey_col_to_pkey_table={
            "UserID": "UserInfo",
            "AdID": "AdsInfo",
        },
        time_col="PhoneRequestDate",
    )
    tables["SearchInfo"] = Table(
        df=search_info_df,
        fkey_col_to_pkey_table={
            "UserID": "UserInfo",
            "LocationID": "Location",
            "CategoryID": "Category",
        },
        pkey_col="SearchID",
        time_col="SearchDate",
    )
    tables["SearchStream"] = Table(
        df=search_stream_df,
        fkey_col_to_pkey_table={
            "SearchID": "SearchInfo",
            "AdID": "AdsInfo",
        },
        time_col="SearchDate",
    )
    tables["UserInfo"] = Table(
        df=user_info_df,
        fkey_col_to_pkey_table={},
        pkey_col="UserID",
    )
    tables["VisitStream"] = Table(
        df=visit_stream_df,
        fkey_col_to_pkey_table={
            "UserID": "UserInfo",
            "AdID": "AdsInfo",
        },
        time_col="ViewDate",
    )
    return tables


def main(out="rel-avito") -> None:
    raw = fetch(URL, SHA) / "avito_100k_integ_test"
    write_hf(out, "rel-avito", VAL_TIMESTAMP, TEST_TIMESTAMP, build(raw))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-avito")
