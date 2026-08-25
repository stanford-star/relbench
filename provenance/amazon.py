r"""Generate the **rel-amazon** database from its raw source (Amazon product reviews).

    python amazon.py [OUT_DIR]      # default OUT_DIR: ./rel-amazon

Source: the McAuley-group Amazon Review Data (v2) hosted at UCSD -- the per-category
product metadata (``meta_Books.json.gz``) and 5-core review (``Books_5.json.gz``) dumps.
Produces the Hugging Face layout (manifest.yaml + db/*.parquet) reproducing
stanford-star/relbench-v1/rel-amazon.
"""

import gzip
import shutil
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.json
from _lib import Table, fetch, write_hf

URL_PREFIX = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2"
PRODUCT_URL = f"{URL_PREFIX}/metaFiles2/meta_Books.json.gz"
REVIEW_URL = f"{URL_PREFIX}/categoryFilesSmall/Books_5.json.gz"
PRODUCT_SHA = "80ed7ac64f5967a140401e8d7bf0587d2e5087492de9e94077a7f554ef6b18f0"
REVIEW_SHA = "ded924d1d1a22bae499f1a1c2b39397104304bfdb24232a2dd0aa50e89cd37bb"
VAL_TIMESTAMP, TEST_TIMESTAMP = "2015-10-01", "2016-01-01"
DESCRIPTION = (
    "Amazon product reviews: customers, products, and time-stamped reviews and "
    "ratings across the Amazon catalog."
)

# Drop reviews before this date (legacy ``Database.from_`` cutoff).
FROM_TIMESTAMP = pd.Timestamp("2008-01-01")


def decompress_gz_file(input_path, output_path):
    with gzip.open(input_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def build(product_path, review_path) -> dict:
    ### product table ###

    print(f"reading product info from {product_path}...")
    tic = time.time()
    ptable = pa.json.read_json(
        product_path,
        parse_options=pa.json.ParseOptions(
            explicit_schema=pa.schema(
                [
                    ("asin", pa.string()),
                    ("category", pa.list_(pa.string())),
                    ("brand", pa.string()),
                    ("title", pa.string()),
                    ("description", pa.list_(pa.string())),
                    ("price", pa.string()),
                ]
            ),
            unexpected_field_behavior="ignore",
        ),
    )
    toc = time.time()
    print(f"done in {toc - tic:.2f} seconds.")

    print("converting to pandas dataframe...")
    tic = time.time()
    pdf = ptable.to_pandas()
    toc = time.time()
    print(f"done in {toc - tic:.2f} seconds.")

    print("processing product info...")
    tic = time.time()

    # asin is not intuitive / recognizable
    pdf.rename(columns={"asin": "product_id"}, inplace=True)

    # somehow the raw data has duplicate product_id's
    pdf.drop_duplicates(subset=["product_id"], inplace=True)

    # price is like "$x,xxx.xx", "$xx.xx", or "$xx.xx - $xx.xx", or garbage html
    # if it's a range, we take the first value
    pdf.loc[:, "price"] = pdf["price"].apply(
        lambda x: (
            None
            if x is None or x == "" or x[0] != "$"
            else float(x.split(" ")[0][1:].replace(",", ""))
        )
    )

    # remove products with missing price
    pdf = pdf.dropna(subset=["price"])

    pdf.loc[:, "category"] = pdf["category"].apply(
        lambda x: None if x is None or len(x) == 0 else x
    )

    # some rows are stored as ['cat1' 'cat2' 'cat3' ...]
    # this function maps them to ['cat1', 'cat2', 'cat3', ...] (list of strings)
    # since otherwise pytorch-frame breaks
    def fix_column(value):
        if isinstance(value, str):
            return value  # Already a string
        elif value is None:
            return None
        else:
            return list(value)

    pdf["category"] = pdf["category"].apply(fix_column)

    # description is either [] or ["some description"]
    pdf.loc[:, "description"] = pdf["description"].apply(
        lambda x: None if x is None or len(x) == 0 else x[0]
    )

    toc = time.time()
    print(f"done in {toc - tic:.2f} seconds.")

    ### review table ###

    print(f"reading review and customer info from {review_path}...")
    tic = time.time()
    rtable = pa.json.read_json(
        review_path,
        parse_options=pa.json.ParseOptions(
            explicit_schema=pa.schema(
                [
                    ("unixReviewTime", pa.int32()),
                    ("reviewerID", pa.string()),
                    ("reviewerName", pa.string()),
                    ("asin", pa.string()),
                    ("overall", pa.float32()),
                    ("verified", pa.bool_()),
                    ("reviewText", pa.string()),
                    ("summary", pa.string()),
                ]
            ),
            unexpected_field_behavior="ignore",
        ),
    )
    toc = time.time()
    print(f"done in {toc - tic:.2f} seconds.")

    print("converting to pandas dataframe...")
    tic = time.time()
    rdf = rtable.to_pandas()
    toc = time.time()
    print(f"done in {toc - tic:.2f} seconds.")

    print("processing review and customer info...")
    tic = time.time()

    rdf.rename(
        columns={
            "unixReviewTime": "review_time",
            "reviewerID": "customer_id",
            "reviewerName": "customer_name",
            "asin": "product_id",
            "overall": "rating",
            "reviewText": "review_text",
        },
        inplace=True,
    )

    rdf.loc[:, "review_time"] = pd.to_datetime(rdf["review_time"], unit="s")

    toc = time.time()
    print(f"done in {toc - tic:.2f} seconds.")

    print("keeping only products common to product and review tables...")
    tic = time.time()
    plist = list(set(pdf["product_id"]) & set(rdf["product_id"]))
    pdf = pdf[pdf["product_id"].isin(plist)]
    rdf = rdf[rdf["product_id"].isin(plist)]
    toc = time.time()
    print(f"done in {toc - tic:.2f} seconds.")

    print("extracting customer table...")
    tic = time.time()
    cdf = (
        rdf[["customer_id", "customer_name"]]
        .drop_duplicates(subset=["customer_id"])
        .copy()
    )
    rdf.drop(columns=["customer_name"], inplace=True)
    toc = time.time()
    print(f"done in {toc - tic:.2f} seconds.")

    # Drop reviews before the cutoff (legacy ``Database.from_``); the product and
    # customer tables have no time column and are left unchanged.
    rdf = rdf.query("review_time >= @FROM_TIMESTAMP")

    return {
        "product": Table(
            df=pdf,
            fkey_col_to_pkey_table={},
            pkey_col="product_id",
            time_col=None,
        ),
        "customer": Table(
            df=cdf,
            fkey_col_to_pkey_table={},
            pkey_col="customer_id",
            time_col=None,
        ),
        "review": Table(
            df=rdf,
            fkey_col_to_pkey_table={
                "customer_id": "customer",
                "product_id": "product",
            },
            pkey_col=None,
            time_col="review_time",
        ),
    }


def main(out="rel-amazon") -> None:
    product_gz = fetch(PRODUCT_URL, PRODUCT_SHA)
    review_gz = fetch(REVIEW_URL, REVIEW_SHA)
    product_path = product_gz.with_suffix("")  # strip .gz
    review_path = review_gz.with_suffix("")
    if not product_path.exists():
        decompress_gz_file(product_gz, product_path)
    if not review_path.exists():
        decompress_gz_file(review_gz, review_path)
    write_hf(
        out,
        "rel-amazon",
        VAL_TIMESTAMP,
        TEST_TIMESTAMP,
        build(product_path, review_path),
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-amazon")
