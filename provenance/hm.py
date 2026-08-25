r"""Generate the **rel-hm** database from its raw source (H&M fashion recommendations).

    python hm.py [OUT_DIR]      # default OUT_DIR: ./rel-hm

Source: the Kaggle "H&M Personalized Fashion Recommendations" competition archive
(``h-and-m-personalized-fashion-recommendations.zip``: the ``customers.csv``,
``articles.csv``, and ``transactions_train.csv`` dumps). The competition data requires
Kaggle credentials to download; once you have a Kaggle API key::

    kaggle competitions download -c h-and-m-personalized-fashion-recommendations

place the resulting zip at
``$RELBENCH_RAW_CACHE/h-and-m-personalized-fashion-recommendations.zip`` (default
``~/.cache/relbench-raw/h-and-m-personalized-fashion-recommendations.zip``); ``fetch``
then uses it without downloading. Produces the Hugging Face layout (manifest.yaml +
db/*.parquet) reproducing stanford-star/relbench-v1/rel-hm.
"""

import sys

import pandas as pd
from _lib import Table, fetch, write_hf

URL = "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations"
SHA = None
FILENAME = "h-and-m-personalized-fashion-recommendations.zip"
VAL_TIMESTAMP, TEST_TIMESTAMP = "2020-09-07", "2020-09-14"
DESCRIPTION = (
    "H&M e-commerce: customers, articles, and time-stamped purchase transactions."
)

# Drop transactions before this date (legacy ``Database.from_`` cutoff).
FROM_TIMESTAMP = pd.Timestamp("2019-09-07")


def build(raw) -> dict:
    articles_df = pd.read_csv(raw / "articles.csv")
    customers_df = pd.read_csv(raw / "customers.csv")
    transactions_df = pd.read_csv(raw / "transactions_train.csv")
    transactions_df["t_dat"] = pd.to_datetime(
        transactions_df["t_dat"], format="%Y-%m-%d"
    )

    # Drop transactions before the cutoff (legacy ``Database.from_``); the article and
    # customer tables have no time column and are left unchanged.
    transactions_df = transactions_df.query("t_dat >= @FROM_TIMESTAMP")

    return {
        "article": Table(
            df=articles_df,
            fkey_col_to_pkey_table={},
            pkey_col="article_id",
        ),
        "customer": Table(
            df=customers_df,
            fkey_col_to_pkey_table={},
            pkey_col="customer_id",
        ),
        "transactions": Table(
            df=transactions_df,
            fkey_col_to_pkey_table={
                "customer_id": "customer",
                "article_id": "article",
            },
            time_col="t_dat",
        ),
    }


def main(out="rel-hm") -> None:
    raw = fetch(URL, SHA, filename=FILENAME)
    write_hf(
        out,
        "rel-hm",
        VAL_TIMESTAMP,
        TEST_TIMESTAMP,
        build(raw),
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-hm")
