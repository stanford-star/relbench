r"""Generate the **rel-arxiv** database from its raw source (arXiv papers, authors,
categories, and citations).

    python arxiv.py [OUT_DIR]      # default OUT_DIR: ./rel-arxiv

Source: a static snapshot of the arXiv relational data, hosted on Dropbox. Produces the
Hugging Face layout (manifest.yaml + db/*.parquet) reproducing stanford-star/relbench-v2-extra/rel-arxiv.
"""

import sys

import pandas as pd
from _lib import Table, fetch, write_hf
from sklearn.preprocessing import LabelEncoder

URL = (
    "https://www.dropbox.com/scl/fi/tjj6r1fqikt4j0rz4qomu/db.zip?rlkey=1ykfkp8pj3hu6n4utz8g9dkx2&st"
    "=azmm56dc&dl=1"
)
SHA = "ff9e03e467e28df959d08c79c453db1f31b525f07ff3c0e0b5e571e732acc63f"
VAL_TIMESTAMP, TEST_TIMESTAMP = "2022-01-01", "2023-01-01"


def build(raw) -> dict:
    papers = pd.read_csv(raw / "1Paper.csv")
    categories = pd.read_csv(raw / "2Category.csv")
    citations = pd.read_csv(raw / "3Citation.csv")
    paperCategories = pd.read_csv(raw / "4Paper_Category.csv")
    authors = pd.read_csv(raw / "5Author.csv")
    paperAuthors = pd.read_csv(raw / "6Paper_Author.csv")

    # Convert category column to integer
    le = LabelEncoder()
    categories["Category"] = le.fit_transform(categories["Category"])

    # Convert date column to pd.Timestamp
    papers["Submission_Date"] = pd.to_datetime(
        papers["Submission_Date"], format="%Y%m%d"
    )
    citations["Submission_Date"] = pd.to_datetime(
        citations["Submission_Date"], format="%Y%m%d"
    )
    paperAuthors["Submission_Date"] = pd.to_datetime(
        paperAuthors["Submission_Date"], format="%Y%m%d"
    )

    # add time column to other tables
    paperCategories = paperCategories.merge(
        papers[["Paper_ID", "Submission_Date"]], on="Paper_ID", how="left"
    )

    # collect all tables in the database as relbench Table objects.
    return {
        "papers": Table(
            df=pd.DataFrame(papers),
            fkey_col_to_pkey_table={"Primary_Category_ID": "categories"},
            pkey_col="Paper_ID",
            time_col="Submission_Date",
        ),
        "categories": Table(
            df=pd.DataFrame(categories),
            fkey_col_to_pkey_table={},
            pkey_col="Category_ID",
            time_col=None,
        ),
        "citations": Table(
            df=pd.DataFrame(citations),
            fkey_col_to_pkey_table={
                "Paper_ID": "papers",
                "References_Paper_ID": "papers",
            },
            pkey_col=None,
            time_col="Submission_Date",
        ),
        "paperCategories": Table(
            df=pd.DataFrame(paperCategories),
            fkey_col_to_pkey_table={
                "Paper_ID": "papers",
                "Category_ID": "categories",
            },
            pkey_col=None,
            time_col="Submission_Date",
        ),
        "authors": Table(
            df=pd.DataFrame(authors),
            fkey_col_to_pkey_table={},
            pkey_col="Author_ID",
            time_col=None,
        ),
        "paperAuthors": Table(
            df=pd.DataFrame(paperAuthors),
            fkey_col_to_pkey_table={"Paper_ID": "papers", "Author_ID": "authors"},
            pkey_col=None,
            time_col="Submission_Date",
        ),
    }


def main(out="rel-arxiv") -> None:
    raw = next(p for p in fetch(URL, SHA).rglob("1Paper.csv")).parent
    write_hf(out, "rel-arxiv", VAL_TIMESTAMP, TEST_TIMESTAMP, build(raw))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-arxiv")
