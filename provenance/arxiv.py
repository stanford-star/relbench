import os

import pandas as pd
import pooch
from sklearn.preprocessing import LabelEncoder

from relbench.base import Database, Dataset, Table


class ArxivDataset(Dataset):
    val_timestamp = pd.Timestamp("2022-01-01")
    test_timestamp = pd.Timestamp("2023-01-01")

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""

        url = (
            "https://www.dropbox.com/scl/fi/tjj6r1fqikt4j0rz4qomu/db.zip?rlkey=1ykfkp8pj3hu6n4utz8g9dkx2&st"
            "=azmm56dc&dl=1"
        )

        path = pooch.retrieve(
            url,
            known_hash="ff9e03e467e28df959d08c79c453db1f31b525f07ff3c0e0b5e571e732acc63f",
            progressbar=True,
            processor=pooch.Unzip(),
        )

        if isinstance(path, list):
            path = os.path.dirname(path[0])

        print("Final dataset directory:", path)

        papers = pd.read_csv(os.path.join(path, "1Paper.csv"))
        categories = pd.read_csv(os.path.join(path, "2Category.csv"))
        citations = pd.read_csv(os.path.join(path, "3Citation.csv"))
        paperCategories = pd.read_csv(os.path.join(path, "4Paper_Category.csv"))
        authors = pd.read_csv(os.path.join(path, "5Author.csv"))
        paperAuthors = pd.read_csv(os.path.join(path, "6Paper_Author.csv"))

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

        # collect all tables in the database as relbench.base.Table objects.
        tables = {
            "papers": Table(
                df=pd.DataFrame(papers),
                fkey_col_to_pkey_table={},
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

        return Database(tables)
