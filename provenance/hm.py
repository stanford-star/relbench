import os
import shutil
from pathlib import Path

import pandas as pd

from relbench.base import Database, Dataset, Table


class HMDataset(Dataset):
    url = (
        "https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations"
    )

    val_timestamp = pd.Timestamp("2020-09-07")
    test_timestamp = pd.Timestamp("2020-09-14")

    def make_db(self) -> Database:
        path = os.path.join("data", "hm-recommendation")
        zip = os.path.join(path, "h-and-m-personalized-fashion-recommendations.zip")
        customers = os.path.join(path, "customers.csv")
        articles = os.path.join(path, "articles.csv")
        transactions = os.path.join(path, "transactions_train.csv")
        if not os.path.exists(customers):
            if not os.path.exists(zip):
                raise RuntimeError(
                    f"Dataset not found. Please download "
                    f"h-and-m-personalized-fashion-recommendations.zip from "
                    f"'{self.url}' and move it to '{path}'. Once you have your"
                    f"Kaggle API key, you can use the following command: "
                    f"kaggle competitions download -c h-and-m-personalized-fashion-recommendations"
                )
            else:
                print("Unpacking")
                shutil.unpack_archive(zip, Path(zip).parent)

        articles_df = pd.read_csv(articles)
        customers_df = pd.read_csv(customers)
        transactions_df = pd.read_csv(transactions)
        transactions_df["t_dat"] = pd.to_datetime(
            transactions_df["t_dat"], format="%Y-%m-%d"
        )

        db = Database(
            table_dict={
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
        )

        db = db.from_(pd.Timestamp("2019-09-07"))

        return db
