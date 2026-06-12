import os
import zipfile
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pooch
from tqdm import tqdm

from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor

# Update URL to Dropbox direct download link
DB_URL = "https://www.dropbox.com/scl/fi/exwygxep7vdvq55uiq28r/db.zip?rlkey=o7q0r8nw758p4wxx1wka9ubuj&st=rg3gvkxg&dl=1"


class RateBeerDataset(Dataset):
    val_timestamp = pd.Timestamp("2018-09-01")
    test_timestamp = pd.Timestamp("2020-01-01")

    name = "rel-ratebeer"

    def _process_timestamps(
        self, df: pd.DataFrame, table_name: str, time_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Convert timestamp columns to datetime and remove rows with NaT in the
        designated time column."""
        # Convert timestamp columns
        for col in ["created_at", "updated_at", "last_edited_at", "opened_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

        # Remove rows with NaT in the designated time_col
        if time_col is not None:
            if time_col in df.columns and df[time_col].isna().any():
                initial_rows = len(df)
                nat_count = df[time_col].isna().sum()
                print(
                    f"\nWarning: Found {nat_count} NaT value(s) in time column '{time_col}' for table '{table_name}'. Removing these rows."
                )
                df = df.dropna(subset=[time_col])
                print(
                    f"Removed {initial_rows - len(df)} rows from {table_name}. New shape: {df.shape}"
                )

        return df

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        path = pooch.retrieve(
            DB_URL,
            known_hash="c3921164da60f8c97e6530d1f2872f7e0d307f8276348106db95c10c2df677ad",
            progressbar=True,
            processor=unzip_processor,
        )

        print("Reading from processed database...")
        tables = {}

        beers = pd.read_csv(os.path.join(path, "db", f"beers.csv"), low_memory=False)
        brewers = pd.read_csv(
            os.path.join(path, "db", f"brewers.csv"), low_memory=False
        )
        beer_styles = pd.read_csv(
            os.path.join(path, "db", f"beer_styles.csv"), low_memory=False
        )
        countries = pd.read_csv(
            os.path.join(path, "db", f"countries.csv"), low_memory=False
        )
        states = pd.read_csv(os.path.join(path, "db", f"states.csv"), low_memory=False)
        users = pd.read_csv(os.path.join(path, "db", f"users.csv"), low_memory=False)
        beer_ratings = pd.read_csv(
            os.path.join(path, "db", f"beer_ratings.csv"), low_memory=False
        )
        beer_upcs = pd.read_csv(
            os.path.join(path, "db", f"beer_upcs.csv"), low_memory=False
        )
        availability = pd.read_csv(
            os.path.join(path, "db", f"availability.csv"), low_memory=False
        )
        favorites = pd.read_csv(
            os.path.join(path, "db", f"favorites.csv"), low_memory=False
        )
        place_ratings = pd.read_csv(
            os.path.join(path, "db", f"place_ratings.csv"), low_memory=False
        )
        places = pd.read_csv(os.path.join(path, "db", f"places.csv"), low_memory=False)
        place_types = pd.read_csv(
            os.path.join(path, "db", f"place_types.csv"), low_memory=False
        )

        # ---------------------- Beers ----------------------
        beers = self._process_timestamps(beers, "beers", "created_at")

        beers.drop(
            columns=[
                "contract_brewer_id",  # 96.27% NA
                "contract_note",  # 99.96% NA
                "featured_beer_id",  # 100% NA
                "producer_style",  # 100% NA
                "LogoImage",  # 92.45% NA
                "beer_jobber_id",  # 99.65% NA
            ],
            inplace=True,
        )

        tables["beers"] = Table(
            df=beers,
            fkey_col_to_pkey_table={
                "brewer_id": "brewers",
                "style_id": "beer_styles",
            },
            pkey_col="beer_id",
            time_col="created_at",
        )

        # ---------------------- Brewers ----------------------
        brewers.drop(
            columns=[
                "newsletter_email",  # 99.85% NA
                "head_brewer",  # 100% NA
                "latitude",  # 100% NA
                "longitude",  # 100% NA
                "msa",  # 83.42% NA
                "instagram",  # 95.55% NA
            ],
            inplace=True,
        )

        tables["brewers"] = Table(
            df=brewers,
            fkey_col_to_pkey_table={
                "country_id": "countries",
                "state_id": "states",
                "type_id": "place_types",
            },
            pkey_col="brewer_id",
        )

        # ---------------------- Beer Styles ----------------------
        tables["beer_styles"] = Table(
            df=beer_styles,
            fkey_col_to_pkey_table={},
            pkey_col="style_id",
        )

        # ---------------------- Countries ----------------------
        tables["countries"] = Table(
            df=countries,
            fkey_col_to_pkey_table={},
            pkey_col="country_id",
        )

        # ---------------------- Users ----------------------
        users = self._process_timestamps(users, "users", "created_at")

        users.drop(
            columns=[
                "favorite_first_added",  # 98.44% NA
                "favorite_last_added",  # 98.44% NA
            ],
            inplace=True,
        )

        tables["users"] = Table(
            df=users,
            fkey_col_to_pkey_table={},
            pkey_col="user_id",
            time_col="created_at",
        )

        # ---------------------- Beer Ratings ----------------------
        beer_ratings = self._process_timestamps(
            beer_ratings, "beer_ratings", "created_at"
        )

        # Fix duplicate rating_id (rating_id = 1759935)
        duplicate_mask = beer_ratings.duplicated(subset=["rating_id"], keep="first")
        if duplicate_mask.any():
            print(f"Found {duplicate_mask.sum()} duplicate rating_id(s), fixing...")
            max_rating_id = beer_ratings["rating_id"].max()
            beer_ratings.loc[duplicate_mask, "rating_id"] += max_rating_id + 1

        beer_ratings.drop(
            columns=[
                "served_in",  # 99.94% NA
                "latitude",  # 100% NA
                "longitude",  # 100% NA
            ],
            inplace=True,
        )

        tables["beer_ratings"] = Table(
            df=beer_ratings,
            fkey_col_to_pkey_table={
                "user_id": "users",
                "beer_id": "beers",
                "availability_id": "availability",
            },
            pkey_col="rating_id",
            time_col="created_at",
        )

        # ---------------------- Availability ----------------------
        availability = self._process_timestamps(
            availability, "availability", time_col=None
        )

        availability.drop(
            columns=[
                "area_code",  # 100% NA
                "rating_id",  # 100% NA
                "tap_lister",  # 100% NA
            ],
            inplace=True,
        )

        tables["availability"] = Table(
            df=availability,
            fkey_col_to_pkey_table={
                "beer_id": "beers",
                "place_id": "places",
                "country_id": "countries",
                "user_id": "users",
            },
            pkey_col="avail_id",
        )

        # ---------------------- Beer UPCs ----------------------
        tables["beer_upcs"] = Table(
            df=beer_upcs,
            fkey_col_to_pkey_table={"beer_id": "beers"},
            pkey_col=None,  # Same UPC may map to multiple beers
        )

        # ---------------------- Favorites ----------------------
        favorites = self._process_timestamps(favorites, "favorites", "created_at")

        tables["favorites"] = Table(
            df=favorites,
            fkey_col_to_pkey_table={
                "user_id": "users",
                "beer_id": "beers",
            },
            pkey_col="favorite_id",
            time_col="created_at",
        )

        # ---------------------- Places ----------------------
        places.drop(
            columns=[
                "email",  # 100% NA
                "opened_at",  # 100% NA
                "phone_country_code",  # 100% NA
                "last_edited_at",  # 99.98% NA
                # "score",                  # 86.23% NA
            ],
            inplace=True,
        )

        tables["places"] = Table(
            df=places,
            fkey_col_to_pkey_table={
                "state_id": "states",
                "type_id": "place_types",
                "country_id": "countries",
            },
            pkey_col="place_id",
        )

        # ---------------------- Place Ratings ----------------------
        place_ratings = self._process_timestamps(
            place_ratings, "place_ratings", "created_at"
        )

        place_ratings.drop(
            columns=[
                "latitude",  # 100% NA
                "longitude",  # 100% NA
            ],
            inplace=True,
        )

        tables["place_ratings"] = Table(
            df=place_ratings,
            fkey_col_to_pkey_table={
                "place_id": "places",
                "user_id": "users",
            },
            pkey_col="rating_id",
            time_col="created_at",
        )

        # ---------------------- Place Types ----------------------
        tables["place_types"] = Table(
            df=place_types,
            fkey_col_to_pkey_table={},
            pkey_col="type_id",
        )

        # ---------------------- States ----------------------
        states.drop(
            columns=[
                "Abbrev",  # 86.66% NA
                # "hasbrewer",              # 79.58% NA
            ],
            inplace=True,
        )

        tables["states"] = Table(
            df=states,
            fkey_col_to_pkey_table={"country_id": "countries"},
            pkey_col="state_id",
        )

        print("\nAll tables loaded successfully!")
        return Database(tables)


if __name__ == "__main__":
    ratebeer_dataset = RateBeerDataset()
    ratebeer_db = ratebeer_dataset.make_db()

    total_rows = 0
    total_cols = 0

    # Print table statistics
    for table_name, table in ratebeer_db.table_dict.items():
        print(f"Number of rows in {table_name} table: {table.df.shape[0]}")
        print(f"Sample of {table_name} table:")
        print(table.df.head())
        print()
        total_rows += table.df.shape[0]
        total_cols += table.df.shape[1]

    print(f"Total number of rows: {total_rows}")
    print(f"Total number of columns: {total_cols}")
