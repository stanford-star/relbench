r"""Generate the **rel-ratebeer** database from its raw source (RateBeer dump).

    python ratebeer.py [OUT_DIR]      # default OUT_DIR: ./rel-ratebeer

Source: a processed RateBeer SQL dump (``db.zip``) hosted on Dropbox. Produces the
Hugging Face layout (manifest.yaml + db/*.parquet) reproducing relbench/core/rel-ratebeer.
"""

import sys
from typing import Optional

import pandas as pd
from _lib import Table, fetch, write_hf

URL = "https://www.dropbox.com/scl/fi/exwygxep7vdvq55uiq28r/db.zip?rlkey=o7q0r8nw758p4wxx1wka9ubuj&st=rg3gvkxg&dl=1"
SHA = "c3921164da60f8c97e6530d1f2872f7e0d307f8276348106db95c10c2df677ad"
VAL_TIMESTAMP, TEST_TIMESTAMP = "2018-09-01", "2020-01-01"


def _process_timestamps(
    df: pd.DataFrame, table_name: str, time_col: Optional[str] = None
) -> pd.DataFrame:
    """Convert timestamp columns to datetime and remove rows with NaT in the designated
    time column."""
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


def build(raw) -> dict:
    r"""Process the raw files into a database."""
    print("Reading from processed database...")
    tables = {}

    beers = pd.read_csv(raw / "beers.csv", low_memory=False)
    brewers = pd.read_csv(raw / "brewers.csv", low_memory=False)
    beer_styles = pd.read_csv(raw / "beer_styles.csv", low_memory=False)
    countries = pd.read_csv(raw / "countries.csv", low_memory=False)
    states = pd.read_csv(raw / "states.csv", low_memory=False)
    users = pd.read_csv(raw / "users.csv", low_memory=False)
    beer_ratings = pd.read_csv(raw / "beer_ratings.csv", low_memory=False)
    beer_upcs = pd.read_csv(raw / "beer_upcs.csv", low_memory=False)
    availability = pd.read_csv(raw / "availability.csv", low_memory=False)
    favorites = pd.read_csv(raw / "favorites.csv", low_memory=False)
    place_ratings = pd.read_csv(raw / "place_ratings.csv", low_memory=False)
    places = pd.read_csv(raw / "places.csv", low_memory=False)
    place_types = pd.read_csv(raw / "place_types.csv", low_memory=False)

    # ---------------------- Beers ----------------------
    beers = _process_timestamps(beers, "beers", "created_at")

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

    # issue #373: drop future-leakage / artifact columns
    beers = beers.drop(
        columns=[
            "view_count",
            "avg_rating",
            "rating_count",
            "real_avg_rating",
            "rating_std_dev",
            "overall_percentile",
            "style_percentile",
            "last_9m_avg",
            "last_9m_count",
            "straight_avg_rating",
            "straight_rating_count",
            "year4_avg",
            "year4_overall",
            "year4_style",
            "year4_count",
            "updated_at",
        ],
        errors="ignore",
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

    # issue #373: drop future-leakage / artifact columns
    brewers = brewers.drop(
        columns=[
            "view_count",
            "score",
            "updated_at",
            "msrepl_tran_version",
        ],
        errors="ignore",
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
    users = _process_timestamps(users, "users", "created_at")

    users.drop(
        columns=[
            "favorite_first_added",  # 98.44% NA
            "favorite_last_added",  # 98.44% NA
        ],
        inplace=True,
    )

    # issue #373: drop future-leakage / artifact columns
    users = users.drop(
        columns=[
            "beer_first_rating",
            "beer_last_rating",
            "beer_rating_count",
            "avg_beer_rating",
            "max_beer_rating",
            "min_beer_rating",
            "place_first_rating",
            "place_last_rating",
            "place_rating_count",
            "avg_place_rating",
            "max_place_rating",
            "min_place_rating",
            "favorite_count",
            "total_activity_count",
            "updated_at",
        ],
        errors="ignore",
    )

    tables["users"] = Table(
        df=users,
        fkey_col_to_pkey_table={},
        pkey_col="user_id",
        time_col="created_at",
    )

    # ---------------------- Beer Ratings ----------------------
    beer_ratings = _process_timestamps(beer_ratings, "beer_ratings", "created_at")

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

    # issue #373: drop future-leakage / artifact columns
    beer_ratings = beer_ratings.drop(
        columns=[
            "updated_at",
        ],
        errors="ignore",
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
    availability = _process_timestamps(availability, "availability", time_col=None)

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
    favorites = _process_timestamps(favorites, "favorites", "created_at")

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

    # issue #373: drop future-leakage / artifact columns
    places = places.drop(
        columns=[
            "avg_rating",
            "weighted_avg",
            "bay_mean",
            "percentile",
            "rating_count",
            "valid_rating_count",
            "score",
            "rating_text",
            "updated_at",
        ],
        errors="ignore",
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
    place_ratings = _process_timestamps(place_ratings, "place_ratings", "created_at")

    place_ratings.drop(
        columns=[
            "latitude",  # 100% NA
            "longitude",  # 100% NA
        ],
        inplace=True,
    )

    # issue #373: drop future-leakage / artifact columns
    place_ratings = place_ratings.drop(
        columns=[
            "updated_at",
        ],
        errors="ignore",
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

    # issue #373: drop future-leakage / artifact columns
    states = states.drop(
        columns=[
            "msrepl_tran_version",
        ],
        errors="ignore",
    )

    tables["states"] = Table(
        df=states,
        fkey_col_to_pkey_table={"country_id": "countries"},
        pkey_col="state_id",
    )

    print("\nAll tables loaded successfully!")
    return tables


def main(out="rel-ratebeer") -> None:
    raw = next(fetch(URL, SHA).rglob("beers.csv")).parent
    write_hf(out, "rel-ratebeer", VAL_TIMESTAMP, TEST_TIMESTAMP, build(raw))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-ratebeer")
