r"""Generate the **rel-event** database from its raw source (Event Recommendation).

    python event.py [OUT_DIR]      # default OUT_DIR: ./rel-event

Source: the Kaggle "Event Recommendation Engine Challenge" competition archive
(``event-recommendation-engine-challenge.zip``). The competition data requires Kaggle
credentials to download; once you have a Kaggle API key::

    kaggle competitions download -c event-recommendation-engine-challenge

place the resulting ``event-recommendation-engine-challenge.zip`` under
``$RELBENCH_RAW_CACHE`` (default ``~/.cache/relbench-raw``) so ``fetch`` can extract it.
Produces the Hugging Face layout (manifest.yaml + db/*.parquet) reproducing
stanford-star/relbench-v1/rel-event.
"""

import sys

import pandas as pd
from _lib import Table, fetch, write_hf

URL = (
    "https://www.kaggle.com/competitions/event-recommendation-engine-challenge/"
    "event-recommendation-engine-challenge.zip"
)
SHA = None  # Kaggle-gated archive; hash not pinned.
VAL_TIMESTAMP, TEST_TIMESTAMP = "2012-11-21", "2012-11-29"


def build(raw) -> dict:
    users = raw / "users.csv"
    user_friends = raw / "user_friends.csv"
    events = raw / "events.csv"
    event_attendees = raw / "event_attendees.csv"

    users_df = pd.read_csv(users, dtype={"user_id": int}, parse_dates=["joinedAt"])
    users_df["birthyear"] = pd.to_numeric(users_df["birthyear"], errors="coerce")
    users_df["joinedAt"] = pd.to_datetime(
        users_df["joinedAt"], errors="coerce", format="mixed"
    ).dt.tz_localize(None)

    events_df = pd.read_csv(events)
    events_df["start_time"] = pd.to_datetime(
        events_df["start_time"], errors="coerce", format="mixed"
    ).dt.tz_localize(None)

    train = raw / "train.csv"
    event_interest_df = pd.read_csv(train)
    event_interest_df["timestamp"] = pd.to_datetime(
        event_interest_df["timestamp"], format="mixed"
    ).dt.tz_localize(None)

    user_friends_df = pd.read_csv(user_friends)
    user_friends_df = (
        user_friends_df.set_index("user")["friends"]
        .str.split(expand=True)
        .stack()
        .reset_index()
    )
    user_friends_df.columns = ["user", "index", "friend"]
    user_friends_flattened_df = user_friends_df.drop("index", axis=1).assign(
        user=lambda df: df["user"].astype(int),
        friend=lambda df: df["friend"].astype(int),
    )

    # Some friends are not present in the user table, so we drop those friends
    # in the user_friends table
    user_friends_flattened_df = user_friends_flattened_df.merge(
        users_df, how="inner", left_on="friend", right_on="user_id"
    )
    user_friends_flattened_df = user_friends_flattened_df[["user", "friend"]]
    # issue #373: published rel-event drops the leftover pandas row-index column.
    user_friends_flattened_df = user_friends_flattened_df.drop(
        columns=["Unnamed: 0"], errors="ignore"
    )

    event_attendees_df = pd.read_csv(event_attendees)
    melted_df = event_attendees_df.melt(
        id_vars=["event"],
        value_vars=["yes", "maybe", "invited", "no"],
        var_name="status",
        value_name="user_ids",
    )
    melted_df = melted_df.dropna()
    melted_df["user_ids"] = melted_df["user_ids"].str.split()
    melted_df["user_ids"] = melted_df["user_ids"].apply(lambda x: [int(i) for i in x])
    exploded_df = melted_df.explode("user_ids")
    exploded_df["user_ids"] = exploded_df["user_ids"].astype(int)
    exploded_df.rename(columns={"user_ids": "user_id"}, inplace=True)
    exploded_df = pd.merge(
        exploded_df,
        events_df[["event_id", "start_time"]],
        left_on="event",
        right_on="event_id",
        how="left",
    )
    exploded_df = exploded_df.drop("event_id", axis=1)
    event_attendees_flattened_df = exploded_df.dropna(subset=["user_id"])
    # issue #373: published rel-event drops the leftover pandas row-index column.
    event_attendees_flattened_df = event_attendees_flattened_df.drop(
        columns=["Unnamed: 0"], errors="ignore"
    )

    return {
        "users": Table(
            df=users_df,
            fkey_col_to_pkey_table={},
            pkey_col="user_id",
            time_col="joinedAt",
        ),
        "events": Table(
            df=events_df,
            fkey_col_to_pkey_table={"user_id": "users"},
            pkey_col="event_id",
            time_col="start_time",
        ),
        "event_attendees": Table(
            df=event_attendees_flattened_df,
            fkey_col_to_pkey_table={
                "event": "events",
                "user_id": "users",
            },
            time_col="start_time",
        ),
        "event_interest": Table(
            df=event_interest_df,
            fkey_col_to_pkey_table={
                "event": "events",
                "user": "users",
            },
            time_col="timestamp",
        ),
        "user_friends": Table(
            df=user_friends_flattened_df,
            fkey_col_to_pkey_table={
                "user": "users",
                "friend": "users",
            },
        ),
    }


def main(out="rel-event") -> None:
    raw = fetch(URL, SHA)
    write_hf(out, "rel-event", VAL_TIMESTAMP, TEST_TIMESTAMP, build(raw))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-event")
