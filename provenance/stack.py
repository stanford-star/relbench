r"""Generate the **rel-stack** database from its raw source (Stack Exchange forum data).

    python stack.py [OUT_DIR]      # default OUT_DIR: ./rel-stack

Source: a Stack Exchange forum dump (Users/Posts/Comments/Votes/PostLinks/Badges/
PostHistory), hosted by RelBench. Produces the Hugging Face layout (manifest.yaml +
db/*.parquet) reproducing stanford-star/relbench/rel-stack.
"""

import os
import sys

import pandas as pd
from _lib import Table, clean_datetime, fetch, write_hf

URL = "https://relbench.stanford.edu/data/relbench-forum-raw.zip"
SHA = "ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f"
# 3 months gap
VAL_TIMESTAMP, TEST_TIMESTAMP = "2020-10-01", "2021-01-01"


def build(raw) -> dict:
    users = pd.read_csv(os.path.join(raw, "Users.csv"))
    comments = pd.read_csv(os.path.join(raw, "Comments.csv"))
    posts = pd.read_csv(os.path.join(raw, "Posts.csv"))
    votes = pd.read_csv(os.path.join(raw, "Votes.csv"))
    postLinks = pd.read_csv(os.path.join(raw, "PostLinks.csv"))
    badges = pd.read_csv(os.path.join(raw, "Badges.csv"))
    postHistory = pd.read_csv(os.path.join(raw, "PostHistory.csv"))

    # tags = pd.read_csv(os.path.join(raw, "Tags.csv")) we remove tag table here since after removing time leakage columns, all information are kept in the posts tags columns

    ## remove time leakage columns (ProfileImageUrl is 100% NaN -- issue #373)
    users.drop(
        columns=[
            "Reputation",
            "Views",
            "UpVotes",
            "DownVotes",
            "LastAccessDate",
            "ProfileImageUrl",
        ],
        inplace=True,
        errors="ignore",
    )

    posts.drop(
        columns=[
            "ViewCount",
            "AnswerCount",
            "CommentCount",
            "FavoriteCount",
            "CommunityOwnedDate",
            "ClosedDate",
            "LastEditDate",
            "LastActivityDate",
            "Score",
            "LastEditorDisplayName",
            "LastEditorUserId",
            "AcceptedAnswerId",
        ],
        inplace=True,
    )

    comments.drop(columns=["Score"], inplace=True)
    votes.drop(columns=["BountyAmount"], inplace=True)

    comments = clean_datetime(comments, "CreationDate")
    badges = clean_datetime(badges, "Date")
    postLinks = clean_datetime(postLinks, "CreationDate")
    postHistory = clean_datetime(postHistory, "CreationDate")
    votes = clean_datetime(votes, "CreationDate")
    users = clean_datetime(users, "CreationDate")
    posts = clean_datetime(posts, "CreationDate")

    tables = {}

    tables["comments"] = Table(
        df=pd.DataFrame(comments),
        fkey_col_to_pkey_table={
            "UserId": "users",
            "PostId": "posts",
        },
        pkey_col="Id",
        time_col="CreationDate",
    )

    tables["badges"] = Table(
        df=pd.DataFrame(badges),
        fkey_col_to_pkey_table={
            "UserId": "users",
        },
        pkey_col="Id",
        time_col="Date",
    )

    tables["postLinks"] = Table(
        df=pd.DataFrame(postLinks),
        fkey_col_to_pkey_table={
            "PostId": "posts",
            "RelatedPostId": "posts",  ## is this allowed? two foreign keys into the same primary
        },
        pkey_col="Id",
        time_col="CreationDate",
    )

    tables["postHistory"] = Table(
        df=pd.DataFrame(postHistory),
        fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
        pkey_col="Id",
        time_col="CreationDate",
    )

    tables["votes"] = Table(
        df=pd.DataFrame(votes),
        fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
        pkey_col="Id",
        time_col="CreationDate",
    )

    tables["users"] = Table(
        df=pd.DataFrame(users),
        fkey_col_to_pkey_table={},
        pkey_col="Id",
        time_col="CreationDate",
    )

    tables["posts"] = Table(
        df=pd.DataFrame(posts),
        fkey_col_to_pkey_table={
            "OwnerUserId": "users",
            "ParentId": "posts",  # notice the self-reference
        },
        pkey_col="Id",
        time_col="CreationDate",
    )

    return tables


def main(out="rel-stack") -> None:
    raw = os.path.join(fetch(URL, SHA), "raw")
    write_hf(out, "rel-stack", VAL_TIMESTAMP, TEST_TIMESTAMP, build(raw))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rel-stack")
