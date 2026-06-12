import os

import pandas as pd
import pooch

from relbench.base import Database, Dataset, Table
from relbench.utils import clean_datetime, unzip_processor


class StackDataset(Dataset):
    # 3 months gap
    val_timestamp = pd.Timestamp("2020-10-01")
    test_timestamp = pd.Timestamp("2021-01-01")

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-forum-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "raw")
        users = pd.read_csv(os.path.join(path, "Users.csv"))
        comments = pd.read_csv(os.path.join(path, "Comments.csv"))
        posts = pd.read_csv(os.path.join(path, "Posts.csv"))
        votes = pd.read_csv(os.path.join(path, "Votes.csv"))
        postLinks = pd.read_csv(os.path.join(path, "PostLinks.csv"))
        badges = pd.read_csv(os.path.join(path, "Badges.csv"))
        postHistory = pd.read_csv(os.path.join(path, "PostHistory.csv"))

        # tags = pd.read_csv(os.path.join(path, "Tags.csv")) we remove tag table here since after removing time leakage columns, all information are kept in the posts tags columns

        ## remove time leakage columns
        users.drop(
            columns=["Reputation", "Views", "UpVotes", "DownVotes", "LastAccessDate"],
            inplace=True,
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

        return Database(tables)
