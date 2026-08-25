r"""Generate the **stanford-star/dbinfer** database family from the 4DBInfer collection.

    python dbinfer.py DATASET [OUT_DIR]   # one dataset (default OUT_DIR: ./DATASET)
    python dbinfer.py --all  [OUT_ROOT]   # the whole collection (default OUT_ROOT: ./dbinfer)

Unlike the single-database generators in this directory, ``dbinfer`` ports a *collection*:
the 4DBInfer relational benchmark (Wang et al., "4DBInfer: A 4D Benchmarking Toolbox for
Graph-Centric Predictive Modeling on Relational DBs", NeurIPS 2024 Datasets & Benchmarks;
https://github.com/awslabs/multi-table-benchmark) exported to RelBench format. The
collection spans the seven multi-table databases listed in ``DATASETS`` below
(``dbinfer-amazon``, ``dbinfer-avs``, ``dbinfer-diginetica``, ``dbinfer-outbrain-small``,
``dbinfer-retailrocket``, ``dbinfer-seznam``, ``dbinfer-stackexchange``).

Source of truth
---------------
This generator reads the **original 4DBInfer artifacts** --
``https://data.dgl.ai/mtbench/20240304-<name>.tar``, the archives that ``dbinfer_bench``'s
own downloader fetches -- and builds both the database and the task labels from them. Each
archive is a self-describing RDB dataset: a ``metadata.yaml`` declaring every table's
columns and their 4DBInfer dtype (``primary_key`` / ``foreign_key`` with ``link_to:
<Table>.<column>`` / ``datetime`` / ``category`` / ``float`` / ``text``), plus
``data/<table>.pqt`` and ``<task>/<split>.pqt`` payloads.

An earlier revision of this generator instead re-served pre-built ``db.zip`` artifacts
produced by the upstream ``dbinfer-relbench-adapter`` export pipeline. Those artifacts are
**corrupt** and must not be used: see ``UPSTREAM_ADAPTER_BUGS`` below.

Port decisions
--------------
1. *Columns*: exactly the columns ``metadata.yaml`` declares. The payload parquet often
   carries extras -- dask/pandas index artifacts (``__index_level_0__``,
   ``__null_dask_index__``) and, in ``stackexchange``, undeclared full-history aggregates
   (``Users.Reputation/Views/UpVotes/DownVotes``, ``Posts.Score/ViewCount/AnswerCount/
   CommentCount/FavoriteCount``, ``Tag.Count``, ``Comments.Score``) plus, in ``avs``, the
   ``repeater``/``repeattrips`` label columns. None of these is part of the benchmark, and
   several encode the future relative to a row's timestamp (``upvote``'s label is derived
   from ``Posts.Score``), so they are dropped. ``DROPPED_UNDECLARED`` records them.
2. *Implicit dimension tables*: 4DBInfer lets a ``foreign_key`` ``link_to`` a table that has
   no payload of its own -- the target exists only as a key domain (``Item``, ``Visitor``,
   ``Customer``, ``Chain``, ``Brand``, ``Category``, ``Company``, ``Session``, ``User``,
   ``Orders``, ``Token``). RelBench needs a table to point at, so each is materialized as a
   single-column key table holding the sorted union of the values observed in every
   referencing column (database *and* task). Without them the foreign-key graph would have
   to be amputated -- and ``avs``/``retailrocket``'s entity tasks would have no entity
   table.
3. *Key reindexing*: ``Database.reindex_pkeys_and_fkeys`` semantics -- a table with a
   primary key is ordered by its time column (when it has one), its key becomes ``0..n-1``,
   and every referencing foreign key is remapped through that same mapping. The mapping is
   applied to the **task** tables too, so a task's entity column indexes its entity table.
   A table with no declared primary key gets ``pkey: null``, as ``rel-amazon/review`` and
   ``rel-hm/transactions`` do. (The previous revision gave such tables a
   ``__synthetic_pk__`` numbered in payload order; because ``Dataset.get_db`` trims by
   timestamp and then requires a *consecutive* primary key, that made every dataset whose
   fact table is not stored in time order -- ``diginetica``, ``outbrain-small``,
   ``retailrocket`` -- fail to load at all.)
4. *Task labels*: served verbatim from the archive (``kind: external``), with two coercions
   only -- ``avs``'s ``'t'/'f'`` and ``stackexchange``'s booleans become ``int8`` 0/1, so
   binary metrics work. Multiclass targets keep their source values (``seznam``'s ``sluzba``
   stays ``'a'..'h'``, matching the ``Dobito``/``Probehnuto`` column it labels).
5. *Splits*: ``val_timestamp`` / ``test_timestamp`` are the earliest label time in any
   task's val / test split, floored to the day, so trimming the database at either cutoff
   cannot expose a label. (The previous revision's cutoffs fell *after* the end of the data,
   so ``get_db(upto_test_timestamp=True)`` trimmed nothing at all.)
6. *Label leakage*: where a task's target is also a declared column of the table 4DBInfer
   derived it from, the task declares ``remove_columns`` so the loader drops it from the
   graph. This is derived, not hand-listed: it catches ``retailrocket/cvr`` ->
   ``View.added_to_cart``, ``outbrain-small/ctr`` -> ``Click.clicked``, ``seznam/charge`` ->
   ``Dobito.sluzba``, ``seznam/prepay`` -> ``Probehnuto.sluzba`` and ``amazon/rating`` ->
   ``Review.rating``, and correctly leaves ``diginetica/ctr`` alone (its ``Click`` table
   declares no ``clicked`` column).

Retrieval tasks (``amazon/purchase``, ``diginetica/purchase``) are 4DBInfer's MRR-scored
candidate-ranking protocol: the train split holds positives only, while val/test enumerate
candidates with ``label`` and ``query_idx`` columns. They are declared
``task_type: recommendation`` -- with both endpoints named, so the labels are at least
joinable -- rather than being forced into the entity-task mould as multiclass targets.
RelBench's default link metric (MAP over a ranked list) is *not* 4DBInfer's MRR over the
supplied candidate set, so score them with your own metric over ``label``/``query_idx``;
each task's description says so.

``mag`` is the one 4DBInfer database this generator does not port: its ``Paper.feat``
column is a dense embedding matrix shipped as ``.npz``, which the RelBench manifest format
(native parquet column dtypes only) has no representation for.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import yaml
from _lib import fetch

from relbench.base.task_base import sort_labels
from relbench.manifest import DatasetManifest, TableSpec, TaskManifest

# 4DBInfer's own download source and data version (``dbinfer_bench/download_config.yaml``).
BASE_URL = "https://data.dgl.ai/mtbench"
VERSION = "20240304"

# Why the previous ``db.zip``-based port had to be discarded. Recorded here because the
# artifacts are still hosted and still look plausible.
UPSTREAM_ADAPTER_BUGS = r"""
``dbinfer-relbench-adapter``'s ``DBInferDatasetAdapter.get_db`` (MockTable.__init__):

  * overwrites every real primary key with ``np.arange(len(df))`` *before* validating
    foreign keys, so the key correspondence is gone by the time the keys are checked;
  * validates each foreign key against ``target_size = len(all_tables[target_table])``,
    where ``all_tables`` maps table name -> {column: array}. ``len`` of that dict is the
    parent's **column count**, so every foreign-key value >= the parent's number of columns
    is set to NaN. In the published artifacts this nulls 99.9-100% of every declared
    foreign key (``dbinfer-seznam/Dobito.client_id``: 554,096/554,096 null;
    ``dbinfer-amazon/Review.product_id``: 13,716,231/13,717,752 null, the 1,521 survivors
    spanning ids 0-5 because ``Product`` has six columns);
  * drops foreign keys whose ``link_to`` target has no payload table, so e.g.
    ``retailrocket/View`` lost both of its foreign keys;
  * in ``DBInferTaskAdapter``, remaps task entity ids through a task-local
    ``sorted(entities) -> 0..n-1`` mapping unrelated to the database's, relabels
    classification targets by sorted-string order, infers ``task_type`` from target
    cardinality (turning ``amazon/rating``, a 4DBInfer regression/RMSE task, into
    multiclass), and coerces integer-valued time columns with ``astype('datetime64[ns]')``,
    producing timestamps like ``1970-01-01 00:00:00.000000022``.
"""

# Undeclared payload columns dropped by decision (1) above, for the record.
DROPPED_UNDECLARED: dict = {
    "amazon": {"Customer": ["customer_name"]},
    "avs": {"History": ["repeattrips", "repeater"]},
    "stackexchange": {
        "Comments": ["Score", "UserDisplayName", "ContentLicense"],
        "PostHistory": ["RevisionGUID", "UserDisplayName", "ContentLicense"],
        "Posts": [
            "Score",
            "ViewCount",
            "LastActivityDate",
            "Tags",
            "AnswerCount",
            "CommentCount",
            "FavoriteCount",
            "LastEditDate",
            "CommunityOwnedDate",
            "ClosedDate",
            "OwnerDisplayName",
            "LastEditorDisplayName",
            "ContentLicense",
        ],
        "Tag": ["Count"],
        "Users": [
            "Reputation",
            "DisplayName",
            "LastAccessDate",
            "WebsiteUrl",
            "Views",
            "UpVotes",
            "DownVotes",
            "AccountId",
        ],
    },
}

# Index artifacts that appear in the payloads and are never part of a table.
INDEX_ARTIFACTS = ("__index_level_0__", "__null_dask_index__")

# Defects in the 4DBInfer archives themselves, surfaced on the dataset cards. Nothing in
# this port can repair them -- the referenced rows are simply not in the source -- but the
# previous revision hid them behind all-null foreign keys, so they are stated plainly.
# Reproduce the numbers with ``provenance/check_dbinfer.py``.
UPSTREAM_DEFECTS: dict = {
    "outbrain-small": """\
**This database has almost no referential integrity, in the source.** 4DBInfer's
`outbrain-small` subsamples each table independently, so the overwhelming majority of its
foreign keys point at rows that were not kept:

| foreign key | distinct values | resolve |
|---|---|---|
| `Click.display_id` -> `Event` | 86,929 | 78 (0.09%) |
| `Click.ad_id` -> `PromotedContent` | 23,391 | 33 (0.14%) |
| `Pageview.document_id` -> `DocumentsMeta` | 436,049 | 437 (0.10%) |
| `Event.document_id` -> `DocumentsMeta` | 5,651 | 891 (15.77%) |
| `DocumentsTopic/Category/Entity.document_id` -> `DocumentsMeta` | 5.5k-11.3k | 4-6 (~0.1%) |

The `ctr` task inherits this: only 58 of 69,543 distinct train `display_id`s (and 5 of
8,693 val, 15 of 8,693 test) exist in `Event`, so `filter_dangling_entities` leaves almost
nothing. Treat this dataset as unusable for relational modeling until upstream republishes
it; there is no full-size `outbrain` archive to fall back on.""",
    "diginetica": """\
Two source-side quirks, both faithful to the archive:

* `QuerySearchstringToken.queryId` -> `Query`: 43,759 of 138,260 values (31.65%) name a
  query that is not in `Query`, so they are null here.
* `Query.userId`, `View.userId`, `Purchase.userId` are 64%/70%/63% null in the source --
  Diginetica sessions are mostly anonymous. That is the data, not a porting loss.""",
    "seznam": """\
The source's task splits are monthly and share their boundary month: `charge`/`prepay`
train runs to `2015-04-01` and val *starts* at `2015-04-01` (likewise val/test at
`2015-07-01`). `val_timestamp`/`test_timestamp` are set to those boundaries, so a database
trimmed at a cutoff still contains the boundary month's rows. Trim strictly below the
cutoff if that matters for your setup.""",
    "amazon": """\
The three tasks are split at different points in time (`churn` from 2015-10-03,
`purchase` from 2015-12-29, `rating` from 2015-12-30). The dataset-level
`val_timestamp`/`test_timestamp` take the earliest of each, so trimming the database at a
cutoff is conservative for every task rather than exact for one.""",
}

# Per-dataset: sha256 of ``<VERSION>-<name>.tar``, one-line description, and the domain
# recorded in the repo's STATS table.
DATASETS: dict = {
    "amazon": dict(
        sha256="28eddd41cdd76b3b48da8d18f627dcd6495ab4747ac56cca7d45053f761578c3",
        domain="E-commerce (reviews)",
        description=(
            "Amazon from the 4DBInfer benchmark: a large product-review dataset "
            "linking users, products and reviews, used for rating prediction and "
            "user purchase/churn prediction."
        ),
    ),
    "avs": dict(
        sha256="60eec9edafa20a322526cff47fb3d0994a54be8015f6c98b615d32150cd4bd8f",
        domain="Retail (Acquire Valued Shoppers)",
        description=(
            "Acquire Valued Shoppers (AVS) from the 4DBInfer benchmark: a retail "
            "dataset of customer transaction histories and promotional offers, used "
            "to predict shopper behavior such as offer repeat purchases."
        ),
    ),
    "diginetica": dict(
        sha256="43738691944d4f22802e170887ccc2444ecc3dd7304f9fd7d9072d10b83c3fe3",
        domain="E-commerce (sessions)",
        description=(
            "Diginetica from the 4DBInfer benchmark: an e-commerce dataset of user "
            "browsing and purchasing sessions over a product catalog (CIKM Cup 2016), "
            "used for click-through-rate and purchase prediction."
        ),
    ),
    "outbrain-small": dict(
        sha256="c1241e5b4f5659a570f3dfb820992b25d42e730260dfd0e134380c8aa488d855",
        domain="Content recommendation",
        description=(
            "Outbrain (small) from the 4DBInfer benchmark: a content-recommendation "
            "dataset of document page views and promoted-content displays/clicks, used "
            "for click-through-rate prediction."
        ),
    ),
    "retailrocket": dict(
        sha256="54df293f8ad8487bdce944b8bf88032e4fdeafddb2d3ef3cb568d4b35188e95e",
        domain="E-commerce (behaviour)",
        description=(
            "RetailRocket from the 4DBInfer benchmark: an e-commerce dataset of visitor "
            "events (views, add-to-cart, transactions) over an item catalog, used to "
            "predict conversion (whether a viewed item is later purchased)."
        ),
    ),
    "seznam": dict(
        sha256="edf063e5f4ba0b0be80d191c2d169749735b17f8422a85857db89de52e2ef1ae",
        domain="Digital advertising",
        description=(
            "Seznam from the 4DBInfer benchmark: a digital-advertising dataset from the "
            "Seznam.cz search engine, containing client prepaid-account charges and "
            "transactions, used to predict which service an account transacts on."
        ),
    ),
    "stackexchange": dict(
        sha256="bad013286e2b24c9a747eb4952eb465090c9af8a4df5f646714d00171804355b",
        domain="Online community (Q&A)",
        description=(
            "StackExchange from the 4DBInfer benchmark: the Cross Validated "
            "(stats.stackexchange.com) community-Q&A dataset of users, posts, votes and "
            "badges, used to predict user churn and post upvotes."
        ),
    ),
}

# Per-task RelBench declarations that ``metadata.yaml`` cannot supply: which table the
# entity column indexes (4DBInfer records the table a label was *derived from*, which for a
# fact-table task is not the entity's table), the link-task endpoints, the human
# description, and the database columns a task must not see.
#
#   (dataset, task) -> dict(entity_table=, entity_col=, description=, remove_columns=,
#                           src_entity_table=, src_entity_col=,
#                           dst_entity_table=, dst_entity_col=)
TASKS: dict = {
    ("amazon", "rating"): dict(
        entity_table="Customer",
        entity_col="customer_id",
        description=(
            "Predict the star rating a customer gives a product "
            "(`product_id` names the product; 4DBInfer scores this as regression/RMSE)."
        ),
    ),
    ("amazon", "purchase"): dict(
        src_entity_table="Customer",
        src_entity_col="customer_id",
        dst_entity_table="Product",
        dst_entity_col="product_id",
        description=(
            "Rank products a customer will purchase. 4DBInfer's MRR retrieval protocol: "
            "`train` holds positives only; `val`/`test` enumerate candidates with `label` "
            "and `query_idx`. Score with MRR over those candidates -- RelBench's default "
            "link metric (MAP) is a different protocol."
        ),
    ),
    ("amazon", "churn"): dict(
        entity_table="Customer",
        entity_col="customer_id",
        description="Predict whether a customer churns (stops reviewing/purchasing).",
    ),
    ("avs", "repeater"): dict(
        entity_table="Customer",
        entity_col="id",
        description=(
            "Predict whether a shopper becomes a repeat buyer of an offer "
            "(`offer`, `chain`, `market`, `offerdate` describe the offer instance). "
            "Source labels `'t'`/`'f'` are encoded 1/0."
        ),
    ),
    ("diginetica", "ctr"): dict(
        entity_table="Query",
        entity_col="queryId",
        description=(
            "Predict whether a displayed item is clicked (click-through rate); "
            "`itemId` names the displayed item."
        ),
    ),
    ("diginetica", "purchase"): dict(
        src_entity_table="Session",
        src_entity_col="sessionId",
        dst_entity_table="Product",
        dst_entity_col="itemId",
        description=(
            "Rank items a session will purchase. 4DBInfer's MRR retrieval protocol: "
            "`train` holds positives only; `val`/`test` enumerate candidates with `label` "
            "and `query_idx`. Score with MRR over those candidates -- RelBench's default "
            "link metric (MAP) is a different protocol."
        ),
    ),
    ("outbrain-small", "ctr"): dict(
        entity_table="Event",
        entity_col="display_id",
        description=(
            "Predict whether a promoted content item is clicked (click-through rate); "
            "`ad_id` names the promoted item."
        ),
    ),
    ("retailrocket", "cvr"): dict(
        entity_table="Visitor",
        entity_col="visitorid",
        description=(
            "Predict whether a viewed item is added to cart (conversion rate); "
            "`itemid` names the viewed item. A 100k-row subsample of `View`."
        ),
    ),
    ("seznam", "charge"): dict(
        entity_table="Client",
        entity_col="client_id",
        description=(
            "Predict which service (`sluzba`, 8 classes) a client's wallet top-up is for."
        ),
    ),
    ("seznam", "prepay"): dict(
        entity_table="Client",
        entity_col="client_id",
        description=(
            "Predict which service (`sluzba`, 8 classes) a client's wallet spend is on."
        ),
    ),
    ("stackexchange", "churn"): dict(
        entity_table="Users",
        entity_col="Id",
        description="Predict whether a user churns (stops participating).",
    ),
    ("stackexchange", "upvote"): dict(
        entity_table="Posts",
        entity_col="Id",
        description=(
            "Predict whether a post receives upvotes. The `Posts.Score` column the label "
            "is derived from is not part of the benchmark schema and is not published."
        ),
    ),
}

# 4DBInfer ``task_type`` -> RelBench ``task_type``. ``classification`` resolves by the
# number of distinct target values.
RETRIEVAL_TYPES = ("retrieval",)

SPLITS = {"train": "train", "validation": "val", "test": "test"}

# Rows per batch when streaming a payload that has no primary key (``avs/Transaction`` is
# 349,655,789 rows).
BATCH_ROWS = 1 << 21

SOURCE = dict(
    label="4DBInfer (NeurIPS 2024)",
    url=(
        "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
        "2fd67447702c8eff5683dda507a1b0a2-Abstract-Datasets_and_Benchmarks_Track.html"
    ),
    bibtex="""@inproceedings{wang2024fourdbinfer,
  title     = {{4DBInfer}: A {4D} Benchmarking Toolbox for Graph-Centric Predictive Modeling on Relational Databases},
  author    = {Wang, Minjie and Gan, Quan and Wipf, David and Cai, Zhenkun and Li, Ning and Tang, Jianheng and Zhang, Yanlin and Zhang, Zizhao and Mao, Zunyao and Song, Yakun and Wang, Yanbo and Li, Jiahang and Zhang, Han and Yang, Guang and Qin, Xiao and Lei, Chuan and Zhang, Muhan and Zhang, Weinan and Faloutsos, Christos and Zhang, Zheng},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024) Datasets and Benchmarks Track},
  year      = {2024}
}""",
)


# --------------------------------------------------------------------------- source access


def _payload(root: Path, source: str) -> Path:
    r"""Resolve a ``metadata.yaml`` ``source`` (a file *or* a partitioned directory)."""
    return root / source


def _dataset(path: Path) -> pds.Dataset:
    return pds.dataset(path, format="parquet")


def _declared(cols: list) -> list:
    return [c["name"] for c in cols]


def _project(path: Path, names: list) -> list:
    r"""Names to actually read: declared columns present in the payload."""
    have = set(_dataset(path).schema.names)
    return [n for n in names if n in have]


# ------------------------------------------------------------------------ dtype coercion


def _coerce(df: pd.DataFrame, specs: dict) -> pd.DataFrame:
    r"""Coerce columns to the dtype 4DBInfer declares for them."""
    for name, dtype in specs.items():
        if name not in df.columns:
            continue
        s = df[name]
        if dtype == "datetime":
            df[name] = pd.to_datetime(s, errors="coerce")
        elif dtype == "float":
            df[name] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif dtype in ("primary_key", "foreign_key"):
            # Keys are integers, except a few string domains (``outbrain-small``'s user
            # uuids, ``diginetica``'s name tokens). ``double`` columns are nullable ints.
            if pd.api.types.is_numeric_dtype(s):
                df[name] = pd.to_numeric(s, errors="coerce").astype("Int64")
            else:
                df[name] = s.astype("string")
        elif dtype == "text":
            df[name] = s.astype("string")
        # 'category' keeps its source representation (str or int).
    return df


# ------------------------------------------------------------------------------- key maps


class KeyMap:
    r"""``original key -> 0..n-1`` for one primary-key table, in that table's final row
    order (``Database.reindex_pkeys_and_fkeys`` semantics).

    Keys may be integers or
    strings (``outbrain-small`` links on user uuids, ``diginetica`` on name tokens).
    """

    def __init__(self, keys):
        s = pd.Series(np.asarray(keys) if not isinstance(keys, pd.Series) else keys)
        self._numeric = pd.api.types.is_numeric_dtype(s)
        if self._numeric:
            idx = pd.Index(pd.to_numeric(s).to_numpy(dtype=np.int64), dtype="int64")
        else:
            idx = pd.Index(s.astype("string").to_numpy(dtype=object), dtype=object)
        if idx.hasnans or idx.has_duplicates:
            raise RuntimeError("primary key has nulls or duplicates")
        self._idx = idx

    def __len__(self) -> int:
        return len(self._idx)

    def apply(self, values) -> pd.api.extensions.ExtensionArray:
        r"""Map ``values`` (original keys) to new indices; unresolvable or missing values
        become NA.

        The new index *is* the position in the parent's final row order.
        """
        s = pd.Series(values).reset_index(drop=True)
        n = len(s)
        if len(self._idx) == 0:
            return pd.arrays.IntegerArray(
                np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool)
            )
        if self._numeric:
            v = pd.to_numeric(s, errors="coerce")
            na = v.isna().to_numpy()
            probe = pd.Index(
                v.fillna(self._idx[0]).to_numpy(dtype=np.int64), dtype="int64"
            )
        else:
            v = s.astype("string")
            na = v.isna().to_numpy()
            probe = pd.Index(
                v.fillna(self._idx[0]).to_numpy(dtype=object), dtype=object
            )
        pos = self._idx.get_indexer(probe)
        miss = (pos < 0) | na
        return pd.arrays.IntegerArray(np.where(miss, 0, pos).astype(np.int64), miss)


# --------------------------------------------------------------------------------- build


def _fk_links(meta: dict) -> list:
    r"""Every ``(owner, column, target_table, target_column)`` foreign key in the
    dataset, across database tables *and* task tables."""
    out = []
    for t in meta["tables"]:
        for c in t["columns"]:
            if c["dtype"] == "foreign_key":
                tt, tc = c["link_to"].split(".")
                out.append((t["name"], c["name"], tt, tc))
    for tk in meta.get("tasks", []):
        for c in tk.get("columns", []):
            if c["dtype"] == "foreign_key":
                tt, tc = c["link_to"].split(".")
                out.append((f"task:{tk['name']}", c["name"], tt, tc))
    return out


def _observed(paths_cols: list) -> pd.Index:
    r"""Sorted unique of the values in a list of ``(payload_path, column)`` pairs.

    Integer
    domains (including ones stored as ``double``) come back as int64, string domains as
    string.
    """
    seen, numeric = [], True
    for path, col in paths_cols:
        d = _dataset(path)
        if col not in d.schema.names:
            continue
        numeric &= pa.types.is_integer(
            d.schema.field(col).type
        ) or pa.types.is_floating(d.schema.field(col).type)
        for batch in d.to_batches(columns=[col], batch_size=BATCH_ROWS):
            a = batch.column(0).drop_null()
            if len(a):
                seen.append(pc.unique(a).to_pandas())
    if not seen:
        return pd.Index([], dtype="int64")
    s = pd.concat(seen, ignore_index=True)
    if numeric:
        u = pd.to_numeric(s, errors="coerce").dropna().astype("int64").unique()
        return pd.Index(np.sort(u), dtype="int64")
    u = s.astype("string").dropna().unique().to_numpy(dtype=object)
    return pd.Index(np.sort(u), dtype=object)


def build(root: Path, name: str, out: Path) -> dict:
    r"""Port one extracted 4DBInfer dataset to the RelBench layout under ``out``."""
    meta = yaml.safe_load((root / "metadata.yaml").read_text())
    tables = {t["name"]: t for t in meta["tables"]}
    tasks = {tk["name"]: tk for tk in meta.get("tasks", [])}
    dropped = DROPPED_UNDECLARED.get(name, {})

    def task_split_path(tk: dict, split: str) -> Path:
        return _payload(root, tk["source"].replace("{split}", split))

    links = _fk_links(meta)
    implicit = sorted({(tt, tc) for _, _, tt, tc in links if tt not in tables})
    print(
        f"  implicit dimension tables: {[t for t, _ in implicit] or 'none'}", flush=True
    )

    # --- key maps -----------------------------------------------------------------------
    keymaps: dict = {}
    loaded: dict = {}  # fully-loaded primary-key tables, ready to write

    for tname, t in tables.items():
        pkey = next(
            (c["name"] for c in t["columns"] if c["dtype"] == "primary_key"), None
        )
        if pkey is None:
            continue
        path = _payload(root, t["source"])
        names = _project(path, _declared(t["columns"]))
        df = _dataset(path).to_table(columns=names).to_pandas()
        df = _coerce(df, {c["name"]: c["dtype"] for c in t["columns"]})
        time_col = t.get("time_column")
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
        keymaps[tname] = KeyMap(pd.Index(df[pkey]))
        df[pkey] = pd.array(np.arange(len(df)), dtype="Int64")
        loaded[tname] = (df, pkey, time_col)
        print(f"  pkey table {tname}: {len(df):,} rows", flush=True)

    for tt, tc in implicit:
        srcs = []
        for owner, col, t2, c2 in links:
            if (t2, c2) != (tt, tc):
                continue
            if owner.startswith("task:"):
                tk = tasks[owner[5:]]
                srcs += [
                    (task_split_path(tk, s), col)
                    for s in SPLITS
                    if task_split_path(tk, s).exists()
                ]
            else:
                srcs.append((_payload(root, tables[owner]["source"]), col))
        keys = _observed(srcs)
        keymaps[tt] = KeyMap(keys)
        loaded[tt] = (
            pd.DataFrame({tc: pd.array(np.arange(len(keys)), dtype="Int64")}),
            tc,
            None,
        )
        print(f"  synthesized {tt}({tc}): {len(keys):,} keys", flush=True)

    # --- database -----------------------------------------------------------------------
    (out / "db").mkdir(parents=True, exist_ok=True)
    specs: dict = {}

    def fkeys_of(t: dict) -> dict:
        return {
            c["name"]: c["link_to"].split(".")[0]
            for c in t["columns"]
            if c["dtype"] == "foreign_key"
        }

    for tname, (df, pkey, time_col) in loaded.items():
        t = tables.get(tname)
        fk = fkeys_of(t) if t else {}
        for col, target in fk.items():
            if col in df.columns:
                df[col] = keymaps[target].apply(df[col])
        df.to_parquet(out / "db" / f"{tname}.parquet", index=False)
        specs[tname] = TableSpec(pkey=pkey, time_col=time_col, fkeys=fk)

    for tname, t in tables.items():
        if tname in loaded:
            continue
        path = _payload(root, t["source"])
        dtypes = {c["name"]: c["dtype"] for c in t["columns"]}
        names = _project(path, _declared(t["columns"]))
        fk = fkeys_of(t)
        time_col = t.get("time_column")
        writer = None
        offset = 0
        for batch in _dataset(path).to_batches(columns=names, batch_size=BATCH_ROWS):
            df = batch.to_pandas()
            df = _coerce(df, dtypes)
            for col, target in fk.items():
                if col in df.columns:
                    df[col] = keymaps[target].apply(df[col])
            offset += len(df)
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out / "db" / f"{tname}.parquet", tbl.schema)
            writer.write_table(tbl)
        if writer is None:
            raise RuntimeError(f"table '{tname}': empty payload {path}")
        writer.close()
        specs[tname] = TableSpec(pkey=None, time_col=time_col, fkeys=fk)
        print(f"  streamed {tname}: {offset:,} rows", flush=True)

    # --- tasks --------------------------------------------------------------------------
    val_ts, test_ts = [], []
    manifests = []
    for tkname, tk in tasks.items():
        decl = {c["name"]: c["dtype"] for c in tk["columns"]}
        fk = {
            c["name"]: c["link_to"].split(".")[0]
            for c in tk["columns"]
            if c["dtype"] == "foreign_key"
        }
        extra = TASKS[(name, tkname)]
        pkey_cols = [c["name"] for c in tk["columns"] if c["dtype"] == "primary_key"]
        target = tk["target_column"]
        time_col = tk.get("time_column")
        tdir = out / "tasks" / tkname
        tdir.mkdir(parents=True, exist_ok=True)

        n_classes, wrote = None, {}
        for raw_split, split in SPLITS.items():
            path = task_split_path(tk, raw_split)
            if not path.exists():
                raise FileNotFoundError(f"task '{tkname}': missing {path}")
            keep = [
                n
                for n in _dataset(path).schema.names
                if n not in INDEX_ARTIFACTS  # keep undeclared label/query_idx
            ]
            df = _dataset(path).to_table(columns=keep).to_pandas()
            df = _coerce(df, decl)
            if time_col and time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            # Entity/foreign keys index the database after reindexing.
            for col, tgt in fk.items():
                if col in df.columns:
                    df[col] = keymaps[tgt].apply(df[col])
            for col in pkey_cols:
                if col in df.columns:
                    df[col] = keymaps[extra["entity_table"]].apply(df[col])
            # Binary targets shipped as 't'/'f' or bool -> int8 0/1.
            if target in df.columns:
                s = df[target]
                if s.dtype == bool:
                    df[target] = s.astype("int8")
                elif set(map(str, pd.unique(s.dropna()))) <= {"t", "f"}:
                    df[target] = (s.astype("string") == "t").astype("int8")
                n_classes = max(n_classes or 0, int(pd.unique(s.dropna()).size))
            # Canonical label order, applied at write time (loading does not sort).
            df = sort_labels(df, [time_col, *fk, *pkey_cols])
            df.to_parquet(tdir / f"{split}.parquet", index=False)
            wrote[split] = len(df)
            if time_col and time_col in df.columns and len(df):
                if split == "val":
                    val_ts.append(df[time_col].min())
                elif split == "test":
                    test_ts.append(df[time_col].min())

        if tk["task_type"] in RETRIEVAL_TYPES:
            task_type = "recommendation"
        elif tk["task_type"] == "regression":
            task_type = "regression"
        elif n_classes == 2:
            task_type = "binary_classification"
        else:
            task_type = "multiclass_classification"

        fields = dict(
            name=tkname,
            kind="external",
            task_type=task_type,
            description=extra["description"],
            time_col=time_col,
        )
        if task_type == "recommendation":
            fields.update(
                src_entity_table=extra["src_entity_table"],
                src_entity_col=extra["src_entity_col"],
                dst_entity_table=extra["dst_entity_table"],
                dst_entity_col=extra["dst_entity_col"],
            )
        else:
            fields.update(
                entity_table=extra["entity_table"],
                entity_col=extra["entity_col"],
                target_col=target,
            )
        # Label leakage: if the target is also a declared column of the table 4DBInfer
        # derived it from, the loader must drop it from the graph.
        origin = tk.get("target_table")
        if origin in tables and target in _declared(tables[origin]["columns"]):
            fields["remove_columns"] = [[origin, target]]
        tm = TaskManifest(**fields)
        tm.validate()
        tm.save(tdir / "manifest.yaml")
        manifests.append(tm)
        print(f"  task {tkname} ({task_type}): {wrote}", flush=True)

    # --- dataset manifest ---------------------------------------------------------------
    # Day granularity, as the core RelBench datasets use. Flooring can only move a cutoff
    # *earlier*, so "no val/test label precedes its cutoff" still holds and a database
    # trimmed at either cutoff is, if anything, more conservative. It also keeps
    # `test - val >= 1 day`, which `TaskBase` requires of the default `timedelta`
    # (`retailrocket`'s raw split points are only 22h07m apart).
    def _floor(ts_list, fallback):
        if not ts_list:
            return pd.Timestamp(fallback)
        return pd.Timestamp(min(ts_list)).floor("D")

    val, test = _floor(val_ts, "1970-01-01"), _floor(test_ts, "1970-01-02")
    if test - val < pd.Timedelta(days=1):
        raise RuntimeError(
            f"{name}: test_timestamp {test} is not at least a day after val_timestamp {val}"
        )
    val, test = str(val), str(test)

    dm = DatasetManifest(
        name=f"dbinfer-{name}",
        description=DATASETS[name]["description"],
        val_timestamp=val,
        test_timestamp=test,
        tables=specs,
    )
    dm.save(out / "manifest.yaml")
    print(f"  val_timestamp={val}  test_timestamp={test}", flush=True)

    # --- card + diagram -----------------------------------------------------------------
    from relbench.schema import render_schema_svg

    render_schema_svg(dm, out / "schema.svg", db_dir=out / "db")
    write_card(out, name, dm, manifests, implicit=[t for t in specs if t not in tables])

    return {"tables": len(specs), "tasks": len(manifests), "val": val, "test": test}


def write_card(out: Path, name: str, dm=None, manifests=None, implicit=None) -> None:
    r"""Write one dataset's ``README.md``.

    Called by :func:`build`, and re-runnable on an already-built directory (it reads the
    manifests back from disk when they are not passed) so the prose can be refreshed
    without regenerating 10 GiB of parquet.
    """
    from relbench.schema import dataset_card

    out = Path(out)
    if dm is None:
        dm = DatasetManifest.load(out / "manifest.yaml")
    if manifests is None:
        manifests = [
            TaskManifest.load(p)
            for p in sorted((out / "tasks").glob("*/manifest.yaml"))
        ]
    if implicit is None:
        # A materialized key domain is a single-column table whose one column is its pkey.
        implicit = [
            t
            for t, spec in dm.tables.items()
            if spec.pkey
            and not spec.fkeys
            and spec.time_col is None
            and len(pds.dataset(out / "db" / f"{t}.parquet").schema.names) == 1
        ]
    dropped = DROPPED_UNDECLARED.get(name, {})

    card = dataset_card(
        dm,
        tasks=manifests,
        repo=f"stanford-star/dbinfer/dbinfer-{name}",
        source=SOURCE,
    )
    note = ["", "## Port notes", ""]
    note += [
        "Built from the original 4DBInfer archive "
        f"(`{BASE_URL}/{VERSION}-{name}.tar`), keeping exactly the columns its "
        "`metadata.yaml` declares. Primary keys are reindexed to `0..n-1` and every "
        "foreign key -- in the database and in the task labels -- is remapped through the "
        "same mapping.",
        "",
    ]
    if implicit:
        note += [
            "4DBInfer links some foreign keys to key domains that have no payload table "
            "of their own; those are materialized here as single-column key tables: "
            + ", ".join(f"`{t}`" for t in sorted(implicit))
            + ".",
            "",
        ]
    if dropped:
        note += ["Undeclared payload columns dropped:", ""]
        for tname in sorted(dropped):
            note.append(f"* `{tname}`: " + ", ".join(f"`{c}`" for c in dropped[tname]))
        note.append("")
    leaky = [(tm.name, tm.remove_columns[0]) for tm in manifests if tm.remove_columns]
    if leaky:
        note += [
            "### Label columns in the database",
            "",
            "4DBInfer derives some labels from a column that is itself part of the "
            "database, so a model reading that row could read its own label:",
            "",
        ]
        note += [f"* `{tk}` -> `{tbl}.{col}`" for tk, (tbl, col) in leaky]
        note += [
            "",
            "Each such task declares `remove_columns`, and the column is left in `db/` so "
            "the database stays faithful to the source. `Dataset.get_db` drops the declared "
            "pairs, so `ds.load_task(...)` hands you a graph without them -- for "
            "every task `kind`, not just autocomplete. If you read the parquet directly, "
            "drop them yourself.",
            "",
        ]
    if name in UPSTREAM_DEFECTS:
        note += ["### Known upstream defects", "", UPSTREAM_DEFECTS[name], ""]
    card = card.replace("\n## Loading\n", "\n".join(note) + "\n## Loading\n", 1)
    (out / "README.md").write_text(card)


def main(dataset: str, out=None) -> None:
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dbinfer dataset '{dataset}'. Known: {sorted(DATASETS)}"
        )
    name = f"dbinfer-{dataset}"
    url = f"{BASE_URL}/{VERSION}-{dataset}.tar"
    sha = DATASETS[dataset]["sha256"] or None
    local = os.environ.get("DBINFER_RAW_ROOT")
    if local:
        root = Path(local) / dataset
        if not (root / "metadata.yaml").exists():
            raise FileNotFoundError(f"no metadata.yaml under {root}")
        print(f"using local extract {root}", flush=True)
    else:
        root = Path(fetch(url, sha)) / dataset
    out = Path(out or name)
    if out.exists():
        shutil.rmtree(out)
    info = build(root, dataset, out)
    print(f"wrote {out}: {json.dumps(info)}", flush=True)


REPO_CARD = """\
---
tags:
- relbench
- relational-deep-learning
pretty_name: RelBench dbinfer datasets
size_categories:
- 100M<n<1B
configs:
- config_name: databases
  data_files:
  - split: eval
    path: STATS/databases.parquet
- config_name: tasks
  data_files:
  - split: eval
    path: STATS/tasks.parquet
---

# RelBench dbinfer datasets

This repository hosts the **dbinfer** family of relational datasets in the RelBench 3.0
manifest format, one subdirectory per dataset. The datasets originate from the
[4DBInfer benchmark](https://github.com/awslabs/multi-table-benchmark) (data version
`{version}`), built directly from the original archives that `dbinfer_bench` itself
downloads (`{base}/{version}-<name>.tar`). Labels are the source's own, served as-is
(every task has `kind: external`).

Each subdirectory is a self-describing RelBench dataset (`manifest.yaml` + plain
`db/*.parquet` + `tasks/<task>/`); open its `schema.svg` for a zoomable
entity-relationship diagram.

## Datasets

{table}

Table counts include the single-column key tables materialized for the foreign-key targets
that 4DBInfer declares without a payload of their own (`Item`, `Visitor`, `Customer`,
`Chain`, `Brand`, `Category`, `Company`, `Session`, `User`, `Orders`, `Token`); see each
dataset card.

> **`dbinfer-outbrain-small` has almost no referential integrity, in the source.** 4DBInfer
> subsampled its tables independently, so ~99.9% of its foreign keys -- and all but 58 of
> 69,543 distinct train entities of its `ctr` task -- point at rows that were not kept. The
> previous revision hid this behind all-null keys. There is no full-size `outbrain` archive
> upstream. Its card has the numbers.

Several tasks derive their label from a column that is also in the database
(`retailrocket/cvr` <- `View.added_to_cart`, `seznam/charge` <- `Dobito.sluzba`,
`seznam/prepay` <- `Probehnuto.sluzba`, `outbrain-small/ctr` <- `Click.clicked`,
`amazon/rating` <- `Review.rating`). Each such task declares `remove_columns`; the column is
kept in `db/` so the database stays faithful to 4DBInfer, and `Dataset.get_db` drops it, so
`ds.load_task(...)` hands you a graph without it. If you read the parquet directly,
drop it yourself.

## Loading

```python
import relbench
ds = relbench.load_dataset("stanford-star/dbinfer/dbinfer-diginetica")
task = ds.load_task("ctr")
db = ds.get_db()
train = task.get_table("train")
```

## Provenance and revision history

Generated by
[`provenance/dbinfer.py`](https://github.com/snap-stanford/relbench/blob/main/provenance/dbinfer.py),
which pins the sha256 of each source archive; verified by
[`provenance/check_dbinfer.py`](https://github.com/snap-stanford/relbench/blob/main/provenance/check_dbinfer.py).
Port decisions -- which columns are kept, how implicit key domains are materialized, how
keys are reindexed -- are documented in that file's module docstring, and each dataset card
lists the payload columns it drops.

**This collection was rebuilt from the original 4DBInfer archives.** The previous revision
was derived from pre-built `db.zip` artifacts produced by the upstream
`dbinfer-relbench-adapter` export pipeline, which corrupted the data in ways that were not
visible from the schema:

* **Every declared foreign key was 99.9-100% null.** The adapter validated foreign keys
  against `len(parent_table)` where the parent was a `{{column: array}}` dict -- i.e.
  against the parent's *column count* -- so all larger key values were set to `NaN`. None
  of the seven databases could be joined.
* **Real primary keys were overwritten** with `np.arange(n)` before that validation, so the
  key correspondence was already gone.
* **Foreign keys to non-materialized key domains were dropped**, amputating the schema
  (`retailrocket/View` was left with no foreign keys at all).
* **Task entity ids were remapped through a task-local mapping** unrelated to the
  database's, so label rows did not reference the database.
* **Task time columns were destroyed** (`1970-01-01 00:00:00.000000022`) by casting the
  source's integer-valued columns with `astype('datetime64[ns]')`.
* **Classification targets were silently relabelled** by sorted-string order, and
  `task_type` was inferred from target cardinality -- turning `amazon/rating`, a 4DBInfer
  regression/RMSE task, into multiclass, and both retrieval tasks into multiclass.
* **Undeclared payload columns were published**, including `Posts.Score` (from which
  `stackexchange/upvote`'s label is derived), `History.repeater` (`avs/repeater`'s label
  itself), and full-history aggregates like `Users.Reputation/UpVotes/Views`.
* **`val_timestamp` / `test_timestamp` fell after the end of the data**, so
  `get_db(upto_test_timestamp=True)` trimmed nothing and gave no temporal protection at all
  (`dbinfer-retailrocket` claimed `2015-09-21` against a last event of `2015-09-18`;
  `dbinfer-diginetica` claimed `2016-11-12`; `dbinfer-seznam` `2015-10-04` against
  `2015-10-01`). They also did not bracket the source's own splits -- seznam's val labels
  start at `2015-04-01` and its test labels at `2015-07-01`.

The current revision fixes all of the above: foreign keys resolve, primary keys are dense,
implicit key domains are materialized as tables, task entity columns index their entity
table, time columns are real timestamps, targets keep their source values, `task_type`
follows the source, and each task declares `remove_columns` for any database column its
label is derived from. **Results computed against the previous revision are not
comparable.**

## Citation

These datasets are from the 4DBInfer benchmark. If you use them, please cite:

```bibtex
{bibtex}
```
"""


def write_repo_card(out_root: Path) -> None:
    r"""Emit the collection-level ``README.md`` (the Hub repo card)."""
    rows = ["| dataset | domain | tables | tasks |", "|---|---|---|---|"]
    for name, meta in DATASETS.items():
        d = out_root / f"dbinfer-{name}"
        man = DatasetManifest.load(d / "manifest.yaml")
        tasks = sorted(p.parent.name for p in d.glob("tasks/*/manifest.yaml"))
        rows.append(
            f"| [`dbinfer-{name}`](dbinfer-{name}) | {meta['domain']} | "
            f"{len(man.tables)} | " + ", ".join(f"`{t}`" for t in tasks) + " |"
        )
    (out_root / "README.md").write_text(
        REPO_CARD.format(
            version=VERSION,
            base=BASE_URL,
            table="\n".join(rows),
            bibtex=SOURCE["bibtex"].strip(),
        )
    )
    print(f"wrote {out_root}/README.md", flush=True)


# Descriptive STATS columns the generic overview builders leave blank for a human.
LICENSE = "see 4DBInfer / original sources"
SOURCE_URL = "https://github.com/awslabs/multi-table-benchmark"


def write_stats(out_root: Path) -> None:
    r"""Build ``STATS/{databases,tasks}.parquet`` for the collection (the tables behind
    the Hub dataset viewer), filling in the descriptive columns the generic builders
    leave for a human."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "byod"))
    import build_databases_overview as bdo
    import build_tasks_overview as bto

    stats = out_root / "STATS"
    stats.mkdir(parents=True, exist_ok=True)

    db_df = bdo._finalize(bdo.build(str(out_root)))
    db_df["domain"] = db_df.name.map(
        {f"dbinfer-{k}": v["domain"] for k, v in DATASETS.items()}
    ).astype("string")
    db_df["license"] = pd.Series(LICENSE, index=db_df.index, dtype="string")
    db_df["source_url"] = pd.Series(SOURCE_URL, index=db_df.index, dtype="string")
    db_df = db_df[[c for c in db_df.columns if db_df[c].notna().any()]]
    db_df.to_parquet(stats / "databases.parquet", index=False)
    print(f"wrote {stats}/databases.parquet ({len(db_df)} databases)", flush=True)

    # ``build`` already finalizes dtypes and drops the columns that never apply.
    tk_df = bto.build(str(out_root))
    tk_df.to_parquet(stats / "tasks.parquet", index=False)
    print(f"wrote {stats}/tasks.parquet ({len(tk_df)} tasks)", flush=True)


def main_all(out_root="dbinfer") -> None:
    out_root = Path(out_root)
    for dataset in DATASETS:
        print(f"=== {dataset} ===", flush=True)
        main(dataset, out_root / f"dbinfer-{dataset}")
    write_repo_card(out_root)
    write_stats(out_root)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        main_all(sys.argv[2] if len(sys.argv) > 2 else "dbinfer")
    elif len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(
            f"usage: python dbinfer.py DATASET [OUT_DIR] | --all [OUT_ROOT]\n"
            f"datasets: {', '.join(DATASETS)}"
        )
