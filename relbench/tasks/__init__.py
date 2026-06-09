import json
import os
import pkgutil
from collections import defaultdict
from functools import lru_cache
from typing import List

import pooch

from relbench.base import AutoCompleteTask, BaseTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import (
    amazon,
    arxiv,
    avito,
    dbinfer,
    event,
    f1,
    hm,
    ratebeer,
    stack,
    tgb,
    trial,
)
from relbench.utils import get_relbench_cache_dir

task_registry = defaultdict(dict)

hashes_str = pkgutil.get_data(__name__, "hashes.json")
hashes = json.loads(hashes_str)

DOWNLOAD_REGISTRY = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/download/",
    registry=hashes,
    env="RELBENCH_CACHE_DIR",
)


def register_task(
    dataset_name: str,
    task_name: str,
    cls: BaseTask,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`BaseTask` subclass with the given name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        cls: The class of the task.
        args: The arguments to instantiate the task.
        kwargs: The keyword arguments to instantiate the task.

    The name is used to enable caching and downloading functionalities.
    `cache_dir` is added to kwargs by default. If you want to override it, you
    can pass `cache_dir` as a keyword argument in `kwargs`.
    """

    cache_dir = f"{get_relbench_cache_dir()}/{dataset_name}/tasks/{task_name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    task_registry[dataset_name][task_name] = (cls, args, kwargs)


def get_task_names(dataset_name: str) -> List[str]:
    r"""Return a list of names of the registered tasks for the given dataset."""
    return list(task_registry[dataset_name].keys())


def download_task(dataset_name: str, task_name: str) -> None:
    r"""Download task from RelBench server into its cache directory.

    The downloaded task tables will be automatically picked up by the task object, when
    `task.get_table(split)` is called.
    """

    DOWNLOAD_REGISTRY.fetch(
        f"{dataset_name}/tasks/{task_name}.zip",
        processor=pooch.Unzip(extract_dir="."),
        progressbar=True,
    )


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str, download=False) -> BaseTask:
    r"""Return a task object by name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        download: If True, download the task from the RelBench server.

    Returns:
        BaseTask: The task object.

    If `download` is True, the task tables (train, val, test) comprising the
    task will be downloaded into the cache from the RelBench server. If you use
    `download=False` the first time, the task tables will be computed from
    scratch using the database.

    Once the task tables are cached, either because of download or computing from
    scratch, the cache will be used. `download=True` will verify that the
    cached task tables matches the RelBench version even in this case.
    """

    from relbench import hf

    if hf.is_registered(dataset_name):
        # Migrated datasets are served by the new manifest/HF loader.
        from relbench.load import load_task as _load_task

        return _load_task(dataset_name, task_name)

    if download:
        download_task(dataset_name, task_name)
    dataset = get_dataset(dataset_name, download=download)
    cls, args, kwargs = task_registry[dataset_name][task_name]
    task = cls(dataset, *args, **kwargs)
    return task


register_task("rel-amazon", "user-churn", amazon.UserChurnTask)
register_task("rel-amazon", "user-ltv", amazon.UserLTVTask)
register_task("rel-amazon", "item-churn", amazon.ItemChurnTask)
register_task("rel-amazon", "item-ltv", amazon.ItemLTVTask)
register_task("rel-amazon", "user-item-purchase", amazon.UserItemPurchaseTask)
register_task("rel-amazon", "user-item-rate", amazon.UserItemRateTask)
register_task("rel-amazon", "user-item-review", amazon.UserItemReviewTask)
register_task(
    "rel-amazon",
    "review-rating",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="review",
    target_col="rating",
    remove_columns=[
        ("review", "review_text"),
        ("review", "summary"),
    ],
)

# Experimental task: review-verified
# register_task(
#     "rel-amazon",
#     "review-verified",
#     AutoCompleteTask,
#     task_type=TaskType.BINARY_CLASSIFICATION,
#     entity_table="review",
#     target_col="verified",
#     remove_columns=[],
# )

register_task("rel-avito", "ad-ctr", avito.AdCTRTask)
register_task("rel-avito", "user-visits", avito.UserVisitsTask)
register_task("rel-avito", "user-clicks", avito.UserClicksTask)
register_task("rel-avito", "user-ad-visit", avito.UserAdVisitTask)
register_task(
    "rel-avito",
    "searchstream-click",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="SearchStream",
    target_col="IsClick",
)
register_task(
    "rel-avito",
    "searchinfo-isuserloggedon",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="SearchInfo",
    target_col="IsUserLoggedOn",
)

register_task("rel-event", "user-attendance", event.UserAttendanceTask)
register_task("rel-event", "user-repeat", event.UserRepeatTask)
register_task("rel-event", "user-ignore", event.UserIgnoreTask)
register_task(
    "rel-event",
    "event_interest-interested",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="event_interest",
    target_col="interested",
    remove_columns=[
        ("event_interest", "not_interested"),
    ],
)

register_task(
    "rel-event",
    "event_interest-not_interested",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="event_interest",
    target_col="not_interested",
    remove_columns=[
        ("event_interest", "interested"),
    ],
)

register_task(
    "rel-event",
    "users-birthyear",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="users",
    target_col="birthyear",
    remove_columns=[],
)

register_task("rel-f1", "driver-position", f1.DriverPositionTask)
register_task("rel-f1", "driver-dnf", f1.DriverDNFTask)
register_task("rel-f1", "driver-top3", f1.DriverTop3Task)
register_task("rel-f1", "driver-circuit-compete", f1.DriverCircuitCompeteTask)
register_task(
    "rel-f1",
    "results-position",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="results",
    target_col="position",
    remove_columns=[
        ("results", "statusId"),
        ("results", "positionOrder"),
        ("results", "points"),
        ("results", "laps"),
        ("results", "milliseconds"),
        ("results", "fastestLap"),
        ("results", "rank"),
    ],
)
register_task(
    "rel-f1",
    "qualifying-position",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="qualifying",
    target_col="position",
    remove_columns=[],
)

# Experimental task: constructor_results-points
# register_task(
#     "rel-f1",
#     "constructor_results-points",
#     AutoCompleteTask,
#     task_type=TaskType.REGRESSION,
#     entity_table="constructor_results",
#     target_col="points",
#     remove_columns=[],
# )

# Experimental task: constructor_standings-position
# register_task(
#     "rel-f1",
#     "constructor_standings-position",
#     AutoCompleteTask,
#     task_type=TaskType.REGRESSION,
#     entity_table="constructor_standings",
#     target_col="position",
#     remove_columns=[
#         ("constructor_standings", "wins"),
#         ("constructor_standings", "points"),
#     ],
# )

register_task("rel-hm", "user-item-purchase", hm.UserItemPurchaseTask)
register_task("rel-hm", "user-churn", hm.UserChurnTask)
register_task("rel-hm", "item-sales", hm.ItemSalesTask)

register_task(
    "rel-hm",
    "transactions-price",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="transactions",
    target_col="price",
    remove_columns=[],
)

register_task("rel-stack", "user-engagement", stack.UserEngagementTask)
register_task("rel-stack", "post-votes", stack.PostVotesTask)
register_task("rel-stack", "user-badge", stack.UserBadgeTask)
register_task("rel-stack", "user-post-comment", stack.UserPostCommentTask)
register_task("rel-stack", "post-post-related", stack.PostPostRelatedTask)
register_task(
    "rel-stack",
    "badges-class",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="badges",
    target_col="Class",
    remove_columns=[("badges", "TagBased"), ("badges", "Name")],
)

# Experimental task: postlinks-linktypeid
# register_task(
#     "rel-stack",
#     "postlinks-linktypeid",
#     AutoCompleteTask,
#     task_type=TaskType.BINARY_CLASSIFICATION,
#     entity_table="postLinks",
#     target_col="LinkTypeId",
#     remove_columns=[],
# )


register_task("rel-trial", "study-outcome", trial.StudyOutcomeTask)
register_task("rel-trial", "study-adverse", trial.StudyAdverseTask)
register_task("rel-trial", "site-success", trial.SiteSuccessTask)
register_task("rel-trial", "condition-sponsor-run", trial.ConditionSponsorRunTask)
register_task("rel-trial", "site-sponsor-run", trial.SiteSponsorRunTask)

register_task(
    "rel-trial",
    "studies-enrollment",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="studies",
    target_col="enrollment",
    remove_columns=[],
)

register_task(
    "rel-trial",
    "studies-has_dmc",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="studies",
    target_col="has_dmc",
    remove_columns=[],
)

register_task(
    "rel-trial",
    "eligibilities-adult",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="eligibilities",
    target_col="adult",
    remove_columns=[
        ("eligibilities", "child"),
        ("eligibilities", "older_adult"),
        ("eligibilities", "minimum_age"),
        ("eligibilities", "maximum_age"),
        ("eligibilities", "population"),
        ("eligibilities", "criteria"),
        ("eligibilities", "gender_description"),
    ],
)

register_task(
    "rel-trial",
    "eligibilities-child",
    AutoCompleteTask,
    task_type=TaskType.BINARY_CLASSIFICATION,
    entity_table="eligibilities",
    target_col="child",
    remove_columns=[
        ("eligibilities", "adult"),
        ("eligibilities", "older_adult"),
        ("eligibilities", "minimum_age"),
        ("eligibilities", "maximum_age"),
        ("eligibilities", "population"),
        ("eligibilities", "criteria"),
        ("eligibilities", "gender_description"),
    ],
)


register_task("rel-arxiv", "paper-citation", arxiv.PaperCitationTask)
register_task("rel-arxiv", "author-category", arxiv.AuthorCategoryTask)
register_task("rel-arxiv", "author-publication", arxiv.AuthorPublicationTask)
register_task("rel-arxiv", "paper-paper-cocitation", arxiv.CoCitationTask)

register_task(
    "rel-salt",
    "item-plant",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocumentitem",
    target_col="PLANT",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "item-shippoint",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocumentitem",
    target_col="SHIPPINGPOINT",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "item-incoterms",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocumentitem",
    target_col="ITEMINCOTERMSCLASSIFICATION",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
    ],
)
register_task(
    "rel-salt",
    "sales-office",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="SALESOFFICE",
    remove_columns=[
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "sales-group",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="SALESGROUP",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "sales-payterms",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="CUSTOMERPAYMENTTERMS",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "sales-shipcond",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="SHIPPINGCONDITION",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "HEADERINCOTERMSCLASSIFICATION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)
register_task(
    "rel-salt",
    "sales-incoterms",
    AutoCompleteTask,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    entity_table="salesdocument",
    target_col="HEADERINCOTERMSCLASSIFICATION",
    remove_columns=[
        ("salesdocument", "SALESOFFICE"),
        ("salesdocument", "SALESGROUP"),
        ("salesdocument", "CUSTOMERPAYMENTTERMS"),
        ("salesdocument", "SHIPPINGCONDITION"),
        ("salesdocumentitem", "PLANT"),
        ("salesdocumentitem", "SHIPPINGPOINT"),
        ("salesdocumentitem", "ITEMINCOTERMSCLASSIFICATION"),
    ],
)

register_task("rel-ratebeer", "beer-churn", ratebeer.BeerRatingChurnTask)
register_task("rel-ratebeer", "user-churn", ratebeer.UserRatingChurnTask)
register_task("rel-ratebeer", "brewer-dormant", ratebeer.BrewerDormantTask)
register_task("rel-ratebeer", "user-count", ratebeer.UserRatingCountTask)
register_task("rel-ratebeer", "user-beer-liked", ratebeer.UserLikedBeerTask)
register_task("rel-ratebeer", "user-place-liked", ratebeer.UserLikedPlaceTask)
register_task("rel-ratebeer", "user-beer-favorite", ratebeer.UserFavoriteBeerTask)

register_task(
    "rel-ratebeer",
    "beer_ratings-total_score",
    AutoCompleteTask,
    task_type=TaskType.REGRESSION,
    entity_table="beer_ratings",
    target_col="total_score",
    remove_columns=[
        ("beer_ratings", "aroma"),
        ("beer_ratings", "flavor"),
        ("beer_ratings", "mouthfeel"),
        ("beer_ratings", "appearance"),
        ("beer_ratings", "overall"),
        ("beer_ratings", "comments"),
        ("beer_ratings", "description_score"),
    ],
)
register_task("dbinfer-avs", "repeater", dbinfer.AVSRepeaterTask)
register_task("dbinfer-mag", "cite", dbinfer.MAGCiteTask)
register_task("dbinfer-mag", "venue", dbinfer.MAGVenueTask)
register_task("dbinfer-diginetica", "ctr", dbinfer.DigineticaCTRTask)
register_task("dbinfer-diginetica", "purchase", dbinfer.DigineticaPurchaseTask)
register_task("dbinfer-retailrocket", "cvr", dbinfer.RetailRocketCVRTask)
register_task("dbinfer-seznam", "charge", dbinfer.SeznamChargeTask)
register_task("dbinfer-seznam", "prepay", dbinfer.SeznamPrepayTask)
register_task("dbinfer-amazon", "rating", dbinfer.AmazonRatingTask)
register_task("dbinfer-amazon", "purchase", dbinfer.AmazonPurchaseTask)
register_task("dbinfer-amazon", "churn", dbinfer.AmazonChurnTask)
register_task("dbinfer-stackexchange", "churn", dbinfer.StackExchangeChurnTask)
register_task("dbinfer-stackexchange", "upvote", dbinfer.StackExchangeUpvoteTask)
register_task("dbinfer-outbrain-small", "ctr", dbinfer.OutbrainCTRTask)

# Temporal Graph Benchmark (TGB)
for dataset_name in [
    "tgbl-wiki",
    "tgbl-wiki-v2",
    "tgbl-review",
    "tgbl-review-v2",
    "tgbl-coin",
    "tgbl-comment",
    "tgbl-flight",
]:
    register_task(
        dataset_name,
        "src-dst-mrr",
        tgb.TGBOneVsManyLinkPredTask,
        spec=tgb.TGBLinkPredSpec(event_table="events"),
        k_value=10,
    )


def _register_thgl_edge_type_tasks(dataset_name: str, edge_types: list[int]) -> None:
    for et in edge_types:
        register_task(
            dataset_name,
            f"edge-type-{int(et)}-mrr",
            tgb.TGBOneVsManyLinkPredTask,
            spec=tgb.TGBLinkPredSpec(event_table=f"events_edge_type_{int(et)}"),
            k_value=10,
        )


_register_thgl_edge_type_tasks("thgl-software", list(range(14)))
_register_thgl_edge_type_tasks("thgl-github", list(range(14)))
_register_thgl_edge_type_tasks("thgl-forum", [0, 1])
_register_thgl_edge_type_tasks("thgl-myket", [0, 1])

for dataset_name in [
    "tgbn-trade",
    "tgbn-genre",
    "tgbn-reddit",
    "tgbn-token",
]:
    register_task(
        dataset_name,
        "node-label-ndcg",
        tgb.TGBNodePropNDCGTask,
        spec=tgb.TGBNodePropSpec(),
        k=10,
    )
