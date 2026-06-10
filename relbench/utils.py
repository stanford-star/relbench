import os


def get_relbench_cache_dir() -> str:
    r"""Directory for RelBench's local cache.

    Honors ``$RELBENCH_CACHE_DIR``; otherwise ``~/.cache/relbench``. (Hub downloads use
    the standard Hugging Face cache; this is for any auxiliary local artifacts.)
    """
    return os.getenv("RELBENCH_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "relbench"
    )
