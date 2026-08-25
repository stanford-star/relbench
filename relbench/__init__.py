from importlib.metadata import PackageNotFoundError, version

from relbench import base
from relbench.load import load_dataset, train_std

try:
    __version__ = version("relbench")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "base",
    "modeling",
    "load_dataset",
    "train_std",
]


def __getattr__(name):
    # `relbench.modeling` pulls in torch / PyTorch Geometric, which data-only users (and the
    # in-browser WebAssembly tutorials) don't need. Import it lazily on first access so plain
    # `import relbench` + load_dataset/get_db stays torch-free.
    if name == "modeling":
        import importlib

        return importlib.import_module("relbench.modeling")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
