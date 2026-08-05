from relbench import base
from relbench.load import load_dataset, train_std

__all__ = [
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
