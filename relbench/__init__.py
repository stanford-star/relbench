from relbench import base
from relbench.load import get_task_names, load_dataset, load_task

__all__ = ["base", "modeling", "load_dataset", "load_task", "get_task_names"]


def __getattr__(name):
    # `relbench.modeling` pulls in torch / PyTorch Geometric, which data-only users (and the
    # in-browser WebAssembly tutorials) don't need. Import it lazily on first access so plain
    # `import relbench` + load_dataset/load_task stays torch-free.
    if name == "modeling":
        import importlib

        return importlib.import_module("relbench.modeling")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
