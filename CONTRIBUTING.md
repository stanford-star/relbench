# Contributing

## Development setup

The repository is a [pixi](https://pixi.sh) workspace:

```bash
pixi install                 # CPU torch + PyTorch Geometric + PyTorch Frame + dev tools
pixi run pytest test         # the whole suite (~10 s)
pixi run pytest test --ignore=test/modeling   # the torch-free part (~4 s)
pixi run pre-commit run --all-files           # black, isort, docformatter, whitespace
```

`pixi run -e gpu ...` selects CUDA builds for the examples. Without pixi,
`pip install -e ".[example,dev]"` plus `pyg-lib` from https://data.pyg.org/whl/ works too.

## Tests

`test/` runs against a small synthetic dataset (`test/conftest.py`) and never touches the
network. Every behaviour a user relies on has a test: split windows and masking, hosted
labels equal regenerated labels, the leaderboard evaluator, manifest validation, the graph
builder and loaders. Add one next to the code you change; keep tests fast (a single batch,
never a training loop).

## Style

Formatting is enforced by pre-commit (black, isort, docformatter) and checked by CI on
Python 3.10 and 3.13 (Linux and Windows) for the torch-free part, and on Linux with torch
for everything. Keep the data layer (`relbench.load`, `relbench.base`, `relbench.submit`)
free of torch imports.

## Adding data

Datasets and tasks are not added to this repository: publish a manifest + parquet folder
to the Hugging Face Hub and load it by its `org/repo[/subdir]` address. `byod/README.md` is
the walkthrough and `byod/reindex_dataset.py` normalizes keys. Open an issue or PR to have
your dataset listed.

## Leaderboard

Submissions go through a GitHub issue created from the "Submit to the RelBench
leaderboard" template, validated by CI; see the Leaderboard section of the README.

## Releases

Bump `version` in `pyproject.toml`, add a `CHANGELOG.md` entry, and push a `v<version>`
tag; `.github/workflows/publish.yml` builds and publishes to PyPI.
