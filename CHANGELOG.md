# Changelog

## 3.0.0 — 2026-08

RelBench 3 replaces the per-dataset Python classes and the download server with a
manifest-driven loader backed by the Hugging Face Hub. See `MIGRATION.md` for the API
changes from 2.x.

- `relbench.load_dataset("<name | org/repo[/subdir] | path>")` is the single entry point;
  tasks come from the dataset object. Datasets are self-describing folders (manifest +
  parquet + tasks), hosted at `stanford-star/relbench-v1` and `stanford-star/relbench-v2-extra`,
  with CTU/ReDeLEx, 4DBInfer and TGB collections in the same format.
- Leaderboard submissions are prediction tables validated locally
  (`python -m relbench.submit`) and published through a GitHub issue workflow.
- Masked recommendation test tables no longer expose the destination lists.
- Autocomplete splits include their boundary rows; tables without a primary key are kept
  sorted by time so positional keys are stable; hosted autocomplete labels regenerated.
- `rel-event/user-repeat`: the label SQL now runs over many timestamps at once, and the
  hosted train split was regenerated from it (3,842 → 2,972 rows; val/test unchanged).
  The old train split kept users whose last two *recorded* windows, however long ago,
  showed attendance rather than the previous two weeks the task defines.
- `rel-stack/post-votes` (train +117 rows, val +14) and `rel-stack/user-post-comment`
  (train +1) regenerated from their SQL; a handful of posts/users were missing from the
  hosted tables. Test splits unchanged.
- Task manifests validate their required fields; external tasks with an upstream
  evaluator (TGB, 4DBInfer) load as data only.
- `make_pkey_fkey_graph(..., remove_columns=task.hidden_columns())` lets one materialized
  cache per dataset serve every task; `Database.reindex_pkeys_and_fkeys` removed in favour
  of `byod/reindex_dataset.py`.
- Provenance generators for every RelBench database, with raw sources mirrored on the Hub.
- Test suite rewritten (fast, offline); CI on Python 3.10/3.13; PyPI publishing on tags.

Earlier releases: https://github.com/stanford-star/relbench/releases
