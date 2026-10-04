# Submitting to the RelBench leaderboard

The [RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard/) ranks
methods by their test-set performance, averaged over a fixed task set. There are three
independent boards — **classification** (12 tasks), **regression** (9), and
**recommendation** (10); the task lists are in `relbench.submit.LEADERBOARD_TASKS`.
You can submit to any of them; each requires predictions for *all* of its tasks.

## 1. Write one prediction CSV per task

Name each file `<dataset>__<task>.csv` and put them all in one directory. You can write
them with the helper below, or with your own code following the [prediction CSV format](#prediction-csv-format).

```python
relbench.submit.write_prediction_table(task, test_pred, "preds/rel-f1__driver-position.csv")
```

`test_pred` is aligned to the rows of `task.get_table("test", mask_input_cols=True)`: a
1-D array for entity tasks, or an `(N, eval_k)` array of destination ids for
recommendation tasks.

### Prediction CSV format

Each CSV is the task's test table with the target replaced by your predictions. Its
columns are the key columns followed by one prediction column. The column names are those
of the task, e.g. `task.entity_col`.

| Task type | Key columns | Prediction column |
|---|---|---|
| binary classification | `entity_col`, `time_col` | `target_col`: probability in `[0, 1]` |
| regression | `entity_col`, `time_col` | `target_col`: value on the original target scale |
| recommendation | `src_entity_col`, `time_col` | `dst_entity_col`: JSON list of the top-`eval_k` destination ids |

There must be exactly one row per test row: no missing rows, no extras, no duplicate keys.

## 2. Validate and package

```bash
python -m relbench.submit preds/
```

This scores every CSV against the test tables, prints a verdict per leaderboard, and
writes a submission zip for each leaderboard whose tasks are all present and valid.

## 3. Open a submission issue

### Automatically (e.g. from an AI agent)

```bash
python -m relbench.submit preds/ --submit \
  --name "My method" --url https://github.com/me/my-method --in-context no
```

After validating and packaging, this uploads the zips to the public dataset repo
`<your-hf-user>/relbench-submissions` on the Hugging Face Hub and opens the submission
issue with links pinned to that commit. It needs a Hugging Face login (`hf auth login` or
`HF_TOKEN`) and a GitHub token (`gh auth login`, `GH_TOKEN`, or `GITHUB_TOKEN`).
`--in-context yes` means the method did *not* train on the target database. Optionally,
`--repro-link` links to code or instructions to reproduce the results, and `--note` adds
a note shown on hover.

### Manually

[Open a submission issue](https://github.com/stanford-star/relbench/issues/new?template=submit.yml),
fill in the short form, and upload the zip file(s).

The submission is validated automatically and the report is posted on the issue; once a
maintainer approves, your entry appears on the leaderboard.

## Editing an entry

To change the name, URL, repro link, or note of a published entry (e.g. to add a repro
link), run

```bash
python -m relbench.submit --edit 393 --repro-link https://github.com/me/my-method
```

where `393` is the number of the submission issue that published the entry. This needs a
GitHub token, as for `--submit`. Alternatively,
[open an edit issue](https://github.com/stanford-star/relbench/issues/new?template=edit.yml)
and fill in only the fields to change. Edits by the entry's author are applied
automatically; others are reviewed by a maintainer. Scores cannot be edited — resubmit
instead.
