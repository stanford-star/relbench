# Submitting to the RelBench leaderboard

The [RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard/) ranks
methods by their test-set performance, averaged over a fixed task set. There are three
independent boards — **classification** (12 tasks), **regression** (9), and
**recommendation** (10); the task lists are in `relbench.submit.LEADERBOARD_TASKS`.
You can submit to any of them; each requires predictions for *all* of its tasks.

## 1. Write one prediction CSV per task

Name each file `<dataset>__<task>.csv` and put them all in one directory. You can write
them with the helper below, or with your own code following the [CSV format](#prediction-csv-format).

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

[Open a submission issue](https://github.com/stanford-star/relbench/issues/new?template=submit.yml),
fill in the short form, and upload the zip file(s).

The submission is validated automatically and the report is posted on the issue; once a
maintainer approves, your entry appears on the leaderboard.
