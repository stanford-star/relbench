import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # RelBench quickstart

        Load a RelBench dataset and task straight from the Hugging Face Hub, explore the
        relational schema, and train a trivial baseline — all with **no per-dataset code**.

        Read the rendered version on the website, or run it yourself:
        `pip install relbench[tutorial]` then `marimo edit quickstart.py`.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    import relbench

    return np, pd, relbench


@app.cell
def _(mo):
    mo.md("## Load the dataset")
    return


@app.cell
def _(relbench):
    dataset = relbench.load_dataset("rel-f1")
    db = dataset.get_db()
    return (db,)


@app.cell
def _(mo):
    mo.md(
        "The dataset is a set of tables linked by a foreign-key graph — all described by "
        "its `manifest.yaml`:"
    )
    return


@app.cell
def _(db, pd):
    pd.DataFrame(
        [
            {
                "table": name,
                "rows": len(table),
                "pkey": table.pkey_col,
                "time_col": table.time_col,
                "fkeys": ", ".join(table.fkey_col_to_pkey_table) or "—",
            }
            for name, table in db.table_dict.items()
        ]
    )
    return


@app.cell
def _(db):
    db.table_dict["results"].df.head()
    return


@app.cell
def _(mo, relbench):
    mo.md(f"## Tasks\n\nAvailable tasks: `{relbench.get_task_names('rel-f1')}`")
    return


@app.cell
def _(relbench):
    task = relbench.load_task("rel-f1", "driver-position")
    train_df = task.get_table("train").df
    return task, train_df


@app.cell
def _(train_df):
    train_df.head()
    return


@app.cell
def _(mo):
    mo.md(
        "## A trivial baseline\n\n"
        "Predict the training-set mean for every test entity, and evaluate with the "
        "task's own metrics:"
    )
    return


@app.cell
def _(np, task, train_df):
    test_table = task.get_table("test", mask_input_cols=False)
    pred = np.full(len(test_table.df), train_df[task.target_col].mean())
    metrics = task.evaluate(pred, test_table)
    metrics
    return


if __name__ == "__main__":
    app.run()
