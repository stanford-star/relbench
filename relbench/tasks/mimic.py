import duckdb
import pandas as pd

from relbench.base import TaskType
from relbench.base.database import Database
from relbench.base.table import Table
from relbench.base.task_entity import EntityTask
from relbench.metrics import accuracy, average_precision, f1, roc_auc


class ICULengthOfStayTask(EntityTask):
    r"""Binary classification: Predict if ICU length of stay is ≥ 3 days during the
    first ICU stay."""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "subject_id"
    entity_table = "patients"
    time_col = "intime"
    target_col = "los_icu_binary"
    timedelta = pd.Timedelta(days=1)
    metrics = [average_precision, accuracy, f1, roc_auc]

    def __init__(self, *args, **kwargs):
        """Forward all arguments to the base class."""
        super().__init__(*args, **kwargs)

        self.test_timestamp = self.dataset.test_timestamp
        self.val_timestamp = self.dataset.val_timestamp

        if self.test_timestamp is not None and self.val_timestamp is not None:
            self.num_eval_timestamps = int(
                (self.test_timestamp - self.val_timestamp) / self.timedelta
            )
            print(f"num_eval_timestamps: {self.num_eval_timestamps}")
            self.num_eval_timestamps = max(1, self.num_eval_timestamps)
        print(f"num_eval_timestamps: {self.num_eval_timestamps}")

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        icu = db.table_dict["icustays"].df
        patients = db.table_dict["patients"].df

        df = duckdb.sql(f"""
            SELECT
                t.timestamp as intime,
                p.subject_id,
                CASE
                    WHEN i.los_icu >= 3 THEN 1
                    ELSE 0
                END as los_icu_binary
            FROM
                timestamp_df t
            LEFT JOIN icu i

                ON i.intime > t.timestamp
                AND i.intime <= t.timestamp + INTERVAL '{self.timedelta}'
            LEFT JOIN patients p
            ON
                i.subject_id = p.subject_id
            WHERE
                i.subject_id IS NOT NULL
                AND i.intime IS NOT NULL
                AND i.los_icu IS NOT NULL

        """).df()

        # Ensure time column is datetime64[ns] for to_unix_time
        if self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(
                df[self.time_col], errors="coerce"
            ).astype("datetime64[ns]")

        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
