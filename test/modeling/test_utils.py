import numpy as np
import pandas as pd

import relbench.modeling.utils as modeling_utils
from relbench.base import Database, Table
from relbench.modeling.utils import get_stype_proposal


def _large_db(num_rows=5_000):
    # More rows than the 1,000 that get_stype_proposal samples, so the sample matters.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": np.arange(num_rows),
            "code": rng.integers(0, 50, num_rows),
            "amount": rng.normal(size=num_rows),
        }
    )
    return Database({"t": Table(df=df, fkey_col_to_pkey_table={}, pkey_col="id")})


def test_get_stype_proposal_samples_the_same_rows(monkeypatch):
    sampled = []
    infer = modeling_utils.infer_df_stype

    def recording_infer(df):
        sampled.append(tuple(df.index))
        return infer(df)

    monkeypatch.setattr(modeling_utils, "infer_df_stype", recording_infer)
    db = _large_db()

    first = get_stype_proposal(db)
    second = get_stype_proposal(db)

    assert len(sampled[0]) == 1_000
    assert sampled[0] == sampled[1]
    assert first == second
