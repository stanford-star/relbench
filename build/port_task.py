r"""Regenerate + validate ONE task in isolation (subprocess: a duckdb segfault/OOM only
fails this task). Writes the regenerated labels into the artifact, and exits 0 iff they
match the canonical hosted labels.

    python build/port_task.py <artifact_dir> <dataset> <task>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import _keys_target, _match, fetch_canonical_labels  # noqa: E402

from relbench.load import load_task  # noqa: E402

out, ds, tname = sys.argv[1], sys.argv[2], sys.argv[3]
task = load_task(out, tname, regenerate=True)
keys, target, is_list = _keys_target(task)
hosted = fetch_canonical_labels(ds, tname)
tdir = Path(out) / "tasks" / tname

all_ok = True
for split in ["train", "val", "test"]:
    df = task.get_table(split, mask_input_cols=False).df
    ok = split in hosted and _match(df, hosted[split], keys, target, is_list)
    all_ok &= ok
    print(f"    {split:<5} regen={len(df):>8} hosted={len(hosted.get(split, [])):>8} "
          f"{'ok' if ok else 'MISMATCH'}", flush=True)
    df.to_parquet(tdir / f"{split}.parquet", index=False)

sys.exit(0 if all_ok else 1)
