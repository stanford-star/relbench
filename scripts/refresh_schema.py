r"""Render + push schema.svg (+ README.md) for RelBench datasets on the Hub.

There is no registry: name datasets by their Hub address.

    # one dataset (root- or sub-path)
    python scripts/refresh_schema.py relbench/rel-f1 --push
    python scripts/refresh_schema.py relbench/v1/rel-f1 --push

    # every sub-dataset in a family repo (one commit per repo)
    python scripts/refresh_schema.py relbench/v1 relbench/redelex --push

    # huge repos: clone first (git lfs) and render from local disk, then push
    python scripts/refresh_schema.py relbench/plurel-v1 --local /path/to/clone --push

Reads manifest + parquet footers from the Hub (paced, with 429 backoff) unless --local is
given. Commits once per repo (HF allows 128 commits/hour/repo). --no-readme pushes only
schema.svg. Resumable: skips sub-datasets already rendered locally.
"""
import sys, time, threading, concurrent.futures as cf
from pathlib import Path

import yaml
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.errors import HfHubHTTPError

from relbench.manifest import DatasetManifest, TaskManifest
from relbench.schema import _short_type, dataset_card, render_schema_svg

args = sys.argv[1:]
do_push = "--push" in args
no_readme = "--no-readme" in args
BATCH = int(args[args.index("--batch")+1]) if "--batch" in args else 400
local = Path(args[args.index("--local")+1]) if "--local" in args else None
specs = [a for a in args if not a.startswith("--") and a not in
         ({args[args.index("--batch")+1]} if "--batch" in args else set()) and
         (local is None or a != str(local))]

api = HfApi()
fs = HfFileSystem()
OUT_ROOT = Path("/lfs/local/0/ranjanr/relbench_schema_out")

# global pacing to stay under HF's resolver limit (~16/s); aim ~12/s
_lock = threading.Lock(); _next = [0.0]
def pace(interval=0.08):
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next[0] - now); _next[0] = max(now, _next[0]) + interval
    if wait:
        time.sleep(wait)

def _retry(fn, on_missing=None):
    for attempt in range(9):
        try:
            pace(); return fn()
        except FileNotFoundError:
            return on_missing
        except HfHubHTTPError as e:
            if "429" in str(e):
                time.sleep(30 + 10*attempt); continue
            return on_missing
        except Exception:
            if attempt == 8:
                return on_missing
            time.sleep(5)
    return on_missing


def list_subdirs(repo_id):
    files = api.list_repo_files(repo_id, repo_type="dataset")
    subs = sorted({f.split("/")[0] for f in files if f.endswith("/manifest.yaml") and f.count("/") == 1})
    if not subs and "manifest.yaml" in files:
        return [""], files
    return subs, files


def reader_for(repo_id, subdir):
    base = f"datasets/{repo_id}/{(subdir + '/') if subdir else ''}db"
    def reader(t):
        if local is not None:
            p = local / subdir / "db" / f"{t}.parquet"
            if not p.exists():
                return None, None
            return [(x.name, _short_type(x.type)) for x in pq.read_schema(p)], pq.read_metadata(p).num_rows
        def go():
            with fs.open(f"{base}/{t}.parquet", "rb") as f:
                pf = pq.ParquetFile(f)
                return [(x.name, _short_type(x.type)) for x in pf.schema_arrow], pf.metadata.num_rows
        return _retry(go, on_missing=(None, None))
    return reader


def load_manifest(repo_id, subdir, rel):
    if local is not None:
        p = local / subdir / rel
        return yaml.safe_load(p.read_text()) if p.exists() else None
    path = f"datasets/{repo_id}/{(subdir + '/') if subdir else ''}{rel}"
    return _retry(lambda: yaml.safe_load(fs.open(path, "rb").read()))


# Original-dataset citations by repo (datasets not listed here are RelBench-native: the card
# cites RelBench only). TGB carries both the TGB and TGB 2.0 papers.
_CTU = """@article{motl2015ctu,
  title   = {The {CTU} Prague Relational Learning Repository},
  author  = {Motl, Jan and Schulte, Oliver},
  journal = {arXiv preprint arXiv:1511.03086},
  year    = {2015}
}"""
_DBINFER = """@inproceedings{wang2024fourdbinfer,
  title     = {{4DBInfer}: A {4D} Benchmarking Toolbox for Graph-Centric Predictive Modeling on Relational Databases},
  author    = {Wang, Minjie and Gan, Quan and Wipf, David and Cai, Zhenkun and Li, Ning and Tang, Jianheng and Zhang, Yanlin and Zhang, Zizhao and Mao, Zunyao and Song, Yakun and Wang, Yanbo and Li, Jiahang and Zhang, Han and Yang, Guang and Qin, Xiao and Lei, Chuan and Zhang, Muhan and Zhang, Weinan and Faloutsos, Christos and Zhang, Zheng},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024) Datasets and Benchmarks Track},
  year      = {2024}
}"""
_TGB = """@inproceedings{huang2023temporal,
  title     = {Temporal Graph Benchmark for Machine Learning on Temporal Graphs},
  author    = {Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Rabbany, Reihaneh},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track},
  year      = {2023}
}

@inproceedings{gastinger2024tgb2,
  title     = {{TGB 2.0}: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs},
  author    = {Gastinger, Julia and Huang, Shenyang and Galkin, Mikhail and Loghmani, Erfan and Parviz, Ali and Poursafaei, Farimah and Danovitch, Jacob and Rossi, Emanuele and Koutis, Ioannis and Stuckenschmidt, Heiner and Rabbany, Reihaneh and Rabusseau, Guillaume},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024) Datasets and Benchmarks Track},
  year      = {2024}
}"""
SOURCES = {
    "relbench/redelex": {"label": "CTU Prague Relational Learning Repository",
                         "url": "https://relational.fel.cvut.cz/", "bibtex": _CTU},
    "relbench/dbinfer": {"label": "4DBInfer (NeurIPS 2024)",
                         "url": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/2fd67447702c8eff5683dda507a1b0a2-Abstract-Datasets_and_Benchmarks_Track.html",
                         "bibtex": _DBINFER},
    "relbench/tgb": {"label": "Temporal Graph Benchmark (TGB / TGB 2.0)",
                     "url": "https://tgb.complexdatalab.com/", "bibtex": _TGB},
}


def render_one(repo_id, subdir, task_rels, out):
    odir = out / subdir if subdir else out
    if (odir / "schema.svg").exists() and (no_readme or (odir / "README.md").exists()):
        return (subdir, None)
    try:
        manifest = DatasetManifest.from_dict(load_manifest(repo_id, subdir, "manifest.yaml"))
        odir.mkdir(parents=True, exist_ok=True)
        render_schema_svg(manifest, odir / "schema.svg", reader=reader_for(repo_id, subdir))
        if not no_readme:
            tasks = []
            for tr in sorted(task_rels):
                d = load_manifest(repo_id, subdir, tr)
                if d:
                    tasks.append(TaskManifest.from_dict(d))
            addr = f"{repo_id}/{subdir}" if subdir else repo_id
            (odir / "README.md").write_text(
                dataset_card(manifest, tasks, repo=addr, source=SOURCES.get(repo_id)))
        return (subdir, None)
    except Exception as exc:
        return (subdir, repr(exc)[:140])


def task_rels_for(subdir, files):
    out = []
    for f in files:
        p = f.split("/")
        if subdir:
            if len(p) >= 4 and p[0] == subdir and p[1] == "tasks" and f.endswith("manifest.yaml"):
                out.append("/".join(p[1:]))
        else:
            if len(p) >= 3 and p[0] == "tasks" and f.endswith("manifest.yaml"):
                out.append(f)
    return out


# group specs by repo (a spec may be a whole repo or a single org/repo/subdir)
repos = {}
for spec in specs:
    parts = spec.strip("/").split("/")
    repo_id = "/".join(parts[:2])
    sub = "/".join(parts[2:])
    repos.setdefault(repo_id, set())
    if sub:
        repos[repo_id].add(sub)

for repo_id, only in repos.items():
    if local is not None:
        all_subs = sorted(p.name for p in local.iterdir() if (p / "manifest.yaml").exists()) or [""]
        files = []
    else:
        all_subs, files = list_subdirs(repo_id)
    subdirs = sorted(only) if only else all_subs
    print(f"{repo_id}: {len(subdirs)} sub-datasets (readme={not no_readme}, local={local is not None})", flush=True)
    out = OUT_ROOT / repo_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    trs = {s: (task_rels_for(s, files) if not no_readme else []) for s in subdirs}

    ok, fail = [], []
    with cf.ThreadPoolExecutor(max_workers=(8 if local is not None else 4)) as ex:
        futs = {ex.submit(render_one, repo_id, s, trs[s], out): s for s in subdirs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            s, err = fut.result()
            (fail if err else ok).append(s)
            if err:
                print(f"  FAIL {s}: {err}", flush=True)
            if i % 200 == 0:
                print(f"  rendered {i}/{len(subdirs)}", flush=True)
    print(f"  rendered ok={len(ok)} fail={len(fail)}", flush=True)

    if do_push and ok:
        per = ["schema.svg"] + ([] if no_readme else ["README.md"])
        for i in range(0, len(ok), BATCH):
            batch = ok[i:i+BATCH]
            patterns = [f"{(s + '/') if s else ''}{fn}" for s in batch for fn in per]
            for attempt in range(6):
                try:
                    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="dataset",
                                      allow_patterns=patterns,
                                      commit_message=f"Refresh ER schema diagrams ({i+1}-{i+len(batch)} of {len(ok)})")
                    break
                except HfHubHTTPError as e:
                    if "429" in str(e):
                        print("  429 on upload; sleeping 60s", flush=True); time.sleep(60)
                    else:
                        raise
            print(f"  pushed {i+1}-{i+len(batch)}", flush=True)
        print(f"[{repo_id}] done", flush=True)
