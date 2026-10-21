import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import dockq_conf as dq

BASE = Path("/mnt/f0771a75-815c-4553-ba74-9aa70f57f995")
TREES = {
    "multi":  {"root": BASE / "500-seeds", "dockq": dq.DOCKQ_500},
    "single": {"root": BASE / "single-seed", "dockq": dq.DOCKQ_SINGLE},
}
DATA_DIR = dq.REPORTS                    # cache of per-model tables


def _plddt(args):
    root, pdb, s, k = args
    f = Path(root) / pdb / f"seed-{s}_sample-{k}" / f"{pdb}_seed-{s}_sample-{k}_confidences.json"
    with open(f) as fh:
        return float(np.mean(json.load(fh)["atom_plddts"]))


def collect(pdb, tree, workers=8, force=False):
    out = DATA_DIR / f"models_{tree}_{pdb}.tsv"
    if out.exists() and not force:
        return pd.read_csv(out, sep="\t")
    cfg = TREES[tree]
    df = dq.model_table(pdb, root=cfg["root"], dockq_tsv=cfg["dockq"])
    jobs = [(cfg["root"], pdb, int(s), int(k)) for s, k in zip(df.seed, df["sample"])]
    with ProcessPoolExecutor(workers) as ex:
        df["plddt_mean"] = list(ex.map(_plddt, jobs, chunksize=20))
    df.insert(0, "pdb_id", pdb)
    df.insert(0, "tree", tree)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    return df


def list_complexes():
    """Complexes present in BOTH trees."""
    ids = [{p.name for p in TREES[t]["root"].iterdir() if p.is_dir()} for t in TREES]
    return sorted(set.intersection(*ids))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_ids", nargs="*", help="default: all complexes in both trees")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    bad = []
    for pdb in (a.pdb_ids or list_complexes()):
        for tree in TREES:
            try:
                d = collect(pdb, tree, a.workers, a.force)
                print(f"[ok] {pdb} {tree}: {len(d)} models, "
                      f"DockQ mean {d.dockq.mean():.3f}, pLDDT mean {d.plddt_mean.mean():.1f}")
            except Exception as e:      # one bad complex must not stop the batch
                bad.append(f"{pdb} {tree}")
                print(f"[FAIL] {pdb} {tree}: {e!r}")
    if bad:
        print(f"[FAIL] {len(bad)} collections: {', '.join(bad)}")
        sys.exit(1)
