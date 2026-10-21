import os
import re, json, argparse
import numpy as np, pandas as pd
from pathlib import Path

REPORTS = Path("/mnt/1857e392-c689-44a8-9c3e-e7f9beab82f6/results/reports")
ROOT_500 = Path("/mnt/f0771a75-815c-4553-ba74-9aa70f57f995/500-seeds")
# run_clustering.sh exports DOCKQ_500/DOCKQ_SINGLE so shell and python steps stay in sync
DOCKQ_500 = Path(os.environ.get("DOCKQ_500") or REPORTS / "dockq_500seeds_75comp_all5_interfaces_20261019.tsv")
DOCKQ_SINGLE = Path(os.environ.get("DOCKQ_SINGLE") or REPORTS / "dockq_singleseed_75comp_all5_interfaces_20261019.tsv")
REF_TABLE = REPORTS / "anarci_residues_ref.tsv"
CLUSTER_DIR = Path("/mnt/f0771a75-815c-4553-ba74-9aa70f57f995/clustering")
SEED_RE = re.compile(r"seed-(\d+)_sample-(\d+)")


def heavy_chains(pdb_id, ref_table=REF_TABLE):
    df = pd.read_csv(ref_table, sep="\t", dtype=str, keep_default_na=False)
    return set(df[(df.pdb_id == pdb_id) & (df.chain_type == "H")].chain)


def dockq_per_model(pdb_id, dockq_tsv=DOCKQ_500, ref_table=REF_TABLE):
    """One DockQ per (seed, sample): max over heavy-chain/antigen interfaces."""
    H = heavy_chains(pdb_id, ref_table)
    assert H, f"no heavy chain found in reference table for {pdb_id}"
    df = pd.read_csv(dockq_tsv, sep="\t")
    df = df[df.pdb_id == pdb_id].copy()
    df["native_set"] = df.native_chains.str.split(r"\s*,\s*")
    df = df[df.native_set.apply(lambda s: bool(H & set(s)))]
    ss = df.model_name.str.extract(SEED_RE.pattern).astype(int)
    df["seed"], df["sample"] = ss[0], ss[1]
    return (df.groupby(["seed", "sample"]).dockq_score.max()
              .rename("dockq").reset_index())


def confidences(pdb_id, seeds, samples, root=ROOT_500):
    rows = []
    for s, k in zip(seeds, samples):
        f = Path(root) / pdb_id / f"seed-{s}_sample-{k}" / \
            f"{pdb_id}_seed-{s}_sample-{k}_summary_confidences.json"
        j = json.load(open(f))
        rows.append({"seed": s, "sample": k, "ranking_score": j["ranking_score"],
                     "iptm": j["iptm"], "ptm": j["ptm"]})
    return pd.DataFrame(rows)


def model_table(pdb_id, root=ROOT_500, dockq_tsv=DOCKQ_500, ref_table=REF_TABLE):
    """All models of a tree (no cluster labels): seed, sample, scores, dockq."""
    pairs = []
    for d in (Path(root) / pdb_id).glob("seed-*_sample-*"):
        m = SEED_RE.fullmatch(d.name)
        if m:
            pairs.append((int(m[1]), int(m[2])))
    pairs.sort()
    seeds, samples = zip(*pairs)
    df = confidences(pdb_id, seeds, samples, root)
    df = df.merge(dockq_per_model(pdb_id, dockq_tsv, ref_table),
                  on=["seed", "sample"], how="left")
    print(f"{pdb_id}: {len(df)} models, no heavy-AG interface "
          f"(DockQ set to 0): {df.dockq.isna().sum()}")
    df["dockq"] = df.dockq.fillna(0.0)
    return df


def evaluate(pdb_id, tag="", root=ROOT_500, dockq_tsv=DOCKQ_500,
             ref_table=REF_TABLE, cluster_dir=CLUSTER_DIR, out_dir=None):
    out_dir = Path(out_dir or cluster_dir)
    lab = pd.read_csv(Path(cluster_dir) / f"{pdb_id}_labels{tag}.tsv", sep="\t")
    df = lab.merge(confidences(pdb_id, lab.seed, lab["sample"], root),
                   on=["seed", "sample"])
    df = df.merge(dockq_per_model(pdb_id, dockq_tsv, ref_table),
                  on=["seed", "sample"], how="left")
    print(f"models: {len(df)}, no heavy-AG interface (DockQ set to 0): {df.dockq.isna().sum()}")
    df["dockq"] = df.dockq.fillna(0.0)
    g = df.groupby("cluster")
    summ = pd.DataFrame({
        "n_models": g.size(), "n_seeds": g.seed.nunique(),
        "dockq_mean": g.dockq.mean(), "dockq_max": g.dockq.max(),
        "dockq_min": g.dockq.min(), "dockq_sd": g.dockq.std(),
        "frac_acc": g.dockq.apply(lambda x: (x >= 0.23).mean()),
        "frac_med": g.dockq.apply(lambda x: (x >= 0.49).mean()),
        "rank_mean": g.ranking_score.mean(), "rank_sd": g.ranking_score.std(),
        "iptm_mean": g.iptm.mean(),
    }).sort_values("n_models", ascending=False)
    print(summ.round(3).to_string())
    top = df.loc[df.ranking_score.idxmax()]
    print(f"\ntop-ranked model: cluster {int(top.cluster)}, DockQ {top.dockq:.3f}, "
          f"ranking_score {top.ranking_score:.3f}")
    best = df.loc[df.dockq.idxmax()]
    print(f"best-DockQ model: cluster {int(best.cluster)}, DockQ {best.dockq:.3f}")
    df.to_csv(out_dir / f"{pdb_id}_models_eval{tag}.tsv", sep="\t", index=False)
    summ.to_csv(out_dir / f"{pdb_id}_clusters_eval{tag}.tsv", sep="\t")
    return df, summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_id", nargs="?", default="8jow")
    ap.add_argument("--tag", default="")
    ap.add_argument("--root", default=str(ROOT_500))
    ap.add_argument("--dockq", default=str(DOCKQ_500))
    ap.add_argument("--ref", default=str(REF_TABLE))
    ap.add_argument("--cluster-dir", default=str(CLUSTER_DIR))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    evaluate(a.pdb_id, a.tag, a.root, a.dockq, a.ref, a.cluster_dir, a.out)
