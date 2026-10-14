import re
import json
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/mnt/f0771a75-815c-4553-ba74-9aa70f57f995/500-seeds")   # 500-seed tree
REPORTS = Path("/mnt/1857e392-c689-44a8-9c3e-e7f9beab82f6/results/reports")
DOCKQ = str(REPORTS / "dockq_500seeds_all5_interfaces.tsv")
REF_TABLE = str(REPORTS / "anarci_residues_ref.tsv")
CLUSTER_DIR = Path("/mnt/f0771a75-815c-4553-ba74-9aa70f57f995/clustering")
SEED_RE = re.compile(r"seed-(\d+)_sample-(\d+)")


def heavy_chains(pdb_id):
    df = pd.read_csv(REF_TABLE, sep="\t", dtype=str, keep_default_na=False)
    return set(df[(df.pdb_id == pdb_id) & (df.chain_type == "H")].chain)


def dockq_per_model(pdb_id):
    """One DockQ per (seed, sample): max over heavy-chain/antigen interfaces, 0 if none."""
    H = heavy_chains(pdb_id)
    assert H, f"no heavy chain found in reference table for {pdb_id}"
    df = pd.read_csv(DOCKQ, sep="\t")
    df = df[df.pdb_id == pdb_id].copy()
    df["native_set"] = df.native_chains.str.split(r"\s*,\s*")
    df = df[df.native_set.apply(lambda s: bool(H & set(s)))]
    ss = df.model_name.str.extract(SEED_RE.pattern).astype(int)
    df["seed"], df["sample"] = ss[0], ss[1]
    return (df.groupby(["seed", "sample"]).dockq_score.max()
              .rename("dockq").reset_index())


def confidences(pdb_id, seeds, samples):
    rows = []
    for s, k in zip(seeds, samples):
        f = ROOT / pdb_id / f"seed-{s}_sample-{k}" / f"{pdb_id}_seed-{s}_sample-{k}_summary_confidences.json"
        j = json.load(open(f))
        rows.append({"seed": s, "sample": k, "ranking_score": j["ranking_score"],
                     "iptm": j["iptm"], "ptm": j["ptm"]})
    return pd.DataFrame(rows)


def evaluate(pdb_id, tag=""):
    lab = pd.read_csv(CLUSTER_DIR / f"{pdb_id}_labels{tag}.tsv", sep="\t")
    df = lab.merge(confidences(pdb_id, lab.seed, lab["sample"]), on=["seed", "sample"])
    df = df.merge(dockq_per_model(pdb_id), on=["seed", "sample"], how="left")
    n_missing = df.dockq.isna().sum()
    print(f"models: {len(df)}, no heavy-AG interface (DockQ set to 0): {n_missing}")
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

    df.to_csv(CLUSTER_DIR / f"{pdb_id}_models_eval{tag}.tsv", sep="\t", index=False)
    summ.to_csv(CLUSTER_DIR / f"{pdb_id}_clusters_eval{tag}.tsv", sep="\t")
    return df, summ


if __name__ == "__main__":
    evaluate("8jow")
