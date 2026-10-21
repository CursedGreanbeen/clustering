import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist

from matrix import kabsch, apply
from parser import OUT_DIR
from knn_assign import knn_vote, loo_thresholds, K
from collect import DATA_DIR, list_complexes

OUT_DIR = Path(OUT_DIR)
METRICS = ["dockq", "ranking_score", "iptm", "ptm", "plddt_mean"]


def resolve(pdb):
    """(labels, cluster-summary, D, n). Prefers versioned _<n> artifacts if present,
    else the plain names that cluster.py/matrix.py currently write; n = n_models."""
    vers = [int(p.stem.rsplit("_", 1)[1]) for p in OUT_DIR.glob(f"{pdb}_labels_*.tsv")
            if p.stem.rsplit("_", 1)[1].isdigit()]
    if vers:
        n = max(vers)
        lab = pd.read_csv(OUT_DIR / f"{pdb}_labels_{n}.tsv", sep="\t")
        summ = pd.read_csv(OUT_DIR / f"{pdb}_clusters_{n}.tsv", sep="\t").set_index("cluster")
        D = np.load(OUT_DIR / f"{pdb}_matrixA_{n}.npy")
    else:
        lab = pd.read_csv(OUT_DIR / f"{pdb}_labels.tsv", sep="\t")
        summ = pd.read_csv(OUT_DIR / f"{pdb}_clusters.tsv", sep="\t").set_index("cluster")
        D = np.load(OUT_DIR / f"{pdb}_matrixA.npy")
    assert len(lab) == len(D), f"{pdb}: labels/matrix size mismatch"
    return lab, summ, D, len(D)


def single_distances(pdb):
    """(5, n) antigen RMSD of each single-seed model to every multi-seed model,
    same frame as knn_assign.assign_single."""
    fr = np.load(OUT_DIR / f"{pdb}_frame.npz")
    single = np.load(OUT_DIR / f"{pdb}_ca_single.npz")
    fit = single["fit"].astype(np.float64)
    ag = single["ag"].astype(np.float64)[:, fr["keep"]]
    R, cP, cQ = kabsch(fit, fr["mean_ref"])
    ag_s = apply(ag, R, cP, cQ)
    N, m = fr["ag_fit"].shape[:2]
    d = cdist(ag_s.reshape(len(ag_s), -1), fr["ag_fit"].reshape(N, -1)) / np.sqrt(m)
    return d, single["seed"], single["sample"]


def cluster_block(sub):
    return {"n_models": int(len(sub)), "n_seeds": int(sub.seed.nunique()),
            "dockq_mean": sub.dockq.mean(), "dockq_max": sub.dockq.max(),
            "frac_acc": (sub.dockq >= 0.23).mean(), "frac_med": (sub.dockq >= 0.49).mean(),
            **{f"{m}_mean": sub[m].mean() for m in METRICS[1:]}}


def analyse(pdb, k=K):
    (DATA_DIR / f"stats_{pdb}.json").unlink(missing_ok=True)  # stale stats must not survive a failed run
    lab, summ, D, n = resolve(pdb)
    multi = pd.read_csv(DATA_DIR / f"models_multi_{pdb}.tsv", sep="\t")
    single = pd.read_csv(DATA_DIR / f"models_single_{pdb}.tsv", sep="\t")

    m = lab.merge(multi, on=["seed", "sample"], how="left", validate="1:1")
    assert m.dockq.notna().all(), f"{pdb}: models without scores"
    labels, seeds = lab.cluster.values, lab.seed.values
    cl = m[m.cluster != -1]
    if cl.empty:
        raise ValueError(f"{pdb}: no clusters")

    # per-cluster table
    ctab = pd.DataFrame({c: cluster_block(g) for c, g in cl.groupby("cluster")}).T
    ctab.index.name = "cluster"
    ctab["medoid"] = summ.loc[ctab.index, "medoid"].astype(int)
    ctab["mean_rmsd"] = summ.loc[ctab.index, "mean_rmsd"]
    best = {"dockq": int(ctab.dockq_mean.idxmax()),
            "rank": int(ctab.ranking_score_mean.idxmax()),
            "plddt": int(ctab["plddt_mean_mean"].idxmax())}

    # single-seed vs multi-seed frame
    d, ss, sk = single_distances(pdb)
    s = pd.DataFrame({"seed": ss, "sample": sk}).merge(single, on=["seed", "sample"], how="left", validate="1:1")
    assert s.dockq.notna().all(), f"{pdb}: single-seed models without scores"
    thr = loo_thresholds(D, labels, seeds, k)
    rows = []
    for i in range(len(s)):
        c, med = knn_vote(d[i], labels, k)
        rows.append({"nearest_cluster": c, "median_knn": med, "threshold": thr[c],
                     "cluster": c if med <= thr[c] else -1, "min_dist": d[i].min()})
    s = pd.concat([s, pd.DataFrame(rows)], axis=1)
    for key, c in best.items():
        s[f"dist_medoid_best_{key}"] = d[:, int(ctab.loc[c, "medoid"])]
        s[f"dist_ratio_best_{key}"] = s[f"dist_medoid_best_{key}"] / thr[c]

    # numbers for the report
    total_seeds = int(multi.seed.nunique())
    sg = {"n_models": int(len(s)), **{f"{x}_mean": s[x].mean() for x in METRICS},
          "frac_acc": (s.dockq >= 0.23).mean(), "frac_med": (s.dockq >= 0.49).mean(),
          "dockq_max": s.dockq.max()}
    stats = {"pdb": pdb, "n_multi": int(n), "total_seeds": total_seeds,
             "n_clusters": int(len(ctab)), "noise_frac": float((labels == -1).mean()),
             "single": sg, "best": {}}
    top = m.loc[m.ranking_score.idxmax()]
    stats["multi_top_ranked"] = {"cluster": int(top.cluster), "dockq": float(top.dockq)}
    stats["single_top_ranked"] = {"dockq": float(s.loc[s.ranking_score.idxmax(), "dockq"])}
    stats["rank_best_is_dockq_best"] = best["rank"] == best["dockq"]
    for key, c in best.items():
        b = {k_: (float(v) if isinstance(v, (float, np.floating)) else int(v))
             for k_, v in ctab.loc[c].items()}
        b.update({"cluster": c, "seed_support": b["n_seeds"] / total_seeds,
                  "threshold": float(thr[c]),
                  "delta_dockq_vs_single": b["dockq_mean"] - sg["dockq_mean"],
                  "delta_rank_vs_single": b["ranking_score_mean"] - sg["ranking_score_mean"],
                  "delta_plddt_vs_single": b["plddt_mean_mean"] - sg["plddt_mean_mean"],
                  "single_dist_median": float(s[f"dist_medoid_best_{key}"].median()),
                  "single_dist_max": float(s[f"dist_medoid_best_{key}"].max()),
                  "single_assigned_to_best": int((s.cluster == c).sum()),
                  "single_assigned_other": int(((s.cluster != c) & (s.cluster != -1)).sum()),
                  "single_rejected": int((s.cluster == -1).sum())})
        stats["best"][key] = b

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / f"single_dist_{pdb}.npy", d.astype(np.float32))
    s.to_csv(DATA_DIR / f"single_{pdb}.tsv", sep="\t", index=False)
    m.to_csv(DATA_DIR / f"multi_{pdb}.tsv", sep="\t", index=False)
    ctab.to_csv(DATA_DIR / f"clusters_{pdb}.tsv", sep="\t")
    tmp = DATA_DIR / f"stats_{pdb}.json.tmp"
    tmp.write_text(json.dumps(stats, indent=1, default=float))
    os.replace(tmp, DATA_DIR / f"stats_{pdb}.json")      # atomic: never a half-written file
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_ids", nargs="*")
    a = ap.parse_args()
    flat, bad = [], []
    for pdb in (a.pdb_ids or list_complexes()):
        try:
            st = analyse(pdb)
            row = {"pdb": pdb, "n_multi": st["n_multi"], "n_clusters": st["n_clusters"],
                   "noise_frac": st["noise_frac"], "single_dockq": st["single"]["dockq_mean"],
                   "rank_best_is_dockq_best": st["rank_best_is_dockq_best"]}
            for key, b in st["best"].items():
                row.update({f"{key}_{x}": b[x] for x in ["cluster", "n_models", "n_seeds", "dockq_mean",
                            "delta_dockq_vs_single", "single_dist_median", "single_assigned_to_best"]})
            flat.append(row)
            print(f"[ok] {pdb}: best-DockQ cluster {st['best']['dockq']['cluster']} "
                  f"DockQ {st['best']['dockq']['dockq_mean']:.3f} vs single {st['single']['dockq_mean']:.3f}")
        except Exception as e:
            bad.append(pdb)
            print(f"[FAIL] {pdb}: {e!r}")
    if flat:
        out = DATA_DIR / "summary.tsv"
        df = pd.DataFrame(flat)
        if out.exists():                       # keep rows of complexes not in this run
            old = pd.read_csv(out, sep="\t")
            df = pd.concat([df, old[~old.pdb.isin(df.pdb)]], ignore_index=True)
        df.sort_values("pdb").to_csv(out, sep="\t", index=False)
    if bad:
        print(f"[FAIL] {len(bad)} complexes: {', '.join(bad)}")
        sys.exit(1)
