import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from matrix import kabsch, apply          # needs the __main__ guard, see below
from parser import OUT_DIR

K = 5
PCT = 95

def knn_vote(d, labels, k=K):
    """d: (n,) distances to the multiseed models. Noise is ignored.
    Returns (cluster, median distance to the k nearest non-noise models)."""
    d = np.where(labels == -1, np.inf, d)
    nn = np.argpartition(d, k)[:k]
    cl, cnt = np.unique(labels[nn], return_counts=True)
    return cl[cnt.argmax()], float(np.median(d[nn]))

def loo_thresholds(D, labels, seeds, k=K, pct=PCT):
    """Per cluster: pct-th percentile of the leave-one-out kNN median distance of its members.
    Same-seed models are excluded as neighbours, because the 5 samples of a seed are correlated."""
    Dm = D.astype(np.float64).copy()
    Dm[seeds[:, None] == seeds[None, :]] = np.inf      # also removes self
    Dm[:, labels == -1] = np.inf
    med = np.median(np.partition(Dm, k, axis=1)[:, :k], axis=1)
    return {c: np.percentile(med[labels == c], pct) for c in np.unique(labels) if c != -1}

def assign_single(pdb_id, tag="", k=K):
    fr = np.load(OUT_DIR / f"{pdb_id}_frame.npz")
    D = np.load(OUT_DIR / f"{pdb_id}_matrixA.npy")
    lab = pd.read_csv(OUT_DIR / f"{pdb_id}_labels{tag}.tsv", sep="\t")
    labels, seeds = lab.cluster.values, lab.seed.values
    single = np.load(OUT_DIR / f"{pdb_id}_ca_single.npz")

    # same frame as the 2500 models
    fit = single["fit"].astype(np.float64)
    ag = single["ag"].astype(np.float64)[:, fr["keep"]]
    R, cP, cQ = kabsch(fit, fr["mean_ref"])
    ag_s = apply(ag, R, cP, cQ)

    n, m = fr["ag_fit"].shape[:2]
    d = cdist(ag_s.reshape(len(ag_s), -1), fr["ag_fit"].reshape(n, -1)) / np.sqrt(m)

    thr = loo_thresholds(D, labels, seeds, k)
    summ = pd.read_csv(OUT_DIR / f"{pdb_id}_clusters{tag}.tsv", sep="\t").set_index("cluster")
    rows = []
    for i in range(len(ag_s)):
        c, med = knn_vote(d[i], labels, k)
        ok = med <= thr[c]
        rows.append({"seed": single["seed"][i], "sample": single["sample"][i],
                     "cluster": c if ok else -1, "nearest_cluster": c,
                     "median_knn": round(med, 3), "threshold": round(thr[c], 3),
                     "cluster_n_seeds": int(summ.loc[c, "n_seeds"]),
                     "min_dist": round(float(d[i].min()), 3)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(OUT_DIR / f"{pdb_id}_single_assign{tag}.tsv", sep="\t", index=False)
    return out

if __name__ == "__main__":
    assign_single("8jow")
