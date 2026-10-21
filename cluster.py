import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from parser import OUT_DIR


def run_hdbscan(D, min_cluster_size=15, min_samples=5, epsilon=0.0, method="eom"):
    """D: (n,n) RMSD matrix. Returns labels (-1 = noise) and membership probabilities."""
    h = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                cluster_selection_epsilon=epsilon, cluster_selection_method=method,
                metric="precomputed")
    h.fit(D.astype(np.float64))
    return h.labels_, h.probabilities_


def summarize(D, labels, seeds):
    """One row per cluster: size, distinct seeds, medoid index, mean intra-cluster RMSD."""
    rows = []
    for c in sorted(set(labels) - {-1}):
        idx = np.where(labels == c)[0]
        sub = D[np.ix_(idx, idx)]
        rows.append({"cluster": c, "n_models": len(idx),
                     "n_seeds": len(np.unique(seeds[idx])),
                     "medoid": int(idx[sub.mean(1).argmin()]),
                     "mean_rmsd": float(sub.sum() / (len(idx) * max(len(idx) - 1, 1)))})
    df = pd.DataFrame(rows)
    return df.sort_values("n_models", ascending=False).reset_index(drop=True) if len(df) else df


def grid(D, seeds, sizes=(10, 15, 25), min_samples=5, epsilon=0.0, method="eom"):
    """Robustness grid over min_cluster_size. Prints clusters, noise fraction, ARI vs. the first run."""
    results = {s: run_hdbscan(D, s, min_samples, epsilon, method) for s in sizes}
    base = results[sizes[0]][0]
    for s, (lab, _) in results.items():
        print(f"min_cluster_size={s:>3}: {len(set(lab) - {-1}):>3} clusters, "
              f"noise {np.mean(lab == -1):.1%}, ARI vs {sizes[0]}: {adjusted_rand_score(base, lab):.3f}")
    return results


def cluster_complex(pdb_id, run="multi", out_prefix=None, min_cluster_size=15, min_samples=5, epsilon=0.0, method="eom"):
    prefix = out_prefix or pdb_id
    D = np.load(OUT_DIR / f"{prefix}_matrixA.npy")
    npz = np.load(OUT_DIR / f"{pdb_id}_ca_{run}.npz")
    seeds, samples = npz["seed"], npz["sample"]

    grid(D, seeds, min_samples=min_samples, epsilon=epsilon, method=method)

    labels, prob = run_hdbscan(D, min_cluster_size, min_samples, epsilon, method)
    summ = summarize(D, labels, seeds)
    print(f"\nmain run (min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
          f"eps={epsilon}, {method}):")
    print(summ.to_string(index=False) if len(summ) else "no clusters found")

    tag = "" if method == "eom" else f"_{method}"
    pd.DataFrame({"model": np.arange(len(labels)), "seed": seeds, "sample": samples,
                  "cluster": labels, "prob": prob}).to_csv(OUT_DIR / f"{pdb_id}_labels{tag}.tsv", sep="\t", index=False)
    summ.to_csv(OUT_DIR / f"{pdb_id}_clusters{tag}.tsv", sep="\t", index=False)
    return labels, prob, summ


if __name__ == "__main__":
    cluster_complex("8jow")
