"""Plots for the 8jow clustering outputs.

Usage (run from this directory):
    /opt/conda/bin/python3.10 plots.py [pdb_id]     # default 8jow

Reads <pdb>_matrixA.npy / _labels.tsv / _models_eval.tsv from OUT_DIR and writes PNGs
next to them (never into this scripts dir):
    <pdb>_mds.png        classical MDS of the RMSD matrix, colored by cluster
    <pdb>_heatmap.png    pairwise RMSD matrix, rows/cols ordered by cluster
    <pdb>_dockq.png      DockQ per cluster (box) + ranking_score vs DockQ (scatter)
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from parser import OUT_DIR

NOISE = "#999999"

def colors(labels):
    cmap = plt.get_cmap("tab10")
    cl = sorted(set(labels) - {-1})
    m = {c: cmap(i % 10) for i, c in enumerate(cl)}
    m[-1] = matplotlib.colors.to_rgba(NOISE)
    return m, cl

def cmdscale(D, dim=2):
    """Classical MDS (metric) of a precomputed distance matrix."""
    n = D.shape[0]
    G = np.asarray(D, dtype=np.float64) ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ G @ H
    w, V = np.linalg.eigh(B)
    i = np.argsort(w)[::-1][:dim]
    return V[:, i] * np.sqrt(np.maximum(w[i], 0))

def mds(pdb_id, run="multi"):
    D = np.load(OUT_DIR / f"{pdb_id}_matrixA.npy")
    lab = pd.read_csv(OUT_DIR / f"{pdb_id}_labels.tsv", sep="\t")
    X = cmdscale(D)
    col, cl = colors(lab.cluster.values)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in [-1] + cl:                                   # noise below, clusters on top
        m = (lab.cluster == c).values
        ax.scatter(X[m, 0], X[m, 1], s=6 if c == -1 else 10, alpha=0.45 if c == -1 else 0.7,
                   c=[col[c]], label=f"cluster {c} (n={m.sum()})" if c != -1 else f"noise (n={m.sum()})",
                   edgecolors="none")
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
    ax.set_title(f"{pdb_id}: 2500 models, RMSD matrix -> 2D (classical MDS)")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{pdb_id}_mds.png", dpi=150)
    plt.close(fig)
    print(f"{OUT_DIR / f'{pdb_id}_mds.png'}")

def heatmap(pdb_id):
    D = np.load(OUT_DIR / f"{pdb_id}_matrixA.npy")
    lab = pd.read_csv(OUT_DIR / f"{pdb_id}_labels.tsv", sep="\t")
    cl = lab.cluster.value_counts().index.tolist()        # big clusters first
    order = np.concatenate([np.where(lab.cluster.values == c)[0] for c in cl])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(D[np.ix_(order, order)], cmap="magma_r", norm=LogNorm(vmin=0.05, vmax=8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{pdb_id}: pairwise antigen RMSD, ordered by cluster (size desc, noise last)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="RMSD (Å, log)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{pdb_id}_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"{OUT_DIR / f'{pdb_id}_heatmap.png'}")

def dockq(pdb_id):
    df = pd.read_csv(OUT_DIR / f"{pdb_id}_models_eval.tsv", sep="\t")
    cl = sorted(c for c in df.cluster.unique() if c != -1)
    groups = [-1] + cl

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    box = [df.loc[df.cluster == c, "dockq"].values for c in groups]
    bp = a1.boxplot(box, tick_labels=[f"c{c}" for c in groups], showfliers=False, patch_artist=True)
    col, _ = colors(np.array(groups))
    for patch, c in zip(bp["boxes"], groups):
        patch.set_facecolor(col[c]); patch.set_alpha(0.7)
    for thr, name in [(0.23, "acceptable"), (0.49, "medium"), (0.80, "high")]:
        a1.axhline(thr, ls=":", lw=0.8, c="k", alpha=0.6)
        a1.text(a1.get_xlim()[1], thr, f" {name}", va="center", fontsize=7)
    a1.set_ylabel("DockQ"); a1.set_title(f"{pdb_id}: DockQ per cluster")

    cmap, _ = colors(df.cluster.values)
    for c in groups:
        m = df[df.cluster == c]
        a2.scatter(m.ranking_score, m.dockq, s=8, alpha=0.5, c=[cmap[c]],
                   label=f"c{c} (n={len(m)})" if c != -1 else f"noise (n={len(m)})", edgecolors="none")
    for thr, name in [(0.49, "medium"), (0.80, "high")]:
        a2.axhline(thr, ls=":", lw=0.8, c="k", alpha=0.6)
    a2.set_xlabel("ranking_score"); a2.set_ylabel("DockQ")
    a2.set_title(f"{pdb_id}: confidence vs accuracy")
    a2.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{pdb_id}_dockq.png", dpi=150)
    plt.close(fig)
    print(f"{OUT_DIR / f'{pdb_id}_dockq.png'}")

if __name__ == "__main__":
    pdb = sys.argv[1] if len(sys.argv) > 1 else "8jow"
    mds(pdb)
    heatmap(pdb)
    dockq(pdb)
