"""Matrix A: pairwise antigen RMSD between models, after Kabsch-fit on the framework.

Usage (run from this directory):
    /opt/conda/bin/python3.10 matrix.py            # runs 8jow via __main__ block
    or:  from matrix import matrix_A; D, keep = matrix_A(pdb_id, residue_tsv)

Input:  <pdb>_ca_<n>.npz in /mnt/f0771a75-.../clustering (from parser.py; largest n wins)
        + anarci_residues.tsv (from results/reports/, needs --numbering columns)
Output: <pdb>_matrixA_<n>.npy   (n,n float32 pairwise RMSD, Å)
        <pdb>_matrixA_<n>_agmask.npy  (bool keep-mask over npz['ag'] atoms)

Method: align all models by Kabsch-fit on `fit` CA (refit to mean), drop low-pLDDT
antigen residues (chains >= 30 res at pLDDT < 70), then plain RMSD of the fitted
antigen coords. The plain-RMSD shortcut matches exact per-pair Kabsch to ~0.004 Å
on 8jow (verified in __main__ check). Symmetric/identical AG chains are reported
but NOT handled yet.
"""
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from parser import build_spec, OUT_DIR

MIN_CHAIN_LEN = 30     # shorter antigen chains are kept whole
PLDDT_CUT = 70.0

def kabsch(P, Q):
    """Batched fit of P (n,k,3) onto Q (k,3) or (n,k,3).
    Returns rotations R (n,3,3), centroids cP, cQ, so that aligned = (X - cP) @ R + cQ."""
    Q = np.broadcast_to(Q, P.shape)
    cP, cQ = P.mean(1, keepdims=True), Q.mean(1, keepdims=True)
    H = np.einsum("nki,nkj->nij", P - cP, Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros_like(U); D[:, 0, 0] = D[:, 1, 1] = 1; D[:, 2, 2] = d
    return U @ D @ Vt, cP, cQ

def apply(X, R, cP, cQ):
    return np.einsum("nki,nij->nkj", X - cP, R) + cQ

def ag_mask(npz, spec):
    """Fixed residue mask for the antigen, same atom order as npz['ag']."""
    chains = np.array([c for c, _ in spec["groups"]["ag"]])
    mean_pl = np.nanmean(npz["ag_plddt"], axis=0)
    keep = np.zeros(len(chains), bool)
    for c in np.unique(chains):
        m = chains == c
        keep[m] = True if spec["seqs"][c].__len__() < MIN_CHAIN_LEN else mean_pl[m] >= PLDDT_CUT
    return keep, chains

def load_ca(pdb_id):
    """Newest/largest <pdb>_ca_<n>.npz in OUT_DIR (the full run wins)."""
    files = list(OUT_DIR.glob(f"{pdb_id}_ca_*.npz"))
    if not files:
        raise FileNotFoundError(f"no {pdb_id}_ca_*.npz in {OUT_DIR}")
    return np.load(max(files, key=lambda p: int(p.stem.rsplit("_", 1)[1])))

def matrix_A(pdb_id, residue_tsv, out_prefix=None):
    npz = load_ca(pdb_id)
    spec = build_spec(residue_tsv, pdb_id)
    fit, ag = npz["fit"].astype(np.float64), npz["ag"].astype(np.float64)
    assert not np.isnan(fit).any() and not np.isnan(ag).any()

    # antigen mask + report
    keep, chains = ag_mask(npz, spec)
    for c in np.unique(chains):
        print(f"AG chain {c}: kept {keep[chains == c].sum()}/{(chains == c).sum()}")
    ag_k = ag[:, keep]

    # report identical antigen chains (symmetry: not handled yet)
    groups = defaultdict(list)
    for c in np.unique(chains):
        groups[spec["seqs"][c]].append(c)
    print("identical AG chain groups:", [g for g in groups.values() if len(g) > 1] or "none")

    # reference: fit to model 0, average, re-fit to the mean
    R, cP, cQ = kabsch(fit, fit[0])
    mean_ref = apply(fit, R, cP, cQ).mean(0)
    R, cP, cQ = kabsch(fit, mean_ref)
    ag_fit = apply(ag_k, R, cP, cQ)

    # matrix A: plain RMSD between fitted antigen coordinates
    n, m = ag_fit.shape[:2]
    D = squareform(pdist(ag_fit.reshape(n, -1)) / np.sqrt(m)).astype(np.float32)

    # approximation check: exact pairwise Kabsch on random pairs
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(50):
        i, j = rng.choice(n, 2, replace=False)
        Rp, cp, cq = kabsch(fit[i:i+1], fit[j])
        a_i = apply(ag_k[i:i+1], Rp, cp, cq)[0]
        exact = np.sqrt(((a_i - ag_k[j]) ** 2).sum(1).mean())
        diffs.append(D[i, j] - exact)
    diffs = np.abs(diffs)
    print(f"|shortcut - exact|: median {np.median(diffs):.3f} Å, max {diffs.max():.3f} Å")

    iu = np.triu_indices(n, 1)
    q = np.percentile(D[iu], [5, 50, 95])
    print(f"pairwise RMSD: 5% {q[0]:.2f}  median {q[1]:.2f}  95% {q[2]:.2f} Å")

    prefix = out_prefix or str(OUT_DIR / f"{pdb_id}_matrixA_{n}")
    np.save(f"{prefix}.npy", D)
    np.save(f"{prefix}_agmask.npy", keep)
    return D, keep

if __name__ == "__main__":
    D, keep = matrix_A("8jow", "/mnt/1857e392-c689-44a8-9c3e-e7f9beab82f6/results/reports/anarci_residues.tsv")
