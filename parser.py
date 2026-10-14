"""Parse AF3 model CIFs into grouped CA coordinate arrays for clustering.

Usage (run from this directory):
    /opt/conda/bin/python3.10 parser.py <root_dir> <pdb_id> [more pdb_ids...] [--workers N]

    <root_dir>   dir containing <pdb_id>/seed-<s>_sample-<k>/<pdb_id>_..._model.cif
                 e.g. /mnt/f0771a75-815c-4553-ba74-9aa70f57f995/500-seeds
    --workers N  parallel model parsing (default 1; nproc=1 here, so keep 1 locally)

Output: /mnt/f0771a75-815c-4553-ba74-9aa70f57f995/clustering/<pdb_id>_ca.npz
with arrays seed, sample, and per group (fit/cdr3/anchor/ag):
coords (n_models, n_atoms, 3) float32 + <group>_plddt (n_models, n_atoms).

Requires anarci_residues.tsv from annotate_cif_with_anarci.py --numbering.
"""
import sys; sys.path.insert(0, "/mnt/1857e392-c689-44a8-9c3e-e7f9beab82f6/scripts")  # model_cif_to_fasta
import re, numpy as np, pandas as pd, gemmi
from pathlib import Path
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from model_cif_to_fasta import residue_to_one_letter

SEED_RE = re.compile(r"seed-(\d+)_sample-(\d+)")
GROUPS = ("fit", "cdr3", "anchor", "ag")
OUT_DIR = Path("/mnt/f0771a75-815c-4553-ba74-9aa70f57f995/clustering")

def build_spec(residue_tsv, pdb_id):
    df = pd.read_csv(residue_tsv, sep="\t", keep_default_na=False,
                     dtype={"chain": str, "imgt": str, "ins": str})
    df = df[df.pdb_id == pdb_id].sort_values(["chain", "res_index"], kind="stable")
    masks = {"fit": df.region.str.startswith("FR"), "cdr3": df.region == "CDR3",
             "anchor": df.h3_anchor == 1, "ag": df.region == "AG"}
    return {
        "seqs": {c: "".join(g.aa) for c, g in df.groupby("chain", sort=False)},
        "groups": {k: list(zip(df.chain[m], df.res_index[m])) for k, m in masks.items()},
    }

def parse_model(cif_path, spec):
    cif_path = Path(cif_path)
    seed, sample = map(int, SEED_RE.search(cif_path.parent.name).groups())
    model = gemmi.read_structure(str(cif_path))[0]
    res = {ch.name: [r for r in ch if r.het_flag == "A"] for ch in model}

    for c, seq in spec["seqs"].items():                     # sequence check
        got = "".join(residue_to_one_letter(r) for r in res.get(c, []))
        if got != seq:
            raise ValueError(f"{cif_path.name}: chain {c} sequence mismatch")

    out = {"seed": seed, "sample": sample}
    for g in GROUPS:
        xyz, b = [], []
        for c, i in spec["groups"][g]:
            a = res[c][i].find_atom("CA", "*")
            xyz.append(a.pos.tolist() if a else [np.nan] * 3)
            b.append(a.b_iso if a else np.nan)
        out[g] = np.asarray(xyz, np.float32)
        out[g + "_plddt"] = np.asarray(b, np.float32)
    return out

def parse_complex(root, pdb_id, residue_tsv, run="multi", workers=1, limit=None, out_dir=OUT_DIR):
    spec = build_spec(residue_tsv, pdb_id)
    cifs = sorted(Path(root, pdb_id).glob("seed-*_sample-*/*_model.cif"))
    if limit:
        cifs = cifs[:limit]
    if workers == 1:
        rows = [parse_model(p, spec) for p in cifs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(parse_model, cifs, repeat(spec), chunksize=10))
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pdb_id}_ca_{run}.npz"
    np.savez_compressed(path, **arrays)
    print(f"{pdb_id} run={run}: {len(cifs)} models -> {path}")
    return arrays
