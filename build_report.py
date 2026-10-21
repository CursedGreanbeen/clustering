import argparse, json, subprocess, datetime, sys
from pathlib import Path
import numpy as np, pandas as pd
from plotly.offline import get_plotlyjs

from collect import DATA_DIR              # = /mnt/1857e392-.../results/reports
from plots_upd import complex_figures

HTML_DIR = DATA_DIR / "html"             # all pages + shared plotly.min.js live here
EPS = 0.05                               # keep in sync with build_index.py
GOOD, BAD = "#1a7f37", "#c62828"
OUTC = ["multi-seed helps", "no difference", "single-seed better"]

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px auto;max-width:1280px;
color:#1c2733;padding:0 16px}
h1{font-size:26px;margin:8px 0 2px} h2{font-size:19px;margin:34px 0 10px;border-bottom:2px solid #dde3e8;padding-bottom:4px}
.sub{color:#66707a;font-size:13px;margin:2px 0 16px}
.foot{color:#66707a;font-size:12px;margin:36px 0 12px;border-top:1px solid #dde3e8;padding-top:8px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}
.card{background:#f4f7f9;border:1px solid #dde3e8;border-radius:8px;padding:10px 16px;min-width:120px}
.card .v{font-size:22px;font-weight:700} .card .l{font-size:12px;color:#66707a}
table{border-collapse:collapse;font-size:13px;margin:10px 0}
th,td{border:1px solid #dde3e8;padding:5px 9px;text-align:right;white-space:nowrap}
th{background:#eef2f5} td:first-child,th:first-child{text-align:left}
tr.hl td{background:#fdf3ef} tr.hl2 td{background:#edf5fa}
.good{color:#1a7f37;font-weight:600} .bad{color:#c62828;font-weight:600}
.verdict{background:#f4f7f9;border-left:4px solid #2e86ab;padding:10px 14px;border-radius:4px;font-size:14px}
"""


def git_hash():
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "?"
    except Exception:
        return "?"


def card(value, label):
    return f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div></div>'


def ensure_plotly_js():
    f = HTML_DIR / "plotly.min.js"
    if not f.exists():
        f.write_text(get_plotlyjs(), encoding="utf-8")


def _delta_cell(d):
    cls = "good" if d > EPS else "bad" if d < -EPS else ""
    return f'<td class="{cls}">{d:+.3f}</td>'


def cluster_table(st, ctab):
    bd, br = st["best"]["dockq"]["cluster"], st["best"]["rank"]["cluster"]
    cols = ["Cluster", "Models", "Seeds", "Seed support", "DockQ mean", "DockQ max", "Acc", "Med",
            "Rank mean", "pLDDT", "Mean RMSD, Å", "Medoid idx"]
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for c, t in ctab.sort_values("n_models", ascending=False).iterrows():
        cls = " class=\"hl\"" if c == bd else " class=\"hl2\"" if c == br else ""
        tag = " ★" if c == bd else " ◆" if c == br else ""
        body += (f'<tr{cls}><td>{c}{tag}</td><td>{int(t.n_models)}</td><td>{int(t.n_seeds)}</td>'
                 f'<td>{t.n_seeds / st["total_seeds"]:.0%}</td><td>{t.dockq_mean:.3f}</td>'
                 f'<td>{t.dockq_max:.3f}</td><td>{t.frac_acc:.0%}</td><td>{t.frac_med:.0%}</td>'
                 f'<td>{t.ranking_score_mean:.3f}</td><td>{t["plddt_mean_mean"]:.1f}</td>'
                 f'<td>{t.mean_rmsd:.2f}</td><td>{int(t.medoid)}</td></tr>')
    return f'<table><thead>{h}</thead><tbody>{body}</tbody></table>'


def single_table(single):
    cols = ["Seed", "Sample", "DockQ", "Ranking", "pLDDT", "Assigned cluster",
            "Dist to medoid, Å", "Dist / threshold"]
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for r in single.sort_values("dockq", ascending=False).itertuples():
        cl = "rejected" if r.cluster == -1 else int(r.cluster)
        body += (f'<tr><td>{int(r.seed)}</td><td>{int(r.sample)}</td><td>{r.dockq:.3f}</td>'
                 f'<td>{r.ranking_score:.3f}</td><td>{r.plddt_mean:.1f}</td><td>{cl}</td>'
                 f'<td>{r.dist_medoid_best_dockq:.2f}</td><td>{r.dist_ratio_best_dockq:.2f}</td></tr>')
    return f'<table><thead>{h}</thead><tbody>{body}</tbody></table>'


def verdict(st):
    bd, br, s = st["best"]["dockq"], st["best"]["rank"], st["single"]
    o = ("multi-seed helps" if br["delta_dockq_vs_single"] > EPS
         else "single-seed better" if br["delta_dockq_vs_single"] < -EPS else "no difference")
    col = GOOD if o == OUTC[0] else BAD if o == OUTC[2] else "#66707a"
    same = "the same cluster" if st["rank_best_is_dockq_best"] else "a different cluster"
    return (f'<div class="verdict">Blind choice (highest mean ranking_score) → mean DockQ '
            f'<b>{br["dockq_mean"]:.3f}</b> vs single-seed <b>{s["dockq_mean"]:.3f}</b> '
            f'(Δ {br["delta_dockq_vs_single"]:+.3f}): <b style="color:{col}">{o}</b>. '
            f'The blind pick is {same} as the oracle (best-DockQ) cluster. '
            f'{bd["single_assigned_to_best"]}/{s["n_models"]} single-seed models fall into the '
            f'best-DockQ cluster, {bd["single_rejected"]}/{s["n_models"]} are rejected by the kNN rule.</div>')


def build_report(pdb):
    st = json.loads((DATA_DIR / f"stats_{pdb}.json").read_text())
    ctab = pd.read_csv(DATA_DIR / f"clusters_{pdb}.tsv", sep="\t", index_col="cluster")
    single = pd.read_csv(DATA_DIR / f"single_{pdb}.tsv", sep="\t")
    bd, br, s = st["best"]["dockq"], st["best"]["rank"], st["single"]
    cards = "".join([
        card(st["n_multi"], "multi-seed models"), card(st["total_seeds"], "seeds"),
        card(st["n_clusters"], "clusters"), card(f'{st["noise_frac"]:.0%}', "noise fraction"),
        card(f'{s["dockq_mean"]:.3f}', "single-seed mean DockQ"),
        card(f'{bd["dockq_mean"]:.3f}', f'best by DockQ (cl {bd["cluster"]})'),
        card(f'{bd["delta_dockq_vs_single"]:+.3f}', "Δ vs single (oracle)"),
        card(f'{br["dockq_mean"]:.3f}', f'best by ranking (cl {br["cluster"]})'),
        card(f'{br["delta_dockq_vs_single"]:+.3f}', "Δ vs single (blind)"),
        card(f'{br["seed_support"]:.0%}', "seed support (blind)"),
        card(f'{bd["single_assigned_to_best"]}/{s["n_models"]}', "single in best-DockQ cl"),
        card(f'{bd["single_rejected"]}/{s["n_models"]}', "single rejected")])
    figs = complex_figures(pdb)
    plots = "".join(f.to_html(full_html=False, include_plotlyjs="directory" if i == 0 else False)
                    for i, f in enumerate(figs.values()))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>AF3 clustering: {pdb}</title>
<style>{CSS}</style></head><body>
<p class="sub"><a href="index.html">← all complexes</a></p>
<h1>{pdb}: multi-seed clusters vs single-seed</h1>
<p class="sub">{st["n_multi"]} multi-seed models ({st["total_seeds"]} seeds) · {st["n_clusters"]} clusters ·
noise {st["noise_frac"]:.0%} · {s["n_models"]} single-seed models · ★ best by DockQ (oracle), ◆ best by ranking (blind)</p>
<div class="cards">{cards}</div>
{verdict(st)}
<h2>Plots</h2>{plots}
<h2>Clusters</h2>{cluster_table(st, ctab)}
<h2>Single-seed models (kNN assignment vs best-DockQ cluster)</h2>{single_table(single)}
<div class="foot">Generated {datetime.date.today()} · git {git_hash()} · DockQ ≥ 0.23 acceptable, ≥ 0.49 medium</div>
</body></html>"""
    (HTML_DIR / f"{pdb}.html").write_text(html, encoding="utf-8")
    print(f"[ok] {pdb}.html -> {HTML_DIR / f'{pdb}.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_ids", nargs="*", help="default: every stats_*.json in DATA_DIR")
    a = ap.parse_args()
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    ensure_plotly_js()
    bad = []
    for pdb in (a.pdb_ids or sorted(p.stem[len("stats_"):] for p in DATA_DIR.glob("stats_*.json"))):
        try:
            build_report(pdb)
        except Exception as e:                  # one bad complex must not stop the batch
            bad.append(pdb)
            print(f"[FAIL] {pdb}: {e!r}")
    if bad:
        print(f"[FAIL] {len(bad)} complexes: {', '.join(bad)}")
        sys.exit(1)
