import argparse, json, datetime
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from collect import DATA_DIR              # = /mnt/1857e392-.../results/reports
from build_report import HTML_DIR, CSS, git_hash, card

EPS = 0.05                                 # |Δ DockQ| below this counts as "no difference"
OUTCOMES = ["multi-seed helps", "no difference", "single-seed better"]
OC_COL = {"multi-seed helps": "#1a7f37", "no difference": "#9aa3ab", "single-seed better": "#c62828"}
PRIMARY = "oracle"                      # "oracle" | "blind"
P, S = ("bd", "br") if PRIMARY == "oracle" else ("br", "bd")   # primary / secondary prefix
LBL = {"bd": "best by DockQ (oracle)", "br": "best by ranking (blind)"}
NAME = {"bd": "oracle", "br": "blind"}
C_DOCKQ, C_RANK = "#e4572e", "#2e86ab"
T = dict(template="plotly_white", height=480, margin=dict(l=60, r=20, t=60, b=60))

JS = """
const t=document.getElementById('tbl');const dir={};
t.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
 const rows=[...t.tBodies[0].rows];dir[i]=!dir[i];
 const val=r=>{const c=r.cells[i];const v=c.dataset.v!==undefined?c.dataset.v:c.textContent;const n=parseFloat(v);return isNaN(n)?v:n};
 rows.sort((a,b)=>{const x=val(a),y=val(b);return (x>y?1:x<y?-1:0)*(dir[i]?1:-1)});
 rows.forEach(r=>t.tBodies[0].appendChild(r));}));
"""
EXTRA_CSS = "th{cursor:pointer;position:sticky;top:0} th:hover{background:#dde3e8} a{color:#2e86ab;text-decoration:none}"


def outcome(delta):
    return OUTCOMES[0] if delta > EPS else OUTCOMES[2] if delta < -EPS else OUTCOMES[1]


def load_rows():
    rows, bad = [], []
    for f in sorted(DATA_DIR.glob("stats_*.json")):
        try:
            st = json.loads(f.read_text())
            bd, br, s = st["best"]["dockq"], st["best"]["rank"], st["single"]
            rows.append({
                "pdb": st["pdb"], "n_multi": st["n_multi"], "seeds": st["total_seeds"],
                "clusters": st["n_clusters"], "noise": st["noise_frac"],
                "single_dockq": s["dockq_mean"], "single_acc": s["frac_acc"],
                "bd_cluster": bd["cluster"], "bd_dockq": bd["dockq_mean"], "bd_delta": bd["delta_dockq_vs_single"],
                "bd_acc": bd["frac_acc"], "bd_n": bd["n_models"], "bd_seeds": bd["n_seeds"],
                "br_cluster": br["cluster"], "br_dockq": br["dockq_mean"], "br_delta": br["delta_dockq_vs_single"],
                "br_acc": br["frac_acc"], "br_support": br["seed_support"],
                "same": bool(st["rank_best_is_dockq_best"]),
                "s_best": bd["single_assigned_to_best"], "s_other": bd["single_assigned_other"],
                "s_rej": bd["single_rejected"], "s_n": s["n_models"],
                "s_dist": bd["single_dist_median"],
                "outcome_blind": outcome(br["delta_dockq_vs_single"]),
                "outcome_oracle": outcome(bd["delta_dockq_vs_single"])})
        except Exception as e:                      # truncated JSON, missing key, old schema
            bad.append(f.name)
            print(f"[SKIP] {f.name}: {e!r}")
    if not rows:
        raise SystemExit("no valid stats_*.json found: run analysis.py first")
    df = pd.DataFrame(rows)
    df["delta"] = df[f"{P}_delta"]
    df["outcome"] = df[f"outcome_{NAME[P]}"]
    return df, bad


def td(v, text=None, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f'<td data-v="{v}"{c}>{text if text is not None else v}</td>'


def table(df):
    cols = ["Complex", "Models", "Seeds", "Clusters", "Noise", "Single DockQ",
            "Best-DockQ cl: DockQ", "Δ vs single (oracle)", "Best-ranking cl: DockQ", "Δ vs single (blind)",
            "Same cluster?", "Best-ranking seed support", "Single in best-DockQ cl",
            "Single rejected", "Single median dist, Å", f"Outcome ({NAME[P]})"]
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for r in df.itertuples():
        g = lambda d: "good" if d > EPS else "bad" if d < -EPS else ""
        body += "<tr>" + "".join([
            f'<td><a href="{r.pdb}.html">{r.pdb}</a></td>',
            td(r.n_multi), td(r.seeds), td(r.clusters), td(r.noise, f"{r.noise:.0%}"),
            td(r.single_dockq, f"{r.single_dockq:.3f}"),
            td(r.bd_dockq, f"{r.bd_dockq:.3f} (cl {r.bd_cluster})"),
            td(r.bd_delta, f"{r.bd_delta:+.3f}", g(r.bd_delta)),
            td(r.br_dockq, f"{r.br_dockq:.3f} (cl {r.br_cluster})"),
            td(r.br_delta, f"{r.br_delta:+.3f}", g(r.br_delta)),
            td(int(r.same), "yes" if r.same else "no"),
            td(r.br_support, f"{r.br_support:.0%}"),
            td(r.s_best, f"{r.s_best} / {r.s_n}"), td(r.s_rej, f"{r.s_rej} / {r.s_n}"),
            td(r.s_dist, f"{r.s_dist:.2f}"),
            f'<td style="color:{OC_COL[r.outcome]};font-weight:600">{r.outcome}</td>'])
    return f'<table id="tbl"><thead>{h}</thead><tbody>{body}</tbody></table>'


def fig_scatter(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#888", dash="dash"),
                             name="no difference", hoverinfo="skip"))
    for name, col, key in [("best by DockQ (oracle)", C_DOCKQ, "bd_dockq"), ("best by ranking (blind)", C_RANK, "br_dockq")]:
        fig.add_trace(go.Scatter(x=df.single_dockq, y=df[key], mode="markers", name=name, text=df.pdb,
                                 marker=dict(color=col, size=9, opacity=.8, line=dict(color="white", width=1)),
                                 hovertemplate="%{text}<br>single %{x:.3f}<br>cluster %{y:.3f}<extra></extra>"))
    fig.update_layout(title="Mean DockQ: best multi-seed cluster vs single-seed (above the line = multi-seed better)",
                      xaxis_title="single-seed mean DockQ", yaxis_title="best-cluster mean DockQ",
                      xaxis_range=[-.02, 1.02], yaxis_range=[-.02, 1.02], **{**T, "height": 560})
    return fig


def fig_delta(df):
    d = df.sort_values("delta")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.pdb, y=d.delta, name=LBL[P],
                         marker_color=[OC_COL[o] for o in d.outcome]))
    fig.add_trace(go.Scatter(x=d.pdb, y=d[f"{S}_delta"], mode="markers", name=LBL[S],
                             marker=dict(symbol="diamond", size=9, color="#111")))
    for y in (EPS, -EPS):
        fig.add_hline(y=y, line_dash="dot", line_color="#888")
    fig.update_layout(title=f"Δ DockQ (best cluster − single-seed) per complex; dotted lines = ±{EPS}",
                      yaxis_title="Δ mean DockQ", xaxis=dict(type="category", tickangle=-60), **T)
    return fig


def fig_outcomes(df):
    fig = go.Figure()
    for name, col, key in [("best by ranking (blind)", C_RANK, "outcome_blind"), ("best by DockQ (oracle)", C_DOCKQ, "outcome_oracle")]:
        fig.add_trace(go.Bar(x=OUTCOMES, y=[int((df[key] == o).sum()) for o in OUTCOMES], name=name, marker_color=col,
                             text=[int((df[key] == o).sum()) for o in OUTCOMES], textposition="outside"))
    fig.update_layout(title="Outcome per complex (number of complexes)", barmode="group", yaxis_title="complexes", **T)
    return fig


def fig_assign(df):
    d = df.sort_values("pdb")
    fig = go.Figure()
    for name, key, col in [("in best-DockQ cluster", "s_best", "#e4572e"), ("in another cluster", "s_other", "#6c8ebf"),
                           ("rejected (outside any cluster)", "s_rej", "#b8bec4")]:
        fig.add_trace(go.Bar(x=d.pdb, y=d[key], name=name, marker_color=col))
    fig.update_layout(title="Where do the single-seed models fall (kNN assignment)?", barmode="stack",
                      yaxis_title="single-seed models", xaxis=dict(type="category", tickangle=-60), **T)
    return fig


def build_index():
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    (HTML_DIR / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")   # first: pages depend on it
    df, bad = load_rows()
    df.to_csv(DATA_DIR / "report_summary.tsv", sep="\t", index=False)
    n = len(df)
    cnt = lambda k, o: int((df[k] == o).sum())
    cards = "".join([
        card(n, "complexes"),
        card(f"{cnt('outcome', OUTCOMES[0])} / {n}", f"multi-seed helps ({NAME[P]} choice)"),
        card(f"{cnt('outcome', OUTCOMES[2])} / {n}", f"single-seed better ({NAME[P]} choice)"),
        card(f"{df[f'{P}_delta'].median():+.3f}", f"median Δ DockQ, {NAME[P]}"),
        card(f"{df[f'{S}_delta'].median():+.3f}", f"median Δ DockQ, {NAME[S]}"),
        card(f"{df.same.mean():.0%}", "ranking picks the best-DockQ cluster"),
        card(f"{df.noise.median():.0%}", "median noise fraction")])
    figs = [fig_scatter(df), fig_delta(df), fig_outcomes(df), fig_assign(df)]
    plots = "".join(f.to_html(full_html=False, include_plotlyjs="directory" if i == 0 else False)
                    for i, f in enumerate(figs))
    ORACLE_NOTE = ("<br><b>Oracle is an upper bound:</b> the best of several clusters is chosen "
               "using the native structure, and it is compared with a 5-model single-seed mean."
               if PRIMARY == "oracle" else "")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>AF3 clustering: overview</title>
<style>{CSS}{EXTRA_CSS}</style></head><body>
<h1>AF3 multi-seed clustering vs single-seed: overview</h1>
<p class="sub">{n} complexes · HDBSCAN (eom) · click a column header to sort, click a complex to open its report</p>
<div class="cards">{cards}</div>
<h2>Plots</h2>{plots}
<h2>All complexes</h2>{table(df)}
<p class="sub">Blind = cluster with the highest mean ranking_score; oracle = cluster with the highest mean DockQ (uses the native structure).
Outcome: Δ DockQ &gt; +{EPS} multi-seed helps, &lt; −{EPS} single-seed better, otherwise no difference.
Single-seed = {int(df.s_n.iloc[0])} models per complex, so its mean is noisy.
{ORACLE_NOTE}</p>
<div class="foot">Generated {datetime.date.today()} · git {git_hash()} · DockQ < 0.23 incorrect, 0.23-0.49 acceptable, 0.49-0.8 medium, >0.8 high
{f'<br><b style="color:#c62828">Skipped (unreadable stats): {", ".join(bad)}</b>' if bad else ''}</div>
<script>{JS}</script></body></html>"""
    (HTML_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"[ok] index.html: {n} complexes -> {HTML_DIR / 'index.html'}"
          + (f" ({len(bad)} skipped: {', '.join(bad)})" if bad else ""))


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    build_index()
