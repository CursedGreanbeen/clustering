import sys, json
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from collect import DATA_DIR          # = /mnt/1857e392-.../results/reports
from analysis import OUT_DIR

GREY, NOISE, C_DOCKQ, C_RANK = "#b8bec4", "#d5d9dd", "#e4572e", "#2e86ab"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
           "#bcbd22", "#393b79", "#e6550d", "#31a354", "#756bb1", "#843c39", "#d6616b", "#3182bd",
           "#a1d99b", "#fdae6b", "#6b6ecf", "#ce6dbd"]          # no greys: grey = noise
MAX_BOXES = 15
T = dict(template="plotly_white", height=480, margin=dict(l=60, r=20, t=120, b=50))
MODES = ["Best highlighted", "Colour by cluster"]


def load(pdb):
    st = json.loads((DATA_DIR / f"stats_{pdb}.json").read_text())
    multi = pd.read_csv(DATA_DIR / f"multi_{pdb}.tsv", sep="\t")
    single = pd.read_csv(DATA_DIR / f"single_{pdb}.tsv", sep="\t")
    ctab = pd.read_csv(DATA_DIR / f"clusters_{pdb}.tsv", sep="\t", index_col="cluster")
    f = OUT_DIR / f"{pdb}_matrixA_{st['n_multi']}.npy"      # versioned name if present...
    if not f.exists():
        f = OUT_DIR / f"{pdb}_matrixA.npy"                  # ...else the plain name cluster.py writes
    D = np.load(f)
    d = np.load(DATA_DIR / f"single_dist_{pdb}.npy")
    assert len(multi) == len(D) == d.shape[1], f"{pdb}: size mismatch"
    return st, multi, single, ctab, D, d


def _hover(df):
    return df.apply(lambda r: f"seed {int(r.seed)}/{int(r['sample'])}<br>cluster {int(r.cluster)}"
                              f"<br>DockQ {r.dockq:.2f}<br>rank {r.ranking_score:.2f}"
                              f"<br>pLDDT {r.plddt_mean:.1f}", axis=1).values


def _slabel(single):
    return [f"{int(s)}/{int(k)} · {q:.2f}" for s, k, q in zip(single.seed, single["sample"], single.dockq)]


def _by_size(ctab):
    return list(ctab.sort_values("n_models", ascending=False).index)


def palette(ctab):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(_by_size(ctab))}


def cluster_scatter(fig, x, y, multi, st, ctab, size=5):
    """One trace per cluster (+ noise) in three copies (one per colouring mode). Returns {mode: [trace idx]}."""
    bd, br = st["best"]["dockq"]["cluster"], st["best"]["rank"]["cluster"]
    cl, txt = multi.cluster.values, _hover(multi)
    pal = palette(ctab)
    sets = {m: [] for m in MODES}
    for mode in MODES:
        for c in [-1] + _by_size(ctab):
            msk = cl == c
            if not msk.any():
                continue
            if c == -1:
                name, col = f"noise, n={int(msk.sum())}", NOISE
            else:
                tag = " ★ best DockQ" if c == bd else " ◆ best ranking" if c == br else ""
                name = f"cl {c}{tag}, n={int(ctab.loc[c, 'n_models'])}, {int(ctab.loc[c, 'n_seeds'])} seeds"
                col = {"Best highlighted": C_DOCKQ if c == bd else C_RANK if c == br else GREY,
                       "Colour by cluster": pal[c]}[mode]
            fig.add_trace(go.Scattergl(x=x[msk], y=y[msk], mode="markers", name=name, text=txt[msk],
                                       hoverinfo="text", visible=(mode == MODES[0]),
                                       marker=dict(color=col, size=size, opacity=.75)))
            sets[mode].append(len(fig.data) - 1)
    return sets


def add_mode_buttons(fig, sets):
    allset = {i for v in sets.values() for i in v}
    n = len(fig.data)
    btn = [dict(label=m, method="update",
                args=[{"visible": [(i not in allset) or (i in sets[m]) for i in range(n)]}])
           for m in MODES]
    fig.update_layout(updatemenus=[dict(type="buttons", direction="right", buttons=btn, x=0, y=1.13,
                                        xanchor="left", yanchor="top", showactive=True)])


def embed(pdb, n, D, d_single):
    """Classical MDS (PCoA) of the RMSD matrix; single-seed models are placed
    with the exact out-of-sample extension (Gower)."""
    f = DATA_DIR / f"embed_{pdb}_{n}.npz"
    if f.exists():
        z = np.load(f)
        return z["X"], z["Xs"], z["var"]
    D2 = D.astype(np.float64) ** 2
    B = -0.5 * (D2 - D2.mean(0) - D2.mean(1)[:, None] + D2.mean())
    w, V = np.linalg.eigh(B)
    i = np.argsort(w)[::-1][:2]
    w2, V2 = w[i], V[:, i]
    X = V2 * np.sqrt(w2)
    var = w2 / w[w > 0].sum()
    Xs = -0.5 * (d_single.astype(np.float64) ** 2 - D2.mean(0)) @ V2 / np.sqrt(w2)
    np.savez(f, X=X, Xs=Xs, var=var)
    return X, Xs, var


def fig_embedding(st, multi, single, ctab, X, Xs, var):
    bd = st["best"]["dockq"]["cluster"]
    fig = go.Figure()
    sets = cluster_scatter(fig, X[:, 0], X[:, 1], multi, st, ctab)
    med = int(ctab.loc[bd, "medoid"])
    lx, ly = [], []
    for i in range(len(Xs)):
        lx += [Xs[i, 0], X[med, 0], None]
        ly += [Xs[i, 1], X[med, 1], None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="#555", width=1, dash="dot"),
                             name="single → best-DockQ medoid", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[X[med, 0]], y=[X[med, 1]], mode="markers", name="medoid (best DockQ)",
                             marker=dict(symbol="star", size=16, color="#111", line=dict(color="white", width=1.5))))
    pos = ["top right", "bottom left", "top left", "bottom right"]
    fig.add_trace(go.Scatter(x=Xs[:, 0], y=Xs[:, 1], mode="markers+text", text=_slabel(single),
                             textposition=[pos[i % 4] for i in range(len(Xs))], textfont=dict(size=11),
                             name="single-seed (seed/sample · DockQ)",
                             marker=dict(symbol="diamond", size=14, color="#111", opacity=.85,
                                         line=dict(color="white", width=1.5))))
    add_mode_buttons(fig, sets)
    fig.update_layout(title="2D map of antigen poses (classical MDS); single-seed placed approximately",
                      xaxis_title=f"MDS1 ({var[0]:.0%} of variance)",
                      yaxis_title=f"MDS2 ({var[1]:.0%} of variance)", legend=dict(font=dict(size=10)), **T)
    return fig


def fig_distance(st, multi, single, ctab, D, d):
    keys = ["dockq"] if st["best"]["dockq"]["cluster"] == st["best"]["rank"]["cluster"] else ["dockq", "rank"]
    names = {"dockq": "best by DockQ", "rank": "best by ranking"}
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=.18,
                        subplot_titles=[f"{names[k]} (cluster {st['best'][k]['cluster']})" for k in keys])
    cl = multi.cluster.values
    pos = ["top center", "bottom center"]
    for r, k in enumerate(keys, 1):
        c = st["best"][k]["cluster"]
        med = int(ctab.loc[c, "medoid"])
        dm = D[med, cl == c]
        p95 = float(np.percentile(dm, 95))
        fig.add_trace(go.Box(x=dm, y0=0, orientation="h", width=.6, boxpoints=False, showlegend=False,
                             marker_color=C_DOCKQ if k == "dockq" else C_RANK, name="cluster members"),
                      row=r, col=1)
        ds = d[:, med]
        o = np.argsort(ds)                       # neighbours in distance get different heights / label sides
        yy = np.empty(len(ds)); yy[o] = 1 + ((np.arange(len(ds)) % 3) - 1) * .16
        tp = np.empty(len(ds), dtype=object); tp[o] = [pos[i % 2] for i in range(len(ds))]
        fig.add_trace(go.Scatter(x=ds, y=yy, mode="markers+text", text=_slabel(single), textposition=list(tp),
                                 showlegend=False, textfont=dict(size=11),
                                 marker=dict(symbol="diamond", size=13, color="#111", opacity=.85,
                                             line=dict(color="white", width=1.2))), row=r, col=1)
        fig.add_vline(x=p95, line_dash="dash", line_color="#333", row=r, col=1,
                      annotation_text="95% of members are closer", annotation_position="top right")
        fig.update_yaxes(tickvals=[0, 1], ticktext=["cluster<br>members", "single-seed<br>(seed/sample · DockQ)"],
                         range=[-.6, 1.7], row=r, col=1)
    fig.update_xaxes(title_text="antigen RMSD to the cluster medoid, Å", row=len(keys), col=1)
    fig.update_layout(title="How far are single-seed models from the best cluster's medoid?",
                      **{**T, "height": 250 + 190 * len(keys)})
    return fig


def fig_dockq(st, multi, single, ctab):
    pal = palette(ctab)
    bd, br = st["best"]["dockq"]["cluster"], st["best"]["rank"]["cluster"]
    order = _by_size(ctab)
    shown, rest = order[:MAX_BOXES], order[MAX_BOXES:]
    fig = go.Figure()

    def box(v, label, col):
        fig.add_trace(go.Box(y=v, name=label, boxpoints="all", jitter=.4, pointpos=0, boxmean=True,
                             marker=dict(color=col, size=3, opacity=.5), line=dict(color=col)))
    for c in shown:
        tag = " ★" if c == bd else " ◆" if c == br else ""
        box(multi.dockq[multi.cluster == c], f"cl {c}{tag}<br>n={int(ctab.loc[c, 'n_models'])}", pal[c])
    if rest:
        v = multi.dockq[multi.cluster.isin(rest)]
        box(v, f"{len(rest)} smaller<br>clusters<br>n={len(v)}", GREY)
    v = multi.dockq[multi.cluster == -1]
    if len(v):
        box(v, f"noise<br>n={len(v)}", NOISE)
    box(single.dockq, f"single-seed<br>n={len(single)}", "#111")
    for y, t in [(0.23, "acceptable 0.23"), (0.49, "medium 0.49")]:
        fig.add_hline(y=y, line_dash="dash", line_color="#888", annotation_text=t, annotation_position="top left")
    fig.update_layout(title="DockQ by cluster (box = quartiles, dashed line inside = mean; ★ best by DockQ, ◆ best by ranking)",
                      yaxis_title="DockQ", showlegend=False, **{**T, "margin": dict(l=60, r=20, t=60, b=50)})
    return fig


def fig_rank_dockq(st, multi, single, ctab):
    fig = go.Figure()
    sets = cluster_scatter(fig, multi.ranking_score.values, multi.dockq.values, multi, st, ctab)
    fig.add_trace(go.Scatter(x=single.ranking_score, y=single.dockq, mode="markers+text", text=_slabel(single),
                             textposition="top center", name="single-seed (seed/sample · DockQ)",
                             marker=dict(symbol="diamond", size=14, color="#111", opacity=.85,
                                         line=dict(color="white", width=1.5))))
    for y in (0.23, 0.49):
        fig.add_hline(y=y, line_dash="dash", line_color="#888")
    add_mode_buttons(fig, sets)
    fig.update_layout(title="Does AF3 ranking_score predict DockQ?", xaxis_title="ranking_score",
                      yaxis_title="DockQ", legend=dict(font=dict(size=10)), **T)
    return fig


def fig_clusters(st, ctab):
    pal = palette(ctab)
    bd, br = st["best"]["dockq"]["cluster"], st["best"]["rank"]["cluster"]
    t = ctab.sort_values("n_models", ascending=False).head(20)
    x = [str(c) for c in t.index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=t.n_models, name="models", marker=dict(
        color=[pal[c] for c in t.index],
        line=dict(color=["#111" if c == bd else C_RANK if c == br else "rgba(0,0,0,0)" for c in t.index],
                  width=[3 if c in (bd, br) else 0 for c in t.index]))), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=100 * t.n_seeds / st["total_seeds"], mode="markers", name="seed support, %",
                             marker=dict(symbol="diamond", size=10, color="black")), secondary_y=True)
    fig.update_yaxes(title_text="models in cluster", secondary_y=False)
    fig.update_yaxes(title_text="seed support, % of all seeds", range=[0, 105], secondary_y=True)
    fig.update_xaxes(title_text="cluster (largest 20)", type="category")
    fig.update_layout(title=f"Clusters: size and seed support (noise: {st['noise_frac']:.0%} of models; "
                            f"black outline = best by DockQ, blue = best by ranking)", **{**T, "margin": dict(l=60, r=20, t=60, b=50)})
    return fig


def complex_figures(pdb):
    st, multi, single, ctab, D, d = load(pdb)
    X, Xs, var = embed(pdb, st["n_multi"], D, d)
    return {"embedding": fig_embedding(st, multi, single, ctab, X, Xs, var),
            "distance": fig_distance(st, multi, single, ctab, D, d),
            "dockq": fig_dockq(st, multi, single, ctab),
            "rank_dockq": fig_rank_dockq(st, multi, single, ctab),
            "clusters": fig_clusters(st, ctab)}


if __name__ == "__main__":
    pdb = sys.argv[1] if len(sys.argv) > 1 else "8jow"
    figs = complex_figures(pdb)
    html = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                   for i, f in enumerate(figs.values()))
    (DATA_DIR / f"preview_{pdb}.html").write_text(html)
    print("written", DATA_DIR / f"preview_{pdb}.html")
