"""Step 09: main figures (Fig. 1-7). Publication style, 300 dpi PNG + LZW TIFF."""
import os
import re
import textwrap

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib_venn import venn2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.fontsize": 6.8, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "mathtext.fontset": "dejavusans",
})

C_PSC, C_IBD, C_NC = "#B03A2E", "#DF8A34", "#4F86A6"
C_MIC, C_MET, C_TGT = "#7FAF7A", "#E8A33D", "#5B7FA6"
C_UP, C_DOWN, C_NS = "#B03A2E", "#2E6DA4", "#C9C9C9"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG, name + ".tif"), dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", name)


def ttl(ax, letter, text, pad=0.014, fontsize=7.8):
    """Panel title placed relative to the gridspec cell (robust to aspect changes)."""
    bb = ax.get_subplotspec().get_position(ax.figure)
    ax.figure.text(bb.x0, bb.y1 + pad, r"$\mathbf{%s}$   %s" % (letter, text),
                   fontsize=fontsize, va="bottom", ha="left")


# =====================================================================
# Figure 1 - study design and disease-gene overlap
# =====================================================================
psc = pd.read_csv(os.path.join(RES, "psc_genes.csv"))
ibd = pd.read_csv(os.path.join(RES, "ibd_genes.csv"))
ov = pd.read_csv(os.path.join(RES, "psc_ibd_overlap_genes.csv"))
mt = pd.read_csv(os.path.join(RES, "metabolite_targets.csv"))
core = pd.read_csv(os.path.join(RES, "core_targets.csv"))["core_target"].tolist()
hub = pd.read_csv(os.path.join(RES, "hub_genes.csv"))["hub_gene"].tolist()

fig = plt.figure(figsize=(7.09, 4.85))
gs = fig.add_gridspec(2, 3, height_ratios=[1.00, 0.96], hspace=0.30, wspace=0.80,
                      width_ratios=[1.00, 1.02, 1.04])

ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(4.4, 35.2); ax.axis("off")
rows = [
    ("DATA", 28.5, [
        "gutMGene v1.0\n1,049 metabolite-gene\nrecords; 618 structures",
        "Open Targets\nPSC (MONDO_0018646)\nUC / CD / IBD",
        "ChEMBL similarity\nTanimoto >= 0.85\npChEMBL >= 5",
        "GEO GSE230524\n220 colonic biopsies\nPSC-IBD / IBD / NC"]),
    ("INTEGRATION", 17.5, [
        "664 gut metabolites\n618 with SMILES",
        "103 PSC-IBD\noverlap genes",
        "899 metabolite\ntarget genes",
        "14 core targets"]),
    ("VALIDATION", 6.5, [
        "STRING PPI\ncytoHubba (6 metrics)\n5 consensus hubs",
        "g:Profiler\nGO / KEGG / Reactome",
        "AutoDock Vina 1.2.7\n5 receptors x 14 ligands",
        "ADMET-AI\nLipinski rules"]),
]
cols_row = [C_MIC, C_PSC, C_MET, C_NC]
for row_label, y, boxes in rows:
    ax.text(0.2, y + 2.6, row_label, fontsize=5.6, fontweight="bold",
            color="#666666", va="center", ha="left")
    for i, txt in enumerate(boxes):
        x0 = 12.5 + i * 21.9
        col = cols_row[i]
        ax.add_patch(FancyBboxPatch((x0, y - 1.0), 20.0, 7.2,
                                    boxstyle="round,pad=0.22,rounding_size=0.7",
                                    linewidth=0.9, edgecolor=col, facecolor=col + "18"))
        ax.text(x0 + 10.0, y + 2.6, txt, ha="center", va="center", fontsize=5.4,
                color="#1A1A1A", linespacing=1.45)
        if i < 3:
            ax.annotate("", xy=(x0 + 21.7, y + 2.6), xytext=(x0 + 20.3, y + 2.6),
                        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#999999",
                                        shrinkA=0, shrinkB=0))
for y0, y1 in [(26.6, 24.9), (15.6, 13.9)]:
    ax.annotate("", xy=(50.0, y1), xytext=(50.0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#999999",
                                shrinkA=0, shrinkB=0))

ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-0.98, 1.06)
ax.set_aspect("equal"); ax.axis("off")
for cx, col in [(-0.30, C_PSC), (0.30, C_IBD)]:
    ax.add_patch(plt.Circle((cx, 0.00), 0.58, facecolor=col, alpha=0.38,
                            edgecolor=col, linewidth=1.0, zorder=2))
ax.text(0.00, 0.00, f"{len(ov)}", ha="center", va="center", fontsize=8.8,
        fontweight="bold", color="white", zorder=4)
ax.text(-0.56, 0.00, f"{len(psc) - len(ov)}", ha="center", va="center",
        fontsize=7.8, color="#5A1A12", zorder=4)
ax.text(0.58, 0.00, f"{len(ibd) - len(ov)}", ha="center", va="center",
        fontsize=7.8, color="#7A4410", zorder=4)
ax.text(-0.52, 0.76, "PSC\n(n = %d)" % len(psc), ha="center", va="center", fontsize=6.3,
        color=C_PSC, fontweight="bold", linespacing=1.35)
ax.text(0.52, 0.76, "IBD\n(n = %d)" % len(ibd), ha="center", va="center", fontsize=6.3,
        color=C_IBD, fontweight="bold", linespacing=1.35)
ttl(ax, "a", "Disease genes (score \u2265 0.2)")

ax = fig.add_subplot(gs[1, 1])
steps = ["PSC genes", "IBD genes", "Overlap", "Metabolite targets", "Core targets"]
vals = [len(psc), len(ibd), len(ov), mt["gene"].nunique(), len(core)]
cols = [C_PSC, C_IBD, "#6C4C8C", C_MET, C_UP]
y = np.arange(len(steps))[::-1]
ax.barh(y, vals, color=cols, height=0.60)
for yy, vv in zip(y, vals):
    ax.text(vv + 14, yy, f"{vv:,}", va="center", fontsize=6.5, color="#333333")
ax.set_yticks(y); ax.set_yticklabels(steps, fontsize=6.2)
ax.set_xlim(0, 1000); ax.set_xlabel("Genes (n)")
ax.set_xticks([0, 250, 500, 750, 1000])
ax.spines[["top", "right"]].set_visible(False)
ttl(ax, "b", "Filtering cascade")

ax = fig.add_subplot(gs[1, 2])
top = ov.head(16).iloc[::-1].reset_index(drop=True)
yy = np.arange(len(top))
is_core = top["symbol"].isin(core).values
is_hub = top["symbol"].isin(hub).values
ax.hlines(yy, 0, top["score"], color="#DDE3EA", lw=1.1, zorder=1)
ax.scatter(top["score"], yy, s=26,
           c=[C_UP if c else "#A9B8C8" for c in is_core],
           edgecolors="white", linewidths=0.7, zorder=3)
ax.scatter(top.loc[is_hub, "score"], yy[is_hub], s=58, facecolors="none",
           edgecolors="#7A1F18", linewidths=0.8, zorder=4)
for i, s_ in enumerate(top["score"]):
    off = 0.036 if is_hub[i] else 0.018
    ax.text(s_ + off, i, f"{s_:.2f}", va="center", fontsize=5.5, color="#555555")
ax.set_yticks(yy)
labs = ax.set_yticklabels([g + ("*" if h else "") for g, h in zip(top["symbol"], is_hub)],
                          fontsize=6.2)
for t, c in zip(labs, is_core):
    if c:
        t.set_color("#7A1F18"); t.set_fontweight("bold")
ax.set_xlim(0, top["score"].max() * 1.24)
ax.set_xlabel("PSC association score")
ax.grid(axis="x", color="#EEEEEE", lw=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4.0, color=C_UP, label="core"),
                   Line2D([], [], marker="o", ls="", ms=4.0, color="#A9B8C8", label="other"),
                   Line2D([], [], marker="o", ls="", ms=6.0, mfc="none", mec="#7A1F18", label="hub")],
          loc="upper center", bbox_to_anchor=(0.50, -0.30), fontsize=5.4,
          handletextpad=0.10, columnspacing=0.85, borderaxespad=0, ncol=3)
ttl(ax, "c", "Top overlap genes")
save(fig, "Fig1_study_design_and_gene_overlap")

# =====================================================================
# Figure 2 - metabolite-target integration and core targets
# =====================================================================
cur = mt[mt["source"] == "gutMGene_curated"]["gene"]
pred = mt[mt["source"] == "ChEMBL_similarity"]["gene"]
tpm = pd.read_csv(os.path.join(RES, "targets_per_metabolite.csv"))
# drop polybrominated diphenyl ether congeners: environmental contaminants with
# no annotated microbial production in gutMGene
tpm = tpm[~tpm["metabolite"].str.contains("bromodiphenyl ether", case=False, na=False)]
ev = pd.read_csv(os.path.join(RES, "core_target_evidence.csv"))
dual = set(pd.read_csv(os.path.join(RES, "core_targets_dual.csv"))["dual_supported_core"])

fig = plt.figure(figsize=(7.09, 4.35))
gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 0.85, 1.10], wspace=0.62)

ax = fig.add_subplot(gs[0, 0])
v = venn2([set(cur), set(pred)], set_labels=("gutMGene\n(curated)", "ChEMBL\n(predicted)"), ax=ax)
v.get_patch_by_id("10").set_color(C_MIC); v.get_patch_by_id("10").set_alpha(0.6)
v.get_patch_by_id("01").set_color(C_MET); v.get_patch_by_id("01").set_alpha(0.6)
v.get_patch_by_id("11").set_color("#6C4C8C"); v.get_patch_by_id("11").set_alpha(0.8)
for t in v.set_labels:
    if t: t.set_fontsize(6.4)
for t in v.subset_labels:
    if t: t.set_fontsize(6.6)
ttl(ax, "a", "Target genes")

ax = fig.add_subplot(gs[0, 1])
top = tpm.head(14).iloc[::-1]
ax.barh(np.arange(len(top)), top["n_targets"], color=C_MET, height=0.64)
ax.set_yticks(np.arange(len(top)))
ax.set_yticklabels([textwrap.shorten(n, 26, placeholder="..") for n in top["metabolite"]], fontsize=5.9)
ax.set_xlabel("Linked targets (n)")
ax.spines[["top", "right"]].set_visible(False)
ttl(ax, "b", "Most polypharmacological")

ax = fig.add_subplot(gs[0, 2])
cnt = ev.groupby(["gene", "source"]).size().unstack(fill_value=0)
for s in ["gutMGene_curated", "ChEMBL_similarity"]:
    if s not in cnt.columns:
        cnt[s] = 0
tot = cnt.sum(axis=1).sort_values()
order = tot.index
yy = np.arange(len(order))
left = np.zeros(len(order))
for s, col, lab in [("gutMGene_curated", C_MIC, "Curated"),
                    ("ChEMBL_similarity", C_MET, "Predicted")]:
    ax.barh(yy, cnt.loc[order, s].values, left=left, color=col, height=0.64, label=lab)
    left += cnt.loc[order, s].values
for i in range(len(order)):
    ax.text(left[i] + 0.6, i, f"{int(left[i])}", va="center", fontsize=6.0)
ax.set_yticks(yy)
ax.set_yticklabels([g + ("*" if g in dual else "") for g in order], fontsize=6.3)
ax.set_xlabel("Linked metabolites (n)")
ax.set_xlim(0, max(left) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", handlelength=1.0, fontsize=5.8)
ttl(ax, "c", "Core targets (*dual)")
save(fig, "Fig2_metabolite_target_integration")

# =====================================================================
# Figure 3 - functional enrichment
# =====================================================================
en = pd.read_csv(os.path.join(RES, "enrichment_gprofiler.csv"))

def dotplot(ax, sub, n, title, letter, wrap=34):
    sub = sub.nsmallest(n, "p_value").iloc[::-1]
    y = np.arange(len(sub))
    lp = -np.log10(sub["p_value"].values)
    size = 20 + (sub["intersection_size"].values / sub["intersection_size"].max()) * 105
    pmin, pmax = lp.min(), lp.max()
    cols = plt.cm.YlOrRd(0.22 + 0.68 * (lp - pmin) / (pmax - pmin + 1e-9))
    ax.scatter(lp, y, s=size, c=cols, edgecolor="#444444", linewidth=0.4, zorder=3)
    labels = ["\n".join(textwrap.wrap(re.sub(r"\s+", " ", t), wrap))[:90] for t in sub["term_name"]]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.9)
    ax.set_xlabel(r"$-\log_{10}$ $P$", fontsize=6.8)
    ax.grid(axis="x", ls=":", lw=0.5, color="#CCCCCC", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, lp.max() * 1.15)
    ttl(ax, letter, title)

# Terms discussed in Results 3.3, selected by interpretation rather than by rank,
# so that every term named in the text is visible in the figure.
SEL = [
    ("KEGG pathway", [
        "Inflammatory bowel disease",
        "Lipid and atherosclerosis",
        "Toll-like receptor signaling pathway"]),
    ("GO molecular function", [
        "DNA-binding transcription factor binding",
        "transcription coregulator binding",
        "nuclear receptor activity",
        "ligand-modulated transcription factor activity",
        "C-X-C chemokine receptor activity",
        "prostaglandin receptor activity"]),
    ("Reactome", ["Signaling by Interleukins"]),
    ("Infectious-disease KEGG maps (not interpreted)",
     ["Chagas disease", "Yersinia infection"]),
]

G_COL = {"KEGG pathway": "#5B7FA6", "GO molecular function": "#E8A33D",
         "Reactome": "#7FAF7A", "infect": "#B9B9B9"}

rows, group_y = [], []
ypos, gap = 0.0, 0.55
for gname, terms in SEL:
    group_y.append((gname, ypos + 0.55))
    for t in terms:
        r = en[en.term_name == t].iloc[0]
        gkey = "infect" if "infectious" in gname.lower() else gname
        rows.append({"label": t, "p": r.p_value, "genes": r.genes.split(";"),
                     "col": G_COL[gkey], "infect": gkey == "infect", "y": ypos})
        ypos -= 0.95
    ypos -= gap

GENES_NR = ["NR1H4", "PPARA", "PPARG", "GPR35", "PTGER4"]
GENES_EF = ["CCL2", "CXCR2", "FOS", "HDAC7", "IL10", "IL2", "NFKB1", "PTPN2", "TLR4"]
GENES = GENES_NR + GENES_EF
NR_BLUE, EF_RED = "#3C6E9A", "#B03A2E"

fig = plt.figure(figsize=(7.09, 5.0))
outer = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.95], wspace=0.62)
ax = fig.add_subplot(outer[0, 0])
dotplot(ax, en[en.source == "GO:BP"], 12, "GO biological process", "a")

inner = outer[0, 1].subgridspec(1, 3, width_ratios=[0.28, 0.60, 1.12], wspace=0.04)
axp = fig.add_subplot(inner[0, 0])
axl = fig.add_subplot(inner[0, 1])
axm = fig.add_subplot(inner[0, 2])

ymin, ymax = rows[-1]["y"] - 0.52, rows[0]["y"] + 2.30
for ax_ in (axp, axl, axm):
    ax_.set_ylim(ymin, ymax)

# ---- significance bars
lp = -np.log10(np.array([r["p"] for r in rows]))
axp.barh([r["y"] for r in rows], lp, height=0.46,
         color=[r["col"] for r in rows], edgecolor="none", zorder=2)
axp.set_xlim(0, 7.0)
axp.set_yticks([])
axp.set_xlabel(r"$-\log_{10}$ $P$", fontsize=6.2, labelpad=2)
axp.set_xticks([0, 2, 4, 6])
axp.tick_params(axis="x", labelsize=5.0, length=2, pad=1.5)
axp.grid(axis="x", ls=":", lw=0.4, color="#E3E3E3", zorder=0)
axp.spines[["top", "right", "left"]].set_visible(False)
ttl(axp, "b", "Selected terms and core-target membership", pad=0.020)

# ---- term labels (own panel so group headers never collide with tick labels)
axl.set_xlim(0, 1)
axl.axis("off")
for r in rows:
    axl.text(0.97, r["y"], "\n".join(textwrap.wrap(r["label"], 23)), ha="right",
             va="center", fontsize=4.9, linespacing=0.90,
             color="#999999" if r["infect"] else "#222222")
for gname, gy in group_y:
    axl.text(0.97, gy, gname, ha="right", va="bottom", fontsize=4.7,
             color="#777777", style="italic")
    axl.plot([0.34, 0.97], [gy - 0.14, gy - 0.14], color="#D5D5D5", lw=0.5)

# ---- membership matrix (squares coloured by functional module)
for r in rows:
    for j, g in enumerate(GENES):
        if g in r["genes"]:
            fc = "#D8D8D8" if r["infect"] else (NR_BLUE if j < len(GENES_NR) else EF_RED)
            axm.add_patch(Rectangle((j - 0.30, r["y"] - 0.30), 0.60, 0.60,
                                    facecolor=fc, edgecolor="none", zorder=3))
axm.set_xlim(-0.45, len(GENES) - 0.55)
axm.set_xticks(range(len(GENES)))
axm.set_xticklabels(GENES, rotation=90, fontsize=4.4, style="italic")
axm.set_yticks([])
axm.tick_params(axis="x", length=2, pad=2.5)
for s in axm.spines.values():
    s.set_visible(False)
axm.plot([len(GENES_NR) - 0.5] * 2, [rows[-1]["y"] - 0.40, rows[0]["y"] + 0.40],
         color="#CCCCCC", lw=0.7, zorder=1)

# ---- module header: two stacked rows so the labels never collide
for (lo, hi, name, fc), dy in [( (0, len(GENES_NR), "Ligand-sensing (NR)", NR_BLUE), 0.0),
                               ( (len(GENES_NR), len(GENES), "Cytokine / NF-\u03baB effector", EF_RED), 0.52)]:
    axm.add_patch(Rectangle((lo - 0.40, ymax - 1.78 + dy), (hi - lo) - 0.18, 0.11,
                            facecolor=fc, edgecolor="none", clip_on=False, alpha=0.9))
    axm.text(lo - 0.40, ymax - 1.62 + dy, name, ha="left", va="bottom",
             fontsize=4.6, color=fc, fontweight="bold", clip_on=False)

handles = [Line2D([], [], color=c, lw=3.0, label=l)
           for l, c in [("KEGG pathway", G_COL["KEGG pathway"]),
                        ("GO molecular function", G_COL["GO molecular function"]),
                        ("Reactome", G_COL["Reactome"]),
                        ("Infectious-disease map (not interpreted)", G_COL["infect"])]]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.4,
           bbox_to_anchor=(0.5, -0.05), handletextpad=0.5, columnspacing=1.1)
fig.text(0.5, -0.082, "Bars: g:SCS-adjusted P. Filled squares: core targets belonging to each term; "
         "columns are ordered by functional module.",
         ha="center", fontsize=5.3, color="#777777")
save(fig, "Fig3_functional_enrichment")

# =====================================================================
# Figure 4 - PPI network and cytoHubba metrics
# =====================================================================
edges = pd.read_csv(os.path.join(RES, "ppi_string_edges.csv"))
metr = pd.read_csv(os.path.join(RES, "ppi_cytohubba_metrics.csv")).set_index("gene")
G = nx.Graph()
for _, r in edges.iterrows():
    G.add_edge(r["source"], r["target"], weight=float(r["combined_score"]))

fig = plt.figure(figsize=(7.09, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30)
ax = fig.add_subplot(gs[0, 0])
pos = nx.circular_layout(G)
wmax = max(d["weight"] for _, _, d in G.edges(data=True))
for u, v, d in G.edges(data=True):
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#C4C4C4",
            lw=0.35 + 2.0 * d["weight"] / wmax, zorder=1)
deg = dict(G.degree())
# per-node label nudges in points (x right, y up), applied to the radial label position
NOFF = {"TLR4": (-4, 0), "PPARG": (0, 4), "HDAC7": (4, 0),
        "PTPN2": (0, -2), "IL10": (0, -1.5), "NFKB1": (0, -1.5)}
for n in G:
    ax.scatter(pos[n][0], pos[n][1], s=70 + 14 * deg[n],
               c=C_PSC if n in hub else "#9AB6C9", edgecolor="white", lw=0.7, zorder=3)
    dx, dy = NOFF.get(n, (0, 0))
    ax.annotate(n, xy=(pos[n][0] * 1.22, pos[n][1] * 1.22), xytext=(dx, dy),
                textcoords="offset points", ha="center", va="center",
                fontsize=6.5 if n in hub else 6.1,
                fontweight="bold" if n in hub else "normal", zorder=4)
ax.set_xlim(-1.62, 1.62); ax.set_ylim(-1.45, 1.45)
ax.set_aspect("equal", adjustable="box")   # keep the ring a true circle
ax.axis("off")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_PSC, markersize=5.5,
                          label="consensus hub"),
                   Line2D([], [], marker="o", ls="", color="#9AB6C9", markersize=5,
                          label="core target")],
          loc="upper center", ncol=2, fontsize=5.8, bbox_to_anchor=(0.5, -0.01),
          columnspacing=1.2, handletextpad=0.4)
ttl(ax, "a", f"STRING PPI network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

ax = fig.add_subplot(gs[0, 1])
mets = ["Degree", "MNC", "MCC", "EPC", "Radiality", "Stress"]
z = (metr[mets] - metr[mets].mean()) / metr[mets].std(ddof=0)
z = z.sort_values("Degree", ascending=False)
im = ax.imshow(z.values, cmap="RdBu_r", aspect="auto", vmin=-2.2, vmax=2.2)
ax.set_xticks(range(len(mets))); ax.set_xticklabels(mets, rotation=40, ha="right", fontsize=6.2)
ax.set_yticks(range(len(z)))
ax.set_yticklabels([f"{g} *" if g in hub else g for g in z.index], fontsize=6.3)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        m, val = mets[j], metr.loc[z.index[i], mets[j]]
        lab = f"{val:.2f}" if m == "EPC" else f"{val:.0f}"
        ax.text(j, i, lab, ha="center", va="center",
                fontsize=5.1, color="white" if abs(z.values[i, j]) > 1.5 else "#222222")
ax.set_xticks(np.arange(-.5, len(mets), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(z), 1), minor=True)
ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("row z-score", fontsize=6.2); cb.ax.tick_params(labelsize=5.8)
ttl(ax, "b", "cytoHubba topology (* hub)")
save(fig, "Fig4_ppi_network_hubs")

# =====================================================================
# Figure 5 - metabolite-target axis and microbiota producers
# =====================================================================
imp = pd.read_csv(os.path.join(RES, "gmt_metabolite_importance.csv")).set_index("metabolite")
mic = pd.read_csv(os.path.join(RES, "gmt_microbe_importance.csv")).set_index("microbe")
mt_core = mt[mt["gene"].isin(core)].drop_duplicates(["metabolite", "gene", "source"])

fig = plt.figure(figsize=(7.09, 6.35))
outer = fig.add_gridspec(1, 2, width_ratios=[1.30, 0.95], wspace=0.46)
right = outer[0, 1].subgridspec(2, 1, height_ratios=[1, 0.88], hspace=0.34)
fig.subplots_adjust(left=0.135, right=0.975, top=0.915, bottom=0.135)

# (a) dot matrix, all 49 metabolites grouped by chemical class
ax = fig.add_subplot(outer[0, 0])
CLS = {
    "SCFA": ["Acetate", "Propionate", "Butyrate", "Hexanoic acid"],
    "Bile acid": ["Lithocholic acid", "Chenodiol", "Isoallolithocholic acid",
                  "Taurochenodeoxycholic acid", "Tauroursodeoxycholic acid",
                  "Isolithocholic acid", "Glycoursodeoxycholic acid", "Ursodiol",
                  "Hyocholate", "Glycochenodeoxycholic acid", "omega-Muricholic acid"],
    "Fatty acid": ["Arachidonic Acid", "Capric Acid", "Arachidic Acid", "Icosapent",
                   "Heptanoic Acid", "Myristic Acid", "Palmitic Acid", "Linoleic Acid",
                   "Oleic Acid", "Stearic Acid", "Palmitoleic Acid", "Elaidate",
                   "Gondoic Acid", "Tetracosanoic acid", "Nervonic Acid", "Erucic Acid",
                   "Behenic Acid", "(9S,10E,12Z)-9-hydroxyoctadeca-10,12-dienoic acid",
                   "10-Oxo-11-octadecenoic acid", "10-keto-12Z-octadecenoic acid"],
    "Flavonoid/phenolic": ["Caffeic Acid", "3-(4-Hydroxyphenyl)propionic acid",
                           "4-Hydroxyphenylacetic acid", "Formononetin",
                           "Isorhamnetin", "Chrysin", "Myricetin"],
    "Eicosanoid": ["11-epi-Prostaglandin F2alpha", "Dinoprostone"],
    "Other": ["Succinate", "3-Indolepropionic acid", "Levothyroxine", "Loganetin",
              "n-Oleoylethanolamine"],
}
CLS_COL = {"SCFA": "#C0392B", "Bile acid": "#7D6608", "Fatty acid": "#1F618D",
           "Flavonoid/phenolic": "#117A65", "Eicosanoid": "#6C3483", "Other": "#555555"}
GENES_NR = ["NR1H4", "PPARA", "PPARG", "GPR35", "PTGER4"]
GENES_EF = ["CCL2", "CXCR2", "FOS", "HDAC7", "IL10", "IL2", "NFKB1", "PTPN2", "TLR4"]
COLS = GENES_NR + GENES_EF
srcmat = mt_core.set_index(["metabolite", "gene"])["source"].to_dict()

yrows, grp_y = {}, []
cur = 0.0
for gname, members in CLS.items():
    grp_y.append((gname, cur, len(members)))
    cur -= 0.62
    ordered = [m for m in imp.index if m in set(members)]
    ordered.sort(key=lambda m: -imp.loc[m, "score"])
    for m in ordered:
        yrows[m] = cur
        cur -= 1.0
    cur -= 0.55

ymax, ymin = 2.52, cur + 0.55
ax.set_xlim(-0.55, len(COLS) - 0.45)
ax.set_ylim(ymin, ymax)
for i, m in enumerate([m for m in yrows]):
    y = yrows[m]
    for j, g in enumerate(COLS):
        s = srcmat.get((m, g))
        if not s:
            continue
        if s == "gutMGene_curated":
            ax.scatter(j, y, s=17, c=C_MIC, edgecolor="#3A3A3A", lw=0.25, zorder=3)
        else:
            ax.scatter(j, y, s=17, facecolor="white", edgecolor=C_MET, lw=0.85, zorder=3)
ax.set_xticks(range(len(COLS)))
ax.set_xticklabels([g + ("*" if g in hub else "") for g in COLS], rotation=90,
                   fontsize=5.0, style="italic")
ax.set_yticks([yrows[m] for m in yrows])
ax.set_yticklabels([textwrap.shorten(m, 30, placeholder="..") for m in yrows], fontsize=4.15)
ax.tick_params(axis="both", length=0, pad=1.6)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(axis="x", ls=":", lw=0.35, color="#E8E8E8")
ax.set_axisbelow(True)
ax.axvline(len(GENES_NR) - 0.5, color="#C9C9C9", lw=0.6, ls="-", zorder=1)

# class headers: coloured label above a faint rule
for gname, gy, n in grp_y:
    ax.text(-0.50, gy + 0.40, f"{gname} (n={n})", ha="left", va="center", fontsize=4.6,
            color=CLS_COL[gname], fontweight="bold")
    ax.plot([-0.55, len(COLS) - 0.45], [gy - 0.02, gy - 0.02],
            color=CLS_COL[gname], lw=0.45, alpha=0.35, zorder=1)

# column module header
for lo, hi, name, fc in [(0, len(GENES_NR), "Ligand-sensing (NR)", "#3C6E9A"),
                         (len(GENES_NR), len(COLS), "Cytokine / NF-\u03baB effector", "#B03A2E")]:
    ax.plot([lo - 0.42, hi - 0.58], [2.30, 2.30], color=fc, lw=1.1,
            clip_on=False, solid_capstyle="butt")
    ax.text((lo + hi - 1) / 2, 2.36, name, ha="center", va="bottom", fontsize=4.8,
            color=fc, fontweight="bold", clip_on=False)
ttl(ax, "a", "Metabolite-target evidence, all 49 metabolites", pad=0.012)

# (b) importance
ax = fig.add_subplot(right[0, 0])
top = imp.head(15).iloc[::-1]
yy = np.arange(len(top))
h1 = top["n_hub"] * 3
h2 = top["n_core_targets"] - top["n_hub"]
h3 = top["n_microbes"] * 0.2
ax.barh(yy, h1, color=C_PSC, height=0.62, label="Hub targets (x3)")
ax.barh(yy, h2, left=h1, color=C_TGT, height=0.62, label="Other core targets")
ax.barh(yy, h3, left=h1 + h2, color=C_MIC, height=0.62, label="Producing taxa (x0.2)")
ax.set_yticks(yy)
ax.set_yticklabels([m if len(m) <= 22 else m[:21].rstrip() + ".." for m in top.index], fontsize=5.6)
ax.set_xlabel("Importance score")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=5.4, handlelength=1.0)
ttl(ax, "b", "Ranked metabolites")

# (c) microbes
ax = fig.add_subplot(right[1, 0])
tm = mic.head(12).iloc[::-1]
ax.barh(np.arange(len(tm)), tm["n_metabolites"], color=C_MIC, height=0.62)
ax.set_yticks(np.arange(len(tm)))
ax.set_yticklabels([textwrap.shorten(m, 26, placeholder="..") for m in tm.index], fontsize=5.6)
ax.set_xlabel("Linked metabolites (n)")
ax.spines[["top", "right"]].set_visible(False)
ttl(ax, "c", "Top producer taxa")
save(fig, "Fig5_metabolite_target_axis")

# =====================================================================
# Figure 6 - docking and ADMET
# =====================================================================
dock = pd.read_csv(os.path.join(RES, "docking_matrix.csv")).set_index("metabolite")
dock = dock.drop(columns=[c for c in ["CCL2"] if c in dock.columns])
order_p = ["NR1H4", "PPARG", "TLR4", "IL2", "PPARA"]
dock = dock[[c for c in order_p if c in dock.columns]]
order_m = dock.mean(axis=1).sort_values().index.tolist()
dock = dock.loc[order_m]
adme = pd.read_csv(os.path.join(RES, "adme_panel_profile.csv")).set_index("name")
aend = adme[["AMES", "hERG", "DILI", "Bioavailability_Ma", "CYP3A4_Veith"]]

fig = plt.figure(figsize=(7.09, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.55, 0.92], wspace=0.62)

SHORT = {"Taurochenodeoxycholic acid": "TCDCA", "Lithocholic acid": "LCA",
         "Deoxycholic Acid": "DCA", "Chenodiol": "CDCA", "Ursodiol": "UDCA",
         "3-Indolepropionic acid": "IPA", "Indole-3-acetic acid": "IAA"}

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(dock.values, cmap="YlGnBu_r", aspect="auto", vmin=-11, vmax=-2)
ax.set_xticks(range(dock.shape[1]))
ax.set_xticklabels(dock.columns, rotation=35, ha="right", fontsize=6.3)
ax.set_yticks(range(dock.shape[0]))
ax.set_yticklabels([SHORT.get(m, m) for m in dock.index], fontsize=5.9)
for i in range(dock.shape[0]):
    for j in range(dock.shape[1]):
        v = dock.values[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=4.9,
                color="white" if v < -6.5 else "#222222")
p = ax.get_position()
cax = fig.add_axes([p.x0 + 0.18 * p.width, p.y0 - 0.10, 0.52 * p.width, 0.013])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Vina affinity (kcal/mol)", fontsize=6.0); cb.ax.tick_params(labelsize=5.6)
ttl(ax, "a", "Docking heatmap")

ax = fig.add_subplot(gs[0, 1])
means = dock.mean(axis=0)
se = dock.std(axis=0) / np.sqrt(dock.shape[0])      # SEM across the 14 docked metabolites
order_b = means.sort_values(ascending=True).index   # strongest binder (most negative) at top
mean_b, se_b = means[order_b], se[order_b]
ypos = np.arange(len(mean_b))[::-1]
REF = -7.0                                          # strong-binding reference
bcols = [C_PSC if m <= REF else "#B9B9B9" for m in mean_b]
ax.barh(ypos, mean_b, xerr=se_b, height=0.58, color=bcols,
        error_kw=dict(lw=0.7, capsize=1.5, ecolor="#555555"))
for yy, m, s, c in zip(ypos, mean_b, se_b, bcols):
    xt = min(m + s + 0.68, -0.45)                   # clear of the SEM cap, inside the bar
    ax.text(xt, yy, f"{m:.2f}", va="center", ha="left", fontsize=5.9,
            color="white" if c == C_PSC else "#2B2B2B")
ax.set_yticks(ypos); ax.set_yticklabels(order_b, fontsize=6.3)
ax.set_xlabel("Mean affinity (kcal/mol)"); ax.set_xlim(-10.8, 0)
ax.axvline(REF, ls=":", lw=0.8, color="#666666", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ttl(ax, "b", "Receptor ranking")

ax = fig.add_subplot(gs[0, 2])
mat = aend.loc[order_m]
norm = np.zeros_like(mat.values, dtype=float)
for j, cname in enumerate(mat.columns):
    v = mat[cname].values.astype(float)
    norm[:, j] = v if cname == "Bioavailability_Ma" else v / max(v.max(), 1e-9)
im = ax.imshow(norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(mat.columns)))
ax.set_xticklabels(["AMES", "hERG", "DILI", "Bio-\navail.", "CYP3A4\ninh."], fontsize=5.2)
ax.set_yticks(range(len(mat)))
_yp = adme.loc[mat.index, "Lipinski_pass"].astype(bool).values
ax.set_yticklabels([SHORT.get(m, m) + ("" if ok else " †") for m, ok in zip(mat.index, _yp)],
                   fontsize=5.0)
for i in range(norm.shape[0]):
    for j in range(norm.shape[1]):
        ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=4.5,
                color="white" if norm[i, j] > 0.78 else "#222222")
ax.set_xticks(np.arange(-.5, len(mat.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(mat), 1), minor=True)
ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
ttl(ax, "c", "Predicted ADMET")
save(fig, "Fig6_docking_admet")

# =====================================================================
# Figure 7 - transcriptomic validation in GSE230524
# =====================================================================
de = pd.read_csv(os.path.join(RES, "geo_de_PSCIBD_vs_IBD.csv")).set_index("gene")
coretab = pd.read_csv(os.path.join(RES, "geo_core_expression.csv"))
mod = pd.read_csv(os.path.join(RES, "geo_module_score.csv"))
hubexp = pd.read_csv(os.path.join(RES, "geo_hub_expression.csv"))

fig = plt.figure(figsize=(7.09, 6.35))
gs = fig.add_gridspec(2, 3, height_ratios=[0.92, 1.05], width_ratios=[1.05, 0.9, 0.95],
                      hspace=0.62, wspace=0.46)

ax = fig.add_subplot(gs[0, 0])
dt = de.loc[[g for g in core if g in de.index]].dropna(subset=["pvalue"]).copy()
dt["ci"] = 1.96 * dt["lfcSE"].fillna(0)
dt = dt.sort_values("log2FoldChange")
yy = np.arange(len(dt))
cols_pt = [C_UP if p_ < 0.05 else "#8A8A8A"
           for p_ in dt["padj"].fillna(1)]
ax.axvline(0, lw=0.6, color="#555555")
ax.hlines(yy, dt["log2FoldChange"] - dt["ci"], dt["log2FoldChange"] + dt["ci"],
          color=cols_pt, lw=0.9, alpha=0.85)
ax.scatter(dt["log2FoldChange"], yy, s=16, c=cols_pt, edgecolor="white", lw=0.4, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels([f"{g}*" if g in hub else g for g in dt.index], fontsize=6.2)
ax.set_xlabel(r"$\log_2$ FC with 95% CI (PSC-IBD vs IBD)")
ax.set_xlim(-0.9, 1.6)
ax.set_ylim(-0.8, len(dt) - 0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_UP, markersize=3.6,
                          label="FDR < 0.05"),
                   Line2D([], [], marker="o", ls="", color="#8A8A8A", markersize=3.6,
                          label="not significant")], loc="upper left", fontsize=5.2)
ttl(ax, "a", "Core targets (48 vs 116)")

ax = fig.add_subplot(gs[0, 1])
groups = ["NC", "IBD", "PSC-IBD"]
cols = [C_NC, C_IBD, C_PSC]
bp = ax.boxplot([mod.loc[mod.group == g, "module_score"] for g in groups], patch_artist=True,
                widths=0.55, showfliers=False, medianprops=dict(color="black", lw=1.0),
                whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
for p, c in zip(bp["boxes"], cols):
    p.set_facecolor(c); p.set_alpha(0.45); p.set_edgecolor("#333333"); p.set_linewidth(0.7)
rng = np.random.default_rng(1)
for i, (g, c) in enumerate(zip(groups, cols)):
    v = mod.loc[mod.group == g, "module_score"].values
    ax.scatter(i + 1 + rng.normal(0, 0.055, len(v)), v, s=3.2, color=c, alpha=0.7, lw=0)
ax.set_xticks(range(1, 4))
ax.set_xticklabels([f"{g}\n(n={int((mod.group == g).sum())})" for g in groups], fontsize=6.2)
ax.set_ylabel("Module score (z)")
ax.spines[["top", "right"]].set_visible(False)
ymax = mod["module_score"].max()
ax.plot([1.9, 1.9, 3.1, 3.1], [ymax * 1.10, ymax * 1.16, ymax * 1.16, ymax * 1.10], lw=0.6, c="#333333")
ax.text(2.5, ymax * 1.18, "P = 0.014", ha="center", fontsize=5.6)
ttl(ax, "b", "Core-target module")

ax = fig.add_subplot(gs[0, 2])
hubs_present = [g for g in hub if g in hubexp.columns]
for i, g in enumerate(hubs_present):
    for j, (grp, c) in enumerate(zip(groups, cols)):
        v = hubexp.loc[hubexp.group == grp, g].values
        xpos = i + (j - 1) * 0.24
        ax.scatter(xpos + rng.normal(0, 0.02, len(v)), v, s=2.0, color=c, alpha=0.5, lw=0)
        ax.plot([xpos - 0.09, xpos + 0.09], [np.median(v)] * 2, color=c, lw=2.2,
                solid_capstyle="round")
ax.set_xticks(range(len(hubs_present)))
ax.set_xticklabels(hubs_present, rotation=0, fontsize=6.1)
ax.set_ylabel(r"$\log_2$ CPM")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markersize=3.2, label=g)
                   for g, c in zip(groups, cols)], loc="upper right", fontsize=5.4)
ttl(ax, "c", "Consensus hubs")

ax = fig.add_subplot(gs[1, :])
genes = [g for g in core if g in coretab.columns]
ct = coretab.copy()
gorder = {g: ct.index[ct["group"] == g].tolist() for g in groups}
idx = [i for g in groups for i in gorder[g]]
sub = ct.loc[idx, genes].astype(float)
z = (sub - sub.mean()) / sub.std(ddof=0)
im = ax.imshow(z.T.values, cmap="RdBu_r", aspect="auto", vmin=-1.6, vmax=1.6,
               interpolation="nearest")
ax.set_yticks(range(len(genes)))
ax.set_yticklabels([g + ("*" if g in hub else "") for g in genes], fontsize=6.1)
ax.set_xticks([])
for x in np.cumsum([len(gorder[g]) for g in groups])[:-1]:
    ax.axvline(x - 0.5, color="white", lw=1.1)
n0 = 0
for g, c in zip(groups, cols):
    n1 = n0 + len(gorder[g])
    ax.add_patch(Rectangle((n0 - 0.5, len(genes) + 0.1), n1 - n0, 0.9, color=c, clip_on=False))
    ax.text((n0 + n1) / 2, len(genes) + 1.55, g, ha="center", fontsize=6.2, color=c,
            fontweight="bold")
    n0 = n1
ax.set_ylim(len(genes) - 0.5, -0.5)
cb = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.012)
cb.set_label("z-score", fontsize=6.2); cb.ax.tick_params(labelsize=5.8)
ttl(ax, "d", "Core-target expression in 220 biopsies ordered by group (row z-score; * hub)",
    pad=0.075)
save(fig, "Fig7_transcriptomic_validation")

print("main figures complete")
