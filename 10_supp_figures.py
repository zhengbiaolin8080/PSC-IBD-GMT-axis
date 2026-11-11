"""Step 10: supplementary figures S1-S4 and the graphical abstract."""
import os
import gzip

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
GEO = os.path.join(DATA, "GEO")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.titlesize": 8.0, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "legend.fontsize": 6.5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "mathtext.fontset": "dejavusans",
})
C_PSC, C_IBD, C_NC = "#B03A2E", "#DF8A34", "#4F86A6"
C_MIC, C_MET, C_TGT = "#7FAF7A", "#E8A33D", "#5B7FA6"
C_UP, C_DOWN, C_NS = "#B03A2E", "#2E6DA4", "#C9C9C9"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIG, name + ".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved", name)


def ttl(ax, letter, text, pad=0.014, fontsize=7.6):
    bb = ax.get_subplotspec().get_position(ax.figure)
    ax.figure.text(bb.x0, bb.y1 + pad, r"$\mathbf{%s}$   %s" % (letter, text),
                   fontsize=fontsize, va="bottom", ha="left")


# =====================================================================
# Figure S1 - GEO quality control and differential expression overview
# =====================================================================
meta = []
for line in gzip.open(os.path.join(GEO, "GSE230524_series_matrix.txt.gz"), "rt", errors="replace"):
    if line.startswith("!Sample_title"):
        meta = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
    if line.startswith("!Sample_characteristics_ch1") and "diagnosis:" in line and not meta:
        continue
diag = None
for line in gzip.open(os.path.join(GEO, "GSE230524_series_matrix.txt.gz"), "rt", errors="replace"):
    if line.startswith("!Sample_characteristics_ch1") and "diagnosis:" in line:
        diag = [x.split("diagnosis:")[1].strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
        break
md = pd.DataFrame({"sample": meta, "diagnosis": diag})
md["group"] = md["diagnosis"].map(lambda d: "PSC-IBD" if d in ("PSC-UC", "PSC-Crohn's")
                                  else "IBD" if d in ("UC", "Crohn's") else "NC" if d == "NC" else None)
md = md.dropna(subset=["group"])

raw = pd.read_csv(os.path.join(GEO, "GSE230524_count_table.txt.gz"), sep="\t", compression="gzip",
                  header=None, skiprows=1, names=["gene"] + [f"s{i}" for i in range(226)], dtype={0: str})
real = gzip.open(os.path.join(GEO, "GSE230524_count_table.txt.gz"), "rt").readline().rstrip("\n").split("\t")
raw.columns = ["gene"] + real
raw = raw.iloc[1:].copy()
for c in raw.columns[1:]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
ent = pd.read_csv(os.path.join(GEO, "entrez2symbol.csv"), dtype={"entrez": str})
raw = raw.merge(ent, left_on="gene", right_on="entrez", how="left")
raw = raw[raw["symbol"].notna()].drop_duplicates("symbol").set_index("symbol")
cnt = raw[[c for c in raw.columns if c in set(md["sample"])]].fillna(0).astype(int)
md = md.set_index("sample").reindex(cnt.columns)
md.index.name = "sample"
md = md.reset_index()
cpm = cnt.div(cnt.sum(axis=0), axis=1) * 1e6
keep = (cpm >= 1).sum(axis=1) >= 0.2 * cnt.shape[1]
logcpm = np.log2(cpm[keep] + 1)

fig = plt.figure(figsize=(7.09, 5.2))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42)

ax = fig.add_subplot(gs[0, 0])
groups = ["NC", "IBD", "PSC-IBD"]
cols = [C_NC, C_IBD, C_PSC]
lib = np.log10(cnt.sum(axis=0) / 1e6 + 0.01)
data = [lib[md.set_index("sample").loc[cnt.columns, "group"] == g].values for g in groups]
bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                medianprops=dict(color="black", lw=1.0))
for p, c in zip(bp["boxes"], cols):
    p.set_facecolor(c); p.set_alpha(0.45); p.set_edgecolor("#333333")
rng = np.random.default_rng(3)
for i, (d, c) in enumerate(zip(data, cols)):
    ax.scatter(i + 1 + rng.normal(0, 0.05, len(d)), d, s=3, color=c, alpha=0.65, lw=0)
ax.set_xticks(range(1, 4)); ax.set_xticklabels(groups)
ax.set_ylabel(r"$\log_{10}$ library size (million reads)")
ax.spines[["top", "right"]].set_visible(False)
ttl(ax, "a", "Sequencing depth")

ax = fig.add_subplot(gs[0, 1])
topg = logcpm.var(axis=1).nlargest(2000).index
X = logcpm.loc[topg].T.values
X = X - X.mean(axis=0)
u, s, vt = np.linalg.svd(X, full_matrices=False)
pc = u[:, :2] * s[:2]
ve = (s ** 2 / np.sum(s ** 2))[:2] * 100
for g, c in zip(groups, cols):
    m = (md["group"] == g).values
    ax.scatter(pc[m, 0], pc[m, 1], s=7, color=c, alpha=0.75, lw=0, label=g)
ax.set_xlabel(f"PC1 ({ve[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ve[1]:.1f}%)")
ax.legend(loc="upper right", fontsize=5.8, handletextpad=0.2)
ax.spines[["top", "right"]].set_visible(False)
ttl(ax, "b", "PCA (2,000 variable genes)")

ax = fig.add_subplot(gs[1, 0])
de = pd.read_csv(os.path.join(RES, "geo_de_PSCIBD_vs_IBD.csv")).set_index("gene")
d = de.dropna(subset=["padj"]).copy()
d["nlp"] = -np.log10(d["pvalue"].clip(lower=1e-300))
sig = (d["padj"] < 0.05) & (d["log2FoldChange"].abs() > 0.5)
ax.scatter(d.loc[~sig, "log2FoldChange"], d.loc[~sig, "nlp"], s=1.2, c=C_NS, alpha=0.5, lw=0)
ax.scatter(d.loc[sig & (d["log2FoldChange"] > 0), "log2FoldChange"],
           d.loc[sig & (d["log2FoldChange"] > 0), "nlp"], s=1.7, c=C_UP, alpha=0.8, lw=0)
ax.scatter(d.loc[sig & (d["log2FoldChange"] < 0), "log2FoldChange"],
           d.loc[sig & (d["log2FoldChange"] < 0), "nlp"], s=1.7, c=C_DOWN, alpha=0.8, lw=0)
for g, dx, dy in []:
    pass
ax.axhline(-np.log10(0.05), ls=":", lw=0.6, color="#777777")
ax.axvline(0.5, ls=":", lw=0.6, color="#777777"); ax.axvline(-0.5, ls=":", lw=0.6, color="#777777")
ax.set_xlabel(r"$\log_2$ FC (PSC-IBD vs IBD)"); ax.set_ylabel(r"$-\log_{10}$ $P$")
ax.set_xlim(-2.5, 2.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_UP, markersize=3.4, label="up"),
                   Line2D([], [], marker="o", ls="", color=C_DOWN, markersize=3.4, label="down"),
                   Line2D([], [], marker="o", ls="", color=C_NS, markersize=3.4, label="n.s.")],
          loc="upper left", fontsize=5.6, handletextpad=0.2)
ttl(ax, "c", f"Volcano, PSC-IBD vs IBD ({int(sig.sum())} DE genes)")

ax = fig.add_subplot(gs[1, 1])
summ = []
for tag, lab in [("PSCIBD_vs_IBD", "PSC-IBD\nvs IBD"), ("PSCIBD_vs_NC", "PSC-IBD\nvs NC"),
                 ("IBD_vs_NC", "IBD\nvs NC")]:
    r = pd.read_csv(os.path.join(RES, f"geo_de_{tag}.csv")).dropna(subset=["padj"])
    s2 = (r["padj"] < 0.05) & (r["log2FoldChange"].abs() > 0.5)
    summ.append({"contrast": lab,
                 "Up": int((s2 & (r["log2FoldChange"] > 0)).sum()),
                 "Down": int((s2 & (r["log2FoldChange"] < 0)).sum())})
sm = pd.DataFrame(summ)
x = np.arange(len(sm))
ax.bar(x - 0.19, sm["Up"], width=0.36, color=C_UP, label="Up")
ax.bar(x + 0.19, sm["Down"], width=0.36, color=C_DOWN, label="Down")
for i, (u, dn) in enumerate(zip(sm["Up"], sm["Down"])):
    ax.text(i - 0.19, u + 60, f"{u}", ha="center", fontsize=5.8)
    ax.text(i + 0.19, dn + 60, f"{dn}", ha="center", fontsize=5.8)
ax.set_xticks(x); ax.set_xticklabels(sm["contrast"], fontsize=6.5)
ax.set_ylabel("DE genes (FDR < 0.05, |FC| > 0.5)")
ax.set_ylim(0, sm[["Up", "Down"]].values.max() * 1.18)
ax.legend(fontsize=6.0, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ttl(ax, "d", "Differential expression yield")
save(fig, "FigS1_geo_qc_and_de")

# =====================================================================
# Figure S2 - full tripartite gut microbiota-metabolite-target network
# =====================================================================
nodes = pd.read_csv(os.path.join(RES, "gmt_nodes.csv"))
edges = pd.read_csv(os.path.join(RES, "gmt_edges.csv"))
hub = pd.read_csv(os.path.join(RES, "hub_genes.csv"))["hub_gene"].tolist()
G = nx.Graph()
for _, r in edges.iterrows():
    G.add_edge(r["source"], r["target"])
types = dict(zip(nodes["id"], nodes["type"]))
pos = {}
mic = [n for n in G.nodes if types.get(n) == "microbiota"]
met = [n for n in G.nodes if types.get(n) == "metabolite"]
tgt = [n for n in G.nodes if types.get(n) == "target"]
for i, n in enumerate(sorted(mic)):
    pos[n] = (-1.0, (i / max(len(mic) - 1, 1) - 0.5) * 2.1)
for i, n in enumerate(sorted(met)):
    pos[n] = (0.0, (0.5 - i / max(len(met) - 1, 1)) * 2.1)
for i, n in enumerate(sorted(tgt)):
    pos[n] = (1.0, (0.5 - i / max(len(tgt) - 1, 1)) * 2.0)

fig, ax = plt.subplots(figsize=(7.09, 7.6))
for u, v in G.edges:
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#E4E4E4", lw=0.22, zorder=1)
for n in G.nodes:
    t = types.get(n)
    c = {"microbiota": C_MIC, "metabolite": C_MET, "target": C_TGT}[t]
    s = 8 if t == "microbiota" else 22 if t == "metabolite" else 78
    ax.scatter(*pos[n], s=s, c=c, edgecolor="white", lw=0.3, zorder=3)
for n in tgt:
    ax.text(1.07, pos[n][1], n + ("*" if n in hub else ""), fontsize=6.2, va="center",
            fontweight="bold" if n in hub else "normal")
for n in met:
    ax.text(-0.05, pos[n][1], n, fontsize=4.3, va="center", ha="right")
ax.text(-1.0, 1.13, f"Microbiota\n(n={len(mic)})", ha="center", fontsize=7, color=C_MIC,
        fontweight="bold", linespacing=1.3)
ax.text(0.0, 1.13, f"Metabolites\n(n={len(met)})", ha="center", fontsize=7, color=C_MET,
        fontweight="bold", linespacing=1.3)
ax.text(1.0, 1.13, f"Targets\n(n={len(tgt)})", ha="center", fontsize=7, color=C_TGT,
        fontweight="bold", linespacing=1.3)
ax.set_xlim(-1.32, 1.30); ax.set_ylim(-1.12, 1.22); ax.axis("off")
ax.set_title("Gut microbiota-metabolite-target network (all bridging metabolites)", fontsize=8)
save(fig, "FigS2_full_gmt_network")

# =====================================================================
# Figure S3 - physicochemical and drug-likeness space
# =====================================================================
lip = pd.read_csv(os.path.join(RES, "adme_lipinski.csv"))
panel = pd.read_csv(os.path.join(DATA, "ligand_smiles.csv"))["name"].tolist()
fig = plt.figure(figsize=(7.09, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.40)
ax = fig.add_subplot(gs[0, 0])
ax.scatter(lip.loc[~lip["Lipinski_pass"], "MW"], lip.loc[~lip["Lipinski_pass"], "LogP"],
           s=12, c="#C9C9C9", edgecolor="#999999", lw=0.25, label="violation")
pl = lip[lip["name"].isin(panel)]
ax.scatter(pl["MW"], pl["LogP"], s=22, c=C_MET, edgecolor="#7A5A1E", lw=0.5, label="docking panel")
for _, r in pl.iterrows():
    if r["MW"] > 400 or r["LogP"] > 3.4:
        ax.annotate(r["name"], (r["MW"], r["LogP"]), xytext=(3, 2), textcoords="offset points",
                    fontsize=4.6)
ax.axvline(500, ls=":", lw=0.7, color="#666666"); ax.axhline(5, ls=":", lw=0.7, color="#666666")
ax.text(505, -1.0, "MW 500", fontsize=5.4, rotation=90, color="#666666")
ax.set_xlabel("Molecular weight (Da)"); ax.set_ylabel("cLogP")
ax.legend(loc="upper left", fontsize=5.8)
ax.spines[["top", "right"]].set_visible(False)
ttl(ax, "a", "Chemical space of 618 metabolites")

ax = fig.add_subplot(gs[0, 1])
props = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB"]
z = (pl[props] - pl[props].mean()) / pl[props].std(ddof=0)
z.index = pl["name"].values
im = ax.imshow(z.values, cmap="PuOr_r", aspect="auto", vmin=-2, vmax=2)
ax.set_xticks(range(len(props))); ax.set_xticklabels(props, fontsize=6.4)
ax.set_yticks(range(len(z))); ax.set_yticklabels(z.index, fontsize=5.6)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, f"{pl[props].values[i, j]:.1f}", ha="center", va="center", fontsize=4.4,
                color="white" if abs(z.values[i, j]) > 1.5 else "#222222")
ax.set_xticks(np.arange(-.5, len(props), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(z), 1), minor=True)
ax.grid(which="minor", color="white", lw=0.8); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("row z-score", fontsize=6.2); cb.ax.tick_params(labelsize=5.6)
ttl(ax, "b", "Properties of the docking panel")
save(fig, "FigS3_physicochemical_space")

# =====================================================================
# Figure S4 - top docking complex: ligand chemistry and residue contacts
# =====================================================================
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

pose = os.path.join(ROOT, "docking", "poses", "NR1H4__Taurochenodeoxycholic_acid.pdbqt")
rec = os.path.join(ROOT, "docking", "receptors", "NR1H4_1OSV_clean.pdb")
ligatoms = []
for line in open(pose):
    if line.startswith(("ATOM", "HETATM")):
        ligatoms.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
L = np.array(ligatoms)

res_min = {}
for line in open(rec):
    if line.startswith("ATOM"):
        p = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        dmin = float(np.min(np.linalg.norm(L - p, axis=1)))
        key = (line[21], line[22:26].strip(), line[17:20].strip())
        if dmin < res_min.get(key, 99):
            res_min[key] = dmin
contacts = sorted([(f"{k[2].title()}{int(k[1])}", v) for k, v in res_min.items() if v < 4.5],
                  key=lambda x: x[1])
tab = pd.read_csv(os.path.join(RES, "docking_matrix.csv")).set_index("metabolite")
aff = float(tab.loc["Taurochenodeoxycholic acid", "NR1H4"])

fig = plt.figure(figsize=(7.09, 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.42)

ax = fig.add_subplot(gs[0, 0])
smiles = pd.read_csv(os.path.join(DATA, "ligand_smiles.csv")).set_index("name").loc[
    "Taurochenodeoxycholic acid", "isomeric_smiles"]
mol = Chem.MolFromSmiles(smiles)
d = rdMolDraw2D.MolDraw2DCairo(1100, 800)
o = d.drawOptions()
o.bondLineWidth = 2
o.padding = 0.06
o.addStereoAnnotation = True
rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
d.FinishDrawing()
ax.imshow(np.array(Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")))
ax.axis("off")
ttl(ax, "a", f"Taurochenodeoxycholic acid (Vina {aff:.1f} kcal/mol)")

ax = fig.add_subplot(gs[0, 1])
names = [c[0] for c in contacts][::-1]
dists = [c[1] for c in contacts][::-1]
cols_c = [C_PSC if x < 3.5 else C_MET if x < 4.0 else "#B9B9B9" for x in dists]
yy = np.arange(len(names))
ax.hlines(yy, 2.35, dists, color=cols_c, lw=1.0)
ax.scatter(dists, yy, s=16, c=cols_c, zorder=3, edgecolor="white", lw=0.3)
for y_, x_ in zip(yy, dists):
    ax.text(x_ + 0.04, y_, f"{x_:.2f}", va="center", fontsize=5.0, color="#444444")
ax.set_yticks(yy)
ax.set_yticklabels(names, fontsize=5.9)
ax.set_xlabel("Minimum ligand-residue distance (A)")
ax.set_xlim(2.35, 4.7)
ax.axvline(3.5, ls=":", lw=0.8, color="#555555")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_PSC, markersize=3.4, label="< 3.5 A"),
                   Line2D([], [], marker="o", ls="", color=C_MET, markersize=3.4, label="3.5-4.0 A"),
                   Line2D([], [], marker="o", ls="", color="#B9B9B9", markersize=3.4, label="> 4.0 A")],
          loc="upper right", fontsize=5.2, handletextpad=0.2)
ttl(ax, "b", "NR1H4 ligand-binding pocket contacts")
save(fig, "FigS4_docking_pose")

# =====================================================================
# Graphical abstract
# =====================================================================
fig, ax = plt.subplots(figsize=(7.09, 3.55))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(1.5, 96, "PSC-IBD overlap syndrome: a gut microbial metabolite-target axis",
        fontsize=8.6, fontweight="bold", color="#1A1A1A", va="top")
ax.text(1.5, 89.5, "Integration of gutMGene, ChEMBL, Open Targets, STRING, docking and public transcriptomes",
        fontsize=6.0, color="#666666", va="top")

cards = [
    (1.5, 22.5, "GUT MICROBIOTA", C_MIC,
     ["227 linked taxa", "SCFA producers depleted", "bile-acid converters up",
      "Parabacteroides spp."]),
    (27.0, 22.5, "METABOLOME", C_MET,
     ["618 structures screened", "49 bridging metabolites", "butyrate score 35.6",
      "TCDCA / LCA / DCA"]),
    (52.5, 22.5, "TARGET AXIS", C_TGT,
     ["899 metabolite targets", "103 PSC x IBD genes", "14 core targets",
      "hubs: CCL2 IL10 IL2", "PPARG TLR4"]),
    (78.0, 20.5, "VALIDATION", C_PSC,
     ["GSE230524 (n=220)", "core module score up", "P = 0.014 vs IBD",
      "NFKB1 / PPARG up"]),
]
y0, h = 39.0, 45.0
for x, w, head, col, lines in cards:
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                linewidth=1.0, edgecolor=col, facecolor=col + "14"))
    ax.add_patch(FancyBboxPatch((x, y0 + h - 9.5), w, 9.5,
                                boxstyle="round,pad=0.25,rounding_size=1.2",
                                linewidth=0, facecolor=col, alpha=0.92))
    ax.text(x + w / 2, y0 + h - 4.8, head, ha="center", va="center", fontsize=6.4,
            color="white", fontweight="bold")
    for k, ln in enumerate(lines):
        ax.text(x + 1.6, y0 + h - 15.5 - 7.0 * k, ln, ha="left", va="center",
                fontsize=5.6, color="#1A1A1A")

for xa, xb in [(24.4, 26.6), (49.9, 52.1), (75.4, 77.6)]:
    ax.annotate("", xy=(xb, y0 + h / 2), xytext=(xa, y0 + h / 2),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#8C8C8C",
                                shrinkA=0, shrinkB=0))

ax.add_patch(FancyBboxPatch((1.5, 8.5), 97.0, 21.0,
                            boxstyle="round,pad=0.3,rounding_size=1.4",
                            linewidth=0.9, edgecolor="#6C4C8C", facecolor="#6C4C8C12"))
ax.text(3.0, 25.0, "Convergent axis", fontsize=6.4, fontweight="bold", color="#6C4C8C",
        va="center")
ax.text(3.0, 18.4, "bile acids  ->  NR1H4 / PPARG  ->  TLR4-NF-kB signalling",
        fontsize=7.2, color="#1A1A1A", va="center")
ax.text(3.0, 12.2, "TCDCA-NR1H4 -10.6 kcal/mol   |   TCDCA-PPARG -10.3   |   TCDCA-TLR4 -9.0 (AutoDock Vina 1.2.7)",
        fontsize=5.8, color="#444444", va="center")
save(fig, "Fig_Graphical_Abstract")

print("supplementary figures complete")
