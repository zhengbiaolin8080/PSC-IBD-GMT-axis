"""Step 08: external validation in GSE230524 (right-colon biopsies).

Groups: PSC-IBD (PSC-UC + PSC-Crohn's) vs IBD (UC + Crohn's) vs non-IBD controls.
Differential expression with PyDESeq2 (Wald test, BH FDR) and a targeted
assessment of the 34 core targets plus the five consensus hubs.
"""
import os
import re
import gzip

import numpy as np
import pandas as pd
import requests
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "GEO")
RES = os.path.join(ROOT, "results")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# ---- metadata -------------------------------------------------------------
mpath = os.path.join(DATA, "GSE230524_series_matrix.txt.gz")
titles, diag = None, None
for line in gzip.open(mpath, "rt", errors="replace"):
    if line.startswith("!Sample_title"):
        titles = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
    if line.startswith("!Sample_characteristics_ch1") and diag is None and "diagnosis:" in line:
        diag = [x.split("diagnosis:")[1].strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
meta = pd.DataFrame({"sample": titles, "diagnosis": diag})
print(meta["diagnosis"].value_counts().to_string())

def grp(d):
    if d in ("PSC-UC", "PSC-Crohn's"):
        return "PSC-IBD"
    if d in ("UC", "Crohn's"):
        return "IBD"
    if d == "NC":
        return "NC"
    return None

meta["group"] = meta["diagnosis"].map(grp)
meta = meta[meta["group"].notna()].reset_index(drop=True)
print(meta["group"].value_counts().to_string())

# ---- counts ---------------------------------------------------------------
cpath = os.path.join(DATA, "GSE230524_count_table.txt.gz")
raw = pd.read_csv(cpath, sep="\t", compression="gzip", header=None, skiprows=1,
                  names=["gene"] + [f"s{i}" for i in range(226)], dtype={0: str})
real_names = gzip.open(cpath, "rt").readline().rstrip("\n").split("\t")
raw.columns = ["gene"] + real_names
raw = raw.iloc[1:].copy()
raw["gene"] = raw["gene"].astype(str)
for c in raw.columns[1:]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw.dropna(subset=raw.columns[1:].tolist(), how="all")
print("count matrix:", raw.shape)

# Entrez -> symbol (NCBI gene esummary; mygene batch endpoint unreliable)
mapf = os.path.join(DATA, "entrez2symbol.csv")
if os.path.exists(mapf):
    ent = pd.read_csv(mapf, dtype={"entrez": str})
else:
    ids = raw["gene"].tolist()
    recs = []
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        r = requests.get(EUTILS + "esummary.fcgi",
                         params={"db": "gene", "id": ",".join(chunk), "retmode": "json"},
                         timeout=120).json()["result"]
        for q in chunk:
            d = r.get(q, {})
            recs.append({"entrez": q, "symbol": d.get("name") if isinstance(d, dict) else None})
    ent = pd.DataFrame(recs)
    ent.to_csv(mapf, index=False)
print("entrez mapped:", ent["symbol"].notna().sum(), "/", len(ent))
raw = raw.merge(ent, left_on="gene", right_on="entrez", how="left")
raw = raw[raw["symbol"].notna()].drop_duplicates("symbol").set_index("symbol")
cnt = raw[[c for c in raw.columns if c in set(meta["sample"])]].astype(int)
print("genes:", cnt.shape[0])
meta = meta.set_index("sample").reindex(cnt.columns)
meta.index.name = "sample"
meta = meta.dropna(subset=["group"]).reset_index()
cnt = cnt[meta["sample"]]
print("aligned samples:", cnt.shape[1], "| groups:", meta["group"].value_counts().to_dict())

lib = cnt.sum(axis=0)
cpm = cnt.div(lib, axis=1) * 1e6
keep = (cpm >= 1).sum(axis=1) >= 0.2 * cnt.shape[1]
cntf = cnt[keep].astype(int)
print("expressed genes:", cntf.shape)
cpmf = cpm[keep]
logcpm = np.log2(cpmf + 1)

# ---- DESeq2 ---------------------------------------------------------------
de_results = {}
try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    ok_deseq = True
except Exception as e:
    print("PyDESeq2 unavailable:", e)
    ok_deseq = False

def run_deseq(sub_meta, ref, alt, tag):
    counts = cntf[sub_meta["sample"]].T
    counts.index = sub_meta["sample"].values
    md = pd.DataFrame({"group": sub_meta["group"].values}, index=counts.index)
    md["group"] = pd.Categorical(md["group"], categories=[ref, alt])
    dds = DeseqDataSet(counts=counts, metadata=md, design="~group", refit_cooks=True, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["group", alt, ref], quiet=True)
    st.summary()
    r = st.results_df.copy()
    r.index.name = "gene"
    r.to_csv(os.path.join(RES, f"geo_de_{tag}.csv"))
    print(tag, "DE genes FDR<0.05:", int((r["padj"] < 0.05).sum()),
          "| up:", int(((r["padj"] < 0.05) & (r["log2FoldChange"] > 1)).sum()),
          "| down:", int(((r["padj"] < 0.05) & (r["log2FoldChange"] < -1)).sum()))
    return r

if ok_deseq:
    for tag, ref, alt in [("PSCIBD_vs_IBD", "IBD", "PSC-IBD"), ("PSCIBD_vs_NC", "NC", "PSC-IBD"),
                          ("IBD_vs_NC", "NC", "IBD")]:
        sub = meta[meta["group"].isin([ref, alt])].reset_index(drop=True)
        de_results[tag] = run_deseq(sub, ref, alt, tag)
else:
    for tag, ref, alt in [("PSCIBD_vs_IBD", "IBD", "PSC-IBD"), ("PSCIBD_vs_NC", "NC", "PSC-IBD"),
                          ("IBD_vs_NC", "NC", "IBD")]:
        a = logcpm.loc[:, meta.loc[meta["group"] == alt, "sample"]]
        b = logcpm.loc[:, meta.loc[meta["group"] == ref, "sample"]]
        stat, p = stats.ttest_ind(a.values, b.values, axis=1, equal_var=False)
        lfc = a.mean(axis=1) - b.mean(axis=1)
        padj = stats.false_discovery_control(p)
        r = pd.DataFrame({"log2FoldChange": lfc, "pvalue": p, "padj": padj}, index=logcpm.index)
        r.index.name = "gene"
        r.to_csv(os.path.join(RES, f"geo_de_{tag}.csv"))
        de_results[tag] = r
        print(tag, "Welch t-test; FDR<0.05:", int((padj < 0.05).sum()))

# ---- targeted validation --------------------------------------------------
core = pd.read_csv(os.path.join(RES, "core_targets.csv"))["core_target"].tolist()
hub = pd.read_csv(os.path.join(RES, "hub_genes.csv"))["hub_gene"].tolist()
core_present = [g for g in core if g in logcpm.index]
missing = [g for g in core if g not in logcpm.index]
print(f"core targets quantified: {len(core_present)}/{len(core)}; missing: {missing}")

rows = []
for tag, r in de_results.items():
    for g in core_present:
        if g in r.index:
            rows.append({"contrast": tag, "gene": g,
                         "log2FC": float(r.loc[g, "log2FoldChange"]),
                         "pvalue": float(r.loc[g, "pvalue"]),
                         "padj": float(r.loc[g, "padj"]),
                         "hub": g in hub})
tgt = pd.DataFrame(rows)
tgt.to_csv(os.path.join(RES, "geo_core_target_DE.csv"), index=False)
print("\nPSC-IBD vs IBD core targets (sorted by padj):")
piv = tgt[tgt["contrast"] == "PSCIBD_vs_IBD"].sort_values("padj")
print(piv.head(20).round(4).to_string(index=False))
print("hubs:", {g: round(float(piv[piv.gene == g]['log2FC'].iloc[0]), 2) if g in set(piv.gene) else None for g in hub})

# module score (mean scaled log2CPM across core targets)
z = logcpm.loc[core_present].apply(lambda s: (s - s.mean()) / s.std(ddof=1), axis=1)
score = z.mean(axis=0)
sm = meta.set_index("sample").loc[score.index].copy()
sm["module_score"] = score.values
sm.index.name = "sample"
sm.reset_index()[["sample", "diagnosis", "group", "module_score"]].to_csv(
    os.path.join(RES, "geo_module_score.csv"), index=False)
kw = stats.kruskal(*[g["module_score"].values for _, g in sm.groupby("group")])
print("\nmodule score by group:")
print(sm.groupby("group")["module_score"].agg(["count", "mean", "std"]).round(3).to_string())
print("Kruskal-Wallis p =", kw.pvalue)
for g1, g2 in [("PSC-IBD", "IBD"), ("PSC-IBD", "NC"), ("IBD", "NC")]:
    u = stats.mannwhitneyu(sm[sm.group == g1]["module_score"], sm[sm.group == g2]["module_score"])
    print(f"  {g1} vs {g2}: p={u.pvalue:.2e}")

# individual hub expression table
hubtab = logcpm.loc[[g for g in hub if g in logcpm.index]].T
hubtab["group"] = meta.set_index("sample").loc[hubtab.index, "group"]
hubtab.to_csv(os.path.join(RES, "geo_hub_expression.csv"))

coretab = logcpm.loc[core_present].T
coretab["group"] = meta.set_index("sample").loc[coretab.index, "group"]
coretab["diagnosis"] = meta.set_index("sample").loc[coretab.index, "diagnosis"]
coretab.to_csv(os.path.join(RES, "geo_core_expression.csv"))

# Fisher test: are core targets over-represented among PSC-IBD vs IBD DE genes?
r0 = de_results["PSCIBD_vs_IBD"]
sig = set(r0.index[(r0["padj"] < 0.05) & (r0["log2FoldChange"].abs() > 0.5)])
tested = set(r0.dropna(subset=["padj"]).index)
a = len(sig & set(core_present)); b = len(sig) - a
c = len(set(core_present)) - a; d = len(tested) - a - b - c
odds, pf = stats.fisher_exact([[a, b], [c, d]])
print(f"\nFisher: core targets among DE (FDR<0.05, |LFC|>0.5): OR={odds:.2f}, p={pf:.3f}, "
      f"{a}/{len(core_present)} vs {len(sig)}/{len(tested)}")
pd.DataFrame([{"n_tested": len(tested), "n_DE": len(sig), "core_in_DE": a,
               "core_total": len(core_present), "odds_ratio": odds, "fisher_p": pf}]).to_csv(
    os.path.join(RES, "geo_core_enrichment_test.csv"), index=False)
for g in [x for x in hub if x in logcpm.index]:
    vals = [hubtab.loc[hubtab.group == k, g].values for k in ["PSC-IBD", "IBD", "NC"]]
    p = stats.kruskal(*vals).pvalue
    print(f"  {g}: PSC-IBD={vals[0].mean():.2f} IBD={vals[1].mean():.2f} NC={vals[2].mean():.2f} KW p={p:.2e}")

# ---- transcriptome-wide context ------------------------------------------
r0 = de_results["PSCIBD_vs_IBD"]
valid = r0.dropna(subset=["padj"])
rank_pct = valid["pvalue"].rank(pct=True)
enr = [rank_pct.get(g, np.nan) for g in core_present]
print("\ncore targets median p-value percentile in PSC-IBD vs IBD:",
      round(float(np.nanmedian(enr)) * 100, 1), "%")
print("fraction of core targets in top 10% of DE ranking:",
      round(float(np.mean(np.array(enr) < 0.10)), 3),
      "| background expectation 0.10")
pd.DataFrame({"gene": core_present, "pvalue": [r0["pvalue"].get(g, np.nan) for g in core_present],
              "p_percentile": enr}).to_csv(os.path.join(RES, "geo_core_target_rank.csv"), index=False)

# ---- inflammatory context panel ------------------------------------------
panel = ["IL17A", "IL17F", "FOXP3", "IFNG", "CXCL9", "CXCL10", "CD19", "IGHG1",
         "IL23A", "IL1B", "TNF", "NOS2"]
panel = [g for g in panel if g in logcpm.index]
prow = []
for g in panel:
    vals = {k: logcpm.loc[g, meta.loc[meta["group"] == k, "sample"]].values
            for k in ["PSC-IBD", "IBD", "NC"]}
    kw = stats.kruskal(*vals.values())
    u1 = stats.mannwhitneyu(vals["PSC-IBD"], vals["IBD"])
    u2 = stats.mannwhitneyu(vals["PSC-IBD"], vals["NC"])
    prow.append({"gene": g, "PSCIBD_mean": vals["PSC-IBD"].mean(), "IBD_mean": vals["IBD"].mean(),
                 "NC_mean": vals["NC"].mean(), "KW_p": kw.pvalue,
                 "p_vs_IBD": u1.pvalue, "p_vs_NC": u2.pvalue,
                 "log2FC_vs_IBD": vals["PSC-IBD"].mean() - vals["IBD"].mean()})
ctx = pd.DataFrame(prow)
ctx.to_csv(os.path.join(RES, "geo_context_panel.csv"), index=False)
print("\nInflammatory context panel (log2 CPM):")
print(ctx.round(3).to_string(index=False))
pd.DataFrame(logcpm.loc[panel].T).assign(
    group=meta.set_index("sample").loc[logcpm.columns, "group"]).to_csv(
    os.path.join(RES, "geo_context_expression.csv"))
