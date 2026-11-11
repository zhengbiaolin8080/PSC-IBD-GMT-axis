"""Step 02: integrate gutMGene curated and ChEMBL-similarity predicted
metabolite->target evidence, then intersect with the PSC-IBD overlap gene set.
"""
import os
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

met = pd.read_csv(os.path.join(DATA, "metabolites.csv"))
met = met[met["smiles"].notna()].reset_index(drop=True)
print("metabolites with SMILES:", len(met))

sim_map = json.load(open(os.path.join(DATA, "chembl_similar.json")))
acts = json.load(open(os.path.join(DATA, "chembl_activities.json")))
tmap = json.load(open(os.path.join(DATA, "chembl_targets.json")))

mol2gene = {}
for a in acts:
    g = (tmap.get(a["target_chembl_id"]) or {}).get("gene")
    if g:
        mol2gene.setdefault(a["molecule_chembl_id"], set()).add(g)

rows = []
for name, hits in sim_map.items():
    genes = set()
    for h in hits:
        genes |= mol2gene.get(h["chembl_id"], set())
    for g in genes:
        rows.append({"metabolite": name, "gene": g, "source": "ChEMBL_similarity"})
pred = pd.DataFrame(rows)

s2 = pd.read_csv(os.path.join(DATA, "Table_S2.csv"))
cur = s2[["Metabolite", "Gene"]].dropna()
cur.columns = ["metabolite", "gene"]
cur["metabolite_norm"] = cur["metabolite"].astype(str).str.strip().str.lower()
met_names = {n.lower(): n for n in met["name"]}
cur = cur[cur["metabolite_norm"].isin(met_names.keys())]
cur["metabolite"] = cur["metabolite_norm"].map(met_names)
cur = cur[["metabolite", "gene"]].copy()
cur["source"] = "gutMGene_curated"

allmt = pd.concat([pred, cur], ignore_index=True).drop_duplicates(["metabolite", "gene", "source"])
# "Caproic Acid" and "Hexanoate" are the same compound (hexanoic acid);
# record them once under a single canonical name.
_ALIAS = {"caproic acid", "hexanoate"}
_hit = allmt["metabolite"].str.strip().str.lower().isin(_ALIAS)
allmt.loc[_hit, "metabolite"] = "Hexanoic acid"
allmt = allmt.drop_duplicates(["metabolite", "gene", "source"])
allmt.to_csv(os.path.join(RES, "metabolite_targets.csv"), index=False)

uni_pred, uni_cur = set(pred["gene"]), set(cur["gene"])
print("ChEMBL predicted genes:", len(uni_pred), " curated genes:", len(uni_cur),
      " union:", len(uni_pred | uni_cur), " dual:", len(uni_pred & uni_cur))

cnt = allmt.groupby("metabolite")["gene"].nunique().sort_values(ascending=False)
cnt.rename("n_targets").to_csv(os.path.join(RES, "targets_per_metabolite.csv"))

overlap = pd.read_csv(os.path.join(RES, "psc_ibd_overlap_genes.csv"))
ds = set(overlap["symbol"])
core = sorted((uni_pred | uni_cur) & ds)
dual = sorted((uni_pred & uni_cur) & ds)
pd.DataFrame({"core_target": core}).to_csv(os.path.join(RES, "core_targets.csv"), index=False)
pd.DataFrame({"dual_supported_core": dual}).to_csv(os.path.join(RES, "core_targets_dual.csv"), index=False)

# evidence detail for core targets
det = allmt[allmt["gene"].isin(core)].copy()
det.to_csv(os.path.join(RES, "core_target_evidence.csv"), index=False)
print("PSC-IBD genes:", len(ds))
print("CORE TARGETS (metabolite targets ∩ PSC-IBD genes):", len(core))
print("dual-supported core:", len(dual))
print(core)
