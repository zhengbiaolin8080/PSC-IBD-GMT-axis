"""Step 05: microbiota-metabolite-target (G-M-T) tripartite network."""
import os
import pandas as pd
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

s3 = pd.read_csv(os.path.join(DATA, "Table_S3.csv"))
mt = pd.read_csv(os.path.join(RES, "metabolite_targets.csv"))
core = set(pd.read_csv(os.path.join(RES, "core_targets.csv"))["core_target"])
hub = set(pd.read_csv(os.path.join(RES, "hub_genes.csv"))["hub_gene"])

gm = s3[["Gut Microbiota", "Metabolite", "Rank"]].dropna(subset=["Gut Microbiota", "Metabolite"])
gm["microbe"] = gm["Gut Microbiota"].astype(str).str.strip()
gm["metabolite"] = gm["Metabolite"].astype(str).str.strip()
gm["rank"] = gm["Rank"].astype(str).str.strip().str.lower()
gm = gm[["microbe", "metabolite", "rank"]].drop_duplicates()

# merge synonymous records of hexanoic acid into one node
_ALIAS = {"caproic acid", "hexanoate"}
_am = gm["metabolite"].str.strip().str.lower().isin(_ALIAS)
gm.loc[_am, "metabolite"] = "Hexanoic acid"
gm = gm.drop_duplicates()

# unify source-specific spellings with the metabolite-target table (case-insensitive)
_ref = {str(n).strip().lower(): str(n).strip() for n in mt["metabolite"].unique()}
gm["metabolite"] = gm["metabolite"].map(lambda s: _ref.get(str(s).strip().lower(), str(s).strip()))
gm = gm.drop_duplicates()

mt2 = mt[mt["gene"].isin(core)].copy()
print("M-T edges to core targets:", len(mt2))
gm["met_norm"] = gm["metabolite"].str.lower()
mt2["met_norm"] = mt2["metabolite"].str.lower()
shared = set(gm["met_norm"]) & set(mt2["met_norm"])
print("metabolites bridging G-M and M-T:", len(shared))

gm_e = gm[gm["met_norm"].isin(shared)][["microbe", "metabolite", "rank"]].drop_duplicates()
mt_e = mt2[mt2["met_norm"].isin(shared)][["metabolite", "met_norm", "gene", "source"]].drop_duplicates()
canon = gm[gm["met_norm"].isin(shared)].drop_duplicates("met_norm").set_index("met_norm")["metabolite"].to_dict()
mt_e["metabolite"] = mt_e["met_norm"].map(canon)

G = nx.Graph()
for _, r in gm_e.iterrows():
    G.add_edge(("M", r["microbe"]), ("m", r["metabolite"]))
for _, r in mt_e.iterrows():
    G.add_edge(("m", r["metabolite"]), ("T", r["gene"]))
print("G-M-T graph:", G.number_of_nodes(), "nodes", G.number_of_edges(), "edges")

hub_hits = mt_e[mt_e["gene"].isin(hub)].groupby("metabolite")["gene"].agg(
    ["nunique", lambda s: ";".join(sorted(s))])
hub_hits.columns = ["n_hub", "hub_genes"]
tgt_cnt = mt_e.groupby("metabolite")["gene"].nunique().rename("n_core_targets")
mic_cnt = gm_e.groupby("metabolite")["microbe"].nunique().rename("n_microbes")
met_imp = pd.concat([mic_cnt, tgt_cnt, hub_hits], axis=1).fillna(0)
met_imp["n_hub"] = met_imp["n_hub"].astype(int)
met_imp["n_core_targets"] = met_imp["n_core_targets"].astype(int)
met_imp["score"] = met_imp["n_hub"] * 3 + met_imp["n_core_targets"] + 0.2 * met_imp["n_microbes"]
met_imp = met_imp.sort_values("score", ascending=False)
met_imp.index.name = "metabolite"
met_imp.to_csv(os.path.join(RES, "gmt_metabolite_importance.csv"))
print(met_imp.head(25).to_string())

mic_imp = gm_e.groupby("microbe")["metabolite"].nunique().rename("n_metabolites").sort_values(ascending=False)
mic_imp.index.name = "microbe"
mic_imp.to_csv(os.path.join(RES, "gmt_microbe_importance.csv"))

nodes = [{"id": n_[1], "type": {"M": "microbiota", "m": "metabolite", "T": "target"}[n_[0]]}
         for n_, d in G.nodes(data=True)]
edges = [{"source": a[1], "target": b[1]} for a, b in G.edges()]
pd.DataFrame(nodes).to_csv(os.path.join(RES, "gmt_nodes.csv"), index=False)
pd.DataFrame(edges).to_csv(os.path.join(RES, "gmt_edges.csv"), index=False)
print("saved nodes/edges:", len(nodes), len(edges))

# which core targets are reached by the top metabolites
print("\nTop-10 metabolite -> targets:")
for m in met_imp.head(10).index:
    tg = sorted(mt_e[mt_e["metabolite"] == m]["gene"].unique())
    print(f"  {m}: {tg}")
