"""Step 01: assemble the PSC-IBD overlap disease-gene set from Open Targets.

PSC  : Open Targets disease MONDO_0018646 ("sclerosing cholangitis", the ontology
       term that represents primary sclerosing cholangitis; its top associated
       targets are the canonical PSC GWAS loci FUT2, IRF5, BACH2, IL2RA, GPR35...).
IBD  : UC (MONDO_0005101), CD (MONDO_0005011) and IBD (MONDO_0005265) gene lists
       retrieved from Open Targets (snapshot carried in data/disease_genes_*.csv).
Overlap: PSC-associated genes that are also associated with at least one IBD
       phenotype = the genetic substrate of PSC-IBD overlap syndrome.
"""
import os
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def ot_query(query, variables=None):
    r = requests.post(OT_URL, json={"query": query, "variables": variables or {}}, timeout=120)
    r.raise_for_status()
    return r.json()


Q = """
query($id:String!,$size:Int!){
  disease(efoId:$id){
    id name
    associatedTargets(page:{index:0,size:$size}){
      count
      rows{ target{ approvedSymbol approvedName } score }
    }
  }
}
"""

# ---- PSC-associated genes -------------------------------------------------
snap = os.path.join(DATA, "psc_opentargets_targets.csv")
if os.path.exists(snap):
    psc = pd.read_csv(snap)
else:
    out = ot_query(Q, {"id": "MONDO_0018646", "size": 3000})["data"]["disease"]
    rows = [{"symbol": r["target"]["approvedSymbol"],
             "name": r["target"]["approvedName"],
             "score": r["score"],
             "disease": "PSC", "disease_id": out["id"]}
            for r in out["associatedTargets"]["rows"] if r["target"].get("approvedSymbol")]
    psc = pd.DataFrame(rows)
    psc.to_csv(snap, index=False)
print("PSC associated targets retrieved:", len(psc))
psc_sel = psc[psc["score"] >= 0.2].drop_duplicates("symbol")
print("PSC genes (score >= 0.2):", len(psc_sel))

# ---- IBD phenotype genes --------------------------------------------------
frames = []
for src, fn in [("IBD", "disease_genes_IBD.csv"), ("UC", "disease_genes_UC.csv"),
                ("CD", "disease_genes_CD.csv")]:
    d = pd.read_csv(os.path.join(DATA, fn))
    d = d[["symbol", "score"]].dropna()
    d["phenotype"] = src
    frames.append(d[d["score"] >= 0.2])
ibd = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "phenotype"])
ibd_genes = set(ibd["symbol"])
print("IBD phenotype genes (score >= 0.2):", len(ibd_genes))

# ---- overlap --------------------------------------------------------------
overlap = psc_sel[psc_sel["symbol"].isin(ibd_genes)].copy()
phen = ibd[ibd["symbol"].isin(overlap["symbol"])].groupby("symbol")["phenotype"].agg(lambda s: ";".join(sorted(set(s))))
overlap["ibd_phenotypes"] = overlap["symbol"].map(phen)
overlap = overlap.sort_values("score", ascending=False)
overlap.to_csv(os.path.join(RES, "psc_ibd_overlap_genes.csv"), index=False)
psc_sel[["symbol", "name", "score"]].to_csv(os.path.join(RES, "psc_genes.csv"), index=False)
ibd.drop_duplicates("symbol")[["symbol", "score"]].to_csv(os.path.join(RES, "ibd_genes.csv"), index=False)
print("PSC-IBD overlap genes:", len(overlap))
print(overlap.head(30).to_string(index=False))
