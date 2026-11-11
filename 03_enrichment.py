"""Step 03: functional enrichment of the PSC-IBD core targets with g:Profiler."""
import os
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

core = pd.read_csv(os.path.join(RES, "core_targets.csv"))["core_target"].tolist()
print("core targets:", len(core), core)

url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
payload = {
    "organism": "hsapiens",
    "query": core,
    "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC"],
    "user_threshold": 0.05,
    "no_evidences": False,
}
r = requests.post(url, json=payload, timeout=240)
print("status", r.status_code)
res = r.json()["result"]
rows = []
for t in res:
    # g:Profiler returns "intersections" aligned to the submitted query order;
    # a non-empty evidence list means that query gene belongs to the term.
    ev = t.get("intersections", [])
    genes = [g for g, e in zip(core, ev) if e] if ev else []
    rows.append({
        "source": t["source"], "term_id": t["native"], "term_name": t["name"],
        "p_value": t["p_value"], "term_size": t["term_size"],
        "query_size": t["query_size"], "intersection_size": t["intersection_size"],
        "precision": t["precision"], "recall": t["recall"],
        "genes": ";".join(genes),
    })
df = pd.DataFrame(rows).sort_values("p_value")
df.to_csv(os.path.join(RES, "enrichment_gprofiler.csv"), index=False)
for src in ["GO:BP", "GO:CC", "GO:MF", "KEGG", "REAC"]:
    sub = df[df["source"] == src]
    print(f"\n### {src} ({len(sub)} sig) top8:")
    for _, x in sub.head(8).iterrows():
        print(f"  {x['term_id']}  {x['term_name'][:70]:70s}  p={x['p_value']:.2e}  n={x['intersection_size']}")
