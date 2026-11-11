"""Step 04: STRING PPI network and cytoHubba-style hub-gene consensus."""
import os
import io
import time
import math
from collections import deque, Counter

import pandas as pd
import networkx as nx
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

core = pd.read_csv(os.path.join(RES, "core_targets.csv"))["core_target"].tolist()
print("core targets:", len(core))

STRING_BASE = "https://version-11-5.string-db.org/api/tsv"


def resolve_exact(genes):
    mapping = {}
    for g in genes:
        r = requests.post(STRING_BASE + "/resolveList",
                          data={"identifiers": g, "species": 9606, "limit": 10,
                                "caller_identity": "psc_ibd_axis_study"}, timeout=60)
        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        hit = df[df["preferredName"].str.upper() == g.upper()]
        if len(hit) == 0:
            hit = df.head(1)
            print(f"  WARNING: no exact match for {g}, using {hit.iloc[0]['preferredName']}")
        mapping[g] = hit.iloc[0]["stringId"]
        time.sleep(0.4)
    return mapping


sym2sid = resolve_exact(core)
print("resolved:", sym2sid)


def string_network(score):
    ids = "%0d".join(sym2sid[g] for g in core)
    r = requests.post(STRING_BASE + "/network",
                      data={"identifiers": ids, "species": 9606, "required_score": score,
                            "caller_identity": "psc_ibd_axis_study"}, timeout=180)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text), sep="\t")


chosen = None
for sc in [900, 700, 400]:
    net = string_network(sc)
    if len(net) == 0:
        print(f"STRING score>={sc}: no edges")
        continue
    e = net[["preferredName_A", "preferredName_B", "score"]]
    nodes = set(e["preferredName_A"]) | set(e["preferredName_B"])
    print(f"STRING score>={sc}: edges={len(e)} nodes={len(nodes)}")
    if chosen is None:
        chosen = (sc, e)
    if len(nodes) >= 0.7 * len(core) and len(e) >= 20:
        chosen = (sc, e)
        break

sc, edges = chosen
edges.columns = ["source", "target", "combined_score"]
edges.to_csv(os.path.join(RES, "ppi_string_edges.csv"), index=False)
print("chosen threshold:", sc)

G = nx.Graph()
for _, r in edges.iterrows():
    G.add_edge(r["source"], r["target"], weight=float(r["combined_score"]))
print("graph:", G.number_of_nodes(), "nodes", G.number_of_edges(), "edges")
# keep core nodes even if isolated
for g in core:
    if g not in G:
        G.add_node(g, weight=0)
lcc = max(nx.connected_components(G), key=len)
H = G.subgraph(lcc).copy()
print("LCC:", H.number_of_nodes(), "nodes")

deg = dict(H.degree())
def mnc(v):
    nbrs = list(H.neighbors(v))
    if not nbrs:
        return 0
    return max((len(c) for c in nx.connected_components(H.subgraph(nbrs))), default=0)
mnc_d = {v: mnc(v) for v in H.nodes()}

mcc_d = {v: 0 for v in H.nodes()}
for c in nx.find_cliques(H):
    w = math.factorial(len(c) - 1)
    for v in c:
        mcc_d[v] += w

N = H.number_of_nodes()
epc_d = {v: 0.0 for v in H.nodes()}
ths = [0.5, 0.6, 0.7, 0.8, 0.9]
for t in ths:
    sub = nx.Graph([(a, b) for a, b, d in H.edges(data=True) if d["weight"] >= t])
    for comp in nx.connected_components(sub):
        frac = len(comp) / N
        for v in comp:
            epc_d[v] += frac
epc_d = {v: epc_d[v] / len(ths) for v in H.nodes()}

spl = dict(nx.all_pairs_shortest_path_length(H))
diam = nx.diameter(H)
rad_d = {v: sum(diam + 1 - d for w, d in spl[v].items() if w != v) / (N - 1) for v in H.nodes()}

stress = {v: 0.0 for v in H.nodes()}
for s0 in H.nodes():
    dist = {s0: 0}; sigma = {s0: 1}
    order = [s0]; dq = deque([s0])
    while dq:
        u = dq.popleft()
        for w in H.neighbors(u):
            if w not in dist:
                dist[w] = dist[u] + 1; dq.append(w); order.append(w)
            if dist[w] == dist[u] + 1:
                sigma[w] = sigma.get(w, 0) + sigma[u]
    delta = {v: 0.0 for v in H.nodes()}
    for w in reversed(order):
        if w == s0:
            continue
        for u in H.neighbors(w):
            if dist.get(u, -1) == dist[w] - 1:
                delta[u] += (sigma[u] / sigma[w]) * (1 + delta[w])
    for v in H.nodes():
        if v != s0:
            stress[v] += delta[v]

met = pd.DataFrame({
    "gene": list(H.nodes()),
    "Degree": [deg[v] for v in H.nodes()],
    "MNC": [mnc_d[v] for v in H.nodes()],
    "MCC": [mcc_d[v] for v in H.nodes()],
    "EPC": [round(epc_d[v], 4) for v in H.nodes()],
    "Radiality": [round(rad_d[v], 4) for v in H.nodes()],
    "Stress": [round(stress[v], 2) for v in H.nodes()],
}).sort_values("Degree", ascending=False)
met.to_csv(os.path.join(RES, "ppi_cytohubba_metrics.csv"), index=False)
print(met.to_string(index=False))

TOP = 5
tops = {m: set(met.nlargest(TOP, m)["gene"]) for m in ["Degree", "MNC", "MCC", "EPC", "Radiality", "Stress"]}
cnt = Counter()
for m, s in tops.items():
    for g in s:
        cnt[g] += 1
vote = pd.DataFrame({"gene": list(cnt), "methods_top5": [cnt[g] for g in cnt]}).sort_values(
    ["methods_top5", "gene"], ascending=[False, True])
vote.to_csv(os.path.join(RES, "hub_votes.csv"), index=False)
inter = set.intersection(*tops.values())
hub = sorted(inter) if len(inter) >= 2 else sorted([g for g, c in cnt.items() if c >= 5])
if len(hub) < 2:
    hub = sorted([g for g, c in cnt.items() if c >= 4])
pd.DataFrame({"hub_gene": hub}).to_csv(os.path.join(RES, "hub_genes.csv"), index=False)
print("intersection of all 6 metrics:", sorted(inter))
print("HUB GENES:", hub)
