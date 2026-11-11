"""Step 07: real AutoDock Vina docking (v1.2.7, fixed seed 42).

Targets: the five cytoHubba consensus hubs (CCL2, IL10, IL2, PPARG, TLR4) plus the
bile-acid receptor NR1H4 (FXR, highest metabolite-side degree) and PPARA (bile-acid
/ fibrate nuclear receptor). Ligands: 14 metabolites from the PSC-IBD panel with
PubChem isomeric SMILES (bile-acid stereochemistry preserved).
"""
import os
import re
import json
import time
import subprocess
import tempfile
import shutil

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
DOCK = os.path.join(ROOT, "docking")
VINA = os.path.join(DOCK, "vina.exe")
for sub in ("receptors", "ligands", "poses", "results"):
    os.makedirs(os.path.join(DOCK, sub), exist_ok=True)

import pandas as pd  # noqa: E402

hub = pd.read_csv(os.path.join(RES, "hub_genes.csv"))["hub_gene"].tolist()
core = pd.read_csv(os.path.join(RES, "core_targets.csv"))["core_target"].tolist()
# CCL2 is a secreted chemokine lacking a definable small-molecule cavity in the
# available structures (all PDB entries are chemokine dimers without a druggable
# pocket); it is therefore excluded from docking and analysed at the network level.
SKIP = {"CCL2"}
proteins = [g for g in hub if g in core and g not in SKIP]
proteins += [g for g in ["NR1H4", "PPARA"] if g in core and g not in hub]
PREF = {"NR1H4": "1OSV", "PPARG": "6MS7", "TLR4": "3VQ2", "IL2": "4NEJ"}
print("proteins:", proteins)

lig = pd.read_csv(os.path.join(DATA, "ligand_smiles.csv"))
print("ligands:", len(lig))

EXCLUDE_HET = {"HOH", "WAT", "SO4", "GOL", "EDO", "PO4", "CL", "NA", "MG", "CA", "ZN", "K", "DMS",
               "ACT", "BME", "MPD", "PEG", "PG4", "ACE", "FMT", "NH2", "NAG", "BMA", "MAN", "FUC",
               "GAL", "GLC", "MLI", "IOD", "NO3", "SCN", "CIT", "TRS", "MES", "TAR", "BUA", "ACY",
               "CO", "CD", "NI", "CU", "FE", "MN", "HG", "SR", "BA", "RB", "CS", "LI", "F", "UNX",
               "UNL", "DOD", "C8E", "LDA", "SDS", "PLM", "MYR", "RET", "HEM", "FAD", "FMN", "NAD",
               "NAP", "NDP", "ADP", "ATP", "GTP", "GDP", "AMP", "ANP", "SEP", "TPO", "PTR", "CSO",
               "CME", "MSE", "LLP", "CSD", "KCX", "PCA", "HYP", "SAC", "TLA", "BCN", "XLS", "1PE",
               "PEE", "PGE", "P6G", "15P", "7PE", "DTT", "I42", "D10", "PX4", "B3P", "2BM", "MBO",
               "P33", "N8E", "POC", "VO4", "MOO", "WO4", "YX1", "SIN", "CMO", "TCN", "NCO", "AZI",
               "CYN", "NO"}


def fetch_pdb(pid):
    for u in (f"https://files.rcsb.org/download/{pid}.pdb",
              f"https://files.rcsb.org/download/{pid}.cif"):
        try:
            t = requests.get(u, timeout=120).text
            if t.startswith("HEADER") or t.startswith("data_"):
                return t
        except Exception:
            pass
    return None


def parse_pdb_ligands(text):
    het = {}
    for line in text.splitlines():
        if line.startswith("HETATM"):
            resn = line[17:20].strip()
            if resn in EXCLUDE_HET:
                continue
            key = (resn, line[21].strip(), line[22:26].strip())
            het.setdefault(key, []).append(line)
    best = None
    for k, lines in het.items():
        heavy = [l for l in lines if l[76:78].strip() not in ("H", "")]
        if best is None or len(heavy) > len(best[1]):
            best = (k, heavy)
    return het, best


def centroid(lines):
    xs = [float(l[30:38]) for l in lines]; ys = [float(l[38:46]) for l in lines]
    zs = [float(l[46:54]) for l in lines]
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


STD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
       "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
VAR = {"GLH": "GLU", "NGLH": "ASN", "CGLH": "GLN", "NLYN": "LYS", "CLYN": "ASN",
       "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYM": "CYS", "CYX": "CYS", "CSS": "CYS",
       "MLY": "LYS", "SEP": "SER", "TPO": "THR", "PTR": "TYR", "CSO": "CYS", "KCX": "LYS",
       "CSD": "CYS", "MSE": "MET", "PCA": "PRO", "HYP": "PRO", "TYS": "TYR", "CGU": "GLU",
       "ALB": "ALA", "CSW": "CYS", "SNN": "ASN", "NEP": "HIS"}


def clean_pdb(src, dst):
    keep = []
    for line in open(src):
        line = line.rstrip("\n").rstrip("\r")
        if not line.startswith("ATOM"):
            continue
        resn = VAR.get(line[17:20].strip(), line[17:20].strip())
        if resn not in STD:
            continue
        keep.append(line[:17] + resn.ljust(3) + line[20:])
    by_res, order = {}, []
    for l in keep:
        key = (l[21], l[22:26].strip(), l[17:20].strip())
        if key not in by_res:
            by_res[key] = []; order.append(key)
        by_res[key].append(l)
    out = []
    for k in order:
        names = {l[12:16].strip() for l in by_res[k]}
        if {"N", "CA", "C"}.issubset(names):
            out.extend(by_res[k])
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    return len(out)


def prepare_receptor(pdb_id, sym):
    txt = fetch_pdb(pdb_id)
    if not txt:
        return None
    het, best = parse_pdb_ligands(txt)
    if not (best and len(best[1]) >= 8):
        return None
    key, lines = best
    ctr = centroid(lines)
    pdb_path = os.path.join(DOCK, "receptors", f"{sym}_{pdb_id}.pdb")
    atom_lines = [l for l in txt.splitlines() if l.startswith(("ATOM", "TER", "END"))]
    open(pdb_path, "w").write("\n".join(atom_lines) + "\n")
    clean = pdb_path.replace(".pdb", "_clean.pdb")
    clean_pdb(pdb_path, clean)
    pdbqt = pdb_path.replace(".pdb", ".pdbqt")
    if not os.path.exists(pdbqt):
        cmd = ["python", "-m", "meeko.cli.mk_prepare_receptor", "--read_pdb", clean,
               "-o", pdbqt.replace(".pdbqt", ""), "-p", "--allow_bad_res", "--default_altloc", "A"]
        subprocess.run(cmd, capture_output=True, timeout=300)
    if not os.path.exists(pdbqt):
        try:
            from openbabel import pybel
            td = tempfile.gettempdir()
            tc = os.path.join(td, f"pscibd_{sym}_{pdb_id}.pdb")
            tq = os.path.join(td, f"pscibd_{sym}_{pdb_id}.pdbqt")
            shutil.copy(clean, tc)
            mol = next(pybel.readfile("pdb", tc))
            mol.OBMol.AddHydrogens()
            mol.write("pdbqt", tq, overwrite=True)
            if os.path.exists(tq):
                shutil.copy(tq, pdbqt)
                os.remove(tq)
            os.remove(tc)
        except Exception:
            pass
    if os.path.exists(pdbqt):
        print(f"  {sym}: {pdb_id} ligand {key[0]} heavy={len(lines)} center=({ctr[0]:.1f},{ctr[1]:.1f},{ctr[2]:.1f}) OK", flush=True)
        return {"pdb": pdb_path, "pid": pdb_id, "center": ctr, "lig": key[0]}
    return None


def auto_pdb(sym):
    q = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{sym}+AND+organism_id:9606+AND+reviewed:true&fields=accession&format=json&size=1"
    try:
        acc = requests.get(q, timeout=60).json()["results"][0]["primaryAccession"]
    except Exception:
        return []
    body = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
            "operator": "exact_match", "value": acc}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 2.8}}]},
        "return_type": "entry",
        "request_options": {"sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
                            "paginate": {"start": 0, "rows": 12}}}
    try:
        rows = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=body, timeout=60).json()
        return [x["identifier"] for x in rows.get("result_set", [])]
    except Exception:
        return []


receptors = {}
for sym in proteins:
    print("preparing", sym, flush=True)
    ok = None
    if sym in PREF:
        ok = prepare_receptor(PREF[sym], sym)
    if ok is None:
        for pid in auto_pdb(sym)[:10]:
            ok = prepare_receptor(pid, sym)
            if ok:
                break
    if ok:
        receptors[sym] = ok
    else:
        print("  !! no preparable structure for", sym, flush=True)
print("prepared receptors:", list(receptors), flush=True)

# ---- ligands --------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy


def lig_pdbqt(name, smiles):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name)[:40]
    out = os.path.join(DOCK, "ligands", safe + ".pdbqt")
    if os.path.exists(out):
        return out
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    prep = MoleculePreparation()
    mols = prep.prepare(mol)
    pqt = PDBQTWriterLegacy.write_string(mols[0])[0]
    open(out, "w").write(pqt)
    return out


ligs = []
for _, r0 in lig.iterrows():
    try:
        ligs.append((r0["name"], lig_pdbqt(r0["name"], r0["isomeric_smiles"])))
        print("ligand ok", r0["name"], flush=True)
    except Exception as e:
        print("ligand FAIL", r0["name"], str(e)[:120], flush=True)

# ---- docking --------------------------------------------------------------
results = []
for sym, r0 in receptors.items():
    rec = r0["pdb"].replace(".pdb", ".pdbqt")
    if not os.path.exists(rec):
        continue
    cx, cy, cz = r0["center"]
    for name, lp in ligs:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", name)[:40]
        outp = os.path.join(DOCK, "poses", f"{sym}__{safe}.pdbqt")
        cmd = [VINA, "--receptor", rec, "--ligand", lp,
               "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
               "--size_x", "24", "--size_y", "24", "--size_z", "24",
               "--exhaustiveness", "8", "--num_modes", "9", "--seed", "42",
               "--cpu", "4", "--out", outp]
        aff = None
        try:
            subprocess.run(cmd, capture_output=True, timeout=900)
            if os.path.exists(outp):
                for line in open(outp):
                    if line.startswith("REMARK VINA RESULT:"):
                        aff = float(line.split()[3]); break
            print(f"{sym:8s} {name[:28]:30s} {aff}", flush=True)
        except Exception as e:
            print("vina FAIL", sym, name, str(e)[:120], flush=True)
        results.append({"protein": sym, "pdb": r0["pid"], "metabolite": name,
                        "affinity_kcal_mol": aff})

df = pd.DataFrame(results)
df.to_csv(os.path.join(RES, "docking_affinities.csv"), index=False)
mat = df.pivot_table(index="metabolite", columns="protein", values="affinity_kcal_mol")
mat.to_csv(os.path.join(RES, "docking_matrix.csv"))
json.dump({"method": "AutoDock Vina v1.2.7", "seed": 42, "exhaustiveness": 8,
           "box": [24, 24, 24], "receptors": {k: {"pdb": v["pid"], "ligand": v["lig"],
           "center": v["center"]} for k, v in receptors.items()},
           "ligands": [l[0] for l in ligs]},
          open(os.path.join(DOCK, "results", "docking_metadata.json"), "w"), indent=1)
print(mat.round(2).to_string())
print("DONE docking")
