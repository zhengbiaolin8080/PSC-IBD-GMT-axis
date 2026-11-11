"""Step 06: drug-likeness (RDKit) and ADMET-AI safety profiling."""
import os
import re
import warnings

import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

met = pd.read_csv(os.path.join(DATA, "metabolites.csv"))
met = met[met["smiles"].notna()].reset_index(drop=True)

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

rows = []
for _, r0 in met.iterrows():
    mol = Chem.MolFromSmiles(r0["smiles"])
    if mol is None:
        continue
    mw = Descriptors.MolWt(mol); hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol); logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol); rb = Lipinski.NumRotatableBonds(mol)
    viol = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    rows.append({"name": r0["name"], "cid": r0["cid"], "MW": round(mw, 2), "HBD": hbd, "HBA": hba,
                 "LogP": round(logp, 2), "TPSA": round(tpsa, 2), "RotB": rb,
                 "Lipinski_violations": viol, "Lipinski_pass": viol == 0})
lip = pd.DataFrame(rows)
lip.to_csv(os.path.join(RES, "adme_lipinski.csv"), index=False)
print("Lipinski pass:", round(lip["Lipinski_pass"].mean(), 3), "of", len(lip))

# ---- ADMET-AI snapshot (computed with ADMET-AI for all 618 metabolites) ----
ad = pd.read_csv(os.path.join(DATA, "admet_ai_predictions.csv"))
print("ADMET-AI matrix:", ad.shape)

panel = pd.read_csv(os.path.join(DATA, "ligand_smiles.csv"))["name"].tolist()
cols = ["name", "AMES", "hERG", "hepatotoxicity", "DILI", "Bioavailability_Ma",
        "CYP3A4_Veith", "Clearance_Hepatocyte_AZ", "LD50_Zhu", "Half_Life_Obach"]
cols = [c for c in cols if c in ad.columns]
sub = ad[ad["name"].isin(panel)][cols].copy()

# The 618-metabolite snapshot was computed from the database's flattened SMILES;
# stereochemistry-blind descriptors collapse UDCA and CDCA onto one structure.
# Recompute the 14-metabolite docking panel from PubChem isomeric SMILES.
lig = pd.read_csv(os.path.join(DATA, "ligand_smiles.csv"))
from rdkit import Chem as _C
from rdkit.Chem import Descriptors as _D, Crippen as _Cr, Lipinski as _L, rdMolDescriptors as _R
from admet_ai import ADMETModel

rows = []
for _, r0 in lig.iterrows():
    mol = _C.MolFromSmiles(r0["isomeric_smiles"])
    mw = _D.MolWt(mol); hbd = _L.NumHDonors(mol); hba = _L.NumHAcceptors(mol)
    logp = _Cr.MolLogP(mol); tpsa = _R.CalcTPSA(mol); rb = _L.NumRotatableBonds(mol)
    viol = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    rows.append({"name": r0["name"], "cid": r0["cid"], "inchikey": r0["inchikey"],
                 "MW": round(mw, 2), "HBD": hbd, "HBA": hba, "LogP": round(logp, 2),
                 "TPSA": round(tpsa, 2), "RotB": rb, "Lipinski_violations": viol,
                 "Lipinski_pass": viol == 0})
plip = pd.DataFrame(rows)
model = ADMETModel()
pred = model.predict(smiles=lig["isomeric_smiles"].tolist())
keep = ["AMES", "hERG", "DILI", "Bioavailability_Ma", "CYP3A4_Veith",
        "Clearance_Hepatocyte_AZ", "LD50_Zhu", "Half_Life_Obach"]
pred = pred[[c for c in keep if c in pred.columns]].reset_index(drop=True)
pred.insert(0, "name", lig["name"].values)
pred.round(4).to_csv(os.path.join(RES, "adme_safety_profile.csv"), index=False)
probe = plip.merge(pred, on="name", how="left")
probe.to_csv(os.path.join(RES, "adme_panel_profile.csv"), index=False)
print(probe.to_string(index=False))
print("panel rows:", len(probe), "| unique InChIKeys:", lig["inchikey"].nunique())
