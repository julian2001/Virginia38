#!/usr/bin/env python
# coding: utf-8
"""
Minimal resilient predictor CLI (RDKit-backed).
Usage:
  admet_predict4.py data_path --save_path SAVE --smiles_column smiles
"""
import argparse
import os
from pathlib import Path
import sys

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors
except Exception as e:
    print(f"ERROR: RDKit import failed: {e}", file=sys.stderr)
    sys.exit(2)

def calc_props(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        qed = QED.qed(mol)
        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        stereo_centers = len(chiral)
        # Lipinski RO5 violations
        lipinski = int((mw > 500) + (logp > 5) + (hba > 10) + (hbd > 5))
    except Exception:
        return None
    return {
        "molecular_weight": mw,
        "logP": logp,
        "hydrogen_bond_acceptors": hba,
        "hydrogen_bond_donors": hbd,
        "tpsa": tpsa,
        "QED": qed,
        "stereo_centers": stereo_centers,
        "Lipinski": lipinski,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_path", help="CSV with a SMILES column")
    ap.add_argument("--save_path", required=True, help="Where to write predictions CSV")
    ap.add_argument("--smiles_column", default="smiles")
    args = ap.parse_args()

    inp = Path(args.data_path)
    outp = Path(args.save_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if args.smiles_column not in df.columns:
        print(f"ERROR: missing column '{args.smiles_column}' in {inp}", file=sys.stderr)
        sys.exit(2)

    rows = []
    for s in df[args.smiles_column].astype(str):
        res = calc_props(s)
        if res is None:
            # keep the row but mark NaNs; still include the smiles string
            rows.append({"smiles": s})
        else:
            rows.append({"smiles": s, **res})

    if not rows:
        # ensure we never write an empty file
        rows = [{"smiles": ""}]

    out_df = pd.DataFrame(rows)
    # keep column order stable
    cols = ["smiles", "molecular_weight", "logP", "hydrogen_bond_acceptors", "hydrogen_bond_donors", "Lipinski", "QED", "stereo_centers", "tpsa"]
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = pd.NA
    out_df = out_df[cols]

    out_df.to_csv(outp, index=False)
    print(f"[admet_predict4] wrote → {outp}")

if __name__ == "__main__":
    main()
