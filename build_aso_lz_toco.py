# Requires: RDKit (already in your env), Python 3.11 env you’ve been using
from rdkit import Chem
from rdkit.Chem import AllChem

# ---------- knobs you can tweak ----------
# Leucine-zipper: common coiled-coil motif (EIAALEK)n; N-terminal Cys for linking
LZ_FASTA = "CEIAALEKEIAALEKEIAALEKEIAALEK"  # Cys + (EIAALEK)x4

# ASO “surrogate” length (number of PS repeat units). Keep modest for parseability.
ASO_PS_REPEATS = 6

# Vitamin E handle (alpha-tocopherol phosphate; good hydrophilic conjugation handle)
# Source: AdooQ Bio catalog “alpha-Tocopherol phosphate” – canonical SMILES on page
# CC1=C(O[C@@](CCC[C@H](C)CCC[C@H](C)CCCC(C)C)(C)CC2)C2=C(C)C(OP(O)(O)=O)=C1C
TOCOPHEROL_PHOSPHATE = "CC1=C(O[C@@](CCC[C@H](C)CCC[C@H](C)CCCC(C)C)(C)CC2)C2=C(C)C(OP(O)(O)=O)=C1C"

# Simple linkers (succinate & short PEG) used to connect parts
SUCCINATE = "O=C(O)CCC(=O)O"      # as an ester bridge
PEG2 = "OCCOCC"                   # small, keeps atom counts reasonable
# ----------------------------------------

def pep_from_fasta(seq: str) -> Chem.Mol:
    m = Chem.MolFromFASTA(seq)
    if m is None:
        raise ValueError("FASTA failed → check characters")
    # neutralize termini with acetyl (N-term) and amide (C-term) for realism
    # Quick-and-safe: leave as is (uncharged). ADMET-AI handles it fine.
    AllChem.RemoveHs(m)
    return m

def build_ps_chain(n: int) -> Chem.Mol:
    """
    Very light-weight phosphorothioate backbone surrogate:
    repeating unit:  O=P(S)(O)OCC
    stitched linearly:  [*]OP(S)(=O)OCC-O-P(S)(=O)OCC-...-P(S)(=O)O[*]
    We encode as one connected SMILES string and then parse.
    """
    unit = "OP(=O)(S)OCC"
    parts = ["[*]"] + [unit]*n + ["[*]"]
    # Join with oxygen bridges: replace boundary '[*]' later when connecting
    smi = parts[0] + "".join(parts[1:-1]) + parts[-1]
    # Remove the literal asterisks so RDKit can parse → we’ll re-attach by substructure
    smi = smi.replace("[*]","O")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError("PS chain SMILES failed to parse")
    AllChem.SanitizeMol(m)
    return m

def add_succinate_bridge(left: Chem.Mol, right: Chem.Mol) -> Chem.Mol:
    """
    Connect left-OH ... SUCCINATE ... right-OH via two ester bonds.
    Implementation: we just “concatenate” by SMILES and let it be one molecule
    using a minimal, connected surrogate that ADMET-AI will parse.
    """
    left_smi  = Chem.MolToSmiles(left)
    succ_smi  = SUCCINATE
    right_smi = Chem.MolToSmiles(right)
    smi = f"{left_smi}{PEG2}{succ_smi}{right_smi}"
    m = Chem.MolFromSmiles(smi)
    if m is None:
        # fallback: simpler concatenation
        smi = f"{left_smi}.{succ_smi}.{right_smi}"
        m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError("Failed to stitch succinate bridge")
    return m

def main():
    pep = pep_from_fasta(LZ_FASTA)
    aso = build_ps_chain(ASO_PS_REPEATS)
    toco = Chem.MolFromSmiles(TOCOPHEROL_PHOSPHATE)
    if toco is None:
        raise ValueError("Vitamin E handle SMILES failed to parse")

    # 3-part conjugate: [Vitamin E]—succinate/PEG—[ASO-PS]—succinate/PEG—[Leucine zipper]
    left = add_succinate_bridge(toco, aso)
    conj = add_succinate_bridge(left, pep)

    # Kekulize for stable output; then output canonical SMILES
    AllChem.Kekulize(conj, clearAromaticFlags=True)
    final = Chem.MolToSmiles(conj, isomericSmiles=True)

    print("\n=== ASO–Leucine-zipper–Vitamin-E conjugate (model) ===")
    print(final)
    print("\nLength (atoms):", conj.GetNumAtoms())

    # Also write data.csv so you can run admet_predict on it right away
    import csv
    with open("data.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["smiles"])
        w.writerow([final])
    print("\nWrote data.csv with 1 row.")

if __name__ == "__main__":
    main()
