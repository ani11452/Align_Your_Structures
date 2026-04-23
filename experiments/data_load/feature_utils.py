import torch
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, ChiralType

# This is essentially the featurization used on GeoMol and Torsional Diffusion

# Define hybridization types + unknown
HYBRIDIZATION_TYPES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    "unknown"
]
HYBRIDIZATION_MAPPING = {h: i for i, h in enumerate(HYBRIDIZATION_TYPES)}
CHIRALITY = {ChiralType.CHI_TETRAHEDRAL_CW: [1, 0, 0],
             ChiralType.CHI_TETRAHEDRAL_CCW: [0, 1, 0],
             ChiralType.CHI_UNSPECIFIED: [0, 0, 1],
             ChiralType.CHI_OTHER: [0, 0, 1]}


def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

# === Individual feature functions ===
def extract_aromatic(atom):
    return [1, 0] if atom.GetIsAromatic() else [0, 1] # Dim 2

def extract_hybridization(atom):
    hyb = atom.GetHybridization()
    onehot = [0] * len(HYBRIDIZATION_TYPES)
    if hyb in HYBRIDIZATION_MAPPING:
        idx = HYBRIDIZATION_MAPPING[hyb]
    else:
        idx = HYBRIDIZATION_MAPPING["unknown"]
    onehot[idx] = 1
    return onehot # Dim 6

def extract_partial_charge(atom):
    return [float(atom.GetDoubleProp("_GasteigerCharge"))] # Dim 1

def implicit_valence_scalar(atom):
    return [atom.GetImplicitValence()]

def implicit_valence(atom):
    return one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) # Dim 8

def extract_degree_scalar(atom):
    return [atom.GetDegree()]

def extract_degree(atom):
    return one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) # Dim 8

def extract_formal_charge_scalar(atom):
    return [atom.GetFormalCharge()] 

def extract_formal_charge(atom):
    return one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]) # Dim 4 

def extract_ring_size(atom):
    mol = atom.GetOwningMol()
    ring_info = mol.GetRingInfo()
    idx = atom.GetIdx()

    # Check membership for ring sizes
    encoding1 = [int(ring_info.IsAtomInRingOfSize(idx, size)) for size in [3, 4, 5, 6, 7, 8]]
    num_rings = ring_info.NumAtomRings(idx)
    encoding2 = [0 for _ in range(4)]
    encoding2[min(num_rings, 3)] = 1

    return encoding1 + encoding2  # Dim 10

def extract_chirality(atom):
    chiral_flag = atom.GetChiralTag()
    if chiral_flag in CHIRALITY:
        return CHIRALITY[chiral_flag]
    else:
        return [0, 0, 1]  # Dim 3

# === Feature registry ===
FEATURE_FUNCTIONS = {
    "aromatic": extract_aromatic,
    "hybridization": extract_hybridization,
    "partial_charge": extract_partial_charge,
    "implicit_valence": implicit_valence,
    "implicit_valence_scalar": implicit_valence_scalar,
    "degree": extract_degree,
    "degree_scalar": extract_degree_scalar,
    "formal_charge": extract_formal_charge,
    "formal_charge_scalar": extract_formal_charge_scalar,
    "ring_size": extract_ring_size,
    "chirality": extract_chirality
}

# === Main node feature extractor ===
def get_node_features(mol, features):
    """
    Compute per-atom features [N, D].
    Only re-computes implicit_valence on a temporary copy.
    """
    # 1) Precompute Gasteiger charges if needed
    if "partial_charge" in features:
        AllChem.ComputeGasteigerCharges(mol)

    all_feats = []
    for atom in mol.GetAtoms():
        feats = []
        for name in features:
            if name not in FEATURE_FUNCTIONS:
                raise ValueError(f"Unknown feature: {name}")
            feats += FEATURE_FUNCTIONS[name](atom)
        all_feats.append(feats)

    return torch.tensor(all_feats, dtype=torch.float)
