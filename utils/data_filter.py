import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

'''
Preprocess out the corrupt molecules in the dataset. Three Criteria

1. Make sure RDKit can sanitize the molecule
2. Some conformers are missing bonds
    -- Remove these so we actually have connected molecules
3. Some molecules have reacted in the data generation process
    -- Remove these so that the listed SMILES and 3D molecule align non-isomerically
'''
def filter_data(data):
    # Error Collectors
    non_valid = []
    valence_issue = []
    frags = []
    not_match = []

    # New data list
    new_data = []

    # Items in data
    for item in tqdm(data, total=len(data), desc="Filtering data"):
        # Remove any molecules that do not pass sanitization
        try:
            # Temporarily disable RDKit logging
            rdkit.RDLogger.DisableLog('rdApp.*')
            Chem.SanitizeMol(item['rdmol'])
            rdkit.RDLogger.EnableLog('rdApp.*')
        except Exception as e:
            non_valid.append(item)
            continue

        # Remove any molecules that have fragmentation in the 3D RDMol
        if len(AllChem.GetMolFrags(item['rdmol'], asMols=False)) > 1:
            frags.append(item)
            continue

        # Remove any molecules that have reacted
        rdkit.RDLogger.DisableLog('rdApp.*')
        smile_to_mol = Chem.MolFromSmiles(item['smiles'])
        if not smile_to_mol:
            valence_issue.append(item)
            rdkit.RDLogger.EnableLog('rdApp.*')
            continue
        mol_2d = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(item['smiles'])), isomericSmiles=False)
        mol_3d = Chem.MolToSmiles(item['rdmol'], isomericSmiles=False)
        rdkit.RDLogger.EnableLog('rdApp.*')
        if mol_2d != mol_3d:
            not_match.append(item)
            continue

        # If all good, retain
        new_data.append(item)
        
    return new_data, not_match, frags, non_valid, valence_issue

def get_smiles_dict(data):
    smile_keys = {}
    for item in tqdm(data):
        if item['smiles'] not in smile_keys:
            smile_keys[item['smiles']] = [item, 0]
        smile_keys[item['smiles']][1] += 1
    return smile_keys