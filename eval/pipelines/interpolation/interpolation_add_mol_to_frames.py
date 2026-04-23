import os, sys
_PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PIPELINE_DIR)

import pickle
import mdtraj as md
import types
md.version = types.SimpleNamespace(version=md.__version__)
file_path = 'qm9_interpolation_frames_final.pkl'
ref_dir = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/test'
# read the pickle file
smile_match = os.path.join(_PIPELINE_DIR, 'smiles', 'qm9_test_smile_match.txt')
with open(file_path, 'rb') as f:
    data = pickle.load(f)
for smiles in data.keys():

    ref_traj_ids = []
    md_traj_ids = []
    with open(smile_match, 'r') as f:
        for line in f:
            line = line.strip()
            if smiles == line.split('\t')[0]:
                full_name = line.split('\t')[1]
                last_number = full_name[-1]
                if last_number in {'1', '2', '3', '4', '6', '7', '8', '9'}:
                    ref_traj_ids.append(full_name)
                elif last_number in {"0", "5"}:
                    md_traj_ids.append(full_name)
                else:
                    raise ValueError(f"Unknown last number {last_number} in {full_name}")
    name = full_name.split('/')[-1]

    ref_traj_paths = [f'{ref_dir}/{traj_id}/traj.xtc' for traj_id in ref_traj_ids]
    md_traj_paths = [f'{ref_dir}/{traj_id}/traj.xtc' for traj_id in md_traj_ids]
    ref_mol_paths = [f'{ref_dir}/{traj_id}/mol.pkl' for traj_id in ref_traj_ids]
    mol_path = "/".join(md_traj_paths[0].split("/")[:-1])+'/mol.pkl'  # Only get atoms from mol, thus all traj of the same molecule share the same mol info
    mol = pickle.load(open(mol_path, 'rb'))
    data[smiles]['rdmol'] = mol
    for mol_path in ref_mol_paths:
        mol_i = pickle.load(open(mol_path, 'rb'))
        for a1, a2 in zip(mol.GetAtoms(), mol_i.GetAtoms()):
            if a1.GetAtomicNum() != a2.GetAtomicNum() or a1.GetDegree() != a2.GetDegree() or \
            a1.GetFormalCharge() != a2.GetFormalCharge() or a1.GetHybridization() != a2.GetHybridization() or a1.GetIsAromatic() != a2.GetIsAromatic():
                raise ValueError(f"Atoms in mol and mol_{i} do not match, smiles: {smiles}")
            for neighbor1, neighbor2 in zip(a1.GetNeighbors(), a2.GetNeighbors()):
                if neighbor1.GetAtomicNum() != neighbor2.GetAtomicNum() or neighbor1.GetDegree() != neighbor2.GetDegree() or neighbor1.GetIdx() != neighbor2.GetIdx() or \
                neighbor1.GetFormalCharge() != neighbor2.GetFormalCharge() or neighbor1.GetHybridization() != neighbor2.GetHybridization() or neighbor1.GetIsAromatic() != neighbor2.GetIsAromatic():
                    raise ValueError(f"Atoms in mol and mol_{i} do not match, smiles: {smiles}")
print('All sanity checks passed')
