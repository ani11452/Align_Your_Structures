import os, sys
_PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PIPELINE_DIR)

import torch
import torchani
import numpy as np
import mdtraj as md
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

# ref_dir = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/test'
ref_dir = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-DRUGS/4fs_HMR15_5ns_actual/test'
# smile_match = os.path.join(_PIPELINE_DIR, 'smiles', 'qm9_test_smile_match.txt')
smile_match = os.path.join(_PIPELINE_DIR, 'smiles', 'drugs_test_smile_match.txt')
# output_path = "model_outputs_official/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_interpolator_pretrain_final_539_25/epoch=399-step=69287-gen.pkl"
# output_path = 'model_outputs_official/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_egtn_539_25/epoch=399-step=69200-gen.pkl'
output_path = "model_outputs_official/drugs_trajectory_official/drugs_noH_1000_kabsch_traj_interpolator_pretrain_fs_25/epoch=399-step=71200-gen_official_merged.pkl"
baseline_output_path = "model_outputs_official/drugs_trajectory_official/drugs_noH_1000_kabsch_traj_egtn_fs/epoch=399-step=71200-gen.pkl"

with open(output_path, 'rb') as f:
    outputs = pickle.load(f)

with open(baseline_output_path, 'rb') as f:
    baseline_outputs = pickle.load(f)

def add_hydrogens_optimize(rdmol, coords, optimize=True, retain_heavy=True):
    """
    Add hydrogens and optionally optimize them for each frame of heavy-atom-only MD.

    Args:
        rdmol: RDKit Mol (heavy atoms only, no Hs) with correct connectivity
        coords: np.ndarray of shape (N_heavy, 3, T)
        optimize: whether to perform MMFF optimization (default True)
        retain_heavy: whether to fix heavy atom positions (default True)

    Returns:
        List of RDKit Mol objects with hydrogens added and optionally optimized
    """
    N_heavy, _, T = coords.shape
    mols_with_h = []

    for t in range(T):
        # Get coordinates for this frame
        heavy_xyz = coords[:, :, t]  # (N_heavy, 3)

        # Create a copy of the heavy mol
        mol = Chem.Mol(rdmol)

        # Create conformer with frame coords
        conf = Chem.Conformer(N_heavy)
        for i in range(N_heavy):
            x, y, z = heavy_xyz[i]
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
        mol.RemoveAllConformers()
        mol.AddConformer(conf)

        # Add hydrogens with idealized coordinates
        mol = Chem.AddHs(mol, addCoords=True)

        if optimize:
            # Get indices of atoms to fix (i.e., heavy atoms)
            if retain_heavy:
                heavy_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
            else:
                heavy_idxs = []

            # Try MMFF first
            if AllChem.MMFFHasAllMoleculeParams(mol):
                try:
                    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94', fixedAtoms=heavy_idxs)
                except:
                    pass
            else:
                # Fallback to UFF if MMFF fails
                try:
                    AllChem.UFFOptimizeMolecule(mol, fixedAtoms=heavy_idxs)
                except:
                    pass

        mols_with_h.append(mol)

    return mols_with_h
def compute_framewise_energies(rdmol, coords, energy_model=None):
    """
    Args:
        rdmol: RDKit Mol with correct heavy-atom topology (no Hs)
        coords: np.ndarray of shape (N_heavy, 3, T)

    Returns:
        energies: list of float, energy (in Hartree) for each frame
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if energy_model:
        model = energy_model
    else:
        model = torchani.models.ANI2x().to(device)
    species_to_tensor = model.species_to_tensor

    # Add hydrogens to each frame
    frame_rdmols = add_hydrogens_optimize(rdmol, coords)

    all_species = []
    all_coordinates = []

    for mol in frame_rdmols:
        conf = mol.GetConformer()
        coords_frame = []
        symbols = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords_frame.append([pos.x, pos.y, pos.z])
            symbols.append(atom.GetSymbol())

        # Ensure consistent atom order
        all_species.append(species_to_tensor(symbols))
        all_coordinates.append(torch.tensor(coords_frame, dtype=torch.float32))

    # Stack into batch tensors: [T, N], [T, N, 3]
    species_tensor = torch.stack(all_species).to(device)          # [T, N]
    coordinates_tensor = torch.stack(all_coordinates).to(device)  # [T, N, 3]

    with torch.no_grad():
        output = model((species_tensor, coordinates_tensor))
        energies = output.energies.cpu().numpy().tolist()  # [T] as Python floats

    return energies


def compute_energy_gen_batch(rdmol, coords):
    batch_energies = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchani.models.ANI2x().to(device)
    for coord in coords:
        batch_energies.append(compute_framewise_energies(rdmol, coord, energy_model=model))
    return batch_energies

results = {}
smiles_range = [90,100]
for smiles in tqdm(list(outputs.keys())[smiles_range[0]:smiles_range[1]]):
    print(smiles)
    if smiles not in baseline_outputs.keys():
        continue
    if any(atom.GetSymbol() not in ["H", "C", "N", "O", "F", "S", "Cl"] for atom in outputs[smiles]['rdmol'].GetAtoms()):
        continue
    try:
        rdmol = outputs[smiles]['rdmol']  # RDKit Mol object with heavy atoms only
        # Get heavy atom coordinates from outputs
        coords = outputs[smiles]['coords']  # Shape: (N_heavy, 3, T)
        # Compute energies for this molecule
        print(coords[0].shape)
        energies = compute_energy_gen_batch(rdmol, coords)
        if len(np.array(energies).flatten()) != 5005:
            continue
        
        # energies is (5, 1001). Discard the first one item and Divide it to (4, 250)
        energies_block = {1:[], 2:[], 3:[], 4:[]}
        for i in range(5):
            energies_block[1].append(energies[i][1:251])
            energies_block[2].append(energies[i][251:501])
            energies_block[3].append(energies[i][501:751])
            energies_block[4].append(energies[i][751:1001])

    except Exception as e:
        print(f"Error computing energies for {smiles}: {e}")
        continue

    try:
        baseline_rdmol = baseline_outputs[smiles]['rdmol']  # RDKit Mol object with heavy atoms only
        baseline_coords = baseline_outputs[smiles]['coords']  # Shape: (N_heavy, 3, T)
        baseline_energies = compute_energy_gen_batch(baseline_rdmol, baseline_coords)
        if len(np.array(baseline_energies).flatten()) != 5005:
            continue
        # baseline_energies is (5, 1001). Discard the first one item and Divide it to (4, 250)
        baseline_energies_block = {1:[], 2:[], 3:[], 4:[]}
        for i in range(5):
            baseline_energies_block[1].append(baseline_energies[i][1:251])
            baseline_energies_block[2].append(baseline_energies[i][251:501])
            baseline_energies_block[3].append(baseline_energies[i][501:751])
            baseline_energies_block[4].append(baseline_energies[i][751:1001])
    except Exception as e:
        print(f"Error computing baseline energies for {smiles}: {e}")
        continue

    ref_traj_ids = []
    with open(smile_match, 'r') as f:
        for line in f:
            line = line.strip()
            if smiles == line.split('\t')[0]:
                full_name = line.split('\t')[1]
                last_number = full_name[-1]
                if last_number in {'1', '2', '3', '4', '6', '7', '8', '9'}:
                    ref_traj_ids.append(full_name)
                    break

    name = full_name.split('/')[-1]
    ref_traj_paths = [f'{ref_dir}/{traj_id}/traj.xtc' for traj_id in ref_traj_ids]
    ref_pdb_paths = [f'{ref_dir}/{traj_id}/system.pdb' for traj_id in ref_traj_ids]
    mol_path = "/".join(ref_traj_paths[0].split("/")[:-1])+'/mol.pkl'  # Only get atoms from mol, thus all traj of the same molecule share the same mol info
    mol = pickle.load(open(mol_path, 'rb'))
    mol_noH = Chem.RemoveHs(mol)  # Remove H atoms from the mol
    important_atoms = list(mol.GetSubstructMatch(mol_noH))
    # Get the reference trajectories as coords: (N_heavy, 3, T)
    ref_coords = []
    
    traj = md.load(ref_traj_paths[0], top=ref_pdb_paths[0])
    coords = traj.xyz[:, important_atoms, :]  # Shape: (T, N_heavy, 3)
    # Transpose to (N_heavy, 3, T)
    coords = coords.transpose(1, 2, 0) * 10  # Convert to Angstroms
    ref_coords.append(coords)
    try:
        ref_energies = compute_energy_gen_batch(mol_noH, ref_coords)
    except Exception as e:
       print(f"Error computing reference energies for {smiles}: {e}")
       continue
    # Save results to a file
    results[smiles] = {
        'energies': energies_block,
        'ref_energies': ref_energies,
        'baseline_energies': baseline_energies_block
    }

output_file = f"drug_block_energy_results_{smiles_range[0]}.pkl"
with open(output_file, 'wb') as f:
    f.write(pickle.dumps(results))