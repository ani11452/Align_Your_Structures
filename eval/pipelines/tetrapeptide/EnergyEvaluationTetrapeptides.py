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
import glob
import os
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ref_dir = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/test'
ref_dir = os.environ.get('TIMEWARP_DATA_ROOT', '') + '/data/4AA-large/4AA-large-processed/test'
# smile_match = os.path.join(_PIPELINE_DIR, 'smiles', 'qm9_test_smile_match.txt')
# smile_match = os.path.join(_PIPELINE_DIR, 'smiles', 'drugs_test_smile_match.txt')
# output_path = "model_outputs_official/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_interpolator_pretrain_final_539_25/epoch=399-step=69287-gen.pkl"
# output_path = 'model_outputs_official/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_egtn_539_25/epoch=399-step=69200-gen.pkl'
output_path = "model_outputs_official/timewarp_trajectory_official/timewarp_interpolator_075_full_test_generations.pkl"
baseline_output_path = "model_outputs_official/timewarp_trajectory_official/epoch=659-step=300960-gen.pkl"

with open(output_path, 'rb') as f:
    outputs = pickle.load(f)

with open(baseline_output_path, 'rb') as f:
    baseline_outputs = pickle.load(f)

# Build SMILES to filename mapping from reference directory
print('Building SMILES to filename mapping from reference directory...')
smiles_to_filename = {}
ref_files = glob.glob(os.path.join(ref_dir, '*.pkl'))
for ref_file in tqdm(ref_files, desc='Loading reference metadata'):
    try:
        with open(ref_file, 'rb') as f:
            ref_data = pickle.load(f)
        if 'smiles' in ref_data:
            smiles = ref_data['smiles']
            filename = os.path.basename(ref_file).replace('.pkl', '')
            smiles_to_filename[smiles] = filename
    except Exception as e:
        continue

# Load ANI2x model once globally to save memory
print("Loading ANI2x model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
GLOBAL_ANI_MODEL = torchani.models.ANI2x().to(device)
print(f"Model loaded on {device}")

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
    if energy_model is None:
        energy_model = GLOBAL_ANI_MODEL
    model = energy_model
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
    for coord in coords:
        batch_energies.append(compute_framewise_energies(rdmol, coord, energy_model=GLOBAL_ANI_MODEL))
    return batch_energies

results = {}
smiles_range = [0, 180]
# Filter to only peptides that have reference data
gener_smiles = [k for k in outputs.keys() if k in smiles_to_filename and k != 'WALL_CLOCK' and isinstance(k, str)]
print(f'Found {len(gener_smiles)} peptides with reference data')

for smiles in tqdm(list(gener_smiles)[smiles_range[0]:smiles_range[1]]):
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
        energies = compute_energy_gen_batch(rdmol, coords)
        if len(np.array(energies).flatten()) != 5005:
            print(f"Expected 5005 energy values, got {len(np.array(energies).flatten())}")
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
            print(f"Expected 5005 baseline energy values, got {len(np.array(baseline_energies).flatten())}")
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

    # Load reference trajectory from preprocessed pickle (CONTINUOUS DATA)
    ref_filename = smiles_to_filename[smiles]
    ref_pkl_path = f'{ref_dir}/{ref_filename}.pkl'
    
    try:
        with open(ref_pkl_path, 'rb') as f:
            ref_data = pickle.load(f)
        ref_traj_full = ref_data['coordinates_mdtraj']  # MDTraj object (T, N, 3) in nm
        mol = ref_data['rdkit_mol']
        mol_noH = Chem.RemoveHs(mol)  # Remove H atoms from the mol
        
        # Downsample reference trajectory with stride of 10 to reduce memory
        ref_traj_downsampled = ref_traj_full[::10]
        
        # Extract heavy atom coordinates from MDTraj object
        # Get the substructure match to identify heavy atoms
        important_atoms = list(mol.GetSubstructMatch(mol_noH))
        
        # Extract coordinates: ref_traj_downsampled.xyz is (T_down, N_all, 3) in nm
        coords_nm = ref_traj_downsampled.xyz[:, important_atoms, :]  # Shape: (T_down, N_heavy, 3) in nm
        
        # Process in blocks of 1000 frames to avoid memory issues
        block_size = 1000
        ref_energies_list = []
        for start_idx in range(0, coords_nm.shape[0], block_size):
            end_idx = min(start_idx + block_size, coords_nm.shape[0])
            coords_block = coords_nm[start_idx:end_idx]
            
            # Transpose to (N_heavy, 3, T_block) and convert to Angstroms
            ref_coords_block = coords_block.transpose(1, 2, 0) * 10
            
            # Compute energies for this block
            block_energies = compute_framewise_energies(mol_noH, ref_coords_block, energy_model=GLOBAL_ANI_MODEL)
            ref_energies_list.extend(block_energies)
        
        ref_energies = [ref_energies_list]  # Wrap in list to match expected format
    except Exception as e:
        print(f"Error computing reference energies for {smiles}: {e}")
        continue
    
    # Save results to a file
    results[smiles] = {
        'energies': energies_block,
        'energies_full': energies,  # Full trajectories (5, 1001)
        'ref_energies': ref_energies,
        'baseline_energies': baseline_energies_block,
        'baseline_energies_full': baseline_energies  # Full baseline trajectories (5, 1001)
    }
    
    # Clear memory after each molecule
    del energies, baseline_energies, ref_energies
    del coords, baseline_coords, ref_coords_block, coords_block
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

output_file = f"tetrapeptide_energy_results_075_{50}.pkl"
with open(output_file, 'wb') as f:
    f.write(pickle.dumps(results))