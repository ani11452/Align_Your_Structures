"""
MDGen 4AA conformer sampling.

Reads the MDGen 4AA_sims dataset (flat directory of per-peptide xtc/pdb files)
and writes ONE combined pickle with N_CONFORMERS KMedoids-selected frames per
peptide.

Output:
  ${MDGEN_DATA_ROOT}/.../tetrapeptide_conformers_mdgen.pkl   (single combined file)

This output is an INTERMEDIATE consumed by data_gen/preprocess/merge_mdwarp.py, which
partitions it by MDGen's own train/val/test split CSVs (holding test out)
and merges with TimeWarp-derived conformers to produce the
'tetrapeptide_conformers_final_{train,val}.pkl' pair consumed by
configs_official/conformer/tetrapeptide/mdwarp_tetrapeptide_pretrain_*.yaml.

See also:
- data_gen/preprocess/sample_conformer_AA.py  (TimeWarp sibling; produces the _large / _val
   intermediates that feed merge_mdwarp.py)
- data_gen/preprocess/process_mdgen.py        (trajectory-side MDGen preprocessor)
"""

import os
import pickle
import tempfile
import numpy as np
import mdtraj as md
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn_extra.cluster import KMedoids
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Configure Matplotlib to run without a display server
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---
# 1. SET THIS to the path of your dataset
DATASET_BASE_DIR = os.path.join(os.environ.get('MDGEN_DATA_ROOT', ''), 'data/4AA_sims')

# 2. Set the name of the output file
OUTPUT_PICKLE_FILE = 'tetrapeptide_conformers_mdgen.pkl'

# 3. Set the number of conformers to select
N_CONFORMERS = 20

# 4. Maximum frames to sample for clustering (Memory Optimization)
MAX_FRAMES_FOR_CLUSTERING = 10_000

# 5. Maximum number of CPU workers for parallel processing
MAX_PARALLEL_WORKERS = 32

# -----------------------
# Robust RDKit molecule construction helpers
# -----------------------

def seq_from_filename(pdb_file: str) -> str:
    """Extract 4-letter tetrapeptide sequence from filename like 'KCWL.pdb'."""
    base = os.path.basename(pdb_file)
    prefix = base.split('.')[0]
    if len(prefix) != 4:
        raise ValueError(f"Unexpected peptide name in filename: {base}")
    return prefix


def build_template_from_seq(seq: str) -> Chem.Mol:
    """Build a chemically-correct peptide template from a 1-letter sequence."""
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        raise ValueError(f"MolFromSequence failed for sequence: {seq}")
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return mol


def traj_to_pdb_block(traj: md.Trajectory) -> str:
    """Convert a single-frame MDTraj trajectory to a PDB block (string)."""
    if traj.n_frames == 0:
        raise ValueError("Empty trajectory passed to traj_to_pdb_block")

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        traj[0].save_pdb(tmp_path)
        with open(tmp_path, 'r') as f:
            pdb_block = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return pdb_block


def create_rdkit_mol_robust(traj: md.Trajectory, pdb_file: str) -> Chem.Mol:
    """Robust RDKit molecule construction using sequence-based template matching."""
    seq = seq_from_filename(pdb_file)
    template = build_template_from_seq(seq)
    pdb_block = traj_to_pdb_block(traj)

    # Load PDB block with proximity bonding to get geometry and connectivity
    pdb_mol = Chem.MolFromPDBBlock(
        pdb_block, removeHs=False, sanitize=False, proximityBonding=True
    )
    if pdb_mol is None:
        raise ValueError("Chem.MolFromPDBBlock returned None")

    pdb_mol = Chem.RemoveHs(pdb_mol)

    if pdb_mol.GetNumAtoms() != template.GetNumAtoms():
        raise ValueError(f"Template/PDB heavy atom mismatch: template={template.GetNumAtoms()}, pdb={pdb_mol.GetNumAtoms()}")

    # Assign correct bond orders from the chemical template
    mol_with_bo = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
    Chem.SanitizeMol(mol_with_bo)
    mol_with_bo.SetProp("_Name", os.path.basename(pdb_file).replace(".pdb", ""))

    return mol_with_bo

# -----------------------

def compute_dihedral_features(traj: md.Trajectory) -> np.ndarray:
    """Computes phi and psi angles, returning concatenated features."""
    phi_indices, raw_phi = md.compute_phi(traj)
    psi_indices, raw_psi = md.compute_psi(traj)

    phi_angles = np.nan_to_num(raw_phi, nan=0.0)
    psi_angles = np.nan_to_num(raw_psi, nan=0.0)

    # Sort columns by residue index for reproducibility
    if phi_indices.size:
        phi_residue_ids = [traj.topology.atom(idxs[1]).residue.index for idxs in phi_indices]
        order = np.argsort(phi_residue_ids)
        phi_angles = phi_angles[:, order]
    
    if psi_indices.size:
        psi_residue_ids = [traj.topology.atom(idxs[1]).residue.index for idxs in psi_indices]
        order = np.argsort(psi_residue_ids)
        psi_angles = psi_angles[:, order]
    
    feature_blocks = []
    if phi_angles.size:
        feature_blocks.append(phi_angles)
    if psi_angles.size:
        feature_blocks.append(psi_angles)
    
    if not feature_blocks:
        raise ValueError("No dihedral angles could be computed for the trajectory.")
    
    return np.concatenate(feature_blocks, axis=1)


def process_peptide(subsampled_traj: md.Trajectory, pdb_file: str, dataset_key: str):
    """
    Processes the pre-subsampled MDTraj object (max 10K frames) to find conformers.
    Uses K-Medoids to find the most representative (medoid) structures.
    Extracts only peptide heavy atoms (no H, no solvent).
    """
    total_frames = len(subsampled_traj)
    topology = subsampled_traj.topology

    if total_frames == 0:
        return None

    # 1a. Extract peptide heavy atoms only (no H, no solvent)
    protein_idx = topology.select("protein")
    if protein_idx.size == 0:
        print(f"\n  Error: No protein atoms found in {dataset_key}")
        return None
    
    heavy_idx = [i for i in protein_idx if topology.atom(i).element.symbol != 'H']
    if not heavy_idx:
        print(f"\n  Error: No heavy protein atoms found in {dataset_key}")
        return None
    
    # Create peptide-heavy-only trajectory
    topology_heavy = topology.subset(heavy_idx)
    subsampled_traj_heavy = subsampled_traj.atom_slice(heavy_idx)

    # 1b. Robust RDKit construction (using the first frame for structure only)
    try:
        rdkit_mol = create_rdkit_mol_robust(subsampled_traj_heavy[:1], pdb_file)
        smiles = Chem.MolToSmiles(rdkit_mol)
    except Exception as err:
        print(f"\n  Error creating RDKit mol for {dataset_key}: {err}")
        return None
    
    # VALIDATION: Check atom count consistency
    rdkit_n_atoms = rdkit_mol.GetNumAtoms()
    traj_n_atoms = subsampled_traj_heavy.n_atoms
    
    if rdkit_n_atoms != traj_n_atoms:
        print(f"\n  ERROR {dataset_key}: Atom count mismatch! RDKit={rdkit_n_atoms}, Traj={traj_n_atoms}")
        return None
    
    # VALIDATION: Check that all atoms are heavy (non-H)
    rdkit_n_heavy = rdkit_mol.GetNumHeavyAtoms()
    if rdkit_n_heavy != rdkit_n_atoms:
        print(f"\n  ERROR {dataset_key}: RDKit mol contains hydrogens! Heavy={rdkit_n_heavy}, Total={rdkit_n_atoms}")
        return None
    
    # VALIDATION: Check bond count (should have bonds for a peptide)
    rdkit_n_bonds = rdkit_mol.GetNumBonds()
    if rdkit_n_bonds == 0:
        print(f"\n  ERROR {dataset_key}: RDKit mol has no bonds!")
        return None

    # 2. Superpose and compute features on the sampled trajectory
    backbone_indices = topology_heavy.select('backbone')
    if backbone_indices.size:
        subsampled_traj_heavy.superpose(subsampled_traj_heavy, frame=0, atom_indices=backbone_indices)
    else:
        subsampled_traj_heavy.superpose(subsampled_traj_heavy, frame=0)

    try:
        dihedrals = compute_dihedral_features(subsampled_traj_heavy)
    except ValueError:
        return None

    # 3. K-Medoids Clustering
    desired_clusters = min(N_CONFORMERS, len(subsampled_traj_heavy))

    kmed = KMedoids(
        n_clusters=desired_clusters,
        metric='euclidean',
        random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmed.fit(dihedrals)

    # 4. Conformer Selection: Use the Medoid (cluster center) as the representative
    selected_conformers = []
    
    # medoid_indices are the indices *within* the subsampled_traj_heavy
    medoid_indices = kmed.medoid_indices_

    for medoid_idx in medoid_indices:
        # Coordinates are retrieved from the heavy-atom-only trajectory
        coordinates = np.array(subsampled_traj_heavy.xyz[medoid_idx], dtype=np.float32)
        
        # VALIDATION: Final check that coordinates match RDKit mol
        if coordinates.shape[0] != rdkit_mol.GetNumAtoms():
            print(f"\n  ERROR {dataset_key}: Conformer coordinate mismatch! Coords={coordinates.shape[0]}, RDKit={rdkit_mol.GetNumAtoms()}")
            continue
        
        selected_conformers.append({"coordinates": coordinates})

    return {
        "residues": dataset_key,
        "smiles": smiles,
        "rdkit_mol": rdkit_mol,
        "conformers": selected_conformers,
    }


def _process_single(job_tuple):
    """
    Worker function for parallel processing. Loads ONLY the sampled frames to save memory.
    Uses md.load with a minimal atom count to get total_frames reliably.
    """
    xtc_file, pdb_file, dataset_key = job_tuple

    try:        
        subsampled_traj = md.load(xtc_file, top=pdb_file, stride=100)
    except Exception as exc:
        print(f"\n  Error loading/sampling {dataset_key}: {exc}")
        return dataset_key, None
        
    # Process the subsampled trajectory
    result = process_peptide(subsampled_traj, pdb_file, dataset_key)

    if result is None:
        print(f"\n  Error processing {dataset_key} (RDKit/Dihedral calculation failed).")
        
    return dataset_key, result

# --- Main Execution ---
def main():
    if not os.path.isdir(DATASET_BASE_DIR):
        print(f"Error: Dataset directory not found at: {DATASET_BASE_DIR}")
        return

    final_conformer_dataset = {}
    
    # --- Print Setup ---
    print(f"{'='*60}")
    print(f"Starting Conformer Generation")
    print(f"{'='*60}")
    print(f"Base Directory: {DATASET_BASE_DIR}")
    print(f"Workers: {MAX_PARALLEL_WORKERS}")
    print(f"Max Frames for Clustering: {MAX_FRAMES_FOR_CLUSTERING}")
    print(f"Output File: {OUTPUT_PICKLE_FILE}")
    print(f"{'='*60}\n")
    
    # --- 1. Identify all jobs ---
    peptide_dirs = sorted(os.listdir(DATASET_BASE_DIR))
    jobs = []
    
    print("Collecting and verifying peptide files...")
    for peptide_name in peptide_dirs:
        peptide_path = os.path.join(DATASET_BASE_DIR, peptide_name)
        if not os.path.isdir(peptide_path) or len(peptide_name) != 4:
            continue
            
        xtc_file = os.path.join(peptide_path, f'{peptide_name}.xtc')
        pdb_file = os.path.join(peptide_path, f'{peptide_name}.pdb')
        
        if os.path.exists(xtc_file) and os.path.exists(pdb_file):
            jobs.append((xtc_file, pdb_file, peptide_name))
    
    if not jobs:
        print("Error: No valid peptide data found. Exiting.")
        return

    # --- 2. Process jobs in parallel ---
    worker_count = min(MAX_PARALLEL_WORKERS, len(jobs))
    print(f"Found {len(jobs)} valid peptides.")
    print(f"Processing in parallel with {worker_count} workers...\n")

    results_list = []
    
    if worker_count == 1:
        # Sequential processing
        for job in tqdm(jobs, desc="Processing peptides", unit="peptide"):
            dataset_key, payload = _process_single(job)
            if payload:
                results_list.append((dataset_key, payload))
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_key = {
                executor.submit(_process_single, job): job[2] for job in jobs
            }
            
            with tqdm(total=len(jobs), desc="Processing peptides", unit="peptide") as pbar:
                for future in as_completed(future_to_key):
                    dataset_key = future_to_key[future]
                    try:
                        key, payload = future.result()
                    except Exception as exc:
                        print(f"\n  Fatal error in worker for {dataset_key}: {exc}")
                        pbar.update(1)
                        continue
                    
                    if payload:
                        results_list.append((key, payload))
                    
                    pbar.update(1)

    # Build final dataset
    successful = 0
    for key, payload in results_list:
        final_conformer_dataset[key] = payload
        successful += 1
    
    failed = len(jobs) - successful
    
    # --- 3. Print statistics ---
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total: {len(jobs)}")
    print(f"  Success rate: {100*successful/len(jobs):.1f}%")
    
    # --- 4. Save the final dataset ---
    print(f"\n{'='*60}")
    print(f"Saving all data to {OUTPUT_PICKLE_FILE}...")
    print(f"{'='*60}")
    
    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(final_conformer_dataset, f)
    
    print("✓ Done!")
    print(f"Total peptides in final dataset: {len(final_conformer_dataset)}")


if __name__ == "__main__":
    main()