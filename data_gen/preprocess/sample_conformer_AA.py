"""
Timewarp 4AA conformer preprocessing — per-split-output variant.

Reads the 4AA-large Timewarp dataset and writes a PER-SPLIT pickle of
representative conformers per peptide (via KMedoids clustering) using
a robust RDKit mol construction pipeline (template matching + heavy-atom
/ bond-count validation).

Output (one run per split, edit OUTPUT_PICKLE_FILE + `splits` in main()):
  ${MD_DATA_ROOT}/tetrapeptide_conformers_{train,val}.pkl   (N_CONFORMERS=20)

Consumed by:
- configs_official/conformer/tetrapeptide/
  mdwarp_tetrapeptide_pretrain_noH_1000_kabsch_conf_basic_es_order_3.yaml
  (train_conf_path / val_conf_path reference per-split pickles).

Compare with data_gen/preprocess/process_timewarp.py, which shares the same
KMedoids / dihedral / plotting internals but produces a single combined
pickle (all splits -> one file, N=10) for the
tetrapeptide_drugs_pretrained_*.yaml config path.
"""

import os
import pickle
import tempfile
import numpy as np
import mdtraj
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn_extra.cluster import KMedoids
import warnings
import matplotlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuration ---

# 1. SET THIS to the path of your "4AA-large" dataset
DATASET_BASE_DIR = os.path.join(os.environ.get('TIMEWARP_DATA_ROOT', ''), 'data/4AA-large/4AA-large')

# 2. Set the name of the output file
OUTPUT_PICKLE_FILE = 'tetrapeptide_conformers_val.pkl'

# 3. Set the number of conformers to select
N_CONFORMERS = 20

# 4. Directory for diagnostic plots
PLOT_OUTPUT_BASE_DIR = './tetrapeptide_plots'

# 5. Maximum frames for clustering/plotting
MAX_FRAMES_FOR_CLUSTERING = 10_000

# 6. Maximum number of CPU workers for parallel processing
MAX_PARALLEL_WORKERS = 8

# -----------------------
# Robust RDKit molecule construction helpers
# -----------------------

def seq_from_filename(pdb_file: str) -> str:
    """
    Extract 4-letter tetrapeptide sequence from filename like 'CIHK-traj-state0.pdb'.
    """
    base = os.path.basename(pdb_file)
    prefix = base.split('-')[0]  # 'CIHK' in 'CIHK-traj-state0.pdb'
    if len(prefix) != 4:
        raise ValueError(f"Unexpected peptide name in filename: {base}")
    return prefix


def build_template_from_seq(seq: str) -> Chem.Mol:
    """
    Build a chemically-correct peptide template from a 1-letter sequence
    using RDKit's MolFromSequence, then strip H's so we only have heavy atoms.
    """
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        raise ValueError(f"MolFromSequence failed for sequence: {seq}")
    
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return mol


def traj_to_pdb_block(traj: mdtraj.Trajectory) -> str:
    """
    Convert a single-frame MDTraj trajectory to a PDB block (string).
    """
    if traj.n_frames == 0:
        raise ValueError("Empty trajectory passed to traj_to_pdb_block")

    frame0 = traj[0]

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.pdb', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        frame0.save_pdb(tmp_path)
        with open(tmp_path, 'r') as f:
            pdb_block = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return pdb_block


def create_rdkit_mol_robust(traj: mdtraj.Trajectory, pdb_file: str) -> Chem.Mol:
    """
    Robust RDKit molecule construction using sequence-based template matching.
    This is much more reliable than MolFromPDBFile for peptides.
    """
    # Extract sequence from filename
    seq = seq_from_filename(pdb_file)
    
    # Build template with correct bond orders
    template = build_template_from_seq(seq)
    
    # Convert first frame to PDB block
    pdb_block = traj_to_pdb_block(traj)
    
    # Load from PDB block with proximity bonding
    pdb_mol = Chem.MolFromPDBBlock(
        pdb_block,
        removeHs=False,
        sanitize=False,
        proximityBonding=True,
    )
    if pdb_mol is None:
        raise ValueError(f"Chem.MolFromPDBBlock returned None")
    
    pdb_mol = Chem.RemoveHs(pdb_mol)
    
    if pdb_mol.GetNumAtoms() != template.GetNumAtoms():
        raise ValueError(
            f"Template/PDB heavy atom mismatch: "
            f"template={template.GetNumAtoms()}, pdb={pdb_mol.GetNumAtoms()}"
        )
    
    # Assign bond orders from template - this is the key step!
    mol_with_bo = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
    Chem.SanitizeMol(mol_with_bo)
    mol_with_bo.SetProp("_Name", os.path.basename(pdb_file).replace(".pdb", ""))
    
    return mol_with_bo

# -----------------------

def create_residue_labels(topology):
    """
    Returns a mapping from residue index to a friendly label (e.g., ALA2).
    """
    labels = {}
    for residue in topology.residues:
        labels[residue.index] = f"{residue.name}{residue.index + 1}"
    return labels

def compute_dihedral_features(traj):
    """
    Computes phi and psi angles, returning concatenated features alongside per-residue mappings.
    """
    phi_indices, raw_phi = mdtraj.compute_phi(traj)
    psi_indices, raw_psi = mdtraj.compute_psi(traj)

    # Convert to numpy arrays with NaNs replaced for clustering stability
    phi_angles = np.nan_to_num(raw_phi, nan=0.0)
    psi_angles = np.nan_to_num(raw_psi, nan=0.0)

    topology = traj.topology

    phi_residue_ids = [topology.atom(idxs[1]).residue.index for idxs in phi_indices]
    psi_residue_ids = [topology.atom(idxs[1]).residue.index for idxs in psi_indices]

    # Ensure columns are sorted by residue index for reproducibility
    if phi_residue_ids:
        order = np.argsort(phi_residue_ids)
        phi_angles = phi_angles[:, order]
        phi_residue_ids = [phi_residue_ids[i] for i in order]

    if psi_residue_ids:
        order = np.argsort(psi_residue_ids)
        psi_angles = psi_angles[:, order]
        psi_residue_ids = [psi_residue_ids[i] for i in order]

    feature_blocks = []
    if phi_angles.size:
        feature_blocks.append(phi_angles)
    if psi_angles.size:
        feature_blocks.append(psi_angles)

    if not feature_blocks:
        raise ValueError("No dihedral angles could be computed for the trajectory.")

    dihedral_features = np.concatenate(feature_blocks, axis=1)

    phi_per_residue = {
        residue_id: phi_angles[:, idx]
        for idx, residue_id in enumerate(phi_residue_ids)
    }
    psi_per_residue = {
        residue_id: psi_angles[:, idx]
        for idx, residue_id in enumerate(psi_residue_ids)
    }

    return dihedral_features, phi_per_residue, psi_per_residue

def save_phi_psi_plots(phi_per_residue, psi_per_residue, cluster_labels,
                       residue_labels, plot_dir, peptide_stub):
    """
    Saves (1) phi/psi 2D distributions per residue and (2) clustering assignments.
    """
    os.makedirs(plot_dir, exist_ok=True)
    common_residue_ids = sorted(set(phi_per_residue) & set(psi_per_residue))
    if not common_residue_ids:
        return {}

    n_cols = len(common_residue_ids)
    frames = len(cluster_labels)
    degrees = lambda arr: np.degrees(arr)

    # Distribution plot
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
    for ax, residue_id in zip(axes.flat, common_residue_ids):
        phi_deg = degrees(phi_per_residue[residue_id])
        psi_deg = degrees(psi_per_residue[residue_id])
        ax.hist2d(
            phi_deg,
            psi_deg,
            bins=72,
            range=[[-180, 180], [-180, 180]],
            cmap="viridis",
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel(r"$\phi$ (deg)")
        ax.set_ylabel(r"$\psi$ (deg)")
        ax.set_title(residue_labels.get(residue_id, f"Residue {residue_id + 1}"))
    fig.suptitle("Phi/Psi Angle Distributions", fontsize=14, y=0.98)
    dist_path = os.path.join(plot_dir, f"{peptide_stub}_phi_psi_distribution.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(dist_path, dpi=200)
    plt.close(fig)

    # Cluster assignments plot
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
    cmap = plt.cm.get_cmap("tab10", np.unique(cluster_labels).size or 1)
    for ax, residue_id in zip(axes.flat, common_residue_ids):
        phi_deg = degrees(phi_per_residue[residue_id])
        psi_deg = degrees(psi_per_residue[residue_id])
        sc = ax.scatter(
            phi_deg,
            psi_deg,
            s=8,
            c=cluster_labels,
            cmap=cmap,
            alpha=0.8,
            linewidths=0,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel(r"$\phi$ (deg)")
        ax.set_ylabel(r"$\psi$ (deg)")
        ax.set_title(residue_labels.get(residue_id, f"Residue {residue_id + 1}"))
    fig.suptitle("Cluster Assignments in Dihedral Space", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    cluster_path = os.path.join(plot_dir, f"{peptide_stub}_cluster_assignments.png")
    fig.savefig(cluster_path, dpi=200)
    plt.close(fig)

    return {
        "distribution_plot": dist_path,
        "cluster_plot": cluster_path,
        "common_residues": [residue_labels.get(rid, str(rid)) for rid in common_residue_ids],
        "n_frames": frames,
    }

def process_peptide(pdb_file, npz_file, dataset_key):
    """
    Processes a single tetrapeptide trajectory and returns detailed conformer data.
    Uses robust RDKit molecule construction with template matching.
    """
    residue_short_names = dataset_key.split('/')[-1]
    if not os.path.exists(pdb_file) or not os.path.exists(npz_file):
        print(f"Skipping {dataset_key}: Missing PDB or NPZ file.")
        return None
    
    # 1. Load topology and trajectory arrays
    try:
        pdb = mdtraj.load_pdb(pdb_file)
        topology = pdb.topology
        with np.load(npz_file) as npz_data:
            positions = npz_data['positions']
            potential_energies = npz_data['energies'][:, 0]
        full_traj = mdtraj.Trajectory(positions, topology)
        if len(full_traj) != len(potential_energies):
            print(f"Warning: Mismatch in frames for {dataset_key}. Skipping.")
            return None
    except Exception as exc:
        print(f"Error loading data for {dataset_key}: {exc}")
        return None

    # 1b. Extract peptide heavy atoms only (no H, no solvent)
    protein_idx = topology.select("protein")
    if protein_idx.size == 0:
        print(f"Error: No protein atoms found in {dataset_key}")
        return None
    
    heavy_idx = [i for i in protein_idx if topology.atom(i).element.symbol != 'H']
    if not heavy_idx:
        print(f"Error: No heavy protein atoms found in {dataset_key}")
        return None
    
    # Create peptide-heavy-only trajectory
    topology_heavy = topology.subset(heavy_idx)
    full_traj_heavy = full_traj.atom_slice(heavy_idx)

    # 2. IMPROVED: Try robust RDKit construction (ONLY use heavy-atom trajectory)
    rdkit_mol = None
    try:
        # Use sequence-based template matching on heavy-atom trajectory
        rdkit_mol = create_rdkit_mol_robust(full_traj_heavy, pdb_file)
    except Exception as err:
        # If template matching fails, skip this peptide
        # NOTE: We do NOT fall back to MolFromPDBFile because that would load
        # the full PDB (with H, water, ions) which won't match our heavy-atom coordinates
        print(f"  Skipping {dataset_key}: RDKit construction failed: {err}")
        return None

    # REQUIREMENT: Must have RDKit mol to continue
    if rdkit_mol is None:
        print(f"  Skipping {dataset_key}: Unable to construct RDKit molecule.")
        return None

    # VALIDATION: Check atom count consistency
    rdkit_n_atoms = rdkit_mol.GetNumAtoms()
    traj_n_atoms = full_traj_heavy.n_atoms
    
    if rdkit_n_atoms != traj_n_atoms:
        print(f"  ERROR {dataset_key}: Atom count mismatch! RDKit={rdkit_n_atoms}, Traj={traj_n_atoms}")
        return None
    
    # VALIDATION: Check that all atoms are heavy (non-H)
    rdkit_n_heavy = rdkit_mol.GetNumHeavyAtoms()
    if rdkit_n_heavy != rdkit_n_atoms:
        print(f"  ERROR {dataset_key}: RDKit mol contains hydrogens! Heavy={rdkit_n_heavy}, Total={rdkit_n_atoms}")
        return None
    
    # VALIDATION: Check bond count (should have bonds for a peptide)
    rdkit_n_bonds = rdkit_mol.GetNumBonds()
    if rdkit_n_bonds == 0:
        print(f"  ERROR {dataset_key}: RDKit mol has no bonds!")
        return None

    smiles = None
    try:
        smiles = Chem.MolToSmiles(rdkit_mol)
    except Exception as exc:
        print(f"Warning: RDKit SMILES generation failed for {pdb_file}: {exc}")
        smiles = None

    # Subsample frames for clustering if needed
    total_frames = len(full_traj_heavy)
    if total_frames <= MAX_FRAMES_FOR_CLUSTERING:
        sample_frame_indices = np.arange(total_frames)
    else:
        sample_frame_indices = np.linspace(
            0, total_frames - 1, MAX_FRAMES_FOR_CLUSTERING, dtype=int
        )
        sample_frame_indices = np.unique(sample_frame_indices)

    sampled_energies = potential_energies[sample_frame_indices]
    analysis_traj = full_traj_heavy[sample_frame_indices]
    backbone_indices = topology_heavy.select('backbone')
    if backbone_indices.size:
        analysis_traj.superpose(analysis_traj, frame=0, atom_indices=backbone_indices)
    else:
        analysis_traj.superpose(analysis_traj, frame=0)
    try:
        dihedrals, phi_per_residue, psi_per_residue = compute_dihedral_features(analysis_traj)
    except ValueError as err:
        print(f"Error computing dihedrals for {dataset_key}: {err}")
        return None

    num_sampled_frames = len(analysis_traj)
    desired_clusters = min(N_CONFORMERS, num_sampled_frames)
    if desired_clusters == 0:
        print(f"Warning: No frames loaded for {dataset_key}. Skipping.")
        return None
    if desired_clusters < N_CONFORMERS:
        print(f"  Warning: Only {num_sampled_frames} frames available; using {desired_clusters} clusters.")

    kmed = KMedoids(
        n_clusters=desired_clusters,
        metric='euclidean',
        random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster_labels = kmed.fit_predict(dihedrals)

    peptide_stub = dataset_key.replace('/', '_')
    plot_dir = os.path.join(
        PLOT_OUTPUT_BASE_DIR,
        os.path.dirname(dataset_key),
        os.path.basename(dataset_key)
    )
    # plot_info = save_phi_psi_plots(
    #     phi_per_residue,
    #     psi_per_residue,
    #     cluster_labels,
    #     residue_labels,
    #     plot_dir,
    #     peptide_stub
    # )

    selected_conformers = []
    for cluster_id in range(desired_clusters):
        member_indices = np.where(cluster_labels == cluster_id)[0]
        if len(member_indices) == 0:
            print(f"  Warning: Cluster {cluster_id} is empty. Skipping.")
            continue

        cluster_energies = sampled_energies[member_indices]
        idx_within_cluster = int(np.argmin(cluster_energies))
        original_traj_idx = int(sample_frame_indices[member_indices[idx_within_cluster]])

        # Get coordinates from the peptide-heavy-only trajectory
        coordinates = np.array(full_traj_heavy.xyz[original_traj_idx], dtype=np.float32)
        
        # VALIDATION: Final check that coordinates match RDKit mol
        if coordinates.shape[0] != rdkit_mol.GetNumAtoms():
            print(f"  ERROR {dataset_key}: Conformer coordinate mismatch! Coords={coordinates.shape[0]}, RDKit={rdkit_mol.GetNumAtoms()}")
            continue
        
        conformer_data = {
            "potential_energy_kj_mol": float(potential_energies[original_traj_idx]),
            "coordinates": coordinates,
        }
        selected_conformers.append(conformer_data)

    return {
        "residues": residue_short_names,
        "smiles": smiles,
        "rdkit_mol": rdkit_mol,
        "conformers": selected_conformers,
    }


def _process_single(job):
    """
    Wrapper for parallel processing.
    """
    pdb_file, npz_file, dataset_key = job
    return dataset_key, process_peptide(pdb_file, npz_file, dataset_key)

# --- Main Execution ---
def main():
    if not os.path.isdir(DATASET_BASE_DIR):
        print(f"Error: Dataset directory not found at: {DATASET_BASE_DIR}")
        print("Please set the 'DATASET_BASE_DIR' variable in the script.")
        return

    final_conformer_dataset = {}
    splits = ['val']

    print(f"Starting conformer generation from: {DATASET_BASE_DIR}")
    print(f"Using {MAX_PARALLEL_WORKERS} parallel workers")
    print(f"Output file: {OUTPUT_PICKLE_FILE}")

    for split in splits:
        split_dir = os.path.join(DATASET_BASE_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: Split directory '{split}' not found. Skipping.")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing {split} set...")
        print(f"{'='*60}")
        
        array_files = sorted(
            [
                fname for fname in os.listdir(split_dir)
                if fname.endswith('-traj-arrays.npz')
            ]
        )

        if not array_files:
            print(f"  Warning: No '*-traj-arrays.npz' files found in {split_dir}.")
            continue

        jobs = []
        for i, npz_name in enumerate(array_files):
            base_name = npz_name.replace('-traj-arrays.npz', '')
            npz_file = os.path.join(split_dir, npz_name)
            pdb_file = os.path.join(split_dir, f"{base_name}-traj-state0.pdb")
            dataset_key = f"{split}/{base_name}"
            jobs.append((pdb_file, npz_file, dataset_key))

        if not jobs:
            continue

        worker_count = min(MAX_PARALLEL_WORKERS, len(jobs))
        print(f"Processing {len(jobs)} peptides with {worker_count} worker(s)...")

        successful = 0
        failed = 0
        
        if worker_count == 1:
            for job in tqdm(jobs, desc=f"{split} progress", unit="peptide"):
                dataset_key, payload = _process_single(job)
                if payload:
                    final_conformer_dataset[dataset_key] = payload
                    successful += 1
                else:
                    failed += 1
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_key = {
                    executor.submit(_process_single, job): job[2] for job in jobs
                }
                
                with tqdm(total=len(jobs), desc=f"{split} progress", unit="peptide") as pbar:
                    for future in as_completed(future_to_key):
                        dataset_key = future_to_key[future]
                        try:
                            key, payload = future.result()
                        except Exception as exc:
                            print(f"\n  Error processing {dataset_key}: {exc}")
                            failed += 1
                            pbar.update(1)
                            continue
                        
                        if payload:
                            final_conformer_dataset[key] = payload
                            successful += 1
                        else:
                            failed += 1
                        pbar.update(1)
        
        print(f"\n{split} results:")
        print(f"  ✓ Successful: {successful}")
        print(f"  ✗ Failed: {failed}")
        print(f"  Total: {len(jobs)}")
        print(f"  Success rate: {100*successful/len(jobs):.1f}%")

    # --- Save the final dataset ---
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total peptides processed successfully: {len(final_conformer_dataset)}")
    print(f"\nSaving all data to {OUTPUT_PICKLE_FILE}...")
    
    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(final_conformer_dataset, f)
        
    print("✓ Done!")

if __name__ == "__main__":
    main()
