"""
Timewarp 4AA trajectory preprocessing.

Reads the raw 4AA-large Timewarp dataset (per-peptide trajectory + pdb +
npz files) and writes a preprocessed trajectory pickle per peptide into
a mirror directory (multiprocessed for throughput).

Output: ${TIMEWARP_DATA_ROOT}/data/4AA-large/4AA-large-processed/{split}/

Consumed by:
- TimewarpTrajectoryDataset in experiments/data_load/data_loader.py
- configs_official/trajectory/timewarp/*.yaml (train/val/test_traj_dir
  point at the {split}/ subdirs this script produces)

This is distinct from data_gen/preprocess/sample_conformer_AA.py (which clusters
trajectories into representative conformers) and from
data_gen/preprocess/process_timewarp.py (which writes one combined conformer pkl
for the tetrapeptide_drugs_pretrained config). This script produces the
trajectory-level artefacts that feed TimewarpTrajectoryDataset, not the
conformer-level ones.
"""

import os
import pickle
import tempfile
import numpy as np
import mdtraj

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_TIMEWARP_ROOT = os.environ.get('TIMEWARP_DATA_ROOT', '')
DATASET_BASE_DIR = os.path.join(_TIMEWARP_ROOT, 'data/4AA-large/4AA-large')
OUTPUT_DIR       = os.path.join(_TIMEWARP_ROOT, 'data/4AA-large/4AA-large-processed')

# You can tune this if you see memory issues.
# e.g. export TIMEWARP_NUM_WORKERS=4
NUM_WORKERS = int(os.environ.get("TIMEWARP_NUM_WORKERS",
                                 max(1, min(8, cpu_count() // 2 or 1))))

PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

# ----------------------------------------------------------------------------- 
# Helpers
# ----------------------------------------------------------------------------- 

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


def peptide_heavy_traj_from_system(pdb_file: str, positions_full: np.ndarray):
    """
    Given a PDB file and full positions (T, N_full, 3), return:
      - traj_pep_heavy: MDTraj trajectory with peptide heavy atoms only
      - top_pep_heavy: topology for peptide heavy atoms
      - indices_pep_heavy: indices of peptide heavy atoms in the full system
    """
    full_traj_pdb = mdtraj.load_pdb(pdb_file)
    top_full = full_traj_pdb.topology

    protein_idx = top_full.select("protein")
    if protein_idx.size == 0:
        raise ValueError(f"No protein atoms found in PDB: {pdb_file}")

    heavy_idx = [i for i in protein_idx if top_full.atom(i).element.symbol != 'H']
    if not heavy_idx:
        raise ValueError(f"No heavy protein atoms found in PDB: {pdb_file}")

    top_pep_heavy = top_full.subset(heavy_idx)
    positions_heavy = positions_full[:, heavy_idx, :]  # (T, N_pep_heavy, 3)

    traj_pep_heavy = mdtraj.Trajectory(positions_heavy, top_pep_heavy)
    traj_pep_heavy.center_coordinates()
    traj_pep_heavy.superpose(traj_pep_heavy, frame=0)

    return traj_pep_heavy, top_pep_heavy, np.array(heavy_idx, dtype=np.int64)


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


def pdb_block_to_peptide_mol(pdb_block: str, seq: str, name: str):
    """
    Convert a peptide-only PDB block (heavy atoms) to an RDKit Mol whose
    bond orders are taken from a sequence-derived template.
    """
    template = build_template_from_seq(seq)  # heavy-atom template

    pdb_mol = Chem.MolFromPDBBlock(
        pdb_block,
        removeHs=False,
        sanitize=False,
        proximityBonding=True,
    )
    if pdb_mol is None:
        raise ValueError(f"Chem.MolFromPDBBlock returned None for {name}")

    pdb_mol = Chem.RemoveHs(pdb_mol)

    if pdb_mol.GetNumAtoms() != template.GetNumAtoms():
        raise ValueError(
            f"Template/PDB heavy atom mismatch for {name}: "
            f"template={template.GetNumAtoms()}, pdb={pdb_mol.GetNumAtoms()}"
        )

    mol_with_bo = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
    Chem.SanitizeMol(mol_with_bo)
    mol_with_bo.SetProp("_Name", name)
    return mol_with_bo


def compare_bonds(topology, rdkit_mol):
    """
    Return sets of (extra, missing) bonds comparing RDKit vs MDTraj topology.
    Assumes both are heavy-atom only.
    """
    topo_bonds = {
        tuple(sorted((bond.atom1.index, bond.atom2.index)))
        for bond in topology.bonds
    }
    rdkit_bonds = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in rdkit_mol.GetBonds()
    }

    extra = rdkit_bonds - topo_bonds      # bonds present only in RDKit
    missing = topo_bonds - rdkit_bonds    # bonds missing from RDKit
    return extra, missing


# ----------------------------------------------------------------------------- 
# Per-peptide logic
# ----------------------------------------------------------------------------- 

def process_peptide(pdb_file, npz_file, dataset_key, max_frames=None):
    errors = []

    # ---- Stage 1: Load arrays and peptide-only heavy-atom trajectory --------
    try:
        with np.load(npz_file, mmap_mode='r') as npz_data:
            positions_full = npz_data['positions']          # (T, N_full, 3)
            potential_energies = npz_data['energies'][:, 0] # (T,)

            if max_frames is not None and positions_full.shape[0] > max_frames:
                positions_full = positions_full[:max_frames]
                potential_energies = potential_energies[:max_frames]

        if positions_full.shape[0] != potential_energies.shape[0]:
            raise ValueError(
                f"Frame mismatch (traj={positions_full.shape[0]}, "
                f"energy={potential_energies.shape[0]})"
            )

        traj_pep_heavy, top_pep_heavy, heavy_idx = peptide_heavy_traj_from_system(
            pdb_file, positions_full
        )
    except Exception as exc:
        errors.append(("load_data_or_slice", repr(exc)))
        return None, errors

    # ---- Stage 2: RDKit molecule via sequence template ----------------------
    try:
        peptide_name = seq_from_filename(pdb_file)  # e.g. 'AACG'
        pdb_block_pep = traj_to_pdb_block(traj_pep_heavy)

        rdkit_mol = pdb_block_to_peptide_mol(
            pdb_block_pep,
            seq=peptide_name,
            name=os.path.basename(pdb_file).replace(".pdb", ""),
        )
    except Exception as exc:
        print(f"Error in RDKit molecule construction for {dataset_key}: {exc}")
        errors.append(("rdkit_mol_template", repr(exc)))
        rdkit_mol = None
        peptide_name = seq_from_filename(pdb_file)  # ensure defined

    # Optional connectivity check
    if rdkit_mol is not None:
        try:
            extra_bonds, missing_bonds = compare_bonds(top_pep_heavy, rdkit_mol)
            if extra_bonds or missing_bonds:
                errors.append(
                    (
                        "connectivity_mismatch",
                        {
                            "extra_bonds_rdkit_only": sorted(extra_bonds),
                            "missing_bonds_not_in_rdkit": sorted(missing_bonds),
                            "n_atoms": rdkit_mol.GetNumAtoms(),
                        },
                    )
                )
        except Exception as exc:
            print(f"Error in connectivity check for {dataset_key}: {exc}")
            errors.append(("connectivity_check_failed", repr(exc)))

    # SMILES
    smiles = None
    if rdkit_mol is not None:
        try:
            smiles = Chem.MolToSmiles(rdkit_mol)
        except Exception as exc:
            errors.append(("smiles", repr(exc)))

    # ---- Stage 3: Pack coordinates + energies (peptide heavy atoms only) ----
    coords = np.asarray(traj_pep_heavy.xyz, dtype=np.float32)  # (T, N_pep, 3)
    coords_nt3 = np.transpose(coords, (1, 0, 2))               # (N_pep, T, 3)

    potential_energies = np.asarray(potential_energies, dtype=np.float32)

    payload = {
        "dataset_key": dataset_key,
        "peptide_name": peptide_name,                     # e.g. 'AACG'
        "smiles": smiles,                                 # may be None
        "rdkit_mol": rdkit_mol,                           # may be None if RDKit failed
        "potential_energies_kj_mol": potential_energies,  # shape (T,)
        "coordinates_np": coords_nt3,                     # (N_pep, T, 3)
        "coordinates_mdtraj": traj_pep_heavy              # mdtraj.Trajectory
    }
    return payload, errors


# ----------------------------------------------------------------------------- 
# Parallel driver: worker and split processing
# ----------------------------------------------------------------------------- 

def _worker_process_one(args):
    """
    Worker function used in multiprocessing Pool.

    Returns:
      (dataset_key, success_bool, errors_list)
    """
    split_name, split_dir, base, max_frames = args

    npz_file = os.path.join(split_dir, f"{base}-traj-arrays.npz")
    pdb_file = os.path.join(split_dir, f"{base}-traj-state0.pdb")
    dataset_key = f"{split_name}/{base}"

    out_path = os.path.join(OUTPUT_DIR, split_name, f"{base}.pkl")
    if os.path.exists(out_path):
        print(f"Warning: {out_path} already exists, skipping {base}")
        return dataset_key, False, []

    payload, errors = process_peptide(pdb_file, npz_file, dataset_key, max_frames=max_frames)

    success = payload is not None

    # Each worker writes its own payload to avoid sending large objects back
    if success:
        os.makedirs(os.path.join(OUTPUT_DIR, split_name), exist_ok=True)
        try:
            with open(out_path, 'wb') as f:
                pickle.dump(payload, f, protocol=PICKLE_PROTOCOL)
        except Exception as exc:
            # Treat this as a serious error
            errors.append(("pickle_dump_failed", repr(exc)))
            success = False

    return dataset_key, success, errors


def process_split(split_name, split_dir, max_frames=None):
    """
    Process all peptides in a split, using multiprocessing.
    Writes:
      - per-peptide payloads: {base}.pkl
      - tetrapeptide_conformers_{split_name}_errors.pkl
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split (workers={NUM_WORKERS})")
    print(f"{'='*60}")

    array_files = sorted(
        fname for fname in os.listdir(split_dir) if fname.endswith('-traj-arrays.npz')
    )
    bases = [fname.replace('-traj-arrays.npz', '') for fname in array_files]

    print(f"Found {len(bases)} peptides in {split_name}")

    if not bases:
        return 0, 0, []

    error_log = []
    total = len(bases)
    total_ok = 0

    args_iter = [
        (split_name, split_dir, base, max_frames) for base in bases
    ]

    if NUM_WORKERS > 1:
        with Pool(processes=NUM_WORKERS) as pool:
            for dataset_key, success, errors in tqdm(
                pool.imap_unordered(_worker_process_one, args_iter, chunksize=4),
                total=total,
                desc=split_name
            ):
                if errors:
                    error_log.append({"dataset": dataset_key, "errors": errors})
                if success:
                    total_ok += 1
    else:
        # Fallback to single-process if NUM_WORKERS=1
        for args in tqdm(args_iter, total=total, desc=split_name):
            dataset_key, success, errors = _worker_process_one(args)
            if errors:
                error_log.append({"dataset": dataset_key, "errors": errors})
            if success:
                total_ok += 1

    # Write error log for this split
    if error_log:
        err_pkl = os.path.join(OUTPUT_DIR, split_name, f"{split_name}_errors.pkl")
        with open(err_pkl, 'wb') as f:
            pickle.dump(error_log, f, protocol=PICKLE_PROTOCOL)
        print(f"{split_name}: {len(error_log)} entries had errors (saved to {err_pkl})")

    return total_ok, total, error_log


# ----------------------------------------------------------------------------- 
# Main
# ----------------------------------------------------------------------------- 

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset base: {DATASET_BASE_DIR}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"Using up to   {NUM_WORKERS} workers")

    splits = [
        ('train', os.path.join(DATASET_BASE_DIR, 'train'), None),    # full train
        ('val',   os.path.join(DATASET_BASE_DIR, 'val'),   50000),   # truncate val
        ('test',  os.path.join(DATASET_BASE_DIR, 'test'),  50000),   # truncate test
    ]

    all_errors = []
    total_ok_all = 0
    total_all = 0

    for split_name, split_dir, max_frames in splits:
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping {split_name}")
            continue

        ok, total, errs = process_split(split_name, split_dir, max_frames=max_frames)
        total_ok_all += ok
        total_all += total
        all_errors.extend(errs)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed successfully: {total_ok_all}/{total_all}")
    print(f"Total peptides with any errors: {len(all_errors)}")


if __name__ == "__main__":
    main()