"""
Preprocess MDGen trajectories for fast loading during training.

This script:
1. Loads each peptide trajectory with stride
2. Removes hydrogens
3. Centers and aligns
4. Converts to Angstroms
5. Saves preprocessed coordinates as compressed .npz

Usage:
    python data_gen/preprocess/process_mdgen.py --stride 100 --num_workers 32
"""

import os
import pickle
import argparse
import numpy as np
import mdtraj as md
from rdkit import Chem
from multiprocessing import Pool
from tqdm import tqdm


def get_keep_atoms(mol, z):
    """Get indices of non-hydrogen atoms."""
    mol_no = Chem.RemoveHs(mol)
    important_atoms = list(mol.GetSubstructMatch(mol_no))
    return important_atoms


def mol_2d(mol):
    """Get atomic numbers from RDKit mol."""
    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
    return atomic_number


def process_one_peptide(args):
    """Process a single peptide trajectory."""
    peptide_name, xtc_file, pdb_file, rdkit_mol, stride, remove_hs = args
    
    peptide_dir = os.path.dirname(xtc_file)
    cache_file = os.path.join(peptide_dir, f'{peptide_name}_cache_stride{stride}.npz')
    
    # Skip if already cached
    if os.path.exists(cache_file):
        return peptide_name, "already_cached", 0
    
    try:
        # Load trajectory with stride
        traj = md.load(xtc_file, top=pdb_file, stride=stride)
        original_frames = len(traj)
        
        # Remove hydrogens if needed
        if remove_hs:
            z_full = mol_2d(rdkit_mol)
            keep_idxs = get_keep_atoms(rdkit_mol, z_full)
            traj = traj.atom_slice(keep_idxs)
        
        # Center and align
        traj.center_coordinates()
        traj.superpose(traj, 0)
        
        # Convert to Angstroms (MDTraj uses nm)
        traj.xyz *= 10
        
        # Save to cache
        np.savez_compressed(
            cache_file,
            xyz=traj.xyz.astype(np.float32),  # (T, N, 3)
            num_atoms=traj.n_atoms,
            num_frames=traj.n_frames,
        )
        
        return peptide_name, "success", traj.n_frames
        
    except Exception as e:
        return peptide_name, f"error: {e}", 0


def main():
    parser = argparse.ArgumentParser(description='Preprocess MDGen trajectories')
    parser.add_argument('--data_path', type=str, 
                        default=os.path.join(os.environ.get('MDGEN_DATA_ROOT', ''), 'data/4AA_sims'),
                        help='Path to MDGen data directory')
    parser.add_argument('--pickle_path', type=str,
                        default='data_gen/tetrapeptide_conformers_mdgen.pkl',
                        help='Path to RDKit molecules pickle')
    parser.add_argument('--stride', type=int, default=100,
                        help='Stride for loading trajectories')
    parser.add_argument('--remove_hs', action='store_true', default=True,
                        help='Remove hydrogen atoms')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of parallel workers')
    args = parser.parse_args()
    
    print("="*60)
    print("MDGen Trajectory Preprocessing")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Pickle path: {args.pickle_path}")
    print(f"Stride: {args.stride}")
    print(f"Remove hydrogens: {args.remove_hs}")
    print(f"Workers: {args.num_workers}")
    print("="*60)
    
    # Load RDKit molecules
    print("\nLoading RDKit molecules...")
    with open(args.pickle_path, 'rb') as f:
        tetrapeptide_confs = pickle.load(f)
    print(f"Loaded {len(tetrapeptide_confs)} peptide structures")
    
    # Collect all peptides to process
    jobs = []
    for peptide_name in sorted(os.listdir(args.data_path)):
        peptide_dir = os.path.join(args.data_path, peptide_name)
        
        if not os.path.isdir(peptide_dir):
            continue
        
        # Check for required files
        xtc_file = os.path.join(peptide_dir, f'{peptide_name}.xtc')
        pdb_file = os.path.join(peptide_dir, f'{peptide_name}.pdb')
        
        if not (os.path.exists(xtc_file) and os.path.exists(pdb_file)):
            continue
        
        # Check if we have RDKit mol
        if peptide_name not in tetrapeptide_confs:
            print(f"Warning: {peptide_name} has trajectory but no RDKit mol in pickle")
            continue
        
        rdkit_mol = tetrapeptide_confs[peptide_name]['rdkit_mol']
        jobs.append((peptide_name, xtc_file, pdb_file, rdkit_mol, args.stride, args.remove_hs))
    
    print(f"\nFound {len(jobs)} peptides to process")
    
    # Process in parallel
    results = []
    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_one_peptide, jobs),
                             total=len(jobs),
                             desc="Processing peptides"):
                results.append(result)
    else:
        # Single-threaded for debugging
        for job in tqdm(jobs, desc="Processing peptides"):
            result = process_one_peptide(job)
            results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    
    success_count = sum(1 for _, status, _ in results if status == "success")
    cached_count = sum(1 for _, status, _ in results if status == "already_cached")
    error_count = sum(1 for _, status, _ in results if status.startswith("error"))
    
    print(f"Successfully processed: {success_count}")
    print(f"Already cached: {cached_count}")
    print(f"Errors: {error_count}")
    
    if error_count > 0:
        print("\nErrors:")
        for name, status, _ in results:
            if status.startswith("error"):
                print(f"  {name}: {status}")
    
    # Show frame counts
    total_frames = sum(frames for _, status, frames in results if status == "success")
    if success_count > 0:
        avg_frames = total_frames / success_count
        print(f"\nAverage frames per peptide: {avg_frames:.0f}")
    
    # Estimate cache size
    if success_count > 0:
        sample_name = next(name for name, status, _ in results if status == "success")
        sample_dir = os.path.join(args.data_path, sample_name)
        cache_file = os.path.join(sample_dir, f'{sample_name}_cache_stride{args.stride}.npz')
        if os.path.exists(cache_file):
            cache_size_mb = os.path.getsize(cache_file) / (1024**2)
            total_size_mb = cache_size_mb * (success_count + cached_count)
            print(f"\nCache size per peptide: ~{cache_size_mb:.1f} MB")
            print(f"Total cache size: ~{total_size_mb:.1f} MB")
    
    print("="*60)
    print("Done! Cache files saved as: {peptide_name}_cache_stride{stride}.npz")
    print("Now you can train with fast data loading!")


if __name__ == "__main__":
    main()

