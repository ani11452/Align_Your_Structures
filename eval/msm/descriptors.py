import os
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints
import itertools
import mdtraj as md
import torch
import numpy as np
import pickle

'''
This will return the torsions in the molecule and their normalization values

This is namely with respect to rotatable bonds in the molecule between heavy atoms

We can then extract out torsions from the ground truth trajectories using

torsion_atoms = [[0, 1, 2, 3], [1, 2, 3, 4]]  # example
torsions = md.compute_dihedrals(traj, torsion_atoms)  # shape (n_frames, n_torsions)
'''
def get_torsions_idx_mol(mol):
    non_ring, ring = TorsionFingerprints.CalculateTorsionLists(mol)
    
    # Extract torsion atom indices and normalization values
    non_ring_torsions = [list(torsion[0][0]) for torsion in non_ring]  # List of torsion atom index lists
    non_ring_norms = [torsion[1] for torsion in non_ring]     # List of normalization values
    
    ring_torsions = [list(torsion[0][0]) for torsion in ring]          # List of ring torsion atom index lists  
    ring_norms = [torsion[1] for torsion in ring]             # List of ring normalization values

    return non_ring_torsions, ring_torsions, non_ring_norms, ring_norms

def get_torsions_in_traj(traj, mol):
    torsion_atoms = get_torsions_idx_mol(mol)[0]
    if len(torsion_atoms) == 0:
        return None
    torsions = md.compute_dihedrals(traj, torsion_atoms)  # shape (n_frames, n_torsions)
    return torsions

def get_torsions_in_gen(gen_traj, mol, eps=1e-8):
    """
      gen_traj (torch.Tensor): Tensor of shape (T, N, 3), where T is the number of time steps,
                                and N is the number of atoms.
      deg (bool): If True, return angles in degrees (default is radians).
      eps (float): A small epsilon value added for numerical stability.
      
    Returns:
      numpy.ndarray: An array of shape (T, M) containing the torsion angles in radians [0, 2π].
    """
    # Ensure torsion_indices is a torch LongTensor on the same device as coords.
    torsion_indices = get_torsions_idx_mol(mol)[0]

    torsion_indices = torch.tensor(torsion_indices, dtype=torch.long, device=gen_traj.device)
    r = gen_traj[:, torsion_indices, :]  # (T, M, 4, 3) 
    
    # Compute bond vectors (shape: T x M x 3):
    if len(torsion_indices) == 0:
        return None
    
    b0 = r[:, :, 0, :] - r[:, :, 1, :] 
    b1 = r[:, :, 1, :] - r[:, :, 2, :] 
    b2 = r[:, :, 2, :] - r[:, :, 3, :]
    
    # Calculate normals to the planes defined by (atoms 0,1,2) and (atoms 1,2,3)
    n1 = torch.cross(b0, b1, dim=-1)  # T x M x 3
    n2 = torch.cross(b1, b2, dim=-1)  # T x M x 3
    
    # Normalize the central bond vector (b1), guarding against zero norms.
    b1_norm = b1 / (torch.norm(b1, dim=-1, keepdim=True) + eps)  # T x M x 3
    
    # Compute m1, which is orthogonal to n1 and b1_norm; used for determining the torsion sign.
    m1 = torch.cross(n1, b1_norm, dim=-1)  # T x M x 3
    
    # Compute dot products for the angle calculation, across the last dimension:
    x = (n1 * n2).sum(dim=-1)  # T x M
    y = (m1 * n2).sum(dim=-1)  # T x M
    
    # Torsion angles (in radians) using arctan2 for correct quadrant detection.
    torsions = torch.atan2(y, x)  # T x M
    
    return torsions.numpy()

'''
This will return the bond angles in the molecule

This is namely with respect to the bonds in the molecule between heavy atoms

We can then extract out bond angles from the ground truth trajectories using

angle_triplets = [[0, 1, 2], [1, 2, 3]]  # list of (i, j, k)
angles = md.compute_angles(traj, angle_triplets)  # shape: (n_frames, n_angles)
'''
def get_bond_angles_idx_mol(mol):
    angle_triplets = []
    for atom in mol.GetAtoms():
        # Only consider heavy atoms as the central atom
        if atom.GetAtomicNum() <= 1:
            continue
        j = atom.GetIdx()
        
        # Get the indices of heavy neighbors (atomic number > 1)
        heavy_neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
        
        # Only proceed if there are at least two heavy neighbors
        if len(heavy_neighbors) < 2:
            continue
        
        # For each unique pair of neighbors, create an angle triplet.
        # Using itertools.combinations ensures each pair is unique (order doesn't matter)
        for i, k in itertools.combinations(heavy_neighbors, 2):
            # Append the triplet [i, j, k]
            angle_triplets.append([i, j, k])
    
    return angle_triplets

def get_bond_angles_in_traj(traj, mol):
    angle_triplets = get_bond_angles_idx_mol(mol)
    angles = md.compute_angles(traj, angle_triplets)  # shape: (n_frames, n_angles)
    return angles

def get_bond_angles_in_gen(gen_traj, mol, eps=1e-8):
    # Get the angle triplets from the molecule
    angle_triplets = get_bond_angles_idx_mol(mol)
    
    # Convert to tensor
    angle_triplets = torch.tensor(angle_triplets)
    
    # Get positions of atoms i, j, k
    pos_i = gen_traj[:, angle_triplets[:, 0]]  # T x M x 3
    pos_j = gen_traj[:, angle_triplets[:, 1]]  # T x M x 3 
    pos_k = gen_traj[:, angle_triplets[:, 2]]  # T x M x 3
    
    # Calculate vectors between atoms
    v1 = pos_i - pos_j  # T x M x 3
    v2 = pos_k - pos_j  # T x M x 3
    
    # Normalize vectors
    v1_norm = torch.norm(v1, dim=-1, keepdim=True)
    v2_norm = torch.norm(v2, dim=-1, keepdim=True)
    v1 = v1 / (v1_norm + eps)
    v2 = v2 / (v2_norm + eps)
    
    # Calculate cosine of angles using dot product
    cos_angles = (v1 * v2).sum(dim=-1)  # T x M
    cos_angles = torch.clamp(cos_angles, -1 + eps, 1 - eps)
    
    # Calculate angles in radians
    angles = torch.acos(cos_angles)  # T x M
            
    return angles.numpy()


'''
This will return the bond lengths in the molecule

This is namely with respect to the bonds in the molecule between heavy atoms

We can then extract out bond lengths from the ground truth trajectories using

bond_lengths = md.compute_distances(traj, mol.GetBonds())  # shape: (n_frames, n_bonds)
'''
def get_bond_lengths_in_traj(traj, mol):
    bonds = mol.GetBonds()
    bond_pairs = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in bonds]
    bond_lengths = md.compute_distances(traj, bond_pairs)
    return bond_lengths

def get_bond_lengths_in_gen(gen_traj, mol):
    # Get the bonds from the molecule
    bonds = mol.GetBonds()
    bond_pairs = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in bonds]
    
    # Convert to tensor
    bond_pairs = torch.tensor(bond_pairs)
    
    # Get positions of atoms i and j
    pos_i = gen_traj[:, bond_pairs[:, 0]]  # T x M x 3
    pos_j = gen_traj[:, bond_pairs[:, 1]]  # T x M x 3
    
    # Calculate distances between bonded atoms
    bond_vectors = pos_i - pos_j  # T x M x 3
    bond_lengths = torch.norm(bond_vectors, dim=-1)  # T x M
    
    return bond_lengths.numpy()


if __name__ == "__main__":
    # Load the trajectory   
    xtc_path = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/train/120_results/NC_O_C_H_O_C_H_1C_C_H_1O_109/traj.xtc'
    pdb_path = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/train/120_results/NC_O_C_H_O_C_H_1C_C_H_1O_109/system.pdb'
    mol_path = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/train/120_results/NC_O_C_H_O_C_H_1C_C_H_1O_109/mol.pkl'

    # Load the trajectory
    print("Loading trajectory")
    traj = md.load(xtc_path, top=pdb_path)
    traj = traj[::100]
    coords = torch.tensor(traj.xyz, dtype=torch.float32)

    print("Loading mol")
    mol = pickle.load(open(mol_path, 'rb'))

    # Get the torsions from traj
    print("Getting torsions from traj") 
    torsions_traj = get_torsions_in_traj(traj, mol)

    # Get the torsions from gen
    print("Getting torsions from gen")
    torsions_gen = get_torsions_in_gen(coords, mol)

    # Plot the torsions
    print(torsions_traj.shape, torsions_gen.shape)
    print(np.max(np.abs(torsions_traj - torsions_gen)))

    # Get the torsions from traj
    print("Getting bond angles from traj")
    bond_angles_traj = get_bond_angles_in_traj(traj, mol)

    # Get the torsions from gen
    print("Getting bond angles from gen")
    bond_angles_gen = get_bond_angles_in_gen(coords, mol)

    # Plot the torsions
    print(bond_angles_traj.shape, bond_angles_gen.shape)
    print(np.max(np.abs(bond_angles_traj - bond_angles_gen)))

    # Get the bond lengths from traj
    print("Getting bond lengths from traj")
    bond_lengths_traj = get_bond_lengths_in_traj(traj, mol)

    # Get the bond lengths from gen
    print("Getting bond lengths from gen")
    bond_lengths_gen = get_bond_lengths_in_gen(coords, mol)

    # Plot the bond lengths
    print(bond_lengths_traj.shape, bond_lengths_gen.shape)
    print(np.max(np.abs(bond_lengths_traj - bond_lengths_gen)))


