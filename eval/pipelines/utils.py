from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints
import mdtraj as md
import numpy as np
import pickle
import pyemma
import matplotlib.pyplot as plt
import rdkit.Chem.Draw as Draw
from scipy.spatial.distance import jensenshannon
import itertools
import torch
import random
random.seed(42)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_torsions_idx_mol(mol):
    '''
    Given a molecule object, this function returns the torsion indices.
    '''
    non_ring, ring = TorsionFingerprints.CalculateTorsionLists(mol)
    # Extract torsion atom indices and normalization values
    non_ring_torsions = [list(torsion[0][0]) for torsion in non_ring]  # List of torsion atom index lists
    ring_torsions = [list(torsion[0][0]) for torsion in ring]          # List of ring torsion atom index lists  
    ring_non_aromatic_torsions = []
    if len(ring_torsions) > 0:
        for torsion in ring_torsions:
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in torsion):  # Remove pure aromatic ring torsions
                continue
            ring_non_aromatic_torsions.append(torsion)

    return non_ring_torsions, ring_torsions, ring_non_aromatic_torsions

def find_decorrelation_time(autocorr, time_step=0.4):  # 0.4 ps
    threshold = 1/np.e
    for i, value in enumerate(autocorr):
        if value <= threshold:
            if i == 0:
                return -1     # If the first value is below threshold, return -1.
            return i * time_step
    return -1  # If it doesn't fall below threshold within the calculated range

def get_ref_and_generated_traj(ref_traj_list, gener_traj_list, cossin, plot_distr=False, plot_structure=False, jsd=False):
    '''
    Given a molecule prefix, load the corresponding trajectory,
    extract the dihedral angles, and return the torsion trajectory object.
    We have 5 independent trajs for each molecules.
    Use the first four as ref, and the last one as generated.
    '''

    ref_pdb_list = [f'{"/".join(ref_traj.split("/")[:-1])}/system.pdb' for ref_traj in ref_traj_list]
    gener_pdb_list = [f'{"/".join(gener_traj.split("/")[:-1])}/system.pdb' for gener_traj in gener_traj_list]
    
    mol_path = "/".join(ref_traj_list[0].split("/")[:-1])+'/mol.pkl'  # Only get dihedral angle from mol, thus all traj of the same molecule share the same mol info
    mol = pickle.load(open(mol_path, 'rb'))
    # get torsion atom indices
    torsion_index = np.array(get_torsions_idx_mol(mol)[0]+get_torsions_idx_mol(mol)[2]).reshape(-1,4)
    ref_traj_all = []
    for i in range(len(ref_pdb_list)):
        feat_dihedral = pyemma.coordinates.featurizer(ref_pdb_list[i])
        feat_dihedral.add_dihedrals(torsion_index, cossin=cossin, periodic=False)
        ref_traj = pyemma.coordinates.load(ref_traj_list[i], features=feat_dihedral)
        ref_traj_all.append(ref_traj)
    if cossin:
        num_dihedrals = feat_dihedral.dimension() // 2  # Each dihedral is represented by two features (cos and sin)
    else:
        num_dihedrals = feat_dihedral.dimension()
    print(f'there are {num_dihedrals} dihedrals in the system')

    gener_traj_all = []
    for i in range(len(gener_pdb_list)):
        feat_dihedral = pyemma.coordinates.featurizer(gener_pdb_list[i])
        feat_dihedral.add_dihedrals(torsion_index, cossin=cossin, periodic=False)
        gener_traj = pyemma.coordinates.load(gener_traj_list[i], features=feat_dihedral)
        gener_traj_all.append(gener_traj)

    if plot_distr:
        # plot the distribution of dihedrals
        fig, ax = plt.subplots(figsize=(12, 8))
        pyemma.plots.plot_feature_histograms(np.concatenate(ref_traj_all), feature_labels=feat_dihedral, ax=ax, color=colors[0])
        pyemma.plots.plot_feature_histograms(np.concatenate(gener_traj_all), feature_labels=feat_dihedral, ax=ax, color=colors[1])
        fig.tight_layout()
        # plt.show()
        ax.set_title('Dihedral Angle Distribution')
    # if plot_structure:
    #     # Create a 2D visualization with RDKit
    #     mol_copy = Chem.Mol(mol)
    #     AllChem.Compute2DCoords(mol_copy)

    #     # Create highlighted depictions for each dihedral
    #     fig = plt.figure(figsize=(12, 8))
    #     for i, indices in enumerate(torsion_index):
    #         plt.subplot(4, 3, i+1)
    #         atoms = [int(idx) for idx in indices]
    #         img = Draw.MolToImage(mol_copy, highlightAtoms=atoms, highlightBonds=[])
    #         plt.imshow(img)
    #         plt.title(f"Dihedral {i+1}: {'-'.join(map(str, atoms))}")
    #         plt.axis('off')
    #     plt.tight_layout()
    #     # Plot dihedral angles over time for the ref trajectories
    #     plt.figure(figsize=(10, 6))
    #     for i in range(num_dihedrals):
    #         plt.plot(np.concatenate(ref_traj_all)[:, i], label=f'Dihedral {i+1}', alpha=0.5)
    #     plt.xlabel('Frame')
    #     plt.ylabel('Dihedral Angle (radians)')
    #     plt.title('Dihedral Angles Over Time')
    #     plt.legend()
    #     plt.tight_layout()
    if jsd:
        jsd_out = {}
        for i, feat in enumerate(feat_dihedral.describe()):
            ref_p = np.histogram(np.concatenate(ref_traj_all)[:,i], range=(-np.pi, np.pi), bins=100)[0]
            gener_p = np.histogram(np.concatenate(gener_traj_all)[:,i], range=(-np.pi, np.pi), bins=100)[0]
            jsd_out[feat] = jensenshannon(ref_p, gener_p)
        return ref_traj_all, gener_traj_all, num_dihedrals, jsd_out
    return ref_traj_all, gener_traj_all, num_dihedrals

def tica_on_ref(ref_traj, lag=10, plot=False):
    """Compute TICA on the dihedral angles.

    Parameters
    ----------
    ref_traj : list of numpy.ndarrays
        The input data.
    lag : int, optional, default=10
        Lag step for the TICA calculation.
    """
    tica = pyemma.coordinates.tica(ref_traj, lag=lag)
    tica_output = tica.get_output()
    tica_concatenated = np.concatenate(tica_output)
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        pyemma.plots.plot_feature_histograms(
            tica_concatenated,
            ['IC {}'.format(i + 1) for i in range(tica.dimension())],
            ax=axes[0])
        axes[0].set_title('lag time = {} steps'.format(lag))
        axes[1].set_title(
            'Density, actual dimension = {}'.format(tica.dimension()))
        pyemma.plots.plot_density(
            *tica_concatenated[:, :2].T, ax=axes[1], cbar=False)
        pyemma.plots.plot_free_energy(
            *tica_concatenated[:, :2].T, ax=axes[2], legacy=False)
        for ax in axes[1:].flat:
            ax.set_xlabel('IC 1')
            ax.set_ylabel('IC 2')
        axes[2].set_title('Pseudo free energy')
        fig.tight_layout()
    return tica, tica_output, tica_concatenated

def tica_projection(generated_traj, tica, name, plot_dir, plot=False, num_points_to_plot = 100):
    """Project the generated trajectory onto the TICA space of reference.

    Parameters
    ----------
    generated_traj : list of numpy.ndarrays
        The input data.
    tica : pyemma.coordinates.tica.TICA
        The TICA object fitted on the reference data.
    """
    tica_generated = tica.transform(generated_traj)
    tica_output = tica.get_output()
    tica_concatenated = np.concatenate(tica_output)
    tica_generated_concatenated = np.concatenate(tica_generated)
    if plot:
        print('!!!!!!!!!!!!!!!!!!!!!!! Please be aware that two plots does not have the same range nor color bar!!!!!!!!!!!!!!!!!!!!!!!!')
        if tica_concatenated.shape[1] > 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            pyemma.plots.plot_free_energy(
                *tica_concatenated[:, :2].T, ax=axes[0], legacy=False, cbar=False)
            pyemma.plots.plot_free_energy(
                *tica_generated_concatenated[:, :2].T, ax=axes[1], legacy=False, cbar=False)

            # Plot the traj projection on the FES, downsampled to num_points 
            pyemma.plots.plot_free_energy(
                *tica_concatenated[:, :2].T, ax=axes[2], legacy=False)
            axes[2].scatter(tica_generated_concatenated[::tica_generated_concatenated.shape[0]//num_points_to_plot, 0], 
                    tica_generated_concatenated[::tica_generated_concatenated.shape[0]//num_points_to_plot, 1], 
                    color='black', label='Points', alpha=0.5, s=15) 
            axes[2].plot(tica_generated_concatenated[::tica_generated_concatenated.shape[0]//num_points_to_plot, 0],
                        tica_generated_concatenated[::tica_generated_concatenated.shape[0]//num_points_to_plot, 1], color='black', label='Connections', alpha=0.5)
            
            axes[0].set_title('FES (Reference)')
            axes[1].set_title('FES (Ours)')
            axes[2].set_title('Projection of generated trajectory on reference FES')
            xlims = [axes[0].get_xlim(), axes[1].get_xlim(), axes[2].get_xlim()]
            ylims = [axes[0].get_ylim(), axes[1].get_ylim(), axes[2].get_ylim()]
            x_min = min(x[0] for x in xlims)
            x_max = max(x[1] for x in xlims)
            y_min = min(y[0] for y in ylims)
            y_max = max(y[1] for y in ylims)
            for ax in axes:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            for ax in axes:
                ax.set_xlabel('TICA 0')
                ax.set_ylabel('TICA 1')
                # no ticks
                ax.set_xticks([])
                ax.set_yticks([])
            
            fig.savefig(f'{plot_dir}/{name}_tica_projection.pdf', bbox_inches='tight')
            plt.close()
    return tica_generated

def kmeans(tica, k=100, plot=False):
    """Run kmeans clustering on the TICA output.

    Parameters
    ----------
    tica : pyemma.coordinates.tica.TICA
        The TICA object fitted on the reference data.
    """
    # Run kmeans clustering on the TICA output.
    tica_concatenated = np.concatenate(tica.get_output())
    cluster = pyemma.coordinates.cluster_kmeans(tica, k=k, max_iter=100, fixed_seed=42)
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        pyemma.plots.plot_feature_histograms(
            tica_concatenated, ['IC {}'.format(i + 1) for i in range(tica.dimension())], ax=axes[0])
        pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=axes[1], cbar=False, alpha=0.1, logscale=True)
        axes[1].scatter(*cluster.clustercenters[:, :2].T, s=15, c='C1')
        axes[1].set_xlabel('IC 1')
        axes[1].set_ylabel('IC 2')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.tight_layout()
        axes[1].set_title('Cluster centers on ref FES')
        # plt.show()
    return cluster, k

'''
This will return the torsions in the molecule and their normalization values

This is namely with respect to rotatable bonds in the molecule between heavy atoms

We can then extract out torsions from the ground truth trajectories using

torsion_atoms = [[0, 1, 2, 3], [1, 2, 3, 4]]  # example
torsions = md.compute_dihedrals(traj, torsion_atoms)  # shape (n_frames, n_torsions)
'''


def get_torsions_in_traj(traj, mol):
    torsion_atoms = get_torsions_idx_mol(mol)[0] + get_torsions_idx_mol(mol)[2]
    if len(torsion_atoms) == 0:
        return None
    torsions = md.compute_dihedrals(traj, torsion_atoms)  # shape (n_frames, n_torsions)
    return torsions

def get_torsions_in_gen(gen_traj, torsion_indices, eps=1e-8):
    """
      gen_traj (torch.Tensor): Tensor of shape (T, N, 3), where T is the number of time steps,
                                and N is the number of atoms.
      deg (bool): If True, return angles in degrees (default is radians).
      eps (float): A small epsilon value added for numerical stability.
      
    Returns:
      numpy.ndarray: An array of shape (T, M) containing the torsion angles in radians [0, 2π].
    """
    # Ensure torsion_indices is a torch LongTensor on the same device as coords.
    torsion_indices = torch.tensor(torsion_indices, dtype=torch.long)
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
        if atom.GetAtomicNum() <= 1:  # The center atom must be heavy (atomic number > 1)
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
    angles = md.compute_angles(traj, angle_triplets)  # shape: (n_frames, n_angles)  # The old version of mdtraj will output Nan if the angle is 180deg. 
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
    mol_noH = Chem.RemoveHs(mol)
    important_atoms = list(mol.GetSubstructMatch(mol_noH))
    bonds = mol.GetBonds()
    bond_pairs = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in bonds if b.GetBeginAtomIdx() in important_atoms and b.GetEndAtomIdx() in important_atoms]
    bond_lengths = md.compute_distances(traj, bond_pairs)
    return bond_lengths

def get_bond_lengths_in_conformer_ref(conformer_traj, mol):
    # Get the bonds from the molecule
    mol_noH = Chem.RemoveHs(mol)
    important_atoms = list(mol.GetSubstructMatch(mol_noH))
    bonds = mol.GetBonds()
    bond_pairs = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in bonds if b.GetBeginAtomIdx() in important_atoms and b.GetEndAtomIdx() in important_atoms]
    
    # Convert to tensor
    bond_pairs = torch.tensor(bond_pairs)
    
    # Get positions of atoms i and j
    pos_i = conformer_traj[:, bond_pairs[:, 0]]  # T x M x 3
    pos_j = conformer_traj[:, bond_pairs[:, 1]]  # T x M x 3
    
    # Calculate distances between bonded atoms
    bond_vectors = pos_i - pos_j  # T x M x 3
    bond_lengths = torch.norm(bond_vectors, dim=-1)  # T x M
    
    return bond_lengths.numpy()

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

def downsample_traj(traj, num_new_traj=10, num_points=500, stepsize=5.2):
    """
    Downsample the ref trajectory to a specified number of points.
    
    Parameters:
    traj (numpy.ndarray): The trajectory data to downsample. The reference data should have shape (12500, N). The stepsize is 0.4 ps.
    stepsize (float): The time step of the trajectory in ps.
    num_points (int): The number of points to downsample to.
    
    Returns:
    A list of numpy.ndarray: The downsampled trajectory.
    """
    assert traj.shape[0] == 12500, "The trajectory should have 12500 frames."
    possible_start_indx = round(traj.shape[0] - round(stepsize*num_points/5000*traj.shape[0])) - 1
    start_idxs = random.sample(range(0, possible_start_indx), k=num_new_traj)
    result = []
    # Downsample the trajectory
    for start_idx in start_idxs:
        start_idx = int(start_idx)
        step = round(stepsize/0.4)
        # Downsample the trajectory to the specified number of points
        downsampled_traj = traj[start_idx::step][:num_points]
        # print(len(downsampled_traj))
        result.append(downsampled_traj)
    
    return result

# test_array = np.arange(0, 12500, 1) * 0.4
# print(test_array[470:])
# result = downsample_traj(test_array, num_new_traj=5, num_points=960, stepsize=5.2)
# for i in range(len(result)):
#     print(f"Downsampled trajectory {i+1}: {result[i][:10]}")

def get_torsion_index_noH(torsion_index, mol):
    '''
    Given a torsion index and a molecule object, this function returns the torsion index without hydrogens.
    
    '''
    # old_to_new_idx = {}
    # new_idx = 0
    # for i, atom in enumerate(mol.GetAtoms()):
    #     if atom.GetAtomicNum() > 1:  # Exclude hydrogens
    #         old_to_new_idx[i] = new_idx
    #         new_idx += 1
    # # Translate the torsion atom indices to the new mol_noH
    # torsion_index_noH = []
    # for torsion in torsion_index:
    #     new_torsion = [old_to_new_idx[i] for i in torsion]
    #     torsion_index_noH.append(new_torsion)

    # Another way to do this is to use the GetSubstructMatch function.
    mol_no = Chem.RemoveHs(mol)
    important_atoms = list(mol.GetSubstructMatch(mol_no))
    torsion_index_noH_2 = []
    for torsion in torsion_index:
        new_torsion = [important_atoms.index(i) for i in torsion]
        torsion_index_noH_2.append(new_torsion)
    smiles = Chem.MolToSmiles(mol)
    # assert torsion_index_noH == torsion_index_noH_2, f"The two methods of getting the torsion index do not match, the smiles is {smiles}."

    return torsion_index_noH_2

def get_metastate_prob(msm, gen_clusters, k, nstates=10):
    microstate_count = np.bincount(np.array(gen_clusters).flatten(),minlength=k) # count the number of occurrences of each microstate in the generated trajectory
    microstate_count = np.array(microstate_count/microstate_count.sum()) # use original microstate idx

    generated_metastate_prob = np.zeros(nstates) # create buffer for reference metastate probabilities
    for state in range(nstates):
        microstate_idx = msm.active_set[msm.metastable_sets[state]]  # metastable sets use transformed microstates idx. active_set uses original microstates idx.
        # print(microstate_idx) # Now the microstate_idx is in the original space
        generated_metastate_prob[state] = np.sum(microstate_count[microstate_idx])
    generated_metastate_prob = generated_metastate_prob/generated_metastate_prob.sum() # normalize the generated metastate probabilities
    
    ref_metastate_prob = np.zeros(nstates)  # create buffer for reference metastate probabilities
    for state in range(nstates):
        microstate_idx_in_state = msm.metastable_sets[state]
        ref_metastate_prob[state] = np.sum(msm.pi[microstate_idx_in_state])
    ref_metastate_prob = ref_metastate_prob/ref_metastate_prob.sum() # normalize the reference metastate probabilities
    assert np.isclose(ref_metastate_prob.sum(), 1.0), "Reference metastate probabilities do not sum to 1."
    assert np.isclose(generated_metastate_prob.sum(), 1.0), f"Generated metastate probabilities do not sum to 1. It is {generated_metastate_prob.sum()}"
    return ref_metastate_prob, generated_metastate_prob

def sample_tp(trans, start_state, end_state, traj_len, n_samples):
    s_1 = start_state
    s_N = end_state
    N = traj_len

    s_t = np.ones(n_samples, dtype=int) * s_1
    states = [s_t]
    for t in range(1, N - 1):
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_t = np.zeros(n_samples, dtype=int)
        for n in range(n_samples):
            s_t[n] = np.random.choice(np.arange(len(trans)), 1, p=probs[n])
        states.append(s_t)
    states.append(np.ones(n_samples, dtype=int) * s_N)
    return np.stack(states, axis=1)

def get_tp_likelihood(tp, trans):
    N = tp.shape[1]
    n_samples = tp.shape[0]
    s_N = tp[0, -1]
    trans_probs = []
    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]

        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]
        trans_probs.append(trans_prob)
    probs = np.stack(trans_probs, axis=1)
    probs[np.isnan(probs)] = 0
    return probs