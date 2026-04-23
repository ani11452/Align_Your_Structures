import os, sys
_PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PIPELINE_DIR)

import mdtraj as md
import types

# Monkey‑patch so that PyEMMA can see mdtraj.version.version
md.version = types.SimpleNamespace(version=md.__version__)

import argparse
import pyemma, tqdm, os, pickle
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
from scipy.stats import wasserstein_distance
from utils import get_ref_and_generated_traj, get_torsions_idx_mol, find_decorrelation_time, tica_on_ref, tica_projection, kmeans, downsample_traj
import mdtraj as md
import torch
from rdkit import Chem
from collections import defaultdict
from utils import get_bond_angles_in_traj, get_bond_lengths_in_traj, get_torsions_in_gen, get_bond_angles_in_gen, get_bond_lengths_in_gen, get_torsion_index_noH, get_metastate_prob
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def main(smiles):
    # print(f'\nThe evaluated molecules is {smiles}')
    ### Get the full name of ref trajectory from smile_match.txt file
    out = {}
    ref_traj_ids = []
    md_traj_ids = []
    gener_traj_ids = []
    with open(args.smile_match, 'r') as f:
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
    ### Read pickle file from gener_dir.
    with open(args.gener_dir, 'rb') as f:
        gener_data = pickle.load(f)

        # if the keys are 2‑tuples, reduce to just the SMILES string
        if isinstance(next(iter(gener_data)), tuple):
            gener_data = {k[0]: gener_data[k] for k in gener_data}

        gen_mol = gener_data[smiles]['rdmol']  # It already has Hs removed.
        gen_mol = Chem.RemoveHs(gen_mol)  # Double check to remove Hs from the molecule
        gener_traj_coords = gener_data[smiles]['coords']  # (10, 9, 3, 500), 10 trajs of 500 length.
    assert gener_traj_coords[0].shape[2] == 500, f'The generated coords should have shape (N, 3, 500), but it has shape {gener_traj_coords[0].shape}'
    
    if args.check_nan:
    # Use the code below to check if the gener traj has nan values.
        for i in range(len(gener_traj_coords)):
            # check if any value from this traj is not nan.
            if np.isnan(gener_traj_coords[i]).any():
                raise ValueError(f"Nan encountered in ref traj at index {i} for molecule {smiles}")
        return smiles, out

    ref_traj_paths = [f'{args.ref_dir}/{traj_id}/traj.xtc' for traj_id in ref_traj_ids]
    md_traj_paths = [f'{args.ref_dir}/{traj_id}/traj.xtc' for traj_id in md_traj_ids]
    ref_pdb_paths = [f'{args.ref_dir}/{traj_id}/system.pdb' for traj_id in ref_traj_ids]
    md_pdb_paths = [f'{args.ref_dir}/{traj_id}/system.pdb' for traj_id in md_traj_ids]

    mol_path = "/".join(ref_traj_paths[0].split("/")[:-1])+'/mol.pkl'  # Only get atoms from mol, thus all traj of the same molecule share the same mol info
    mol = pickle.load(open(mol_path, 'rb'))
    torsion_index = np.array(get_torsions_idx_mol(mol)[0]+get_torsions_idx_mol(mol)[2]).reshape(-1,4)   # 0 is non-ring torsion, 2 is ring but non-aromatic torsion
    # Check all mol from each ref folder give the same torsion index.
    # mol_1 = pickle.load(open("/".join(ref_traj_paths[1].split("/")[:-1])+'/mol.pkl', 'rb'))
    # mol_2 = pickle.load(open("/".join(ref_traj_paths[2].split("/")[:-1])+'/mol.pkl', 'rb'))
    # mol_3 = pickle.load(open("/".join(ref_traj_paths[3].split("/")[:-1])+'/mol.pkl', 'rb'))
    # mol_4 = pickle.load(open("/".join(md_traj_paths[0].split("/")[:-1])+'/mol.pkl', 'rb'))
    # torsion_index_1 = np.array(get_torsions_idx_mol(mol_1)[0]+get_torsions_idx_mol(mol_1)[2]).reshape(-1,4)
    # torsion_index_2 = np.array(get_torsions_idx_mol(mol_2)[0]+get_torsions_idx_mol(mol_2)[2]).reshape(-1,4)
    # torsion_index_3 = np.array(get_torsions_idx_mol(mol_3)[0]+get_torsions_idx_mol(mol_3)[2]).reshape(-1,4)
    # torsion_index_4 = np.array(get_torsions_idx_mol(mol_4)[0]+get_torsions_idx_mol(mol_4)[2]).reshape(-1,4)
    # Check if all values in the numpy array are the same
    # assert np.all(torsion_index == torsion_index_1), f'Torsion index mismatch: {torsion_index} vs {torsion_index_1} for molecule {smiles}'
    # assert np.all(torsion_index == torsion_index_2), f'Torsion index mismatch: {torsion_index} vs {torsion_index_2} for molecule {smiles}'
    # assert np.all(torsion_index == torsion_index_3), f'Torsion index mismatch: {torsion_index} vs {torsion_index_3} for molecule {smiles}'
    # assert np.all(torsion_index == torsion_index_4), f'Torsion index mismatch: {torsion_index} vs {torsion_index_4} for molecule {smiles}'

    torsion_index_noH = get_torsion_index_noH(torsion_index, mol)  # used for gener mol without Hs.
    ### Check whether torsion_index has the same atoms as torsion_index_noH
    for i in range(len(torsion_index_noH)):
        torsion_noH = torsion_index_noH[i]
        torsion = torsion_index[i]
        for idx_noH, idx in zip(torsion_noH, torsion):  # Get atom info from mol and gen_mol and compare
            atom_noH = gen_mol.GetAtomWithIdx(int(idx_noH))
            atom = mol.GetAtomWithIdx(int(idx))
            assert atom.GetSymbol() == atom_noH.GetSymbol(), f"Atom mismatch: {atom.GetSymbol()} vs {atom_noH.GetSymbol()} for molecule {smiles}"

    ### Load ref torsion
    feat_dihedral = pyemma.coordinates.featurizer(ref_pdb_paths[0])
    try:
        feat_dihedral.add_dihedrals(torsion_index, cossin=False, periodic=False)
    except TypeError as e:
        print(f"Error adding dihedrals: {e} for molecule {smiles}. Check the shape of torsion_index.")
    ref_torsion_tensor = pyemma.coordinates.load(ref_traj_paths, features=feat_dihedral)  # A list of all torsion traj. Each traj should have (T, N)

    # Down-sample the ref traj to have step size of 5.2 ps.
    # In each traj, sample 30 down-sampled trajs. Then we have 4 * 30 in total. The new list should have shape (500, N) with length 120.
    downsampled_ref_torsion = []
    for traj in ref_torsion_tensor:
        downsampled_traj = downsample_traj(traj, num_new_traj=30, num_points=500, stepsize=args.time_step)
        downsampled_ref_torsion += downsampled_traj
    assert len(downsampled_ref_torsion) == 120, f'The ref torsion should have length 120, but it has length {len(downsampled_ref_torsion)}, the molecules is {smiles}'
    assert downsampled_ref_torsion[0].shape[0] == 500, f'The downsampled ref torsion should have shape (500, N), but it has shape {downsampled_ref_torsion[0].shape}'
    ref_torsion_tensor = downsampled_ref_torsion

    ### Load md torsion
    feat_dihedral = pyemma.coordinates.featurizer(md_pdb_paths[0])
    feat_dihedral.add_dihedrals(torsion_index, cossin=False, periodic=False)
    md_torsion_tensor = pyemma.coordinates.load(md_traj_paths, features=feat_dihedral)
    if not isinstance(md_torsion_tensor, list):
        md_torsion_tensor = [md_torsion_tensor]
    # Down-sample the md traj to have step size of 5.2 ps.
    downsampled_md_torsion = []
    for traj in md_torsion_tensor:
        downsampled_traj = downsample_traj(traj, num_new_traj=10, num_points=500, stepsize=args.time_step)
        downsampled_md_torsion += downsampled_traj
    assert len(downsampled_md_torsion) == 10, f'The md torsion should have length 10, but it has length {len(downsampled_md_torsion)}'
    assert downsampled_md_torsion[0].shape[0] == 500, f'The downsampled md torsion should have shape (500, N), but it has shape {downsampled_md_torsion[0].shape}'
    md_torsion_tensor = downsampled_md_torsion

    ### Load gen traj
    gener_traj_tensor = []
    for i in range(len(gener_traj_coords)):
        gener_traj_tensor.append(torch.permute(torch.from_numpy(gener_traj_coords[i] / 10), (2,0,1)))      # /10 convert to nm
    # assert len(gener_traj_tensor) == 10, f'The gener traj should have length 10, but it has length {len(gener_traj_tensor)}'
    assert gener_traj_tensor[0].shape[0] == 500, f'The gener traj should have shape (500, N, 3), but it has shape {gener_traj_tensor[0].shape}'
    
    # Load gen torsion
    gener_torsion_tensor = []
    for i in range(len(gener_traj_tensor)):
        torsions = get_torsions_in_gen(gener_traj_tensor[i], torsion_index_noH)
        gener_torsion_tensor.append(torsions)
    if np.any(np.isnan(np.array(gener_torsion_tensor))):
        return smiles, {}

    if args.plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        pyemma.plots.plot_feature_histograms(np.concatenate(gener_torsion_tensor), feature_labels=feat_dihedral, ax=ax, color='blue')
        pyemma.plots.plot_feature_histograms(np.concatenate(md_torsion_tensor), feature_labels=feat_dihedral, ax=ax, color='yellow')
        pyemma.plots.plot_feature_histograms(np.concatenate(ref_torsion_tensor), feature_labels=feat_dihedral, ax=ax, color='red')
        ax.set_ylabel('Feature No.')
        ax.set_xlabel('Torsion angle (rad)')
        fig.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_torsion_hist.png')
        plt.close()

    jsd_out = {}
    w1_out = {}
    jsd_md_out = {}
    w1_md_out = {}
    jsd_out_single_traj = defaultdict(list)
    w1_out_single_traj = defaultdict(list)
    for i, feat in enumerate(feat_dihedral.describe()):  # Loop through each torsion angle
        ref_p, _ = np.histogram(np.concatenate(ref_torsion_tensor)[:,i], range=(-np.pi, np.pi), bins=100)
        ref_p = ref_p / np.sum(ref_p)
        md_p, _ = np.histogram(np.concatenate(md_torsion_tensor)[:,i], range=(-np.pi, np.pi), bins=100)
        md_p = md_p / np.sum(md_p)
        gener_p, _ = np.histogram(np.concatenate(gener_torsion_tensor)[:,i], range=(-np.pi, np.pi), bins=100)
        gener_p = gener_p / np.sum(gener_p)
        jsd_out[feat] = jensenshannon(ref_p, gener_p)
        w1_out[feat] = wasserstein_distance(np.concatenate(ref_torsion_tensor)[:,i], np.concatenate(gener_torsion_tensor)[:,i])
        jsd_md_out[feat] = jensenshannon(ref_p, md_p)
        w1_md_out[feat] = wasserstein_distance(np.concatenate(ref_torsion_tensor)[:,i], np.concatenate(md_torsion_tensor)[:,i])
        ### Then compare single gen traj with ref
        for j in range(len(gener_torsion_tensor)):
            single_gener_p, _ = np.histogram(gener_torsion_tensor[j][:,i], range=(-np.pi, np.pi), bins=100)
            if np.sum(single_gener_p) == 0:
                jsd_out_single_traj[f'single_trace_{j}'].append(-1)
                w1_out_single_traj[f'single_trace_{j}'].append(-1)  # Just hard code -1 distance to avoid error in the jsd calculation.
            else:
                single_gener_p = single_gener_p / np.sum(single_gener_p)
                jsd_out_single_traj[f'single_trace_{j}'].append(jensenshannon(ref_p, single_gener_p))
                w1_out_single_traj[f'single_trace_{j}'].append(wasserstein_distance(np.concatenate(ref_torsion_tensor)[:,i], gener_torsion_tensor[j][:,i]))
    out['JSD_torsion_gen'] = jsd_out
    out['JSD_torsion_gen_single_traj'] = jsd_out_single_traj
    out['JSD_torsion_md'] = jsd_md_out
    out['W1_torsion_gen'] = w1_out
    out['W1_torsion_md'] = w1_md_out
    out['W1_torsion_gen_single_traj'] = w1_out_single_traj

    ########### Bond angle eval
    ref_bond_angles = []
    for i in range(len(ref_pdb_paths)):
        traj = md.load(ref_traj_paths[i], top=ref_pdb_paths[i])
        bond_angles = get_bond_angles_in_traj(traj, mol=mol)
        assert bond_angles.shape[0] == 12500, f'The ref bond angles should have shape (12500, N), but it has shape {bond_angles.shape}'
        bond_angles = downsample_traj(bond_angles, num_new_traj=30, num_points=500, stepsize=args.time_step)
        assert len(bond_angles) == 30, f'The ref bond angles should have length 30, but it has length {len(bond_angles)}'
        assert bond_angles[0].shape[0] == 500, f'The downsampled ref bond angles should have shape (500, N), but it has shape {bond_angles[0].shape}'
        if np.isnan(bond_angles).any():
            print(f"Nan encountered in ref traj at index {i} for molecule {smiles}")
            raise ValueError(f"Nan encountered in ref traj at index {i} for molecule {smiles}")
        ref_bond_angles += bond_angles
    ref_bond_angles = np.concatenate(ref_bond_angles, axis=0)
    assert ref_bond_angles.shape[0] == 60000, f'The ref bond angles should have shape (120*500, N), but it has shape {ref_bond_angles.shape}'
    
    md_bond_angles = []
    for i in range(len(md_pdb_paths)):
        traj = md.load(md_traj_paths[i], top=md_pdb_paths[i])
        bond_angles = get_bond_angles_in_traj(traj, mol=mol)
        assert bond_angles.shape[0] == 12500, f'The md bond angles should have shape (12500, N), but it has shape {bond_angles.shape}'
        bond_angles = downsample_traj(bond_angles, num_new_traj=10, num_points=500, stepsize=args.time_step)
        assert len(bond_angles) == 10, f'The md bond angles should have length 10, but it has length {len(bond_angles)}'
        assert bond_angles[0].shape[0] == 500, f'The downsampled md bond angles should have shape (500, N), but it has shape {bond_angles[0].shape}'
        if np.isnan(bond_angles).any():
            print(f"Nan encountered in md traj at index {i} for molecule {smiles}")
            raise ValueError(f"Nan encountered in md traj at index {i} for molecule {smiles}")
        md_bond_angles += bond_angles
    md_bond_angles = np.concatenate(md_bond_angles, axis=0)
    assert md_bond_angles.shape[0] == 5000, f'The md bond angles should have shape (10*500, N), but it has shape {md_bond_angles.shape}'

    gener_bond_angles_tensor = []
    for i in range(len(gener_traj_tensor)):
        bond_angles = get_bond_angles_in_gen(gener_traj_tensor[i], mol=gen_mol)
        gener_bond_angles_tensor.append(bond_angles)
    assert gener_bond_angles_tensor[0].shape[0] == 500, f'The gener bond angles should have shape (10*500, N), but it has shape {gener_bond_angles_tensor.shape}'

    out['JSD_bond_angle_gen'] = []
    out['JSD_bond_angle_md'] = []
    out['W1_bond_angle_gen'] = []
    out['W1_bond_angle_md'] = []
    out['JSD_bond_angle_gen_single_traj'] = defaultdict(list)
    out['W1_bond_angle_gen_single_traj'] = defaultdict(list)
    for i in range(ref_bond_angles.shape[1]):
        ref_p, _ = np.histogram(ref_bond_angles[:,i],range=(0,np.pi), bins=100)
        ref_p = ref_p / np.sum(ref_p)
        md_p, _ = np.histogram(md_bond_angles[:,i],range=(0,np.pi), bins=100)
        md_p = md_p / np.sum(md_p)
        gener_p, _ = np.histogram(np.concatenate(gener_bond_angles_tensor, axis=0)[:,i],range=(0,np.pi), bins=100)
        gener_p = gener_p / np.sum(gener_p)
        out['JSD_bond_angle_gen'].append(jensenshannon(ref_p, gener_p))
        out['JSD_bond_angle_md'].append(jensenshannon(ref_p, md_p))
        out['W1_bond_angle_gen'].append(wasserstein_distance(ref_bond_angles[:,i], np.concatenate(gener_bond_angles_tensor, axis=0)[:,i]))
        out['W1_bond_angle_md'].append(wasserstein_distance(ref_bond_angles[:,i], md_bond_angles[:,i]))
        # Then compare single gen traj with ref
        for j in range(len(gener_bond_angles_tensor)):
            single_gener_p, _ = np.histogram(gener_bond_angles_tensor[j][:,i], range=(0,np.pi), bins=100)
            if np.sum(single_gener_p) == 0:
                out['JSD_bond_angle_gen_single_traj'][f'single_trace_{j}'].append(-1)  # HARD-CODE -1 distance to avoid error in the jsd calculation.
                out['W1_bond_angle_gen_single_traj'][f'single_trace_{j}'].append(-1)      
            else:
                single_gener_p = single_gener_p / np.sum(single_gener_p)
                out['JSD_bond_angle_gen_single_traj'][f'single_trace_{j}'].append(jensenshannon(ref_p, single_gener_p))
                out['W1_bond_angle_gen_single_traj'][f'single_trace_{j}'].append(wasserstein_distance(ref_bond_angles[:,i], gener_bond_angles_tensor[j][:,i]))
    if args.plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        # Only plot the first 10 bond angles
        pyemma.plots.plot_feature_histograms(ref_bond_angles[:,:10], ax=ax, color='red', label='Reference')
        pyemma.plots.plot_feature_histograms(md_bond_angles[:,:10], ax=ax, color='yellow', label='MD')
        pyemma.plots.plot_feature_histograms(np.concatenate(gener_bond_angles_tensor, axis=0)[:,:10], ax=ax, color='blue', label='Generated')
        ax.set_ylabel('Feature No.')
        ax.set_xlabel('Bond angle (rad)')
        ax.set_title('Bond Angle Distribution')
        fig.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_bond_angle_distribution.png')
        plt.close()

    ########## Bond length eval
    ref_bond_lengths = []
    for i in range(len(ref_pdb_paths)):
        traj = md.load(ref_traj_paths[i], top=ref_pdb_paths[i])
        bond_lengths = get_bond_lengths_in_traj(traj, mol=mol)
        assert bond_lengths.shape[0] == 12500, f'The ref bond lengths should have shape (12500, N), but it has shape {bond_lengths.shape}'
        bond_lengths = downsample_traj(bond_lengths, num_new_traj=30, num_points=500, stepsize=args.time_step)
        assert len(bond_lengths) == 30, f'The ref bond lengths should have length 30, but it has length {len(bond_lengths)}'
        assert bond_lengths[0].shape[0] == 500, f'The downsampled ref bond lengths should have shape (500, N), but it has shape {bond_lengths[0].shape}'
        ref_bond_lengths += bond_lengths
    ref_bond_lengths = np.concatenate(ref_bond_lengths, axis=0)
    assert ref_bond_lengths.shape[0] == 60000, f'The ref bond lengths should have shape (120*500, N), but it has shape {ref_bond_lengths.shape}'
    assert np.max(ref_bond_lengths) < 0.22, f'The ref bond lengths should be less than 0.22, but it has max {np.max(ref_bond_lengths)}'
    assert np.min(ref_bond_lengths) > 0.1, f'The ref bond lengths should be greater than 0.1, but it has min {np.min(ref_bond_lengths)}'
    md_bond_lengths = []
    for i in range(len(md_pdb_paths)):
        traj = md.load(md_traj_paths[i], top=md_pdb_paths[i])
        bond_lengths = get_bond_lengths_in_traj(traj, mol=mol)
        assert bond_lengths.shape[0] == 12500, f'The md bond lengths should have shape (12500, N), but it has shape {bond_lengths.shape}'
        bond_lengths = downsample_traj(bond_lengths, num_new_traj=10, num_points=500, stepsize=args.time_step)
        assert len(bond_lengths) == 10, f'The md bond lengths should have length 10, but it has length {len(bond_lengths)}'
        assert bond_lengths[0].shape[0] == 500, f'The downsampled md bond lengths should have shape (500, N), but it has shape {bond_lengths[0].shape}'
        md_bond_lengths += bond_lengths
    md_bond_lengths = np.concatenate(md_bond_lengths, axis=0)
    assert md_bond_lengths.shape[0] == 5000, f'The md bond lengths should have shape (10*500, N), but it has shape {md_bond_lengths.shape}'

    gener_bond_lengths_tensor = []
    for i in range(len(gener_traj_tensor)):
        bond_lengths = get_bond_lengths_in_gen(gener_traj_tensor[i], mol=gen_mol)
        gener_bond_lengths_tensor.append(bond_lengths)
    assert gener_bond_lengths_tensor[0].shape[0] == 500, f'The gener bond lengths should have shape (10*500, N), but it has shape {gener_bond_lengths_tensor.shape}'
    assert ref_bond_lengths.shape[1] == gener_bond_lengths_tensor[0].shape[1], f'The ref bond lengths should have the same shape as gener bond lengths, but they have shape {ref_bond_lengths.shape} and {gener_bond_lengths_tensor.shape}. It is because ref exclude all H, but gener include some necessary H. Change the function get_bonds_in_traj to exclude H.'
    out['JSD_bond_length_gen'] = []
    out['JSD_bond_length_md'] = []
    out['W1_bond_length_gen'] = []
    out['W1_bond_length_md'] = []
    out['JSD_bond_length_gen_single_traj'] = defaultdict(list)
    out['W1_bond_length_gen_single_traj'] = defaultdict(list)

    for i in range(ref_bond_lengths.shape[1]):
        ref_p, _ = np.histogram(ref_bond_lengths[:,i], range=(0.1,0.22), bins=100)  # The bond length is usually between 100 and 220 pm.
        ref_p = ref_p / np.sum(ref_p)
        md_p, _ = np.histogram(md_bond_lengths[:,i], range=(0.1,0.22), bins=100)
        md_p = md_p / np.sum(md_p)
        gener_p, _ = np.histogram(np.concatenate(gener_bond_lengths_tensor,axis=0)[:,i], range=(0.1,0.22), bins=100)
        gener_p = gener_p / np.sum(gener_p)
        out['JSD_bond_length_gen'].append(jensenshannon(ref_p, gener_p))
        out['JSD_bond_length_md'].append(jensenshannon(ref_p, md_p))
        out['W1_bond_length_gen'].append(wasserstein_distance(ref_bond_lengths[:,i], np.concatenate(gener_bond_lengths_tensor,axis=0)[:,i]))
        out['W1_bond_length_md'].append(wasserstein_distance(ref_bond_lengths[:,i], md_bond_lengths[:,i]))
        for j in range(len(gener_bond_lengths_tensor)):
            single_gener_p, _ = np.histogram(gener_bond_lengths_tensor[j][:,i], range=(0.1,0.22), bins=100)
            if np.sum(single_gener_p) == 0:
                out['JSD_bond_length_gen_single_traj'][f'single_trace_{j}'].append(-1)
                out['W1_bond_length_gen_single_traj'][f'single_trace_{j}'].append(-1)  # Just hard code -1 distance to avoid error in the jsd calculation.
            else:
                single_gener_p = single_gener_p / np.sum(single_gener_p)
                out['JSD_bond_length_gen_single_traj'][f'single_trace_{j}'].append(jensenshannon(ref_p, single_gener_p))
                out['W1_bond_length_gen_single_traj'][f'single_trace_{j}'].append(wasserstein_distance(ref_bond_lengths[:,i], gener_bond_lengths_tensor[j][:,i]))

    if args.plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        pyemma.plots.plot_feature_histograms(ref_bond_lengths[:, :10], ax=ax, color='red', label='Reference')
        pyemma.plots.plot_feature_histograms(md_bond_lengths[:, :10], ax=ax, color='yellow', label='MD')
        pyemma.plots.plot_feature_histograms(np.concatenate(gener_bond_lengths_tensor,axis=0)[:, :10], ax=ax, color='blue', label='Generated')
        ax.set_title('Bond Length Distribution')
        ax.set_ylabel('Feature No.')
        ax.set_xlabel('Bond length (nm)')
        fig.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_bond_length_distribution.png')
        plt.close()

    ############ Torsion decorrelations
    out['decorrelation_ref_in_ps'] = {}
    out['decorrelation_gen_in_ps'] = {}
    out['decorrelation_md_in_ps'] = {}
    fig, axs = plt.subplots(1,3,figsize=(15, 5))
    for i, feat in enumerate(feat_dihedral.describe()):
        decorrelation_times_ref = []
        decorrelation_times_gen = []
        decorrelation_times_md = []
        for j in range(len(ref_torsion_tensor)):  # Calculate for each torsion angle in each ref trajectory
            this_torsion = ref_torsion_tensor[j][:,i]
            autocorr = acovf(np.sin(this_torsion), demean=False, adjusted=True, nlag=498) + acovf(np.cos(this_torsion), demean=False, adjusted=True, nlag=498)
            baseline = np.sin(this_torsion).mean()**2 + np.cos(this_torsion).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if 1-baseline < 1e-8:
                decorrelation_times_ref.append(-1)
            else:
                autocorr = (autocorr - baseline) / (1-baseline)
                autocorr = np.array(autocorr, dtype=np.float32)
                decorrelation_times_ref.append(find_decorrelation_time(autocorr, time_step=args.time_step))
        out['decorrelation_ref_in_ps'][feat] = decorrelation_times_ref
        if args.plot:
            axs[0].plot(lags*args.time_step, autocorr, label=f'Torsion {i}')
            axs[0].set_title(f'Ref')
        for j in range(len(md_torsion_tensor)):
            this_torsion = md_torsion_tensor[j][:,i]
            autocorr = acovf(np.sin(this_torsion), demean=False, adjusted=True, nlag=498) + acovf(np.cos(this_torsion), demean=False, adjusted=True, nlag=498)
            baseline = np.sin(this_torsion).mean()**2 + np.cos(this_torsion).mean()**2
            # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
            lags = 1 + np.arange(len(autocorr))
            if 1-baseline < 1e-8:
                decorrelation_times_md.append(-1)
            else:
                autocorr = (autocorr - baseline) / (1-baseline)
                autocorr = np.array(autocorr, dtype=np.float32)
                decorrelation_times_md.append(find_decorrelation_time(autocorr, time_step=args.time_step))
        out['decorrelation_md_in_ps'][feat] = decorrelation_times_md
        if args.plot:
            axs[1].plot(lags*args.time_step, autocorr, label=f'Torsion {i}')
            axs[1].set_title(f'MD')
        for j in range(len(gener_traj_tensor)):
            this_torsion = gener_torsion_tensor[j][:,i]
            autocorr = acovf(np.sin(this_torsion), demean=False, adjusted=True, nlag=498) + acovf(np.cos(this_torsion), demean=False, adjusted=True, nlag=498)
            baseline = np.sin(this_torsion).mean()**2 + np.cos(this_torsion).mean()**2
            lags = 1 + np.arange(len(autocorr))
            if 1-baseline < 1e-8:
                decorrelation_times_gen.append(-1)
            else:
                autocorr = (autocorr - baseline) / (1-baseline)
                autocorr = np.array(autocorr, dtype=np.float32)
                decorrelation_times_gen.append(find_decorrelation_time(autocorr, time_step=args.time_step))
        out['decorrelation_gen_in_ps'][feat] = decorrelation_times_gen
        if args.plot:
            axs[2].plot(lags*args.time_step, autocorr, label=f'Torsion {i}')
            axs[2].set_title(f'Gen')
    if args.plot:
        for i in range(3):
            axs[i].set_xlabel('Lag time (ps)')
            axs[i].set_ylabel('Autocorrelation')
            axs[i].legend()
            axs[i].set_xscale('log')
            axs[i].set_ylim(-0.1, 1.1)
            axs[i].axhline(y=1/np.e, color='gray', linestyle='--', label='1/e')
        fig.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_torsion_autocorrelation.png')
        plt.close()

    ####### TICA #############
    # Reload the ref traj and feat_dihedral, because now we need to use sin and cos of torsion instead of the original torsion angle.
    feat_dihedral = pyemma.coordinates.featurizer(ref_pdb_paths[0])
    feat_dihedral.add_dihedrals(torsion_index, cossin=True, periodic=False)
    # print(feat_dihedral.describe())
    ref_torsion_tensor = pyemma.coordinates.load(ref_traj_paths, features=feat_dihedral)  # A list of all torsion traj. Each traj should have (N, T)
    downsampled_ref_torsion = []
    for traj in ref_torsion_tensor:
        downsampled_traj = downsample_traj(traj, num_new_traj=30, num_points=500, stepsize=args.time_step)
        downsampled_ref_torsion += downsampled_traj
    assert len(downsampled_ref_torsion) == 120, f'The ref torsion should have length 120, but it has length {len(downsampled_ref_torsion)}'
    assert downsampled_ref_torsion[0].shape[0] == 500, f'The downsampled ref torsion should have shape (500, N), but it has shape {downsampled_ref_torsion[0].shape}'
    ref_torsion_tensor = downsampled_ref_torsion

    # Reload md torsion
    feat_dihedral = pyemma.coordinates.featurizer(md_pdb_paths[0])
    feat_dihedral.add_dihedrals(torsion_index, cossin=True, periodic=False)
    md_torsion_tensor = pyemma.coordinates.load(md_traj_paths, features=feat_dihedral)
    if not isinstance(md_torsion_tensor, list):
        md_torsion_tensor = [md_torsion_tensor]
    downsampled_md_torsion = []
    for traj in md_torsion_tensor:
        downsampled_traj = downsample_traj(traj, num_new_traj=10, num_points=500, stepsize=args.time_step)
        downsampled_md_torsion += downsampled_traj
    assert len(downsampled_md_torsion) == 10, f'The md torsion should have length 10, but it has length {len(downsampled_md_torsion)}'
    assert downsampled_md_torsion[0].shape[0] == 500, f'The downsampled md torsion should have shape (500, N), but it has shape {downsampled_md_torsion[0].shape}'
    md_torsion_tensor = downsampled_md_torsion

    # Reload gener torsion
    gener_torsion_tensor = []    # (num_traj, 500, 2N)
    for i in range(len(gener_traj_tensor)):
        torsions = get_torsions_in_gen(gener_traj_tensor[i], torsion_index_noH)  # (500,N)
        torsions_2dim = []
        for j in range(torsions.shape[1]):
            torsions_cossin = np.concatenate([np.cos(torsions[:,j])[:,None], np.sin(torsions[:,j])[:,None]], axis=1)
            assert torsions_cossin.shape == (500, 2), f'The torsion should have shape (500, 2), but it has shape {torsions_cossin.shape}'
            torsions_2dim.append(torsions_cossin)
        torsions_2dim = np.concatenate(torsions_2dim,axis=1)
        assert torsions_2dim.shape[0] == 500, f'The torsion should have shape (500, 2N), but it has shape {torsions_2dim.shape}'
        assert torsions_2dim.shape[1] % 2 == 0, f'The torsion should have shape (500, 2N), but it has shape {torsions_2dim.shape}'
        gener_torsion_tensor.append(torsions_2dim)
    assert ref_torsion_tensor[0].shape == gener_torsion_tensor[0].shape, f'The shape of ref and gener torsion tensor should be both 500, but they are {ref_torsion_tensor[0].shape} and {gener_torsion_tensor[0].shape}'
    
    tica, ref_tica, ref_concatenate = tica_on_ref(ref_torsion_tensor, lag=args.tica_lag, plot=False)

    out['JSD_TICA_gen'] = {}
    out['JSD_TICA_md'] = {}
    out['W1_TICA_gen'] = {}
    out['W1_TICA_md'] = {}
    out['JSD_TICA_gen_single_traj'] = {}
    out['W1_TICA_gen_single_traj'] = {}
    out['JSD_TICA_gen_single_traj']['TICA-0'] = defaultdict(list)
    out['W1_TICA_gen_single_traj']['TICA-0'] = defaultdict(list)
    out['JSD_TICA_gen_single_traj']['TICA-0,1'] = defaultdict(list)
    
    gen_tica = tica_projection(gener_torsion_tensor, tica, plot_dir=plot_dir, name=name, plot=args.plot, num_points_to_plot=100)
    md_tica = tica_projection(md_torsion_tensor, tica, plot_dir=plot_dir, name=name, plot=False, num_points_to_plot=100)

    tica_0_min = min(ref_concatenate[:,0].min(), np.concatenate(gen_tica)[:,0].min(), np.concatenate(md_tica)[:,0].min())
    tica_0_max = max(ref_concatenate[:,0].max(), np.concatenate(gen_tica)[:,0].max(), np.concatenate(md_tica)[:,0].max())

    ref_p, _ = np.histogram(ref_concatenate[:,0], range=(tica_0_min, tica_0_max), bins=100)
    ref_p = ref_p / np.sum(ref_p)
    traj_p, _ = np.histogram(np.concatenate(gen_tica)[:,0], range=(tica_0_min, tica_0_max), bins=100)
    traj_p = traj_p / np.sum(traj_p)
    md_p, _ = np.histogram(np.concatenate(md_tica)[:,0], range=(tica_0_min, tica_0_max), bins=100)
    md_p = md_p / np.sum(md_p)
    out['JSD_TICA_gen']['TICA-0'] = jensenshannon(ref_p, traj_p)
    out['JSD_TICA_md']['TICA-0'] = jensenshannon(ref_p, md_p)
    out['W1_TICA_gen']['TICA-0'] = wasserstein_distance(ref_concatenate[:,0], np.concatenate(gen_tica)[:,0])
    out['W1_TICA_md']['TICA-0'] = wasserstein_distance(ref_concatenate[:,0], np.concatenate(md_tica)[:,0])
    for j in range(len(gen_tica)):
        single_gener_p, _ = np.histogram(gen_tica[j][:,0], range=(tica_0_min, tica_0_max), bins=100)
        if np.sum(single_gener_p) == 0:
            out['JSD_TICA_gen_single_traj']['TICA-0'][f'single_trace_{j}'].append(-1)
            out['W1_TICA_gen_single_traj']['TICA-0'][f'single_trace_{j}'].append(-1)
        else:
            single_gener_p = single_gener_p / np.sum(single_gener_p)
            out['JSD_TICA_gen_single_traj']['TICA-0'][f'single_trace_{j}'].append(jensenshannon(ref_p, single_gener_p))
            out['W1_TICA_gen_single_traj']['TICA-0'][f'single_trace_{j}'].append(wasserstein_distance(ref_concatenate[:,0], gen_tica[j][:,0]))
    if ref_tica[0].shape[1] > 2:  # has more than 1 tica component
        tica_1_min = min(ref_concatenate[:,1].min(), np.concatenate(gen_tica)[:,1].min(), np.concatenate(md_tica)[:,1].min())
        tica_1_max = max(ref_concatenate[:,1].max(), np.concatenate(gen_tica)[:,1].max(), np.concatenate(md_tica)[:,1].max())
        ref_p = np.histogram2d(*ref_concatenate[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=100)[0]
        ref_p = ref_p / np.sum(ref_p)
        traj_p = np.histogram2d(*np.concatenate(gen_tica)[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=100)[0]
        traj_p = traj_p / np.sum(traj_p)
        md_p = np.histogram2d(*np.concatenate(md_tica)[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=100)[0]
        md_p = md_p / np.sum(md_p)
        out['JSD_TICA_gen']['TICA-0,1'] = jensenshannon(ref_p.flatten(), traj_p.flatten())
        out['JSD_TICA_md']['TICA-0,1'] = jensenshannon(ref_p.flatten(), md_p.flatten())
        for j in range(len(gen_tica)):
            single_gener_p = np.histogram2d(*gen_tica[j][:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=100)[0]
            if np.sum(single_gener_p) == 0:
                out['JSD_TICA_gen_single_traj']['TICA-0,1'][f'single_trace_{j}'].append(-1)
            else:
                single_gener_p = single_gener_p / np.sum(single_gener_p)
                out['JSD_TICA_gen_single_traj']['TICA-0,1'][f'single_trace_{j}'].append(jensenshannon(ref_p.flatten(), single_gener_p.flatten()))

        #### TICA FES Plots
        if args.plot:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            pyemma.plots.plot_free_energy(*ref_concatenate[:, :2].T, ax=axs[0], cbar=False)
            pyemma.plots.plot_free_energy(*np.concatenate(gen_tica)[:, :2].T, ax=axs[1], cbar=False)
            pyemma.plots.plot_free_energy(*np.concatenate(md_tica)[:, :2].T, ax=axs[2], cbar=False)
            axs[0].set_title('TICA FES (Ref)')
            axs[1].set_title('TICA FES (Gen)')
            axs[2].set_title('TICA FES (MD)')
            xlims = [axs[0].get_xlim(), axs[1].get_xlim(), axs[2].get_xlim()]
            ylims = [axs[0].get_ylim(), axs[1].get_ylim(), axs[2].get_ylim()]
            x_min = min(x[0] for x in xlims)
            x_max = max(x[1] for x in xlims)
            y_min = min(y[0] for y in ylims)
            y_max = max(y[1] for y in ylims)
            for ax in axs:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            plt.savefig(f'{plot_dir}/{name}_TICA_FES.png')
            plt.close()

    ###### Markov state model stuff #################
    if not args.no_msm:
        clusters, k = kmeans(tica, k=100)
        ref_clusters = clusters.dtrajs
        gen_clusters = clusters.transform(gen_tica)
        md_clusters = clusters.transform(md_tica)
        try:
            msm = pyemma.msm.estimate_markov_model(ref_clusters, lag=args.msm_lag)
            nstates = 10
            msm.pcca(nstates)
            ref_p, gen_p = get_metastate_prob(msm, gen_clusters, k, nstates)
            ref_p, md_p = get_metastate_prob(msm, md_clusters, k, nstates)
            out['ref_metastable_p'] = ref_p
            out['gen_metastable_p'] = gen_p
            out['md_metastable_p'] = md_p
            out['JSD_msm_gen'] = jensenshannon(ref_p, gen_p)
            out['JSD_msm_md'] = jensenshannon(ref_p, md_p)
            out['W1_msm_gen'] = wasserstein_distance(np.arange(nstates),np.arange(nstates), ref_p, gen_p)
            out['W1_msm_md'] = wasserstein_distance(np.arange(nstates),np.arange(nstates), ref_p, md_p)

        except Exception as e:
            print('ERROR', e, smiles, flush=True)

    return smiles, out

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=float, default=5.2)
parser.add_argument('--ref_dir', type=str, default=os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/test')
parser.add_argument('--gener_dir', type=str, default='model_outputs_official/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_interpolator_pretrain_final/epoch=199-step=138400-gen.pkl')
# naive EGTN: model_outputs_official/qm9_trajectory_official/qm9_noH_1000_kabsch_traj_egtn/epoch=199-step=138400-gen.pkl
parser.add_argument('--smile_match', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'qm9_test_smile_match.txt'))
parser.add_argument('--gener_smiles', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'eval_qm9_unconditional_generation_smiles.txt'))
parser.add_argument('--save', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save_name', type=str, default='evalout')
parser.add_argument('--no_msm', action='store_true')
parser.add_argument('--truncate', type=int, default=None)
parser.add_argument('--tica_lag', type=int, default=2)
parser.add_argument('--msm_lag', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--check_nan', action='store_true')
args = parser.parse_args()

with open(args.gener_smiles, 'r') as f:
    gener_smiles_tot = [line.strip() for line in f.readlines()]

with open(args.gener_dir, 'rb') as f:
    gen_dict = pickle.load(f)

if type(list(gen_dict.keys())[0]) is tuple:
    new_gen_dict = {k[0]:gen_dict[k] for k in gen_dict}
    gen_dict = new_gen_dict
generated_smiles = set(gen_dict.keys())

gener_smiles = [smi for smi in gener_smiles_tot if smi in generated_smiles]
print('number of unique molecules with generated trajectories', len(gener_smiles))

plot_dir = None
if args.plot:
    plot_dir = f'{args.save_name}_plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

if args.num_workers > 1:
    p = Pool(args.num_workers)
    p.__enter__()
    __map__ = p.imap
else:
    __map__ = map
out = dict(tqdm.tqdm(__map__(main, gener_smiles), total=len(gener_smiles)))
if args.num_workers > 1:
    p.__exit__(None, None, None)

if args.save:
    with open(f"{args.save_name}_evalout.pkl", 'wb') as f:
        f.write(pickle.dumps(out))