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
from utils import get_ref_and_generated_traj, get_torsions_idx_mol, find_decorrelation_time, tica_on_ref, tica_projection, kmeans
import mdtraj as md
import torch
from rdkit import Chem
from collections import defaultdict
from utils import get_bond_angles_in_traj, get_bond_lengths_in_traj, get_torsions_in_gen, get_bond_angles_in_gen, get_bond_lengths_in_gen, get_torsion_index_noH, get_metastate_prob
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def compute_rmsd(traj: md.Trajectory) -> np.ndarray:
    # compute RMSD using only alpha-carbons (CA)
    ca_idxs = traj.topology.select('name CA')
    # align frames to frame 0 using CA atoms, then compute CA-only RMSD
    traj.superpose(traj[0], atom_indices=ca_idxs)
    rmsd_ca = md.rmsd(traj, traj, frame=0, atom_indices=ca_idxs)
    return rmsd_ca

def compute_backbone_sidechain(traj: md.Trajectory):
    """Extract backbone (phi, psi) and sidechain (chi1-4) dihedral angles using MDTraj."""
    backbone = {}
    sidechain = {}
    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    if phi.size:
        backbone["phi"] = phi
    if psi.size:
        backbone["psi"] = psi
    for name, fn in [("chi1", md.compute_chi1), ("chi2", md.compute_chi2), 
                      ("chi3", md.compute_chi3), ("chi4", md.compute_chi4)]:
        try:
            _, arr = fn(traj)
        except ValueError:
            arr = np.empty((traj.n_frames, 0))
        if arr.size:
            sidechain[name] = arr
    return backbone, sidechain

def get_backbone_sidechain_rmsd_features(traj: md.Trajectory, cossin=False, include_rmsd=True):
    """
    Extract backbone and sidechain features from an MDTraj trajectory, as well as the CA RMSD.
    Returns concatenated array of [phi, psi, chi1, chi2, chi3, chi4, (rmsd_ca)] angles.
    If cossin=True, returns [cos(phi), sin(phi), cos(psi), sin(psi), ..., (rmsd_ca)] for TICA.
    
    Args:
        traj: MDTraj trajectory object
        cossin: If True, return cos/sin representation for each angle
    
    Returns:
        np.ndarray: Feature array of shape (n_frames, n_features)
    """
    bb, sc = compute_backbone_sidechain(traj)
    rmsd = compute_rmsd(traj).reshape(-1, 1)  # (n_frames, 1)
    features = []
    for angle_name in ['phi', 'psi']:
        if angle_name in bb and bb[angle_name].size > 0:
            angles = bb[angle_name]
            if cossin:
                features.append(np.cos(angles))
                features.append(np.sin(angles))
            else:
                features.append(angles)
    
    for angle_name in ['chi1', 'chi2', 'chi3', 'chi4']:
        if angle_name in sc and sc[angle_name].size > 0:
            angles = sc[angle_name]
            if cossin:
                features.append(np.cos(angles))
                features.append(np.sin(angles))
            else:
                features.append(angles)
    if include_rmsd:
        features.append(rmsd)
    if features:
        return np.concatenate(features, axis=1)
    return np.empty((traj.n_frames, 0))

def main(smiles):
    # print(f'\nThe evaluated molecules is {smiles}')
    out = {}
    
    # Get the reference filename for this SMILES
    if smiles not in smiles_to_filename:
        print(f"No reference file found for {smiles}")
        return smiles, {}
    
    ref_filename = smiles_to_filename[smiles]
    
    # Load reference trajectory from preprocessed pickle
    ref_pkl_path = f'{args.ref_dir}/{ref_filename}.pkl'
    try:
        with open(ref_pkl_path, 'rb') as f:
            ref_data = pickle.load(f)
        ref_traj_full = ref_data['coordinates_mdtraj']  # MDTraj object (T, N, 3) in nm
        mol = ref_data['rdkit_mol']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading reference for {smiles} from {ref_filename}: {e}")
        return smiles, {}
    
    name = ref_filename  # Use amino acid sequence as the name
    
    ### Read pickle file from gener_dir
    with open(args.gener_dir, 'rb') as f:
        gener_data = pickle.load(f)

        # if the keys are 2‑tuples, reduce to just the SMILES string
        if isinstance(next(iter(gener_data)), tuple):
            gener_data = {k[0]: gener_data[k] for k in gener_data}
            
        gen_mol = gener_data[smiles]['rdmol']  # It already has Hs removed.
        gen_mol = Chem.RemoveHs(gen_mol)  # Double check to remove Hs from the molecule
        gener_traj_coords = gener_data[smiles]['coords']  # (5, 31, 3, 1001)
    
    # Check for NaN values in generated coordinates
    for i in range(len(gener_traj_coords)):
        if np.isnan(gener_traj_coords[i]).any():
            print(f"WARNING: NaN encountered in generated traj at index {i} for molecule {smiles}, skipping")
            return smiles, {}
    
    if args.check_nan:
        # Early return for NaN checking mode
        return smiles, out
    torsion_index = np.array(get_torsions_idx_mol(mol)[0]+get_torsions_idx_mol(mol)[2]).reshape(-1,4)
    torsion_index_noH = get_torsion_index_noH(torsion_index, mol)
    
    ### Downsample reference trajectory with stride of 10 (like training)
    ref_traj_downsampled = ref_traj_full[::10]
    
    # Extract backbone and sidechain from full downsampled reference
    ref_bb_full, ref_sc_full = compute_backbone_sidechain(ref_traj_downsampled)
    # Extract CA RMSD from reference trajectory
    ref_rmsd_ca = compute_rmsd(ref_traj_downsampled)
    
    ### Load generated trajectories
    gener_traj_tensor = []
    for i in range(len(gener_traj_coords)):
        gener_traj_tensor.append(torch.permute(torch.from_numpy(gener_traj_coords[i] / 10), (2,0,1))[1:]) 
    assert gener_traj_tensor[0].shape[2] == 3, f'The gener traj should have shape (1000, N, 3), but it has shape {gener_traj_tensor[0].shape}'
    
    # Extract backbone and sidechain from each generated trajectory, as well as the CA RMSD
    gener_backbone_all = []
    gener_sidechain_all = []
    gener_rmsd_ca_all = []
    gener_traj_objects = []  # Store MDTraj objects for feature extraction
    for gen_coords in gener_traj_tensor:
        gen_traj_obj = md.Trajectory(gen_coords.numpy(), ref_traj_full.topology)
        bb, sc = compute_backbone_sidechain(gen_traj_obj)
        rmsd_ca = compute_rmsd(gen_traj_obj)
        gener_backbone_all.append(bb)
        gener_sidechain_all.append(sc)
        gener_rmsd_ca_all.append(rmsd_ca)
        gener_traj_objects.append(gen_traj_obj)
    
    # Plot CA RMSD for reference (top) and concatenated generated trajectories (bottom)
    if args.plot and len(gener_rmsd_ca_all):
        gen_rmsd_concat = np.concatenate(gener_rmsd_ca_all)
        ref_len = len(ref_rmsd_ca)
        gen_len = len(gen_rmsd_concat)
        if ref_len != gen_len:
            min_len = min(ref_len, gen_len)
            print(f"WARNING: RMSD length mismatch for {smiles}: ref={ref_len}, gen={gen_len}. Trimming to {min_len} frames.")
            ref_rmsd_plot = ref_rmsd_ca[:min_len]
            gen_rmsd_plot = gen_rmsd_concat[:min_len]
        else:
            ref_rmsd_plot = ref_rmsd_ca
            gen_rmsd_plot = gen_rmsd_concat

        time_ref = np.arange(len(ref_rmsd_plot)) * args.time_step
        time_gen = np.arange(len(gen_rmsd_plot)) * args.time_step

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(time_ref, ref_rmsd_plot, color=colors[0])
        axs[0].set_ylabel('CA RMSD (nm)')
        axs[0].set_title('Reference CA RMSD')
        axs[1].plot(time_gen, gen_rmsd_plot, color=colors[1])
        axs[1].set_ylabel('CA RMSD (nm)')
        axs[1].set_xlabel('Time (ps)')
        axs[1].set_title('Generated CA RMSD (5 concatenated trajectories)')
        fig.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_rmsd.png')
        plt.close()
    
    ### Extract feature arrays for decorrelation, TICA, and MSM
    # Raw angles (for decorrelation and JSD)
    ref_features = get_backbone_sidechain_rmsd_features(ref_traj_downsampled, cossin=False, include_rmsd=False)
    gener_features = [get_backbone_sidechain_rmsd_features(traj_obj, cossin=False, include_rmsd=False) 
                      for traj_obj in gener_traj_objects]
    
    # Cos/sin representation (for TICA and MSM)
    ref_features_cossin = get_backbone_sidechain_rmsd_features(ref_traj_downsampled, cossin=True, include_rmsd=args.feat_rmsd)
    gener_features_cossin = [get_backbone_sidechain_rmsd_features(traj_obj, cossin=True, include_rmsd=args.feat_rmsd) 
                             for traj_obj in gener_traj_objects]
    
    # Check for NaN values in generated features
    if any(np.isnan(feat).any() for feat in gener_features):
        print(f"WARNING: NaN values detected in generated features for {smiles}, skipping this molecule")
        return smiles, {}
    
    if any(np.isnan(feat).any() for feat in gener_features_cossin):
        print(f"WARNING: NaN values detected in generated cos/sin features for {smiles}, skipping this molecule")
        return smiles, {}
    
    # Check for empty feature arrays
    if ref_features.shape[1] == 0 or any(feat.shape[1] == 0 for feat in gener_features):
        print(f"WARNING: Empty feature arrays for {smiles}, skipping this molecule")
        return smiles, {}
    
    ### Compute JSD and W1 for backbone angles
    out['JSD_backbone_gen'] = {}
    out['W1_backbone_gen'] = {}
    
    for angle_name in ['phi', 'psi']:
        if angle_name in ref_bb_full and ref_bb_full[angle_name].size > 0:
            # Concatenate all generated angles
            gen_angles = [bb[angle_name] for bb in gener_backbone_all if angle_name in bb]
            if gen_angles:
                gen_concat = np.concatenate(gen_angles, axis=0)
                ref_angles = ref_bb_full[angle_name]
                
                # Flatten all angles (across residues and frames) for single distribution comparison
                ref_flat = ref_angles.ravel()
                gen_flat = gen_concat.ravel()
                
                ref_p, _ = np.histogram(ref_flat, range=(-np.pi, np.pi), bins=100)
                ref_p = ref_p / np.sum(ref_p)
                gen_p, _ = np.histogram(gen_flat, range=(-np.pi, np.pi), bins=100)
                gen_p = gen_p / np.sum(gen_p)
                
                out['JSD_backbone_gen'][angle_name] = float(jensenshannon(ref_p, gen_p))
                out['W1_backbone_gen'][angle_name] = float(wasserstein_distance(ref_flat, gen_flat))
    
    ### Compute JSD and W1 for sidechain angles
    out['JSD_sidechain_gen'] = {}
    out['W1_sidechain_gen'] = {}
    
    for angle_name in ['chi1', 'chi2', 'chi3', 'chi4']:
        if angle_name in ref_sc_full and ref_sc_full[angle_name].size > 0:
            gen_angles = [sc[angle_name] for sc in gener_sidechain_all if angle_name in sc]
            if gen_angles:
                gen_concat = np.concatenate(gen_angles, axis=0)
                ref_angles = ref_sc_full[angle_name]
                
                # Flatten all angles (across residues and frames) for single distribution comparison
                ref_flat = ref_angles.ravel()
                gen_flat = gen_concat.ravel()
                
                ref_p, _ = np.histogram(ref_flat, range=(-np.pi, np.pi), bins=100)
                ref_p = ref_p / np.sum(ref_p)
                gen_p, _ = np.histogram(gen_flat, range=(-np.pi, np.pi), bins=100)
                gen_p = gen_p / np.sum(gen_p)
                
                out['JSD_sidechain_gen'][angle_name] = float(jensenshannon(ref_p, gen_p))
                out['W1_sidechain_gen'][angle_name] = float(wasserstein_distance(ref_flat, gen_flat))
    
    # Plot backbone and sidechain angle histograms
    if args.plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        # Plot using the feature arrays we created
        pyemma.plots.plot_feature_histograms(np.concatenate(gener_features), ax=ax, color='blue', label='Generated')
        pyemma.plots.plot_feature_histograms(ref_features, ax=ax, color='red', label='Reference')
        ax.set_ylabel('Feature No.')
        ax.set_xlabel('Angle (rad)')
        ax.set_title('Backbone and Sidechain Angle Distribution')
        fig.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_backbone_sidechain_hist.png')
        plt.close()

    ########### Bond angle eval
    # Extract bond angles from reference (using downsampled trajectory)
    ref_bond_angles = get_bond_angles_in_traj(ref_traj_downsampled, mol=mol)
    
    # MD baseline - commented out for tetrapeptides
    # md_bond_angles = []
    # for i in range(len(md_pdb_paths)):
    #     traj = md.load(md_traj_paths[i], top=md_pdb_paths[i])
    #     bond_angles = get_bond_angles_in_traj(traj, mol=mol)
    #     assert bond_angles.shape[0] == 12500, f'The md bond angles should have shape (12500, N), but it has shape {bond_angles.shape}'
    #     bond_angles = downsample_traj(bond_angles, num_new_traj=5, num_points=960, stepsize=args.time_step)
    #     assert len(bond_angles) == 5, f'The md bond angles should have length 5, but it has length {len(bond_angles)}'
    #     assert bond_angles[0].shape[0] == 960, f'The downsampled md bond angles should have shape (960, N), but it has shape {bond_angles[0].shape}'
    #     if np.isnan(bond_angles).any():
    #         print(f"Nan encountered in md traj at index {i} for molecule {smiles}")
    #         raise ValueError(f"Nan encountered in md traj at index {i} for molecule {smiles}")
    #     md_bond_angles += bond_angles
    # md_bond_angles = np.concatenate(md_bond_angles, axis=0)
    # assert md_bond_angles.shape[0] == 4800, f'The md bond angles should have shape (5*960, N), but it has shape {md_bond_angles.shape}'

    # Extract bond angles from generated trajectories
    gener_bond_angles_tensor = []
    for i in range(len(gener_traj_tensor)):
        bond_angles = get_bond_angles_in_gen(gener_traj_tensor[i], mol=gen_mol)
        gener_bond_angles_tensor.append(bond_angles)

    # Compute JSD and W1 for bond angles
    out['JSD_bond_angle_gen'] = []
    out['W1_bond_angle_gen'] = []
    
    for i in range(ref_bond_angles.shape[1]):
        ref_p, _ = np.histogram(ref_bond_angles[:,i], range=(0,np.pi), bins=100)
        ref_p = ref_p / np.sum(ref_p)
        gener_p, _ = np.histogram(np.concatenate(gener_bond_angles_tensor, axis=0)[:,i], range=(0,np.pi), bins=100)
        gener_p = gener_p / np.sum(gener_p)
        out['JSD_bond_angle_gen'].append(jensenshannon(ref_p, gener_p))
        out['W1_bond_angle_gen'].append(wasserstein_distance(ref_bond_angles[:,i], np.concatenate(gener_bond_angles_tensor, axis=0)[:,i]))
    
    if args.plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        # Only plot the first 10 bond angles
        pyemma.plots.plot_feature_histograms(np.concatenate(gener_bond_angles_tensor, axis=0)[:,:10], ax=ax, color='blue', label='Generated')
        pyemma.plots.plot_feature_histograms(ref_bond_angles[:,:10], ax=ax, color='red', label='Reference')
        ax.set_ylabel('Feature No.')
        ax.set_xlabel('Bond angle (rad)')
        ax.set_title('Bond Angle Distribution')
        fig.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_bond_angle_distribution.png')
        plt.close()

    ########## Bond length eval
    # Extract bond lengths from reference
    ref_bond_lengths = get_bond_lengths_in_traj(ref_traj_downsampled, mol=mol)
    
    # MD baseline - commented out for tetrapeptides
    # md_bond_lengths = []
    # for i in range(len(md_pdb_paths)):
    #     traj = md.load(md_traj_paths[i], top=md_pdb_paths[i])
    #     bond_lengths = get_bond_lengths_in_traj(traj, mol=mol)
    #     assert bond_lengths.shape[0] == 12500, f'The md bond lengths should have shape (12500, N), but it has shape {bond_lengths.shape}'
    #     bond_lengths = downsample_traj(bond_lengths, num_new_traj=5, num_points=960, stepsize=args.time_step)
    #     assert len(bond_lengths) == 5, f'The md bond lengths should have length 5, but it has length {len(bond_lengths)}'
    #     assert bond_lengths[0].shape[0] == 960, f'The downsampled md bond lengths should have shape (960, N), but it has shape {bond_lengths[0].shape}'
    #     md_bond_lengths += bond_lengths
    # md_bond_lengths = np.concatenate(md_bond_lengths, axis=0)
    # assert md_bond_lengths.shape[0] == 4800, f'The md bond lengths should have shape (5*960, N), but it has shape {md_bond_lengths.shape}'

    # Extract bond lengths from generated trajectories
    gener_bond_lengths_tensor = []
    for i in range(len(gener_traj_tensor)):
        bond_lengths = get_bond_lengths_in_gen(gener_traj_tensor[i], mol=gen_mol)
        gener_bond_lengths_tensor.append(bond_lengths)
    
    # Compute JSD and W1 for bond lengths
    out['JSD_bond_length_gen'] = []
    out['W1_bond_length_gen'] = []

    for i in range(ref_bond_lengths.shape[1]):
        ref_p, _ = np.histogram(ref_bond_lengths[:,i], range=(0.1,0.22), bins=100)
        ref_p = ref_p / np.sum(ref_p)
        gener_p, _ = np.histogram(np.concatenate(gener_bond_lengths_tensor,axis=0)[:,i], range=(0.1,0.22), bins=100)
        gener_p = gener_p / np.sum(gener_p)
        out['JSD_bond_length_gen'].append(jensenshannon(ref_p, gener_p))
        out['W1_bond_length_gen'].append(wasserstein_distance(ref_bond_lengths[:,i], np.concatenate(gener_bond_lengths_tensor,axis=0)[:,i]))

    if args.plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        pyemma.plots.plot_feature_histograms(np.concatenate(gener_bond_lengths_tensor,axis=0)[:, :10], ax=ax, color='blue', label='Generated')
        pyemma.plots.plot_feature_histograms(ref_bond_lengths[:, :10], ax=ax, color='red', label='Reference')
        ax.set_title('Bond Length Distribution')
        ax.set_ylabel('Feature No.')
        ax.set_xlabel('Bond length (nm)')
        fig.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_bond_length_distribution.png')
        plt.close()

    ############ Backbone/Sidechain Decorrelations
    out['decorrelation_ref_in_ps'] = {}
    out['decorrelation_gen_in_ps'] = {}
    
    if args.plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute decorrelation for reference trajectory
    max_lag_ref = min(900, ref_features.shape[0] - 1)
    for i in range(ref_features.shape[1]):
        this_angle = ref_features[:, i]
        autocorr = acovf(np.sin(this_angle), demean=False, adjusted=True, nlag=max_lag_ref) + \
                   acovf(np.cos(this_angle), demean=False, adjusted=True, nlag=max_lag_ref)
        baseline = np.sin(this_angle).mean()**2 + np.cos(this_angle).mean()**2
        lags = 1 + np.arange(len(autocorr))
        
        if 1 - baseline < 1e-8:
            out['decorrelation_ref_in_ps'][f'feature_{i}'] = -1
        else:
            autocorr = (autocorr - baseline) / (1 - baseline)
            autocorr = np.array(autocorr, dtype=np.float32)
            out['decorrelation_ref_in_ps'][f'feature_{i}'] = find_decorrelation_time(autocorr, time_step=args.time_step)
            
            if args.plot:
                axs[0].plot(lags * args.time_step, autocorr, label=f'Feature {i}')
    
    if args.plot:
        axs[0].set_xlabel('Lag time (ps)')
        axs[0].set_ylabel('Autocorrelation')
        axs[0].set_title('Reference')
        axs[0].set_xscale('log')
        axs[0].set_ylim(-0.1, 1.1)
        axs[0].axhline(y=1/np.e, color='gray', linestyle='--', alpha=0.5)
    
    # Compute decorrelation for generated trajectories
    for i in range(ref_features.shape[1]):
        decorrelation_times_gen = []
        for j, gen_feat in enumerate(gener_features):
            max_lag_gen = min(240, gen_feat.shape[0] - 1)
            this_angle = gen_feat[:, i]
            autocorr = acovf(np.sin(this_angle), demean=False, adjusted=True, nlag=max_lag_gen) + \
                       acovf(np.cos(this_angle), demean=False, adjusted=True, nlag=max_lag_gen)
            baseline = np.sin(this_angle).mean()**2 + np.cos(this_angle).mean()**2
            lags = 1 + np.arange(len(autocorr))
            
            if 1 - baseline < 1e-8:
                decorrelation_times_gen.append(-1)
            else:
                autocorr = (autocorr - baseline) / (1 - baseline)
                autocorr = np.array(autocorr, dtype=np.float32)
                decorrelation_times_gen.append(find_decorrelation_time(autocorr, time_step=args.time_step))
                
                if args.plot and j == 0:  # Only plot first generated trajectory to avoid clutter
                    axs[1].plot(lags * args.time_step, autocorr, label=f'Feature {i}')
        
        out['decorrelation_gen_in_ps'][f'feature_{i}'] = decorrelation_times_gen
    
    if args.plot:
        axs[1].set_xlabel('Lag time (ps)')
        axs[1].set_ylabel('Autocorrelation')
        axs[1].set_title('Generated')
        axs[1].set_xscale('log')
        axs[1].set_ylim(-0.1, 1.1)
        axs[1].axhline(y=1/np.e, color='gray', linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(f'{plot_dir}/{name}_backbone_sidechain_autocorrelation.pdf', bbox_inches='tight')
        plt.close()

    ####### TICA on backbone/sidechain features #############
    # Use the cos/sin features we already extracted
    # tica_on_ref expects a list of arrays, so wrap ref_features_cossin in a list
    tica, ref_tica, ref_concatenate = tica_on_ref([ref_features_cossin], lag=args.tica_lag, plot=False)

    out['JSD_TICA_gen'] = {}
    out['JSD_TICA_md'] = {}
    out['W1_TICA_gen'] = {}
    out['W1_TICA_md'] = {}
    out['JSD_TICA_gen_single_traj'] = {}
    out['W1_TICA_gen_single_traj'] = {}
    out['JSD_TICA_gen_single_traj']['TICA-0'] = defaultdict(list)
    out['W1_TICA_gen_single_traj']['TICA-0'] = defaultdict(list)
    out['JSD_TICA_gen_single_traj']['TICA-0,1'] = defaultdict(list)
    
    # Use our backbone/sidechain features for TICA projection
    gen_tica = tica_projection(gener_features_cossin, tica, plot_dir=plot_dir, name=name, plot=args.plot, num_points_to_plot=100)
    
    # MD baseline - commented out for tetrapeptides
    # md_tica = tica_projection(md_torsion_tensor, tica, plot_dir=plot_dir, name=name, plot=False, num_points_to_plot=100)

    # tica_0_min = min(ref_concatenate[:,0].min(), np.concatenate(gen_tica)[:,0].min(), np.concatenate(md_tica)[:,0].min())
    # tica_0_max = max(ref_concatenate[:,0].max(), np.concatenate(gen_tica)[:,0].max(), np.concatenate(md_tica)[:,0].max())
    tica_0_min = min(ref_concatenate[:,0].min(), np.concatenate(gen_tica)[:,0].min())
    tica_0_max = max(ref_concatenate[:,0].max(), np.concatenate(gen_tica)[:,0].max())

    ref_p, _ = np.histogram(ref_concatenate[:,0], range=(tica_0_min, tica_0_max), bins=100)
    ref_p = ref_p / np.sum(ref_p)
    traj_p, _ = np.histogram(np.concatenate(gen_tica)[:,0], range=(tica_0_min, tica_0_max), bins=100)
    traj_p = traj_p / np.sum(traj_p)
    # md_p, _ = np.histogram(np.concatenate(md_tica)[:,0], range=(tica_0_min, tica_0_max), bins=100)
    # md_p = md_p / np.sum(md_p)
    out['JSD_TICA_gen']['TICA-0'] = jensenshannon(ref_p, traj_p)
    # out['JSD_TICA_md']['TICA-0'] = jensenshannon(ref_p, md_p)
    out['W1_TICA_gen']['TICA-0'] = wasserstein_distance(ref_concatenate[:,0], np.concatenate(gen_tica)[:,0])
    # out['W1_TICA_md']['TICA-0'] = wasserstein_distance(ref_concatenate[:,0], np.concatenate(md_tica)[:,0])
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
        # tica_1_min = min(ref_concatenate[:,1].min(), np.concatenate(gen_tica)[:,1].min(), np.concatenate(md_tica)[:,1].min())
        # tica_1_max = max(ref_concatenate[:,1].max(), np.concatenate(gen_tica)[:,1].max(), np.concatenate(md_tica)[:,1].max())
        tica_1_min = min(ref_concatenate[:,1].min(), np.concatenate(gen_tica)[:,1].min())
        tica_1_max = max(ref_concatenate[:,1].max(), np.concatenate(gen_tica)[:,1].max())
        ref_p = np.histogram2d(*ref_concatenate[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]  # 50x50 bins for 2D (matches MDGen)
        ref_p = ref_p / np.sum(ref_p)
        traj_p = np.histogram2d(*np.concatenate(gen_tica)[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]  # 50x50 bins for 2D
        traj_p = traj_p / np.sum(traj_p)
        # md_p = np.histogram2d(*np.concatenate(md_tica)[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]
        # md_p = md_p / np.sum(md_p)
        out['JSD_TICA_gen']['TICA-0,1'] = jensenshannon(ref_p.flatten(), traj_p.flatten())
        # out['JSD_TICA_md']['TICA-0,1'] = jensenshannon(ref_p.flatten(), md_p.flatten())
        for j in range(len(gen_tica)):
            single_gener_p = np.histogram2d(*gen_tica[j][:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=50)[0]  # 50x50 bins for 2D
            if np.sum(single_gener_p) == 0:
                out['JSD_TICA_gen_single_traj']['TICA-0,1'][f'single_trace_{j}'].append(-1)
            else:
                single_gener_p = single_gener_p / np.sum(single_gener_p)
                out['JSD_TICA_gen_single_traj']['TICA-0,1'][f'single_trace_{j}'].append(jensenshannon(ref_p.flatten(), single_gener_p.flatten()))

        #### TICA FES Plots
        if args.plot:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            pyemma.plots.plot_free_energy(*ref_concatenate[:, :2].T, ax=axs[0], cbar=False)
            pyemma.plots.plot_free_energy(*np.concatenate(gen_tica)[:, :2].T, ax=axs[1], cbar=False)
            # pyemma.plots.plot_free_energy(*np.concatenate(md_tica)[:, :2].T, ax=axs[2], cbar=False)
            axs[0].set_title('TICA FES (Ref)')
            axs[1].set_title('TICA FES (Gen)')
            # axs[2].set_title('TICA FES (MD)')

            xlims = [axs[0].get_xlim(), axs[1].get_xlim()]
            ylims = [axs[0].get_ylim(), axs[1].get_ylim()]
            # xlims = [axs[0].get_xlim(), axs[1].get_xlim(), axs[2].get_xlim()]
            # ylims = [axs[0].get_ylim(), axs[1].get_ylim(), axs[2].get_ylim()]
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
        # md_clusters = clusters.transform(md_tica)
        try:
            msm = pyemma.msm.estimate_markov_model(ref_clusters, lag=args.msm_lag)
            nstates = 10
            msm.pcca(nstates)
            ref_p, gen_p = get_metastate_prob(msm, gen_clusters, k, nstates)
            # ref_p, md_p = get_metastate_prob(msm, md_clusters, k, nstates)
            out['ref_metastable_p'] = ref_p
            out['gen_metastable_p'] = gen_p
            # out['md_metastable_p'] = md_p
            out['JSD_msm_gen'] = jensenshannon(ref_p, gen_p)
            # out['JSD_msm_md'] = jensenshannon(ref_p, md_p)
            out['W1_msm_gen'] = wasserstein_distance(np.arange(nstates),np.arange(nstates), ref_p, gen_p)
            # out['W1_msm_md'] = wasserstein_distance(np.arange(nstates),np.arange(nstates), ref_p, md_p)

        except Exception as e:
            print('ERROR', e, smiles, flush=True)

    return smiles, out

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=float, default=10.0)  # 10 ps frame rate for tetrapeptides
parser.add_argument('--ref_dir', type=str, default=os.environ.get('TIMEWARP_DATA_ROOT', '') + '/data/4AA-large/4AA-large-processed/test')
parser.add_argument('--gener_dir', type=str, default='model_outputs_official/timewarp_trajectory_official/timewarp_interpolator_og_full_test_generations.pkl')
parser.add_argument('--save', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save_name', type=str, default='evalout')
parser.add_argument('--no_msm', action='store_true')
parser.add_argument('--tica_lag', type=int, default=10)  # 10 frames × 10 ps = 100 ps lag (matches MDGen)
parser.add_argument('--msm_lag', type=int, default=10)  # 10 frames × 10 ps = 100 ps lag (matches MDGen)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--check_nan', action='store_true')
parser.add_argument('--feat_rmsd', action='store_true')
args = parser.parse_args()

# Load generated trajectories
with open(args.gener_dir, 'rb') as f:
    gen_dict = pickle.load(f)

# Handle tuple keys if present
if type(list(gen_dict.keys())[0]) is tuple:
    new_gen_dict = {k[0]: gen_dict[k] for k in gen_dict}
    gen_dict = new_gen_dict

# Filter out non-peptide keys
gener_smiles_all = [k for k in gen_dict.keys() if k != 'WALL_CLOCK' and isinstance(k, str) and 'C' in k]

# Build SMILES to filename mapping from reference directory
print('Building SMILES to filename mapping from reference directory...')
smiles_to_filename = {}
import glob
ref_files = glob.glob(os.path.join(args.ref_dir, '*.pkl'))
for ref_file in tqdm.tqdm(ref_files, desc='Loading reference metadata'):
    try:
        with open(ref_file, 'rb') as f:
            ref_data = pickle.load(f)
        if 'smiles' in ref_data:
            smiles = ref_data['smiles']
            filename = os.path.basename(ref_file).replace('.pkl', '')
            smiles_to_filename[smiles] = filename
    except Exception as e:
        continue

# Filter to only evaluate peptides that have reference data
gener_smiles = [smi for smi in gener_smiles_all if smi in smiles_to_filename]
print(f'Found {len(gener_smiles)} peptides with both generated and reference data')
print(f'Skipped {len(gener_smiles_all) - len(gener_smiles)} peptides without reference data')

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