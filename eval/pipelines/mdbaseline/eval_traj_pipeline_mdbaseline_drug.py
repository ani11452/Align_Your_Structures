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

    ref_traj_paths = [f'{args.ref_dir}/{traj_id}/traj.xtc' for traj_id in ref_traj_ids]
    md_traj_paths = [f'{args.ref_dir}/{traj_id}/traj.xtc' for traj_id in md_traj_ids]
    ref_pdb_paths = [f'{args.ref_dir}/{traj_id}/system.pdb' for traj_id in ref_traj_ids]
    md_pdb_paths = [f'{args.ref_dir}/{traj_id}/system.pdb' for traj_id in md_traj_ids]

    mol_path = "/".join(ref_traj_paths[0].split("/")[:-1])+'/mol.pkl'  # Only get atoms from mol, thus all traj of the same molecule share the same mol info
    mol = pickle.load(open(mol_path, 'rb'))
    torsion_index = np.array(get_torsions_idx_mol(mol)[0]+get_torsions_idx_mol(mol)[2]).reshape(-1,4)   # 0 is non-ring torsion, 2 is ring but non-aromatic torsion

    ### Load ref torsion
    feat_dihedral = pyemma.coordinates.featurizer(ref_pdb_paths[0])
    feat_dihedral.add_dihedrals(torsion_index, cossin=False, periodic=False)
    ref_torsion_tensor = pyemma.coordinates.load(ref_traj_paths, features=feat_dihedral)  # A list of all torsion traj. Each traj should have (T, N)

    # Down-sample the ref traj to have step size of 5.2 ps.
    # In each traj, sample 5 down-sampled trajs. Then we have 4 * 30 in total. The new list should have shape (960, N) with length 20.
    downsampled_ref_torsion = []
    for traj in ref_torsion_tensor:
        downsampled_traj = downsample_traj(traj, num_new_traj=5, num_points=960, stepsize=args.time_step)
        downsampled_ref_torsion += downsampled_traj
    assert len(downsampled_ref_torsion) == 20, f'The ref torsion should have length 20, but it has length {len(downsampled_ref_torsion)}, the molecules is {smiles}'
    assert downsampled_ref_torsion[0].shape[0] == 960, f'The downsampled ref torsion should have shape (960, N), but it has shape {downsampled_ref_torsion[0].shape}'
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
        downsampled_traj = downsample_traj(traj, num_new_traj=5, num_points=960, stepsize=args.time_step)
        downsampled_md_torsion += downsampled_traj
    assert len(downsampled_md_torsion) == 5, f'The md torsion should have length 10, but it has length {len(downsampled_md_torsion)}'
    assert downsampled_md_torsion[0].shape[0] == 960, f'The downsampled md torsion should have shape (960, N), but it has shape {downsampled_md_torsion[0].shape}'
    md_torsion_tensor_full = downsampled_md_torsion
    rep_nums = [960, 768, 576, 384, 192, 100, 50]
    rep_names = ['960points', '768points', '576points', '384points', '192points', '100points', '50points']
    for j, rep_num in enumerate(rep_nums):
        md_torsion_tensor = [traj[:rep_num] for traj in md_torsion_tensor_full]
        assert md_torsion_tensor[0].shape[0] == rep_num, f'The downsampled md torsion should have shape ({rep_num}, N), but it has shape {md_torsion_tensor[0].shape}'
        out[f'JSD_torsion_md_{rep_names[j]}'] = {}
        out[f'W1_torsion_md_{rep_names[j]}'] = {}
        for i, feat in enumerate(feat_dihedral.describe()):  # Loop through each torsion angle
            ref_p, _ = np.histogram(np.concatenate(ref_torsion_tensor)[:,i], range=(-np.pi, np.pi), bins=100)
            ref_p = ref_p / np.sum(ref_p)
            md_p, _ = np.histogram(np.concatenate(md_torsion_tensor)[:,i], range=(-np.pi, np.pi), bins=100)
            md_p = md_p / np.sum(md_p)
            out[f'JSD_torsion_md_{rep_names[j]}'][feat] = jensenshannon(ref_p, md_p)
            out[f'W1_torsion_md_{rep_names[j]}'][feat] = wasserstein_distance(np.concatenate(ref_torsion_tensor)[:,i], np.concatenate(md_torsion_tensor)[:,i])
            
    ########### Bond angle eval
    ref_bond_angles = []
    for i in range(len(ref_pdb_paths)):
        traj = md.load(ref_traj_paths[i], top=ref_pdb_paths[i])
        bond_angles = get_bond_angles_in_traj(traj, mol=mol)
        assert bond_angles.shape[0] == 12500, f'The ref bond angles should have shape (12500, N), but it has shape {bond_angles.shape}'
        bond_angles = downsample_traj(bond_angles, num_new_traj=5, num_points=960, stepsize=args.time_step)
        assert len(bond_angles) == 5, f'The ref bond angles should have length 5, but it has length {len(bond_angles)}'
        assert bond_angles[0].shape[0] == 960, f'The downsampled ref bond angles should have shape (960, N), but it has shape {bond_angles[0].shape}'
        if np.isnan(bond_angles).any():
            print(f"Nan encountered in ref traj at index {i} for molecule {smiles}")
            raise ValueError(f"Nan encountered in ref traj at index {i} for molecule {smiles}")
        ref_bond_angles += bond_angles
    ref_bond_angles = np.concatenate(ref_bond_angles, axis=0)
    assert ref_bond_angles.shape[0] == 19200, f'The ref bond angles should have shape (20*960, N), but it has shape {ref_bond_angles.shape}'
    
    md_bond_angles_full = []
    for i in range(len(md_pdb_paths)):
        traj = md.load(md_traj_paths[i], top=md_pdb_paths[i])
        bond_angles = get_bond_angles_in_traj(traj, mol=mol)
        assert bond_angles.shape[0] == 12500, f'The md bond angles should have shape (12500, N), but it has shape {bond_angles.shape}'
        bond_angles = downsample_traj(bond_angles, num_new_traj=5, num_points=960, stepsize=args.time_step)
        assert len(bond_angles) == 5, f'The md bond angles should have length 5, but it has length {len(bond_angles)}'
        assert bond_angles[0].shape[0] == 960, f'The downsampled md bond angles should have shape (960, N), but it has shape {bond_angles[0].shape}'
        if np.isnan(bond_angles).any():
            print(f"Nan encountered in md traj at index {i} for molecule {smiles}")
            raise ValueError(f"Nan encountered in md traj at index {i} for molecule {smiles}")
        md_bond_angles_full += bond_angles
    for j, rep_num in enumerate(rep_nums):
        md_bond_angles = [traj[:rep_num] for traj in md_bond_angles_full]
        md_bond_angles = np.concatenate(md_bond_angles, axis=0)
        assert md_bond_angles.shape[0] == 5*rep_num, f'The md bond angles should have shape (10*960, N), but it has shape {md_bond_angles.shape}'

        out[f'JSD_bond_angle_md_{rep_names[j]}'] = []
        out[f'W1_bond_angle_md_{rep_names[j]}'] = []

        for i in range(ref_bond_angles.shape[1]):
            ref_p, _ = np.histogram(ref_bond_angles[:,i],range=(0,np.pi), bins=100)
            ref_p = ref_p / np.sum(ref_p)
            md_p, _ = np.histogram(md_bond_angles[:,i],range=(0,np.pi), bins=100)
            md_p = md_p / np.sum(md_p)

            out[f'JSD_bond_angle_md_{rep_names[j]}'].append(jensenshannon(ref_p, md_p))
            out[f'W1_bond_angle_md_{rep_names[j]}'].append(wasserstein_distance(ref_bond_angles[:,i], md_bond_angles[:,i]))
            
    ########## Bond length eval
    ref_bond_lengths = []
    for i in range(len(ref_pdb_paths)):
        traj = md.load(ref_traj_paths[i], top=ref_pdb_paths[i])
        bond_lengths = get_bond_lengths_in_traj(traj, mol=mol)
        assert bond_lengths.shape[0] == 12500, f'The ref bond lengths should have shape (12500, N), but it has shape {bond_lengths.shape}'
        bond_lengths = downsample_traj(bond_lengths, num_new_traj=5, num_points=960, stepsize=args.time_step)
        assert len(bond_lengths) == 5, f'The ref bond lengths should have length 5, but it has length {len(bond_lengths)}'
        assert bond_lengths[0].shape[0] == 960, f'The downsampled ref bond lengths should have shape (960, N), but it has shape {bond_lengths[0].shape}'
        ref_bond_lengths += bond_lengths
    ref_bond_lengths = np.concatenate(ref_bond_lengths, axis=0)
    assert ref_bond_lengths.shape[0] == 19200, f'The ref bond lengths should have shape (20*960, N), but it has shape {ref_bond_lengths.shape}'
    assert np.max(ref_bond_lengths) < 0.22, f'The ref bond lengths should be less than 0.22, but it has max {np.max(ref_bond_lengths)}'
    assert np.min(ref_bond_lengths) > 0.1, f'The ref bond lengths should be greater than 0.1, but it has min {np.min(ref_bond_lengths)}'
    md_bond_lengths_full = []
    for i in range(len(md_pdb_paths)):
        traj = md.load(md_traj_paths[i], top=md_pdb_paths[i])
        bond_lengths = get_bond_lengths_in_traj(traj, mol=mol)
        assert bond_lengths.shape[0] == 12500, f'The md bond lengths should have shape (12500, N), but it has shape {bond_lengths.shape}'
        bond_lengths = downsample_traj(bond_lengths, num_new_traj=5, num_points=960, stepsize=args.time_step)
        assert len(bond_lengths) == 5, f'The md bond lengths should have length 10, but it has length {len(bond_lengths)}'
        assert bond_lengths[0].shape[0] == 960, f'The downsampled md bond lengths should have shape (960, N), but it has shape {bond_lengths[0].shape}'
        md_bond_lengths_full += bond_lengths
    for j, rep_num in enumerate(rep_nums):
        md_bond_lengths = [traj[:rep_num] for traj in md_bond_lengths_full]
        md_bond_lengths = np.concatenate(md_bond_lengths, axis=0)
        assert md_bond_lengths.shape[0] == 5*rep_num, f'The md bond lengths should have shape (10*960, N), but it has shape {md_bond_lengths.shape}'
    
        out[f'JSD_bond_length_md_{rep_names[j]}'] = []
        out[f'W1_bond_length_md_{rep_names[j]}'] = []

        for i in range(ref_bond_lengths.shape[1]):
            ref_p, _ = np.histogram(ref_bond_lengths[:,i], range=(0.1,0.22), bins=100)  # The bond length is usually between 100 and 220 pm.
            ref_p = ref_p / np.sum(ref_p)
            md_p, _ = np.histogram(md_bond_lengths[:,i], range=(0.1,0.22), bins=100)
            md_p = md_p / np.sum(md_p)
            
            out[f'JSD_bond_length_md_{rep_names[j]}'].append(jensenshannon(ref_p, md_p))
            out[f'W1_bond_length_md_{rep_names[j]}'].append(wasserstein_distance(ref_bond_lengths[:,i], md_bond_lengths[:,i]))

    ############ Torsion decorrelations
    out['decorrelation_ref_in_ps'] = {}
    for k, rep_num in enumerate(rep_nums):
        out[f'decorrelation_md_in_ps_{rep_names[k]}'] = {}

    for i, feat in enumerate(feat_dihedral.describe()):
        decorrelation_times_ref = []
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
        for k, rep_num in enumerate(rep_nums):
            md_torsion_tensor = [traj[:rep_num] for traj in md_torsion_tensor_full]
            for j in range(len(md_torsion_tensor)):
                this_torsion = md_torsion_tensor[j][:,i]
                autocorr = acovf(np.sin(this_torsion), demean=False, adjusted=True, nlag=rep_num-2) + acovf(np.cos(this_torsion), demean=False, adjusted=True, nlag=rep_num-2)
                baseline = np.sin(this_torsion).mean()**2 + np.cos(this_torsion).mean()**2
                # E[(X(t) - E[X(t)]) * (X(t+dt) - E[X(t+dt)])] = E[X(t)X(t+dt) - E[X(t)]X(t+dt) - X(t)E[X(t+dt)] + E[X(t)]E[X(t+dt)]] = E[X(t)X(t+dt)] - E[X]**2
                lags = 1 + np.arange(len(autocorr))
                if 1-baseline < 1e-8:
                    decorrelation_times_md.append(-1)
                else:
                    autocorr = (autocorr - baseline) / (1-baseline)
                    autocorr = np.array(autocorr, dtype=np.float32)
                    decorrelation_times_md.append(find_decorrelation_time(autocorr, time_step=args.time_step))
            out[f'decorrelation_md_in_ps_{rep_names[k]}'][feat] = decorrelation_times_md

    ####### TICA #############
    # Reload the ref traj and feat_dihedral, because now we need to use sin and cos of torsion instead of the original torsion angle.
    feat_dihedral = pyemma.coordinates.featurizer(ref_pdb_paths[0])
    feat_dihedral.add_dihedrals(torsion_index, cossin=True, periodic=False)
    # print(feat_dihedral.describe())
    ref_torsion_tensor = pyemma.coordinates.load(ref_traj_paths, features=feat_dihedral)  # A list of all torsion traj. Each traj should have (N, T)
    downsampled_ref_torsion = []
    for traj in ref_torsion_tensor:
        downsampled_traj = downsample_traj(traj, num_new_traj=5, num_points=960, stepsize=args.time_step)
        downsampled_ref_torsion += downsampled_traj
    assert len(downsampled_ref_torsion) == 20, f'The ref torsion should have length 20, but it has length {len(downsampled_ref_torsion)}'
    assert downsampled_ref_torsion[0].shape[0] == 960, f'The downsampled ref torsion should have shape (960, N), but it has shape {downsampled_ref_torsion[0].shape}'
    ref_torsion_tensor = downsampled_ref_torsion

    # Reload md torsion
    feat_dihedral = pyemma.coordinates.featurizer(md_pdb_paths[0])
    feat_dihedral.add_dihedrals(torsion_index, cossin=True, periodic=False)
    md_torsion_tensor = pyemma.coordinates.load(md_traj_paths, features=feat_dihedral)
    if not isinstance(md_torsion_tensor, list):
        md_torsion_tensor = [md_torsion_tensor]
    downsampled_md_torsion = []
    for traj in md_torsion_tensor:
        downsampled_traj = downsample_traj(traj, num_new_traj=5, num_points=960, stepsize=args.time_step)
        downsampled_md_torsion += downsampled_traj
    assert len(downsampled_md_torsion) == 5, f'The md torsion should have length 10, but it has length {len(downsampled_md_torsion)}'
    assert downsampled_md_torsion[0].shape[0] == 960, f'The downsampled md torsion should have shape (960, N), but it has shape {downsampled_md_torsion[0].shape}'
    md_torsion_tensor_full = downsampled_md_torsion

    tica, ref_tica, ref_concatenate = tica_on_ref(ref_torsion_tensor, lag=args.tica_lag, plot=False)
    clusters, k = kmeans(tica, k=100)
    ref_clusters = clusters.dtrajs
    for j, rep_num in enumerate(rep_nums):
        md_torsion_tensor = [traj[:rep_num] for traj in md_torsion_tensor_full]
        assert md_torsion_tensor[0].shape[0] == rep_num, f'The downsampled md torsion should have shape ({rep_num}, N), but it has shape {md_torsion_tensor[0].shape}'
        out[f'JSD_TICA_md_{rep_names[j]}'] = {}
        out[f'W1_TICA_md_{rep_names[j]}'] = {}

        md_tica = tica_projection(md_torsion_tensor, tica, plot_dir=None, name=name, plot=False, num_points_to_plot=100)
        tica_0_min = min(ref_concatenate[:,0].min(), np.concatenate(md_tica)[:,0].min())
        tica_0_max = max(ref_concatenate[:,0].max(), np.concatenate(md_tica)[:,0].max())

        ref_p, _ = np.histogram(ref_concatenate[:,0], range=(tica_0_min, tica_0_max), bins=100)
        ref_p = ref_p / np.sum(ref_p)

        md_p, _ = np.histogram(np.concatenate(md_tica)[:,0], range=(tica_0_min, tica_0_max), bins=100)
        md_p = md_p / np.sum(md_p)

        out[f'JSD_TICA_md_{rep_names[j]}']['TICA-0'] = jensenshannon(ref_p, md_p)
        out[f'W1_TICA_md_{rep_names[j]}']['TICA-0'] = wasserstein_distance(ref_concatenate[:,0], np.concatenate(md_tica)[:,0])
        
        if ref_tica[0].shape[1] > 2:  # has more than 1 tica component
            tica_1_min = min(ref_concatenate[:,1].min(), np.concatenate(md_tica)[:,1].min())
            tica_1_max = max(ref_concatenate[:,1].max(), np.concatenate(md_tica)[:,1].max())
            ref_p = np.histogram2d(*ref_concatenate[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=100)[0]
            ref_p = ref_p / np.sum(ref_p)
            md_p = np.histogram2d(*np.concatenate(md_tica)[:,:2].T, range=((tica_0_min, tica_0_max),(tica_1_min, tica_1_max)), bins=100)[0]
            md_p = md_p / np.sum(md_p)
            out[f'JSD_TICA_md_{rep_names[j]}']['TICA-0,1'] = jensenshannon(ref_p.flatten(), md_p.flatten())

        md_clusters = clusters.transform(md_tica)
        try:
            msm = pyemma.msm.estimate_markov_model(ref_clusters, lag=args.msm_lag)
            nstates = 10
            msm.pcca(nstates)
            ref_p, md_p = get_metastate_prob(msm, md_clusters, k, nstates)
            out[f'JSD_msm_md_{rep_names[j]}'] = jensenshannon(ref_p, md_p)
            out[f'W1_msm_md_{rep_names[j]}'] = wasserstein_distance(np.arange(nstates),np.arange(nstates), ref_p, md_p)
        except Exception as e:
            print('ERROR', e, smiles, flush=True)

    return smiles, out

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=float, default=5.2)
parser.add_argument('--ref_dir', type=str, default=os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-DRUGS/4fs_HMR15_5ns_actual/test')
parser.add_argument('--gener_dir', type=str, default='model_outputs_official/drugs_trajectory_official/drugs_noH_1000_kabsch_traj_interpolator_pretrain_fs_25/epoch=199-step=35600-gen.pkl')
parser.add_argument('--smile_match', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'drugs_test_smile_match.txt'))
parser.add_argument('--gener_smiles', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'drugs_test_smiles.txt'))
parser.add_argument('--save', action='store_true')
parser.add_argument('--save_name', type=str, default='mdbaseline_drugs')
parser.add_argument('--no_msm', action='store_true')
parser.add_argument('--truncate', type=int, default=None)
parser.add_argument('--tica_lag', type=int, default=10)
parser.add_argument('--msm_lag', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=1)

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