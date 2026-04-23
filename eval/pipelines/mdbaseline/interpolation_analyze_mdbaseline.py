import os, sys
_PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PIPELINE_DIR)

import mdtraj as md
import types

# Monkey‑patch so that PyEMMA can see mdtraj.version.version
# md.version = types.SimpleNamespace(version=md.__version__)

import argparse
import pyemma, tqdm, os, pickle
from multiprocessing import Pool
from scipy.spatial.distance import jensenshannon
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
from utils import get_ref_and_generated_traj, get_torsions_idx_mol, find_decorrelation_time, tica_on_ref, tica_projection, kmeans, downsample_traj, sample_tp, get_tp_likelihood
import mdtraj as md
import torch
from rdkit import Chem
from collections import defaultdict
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
    
    ref_traj_paths = [f'{args.ref_dir}/{traj_id}/traj.xtc' for traj_id in ref_traj_ids]
    md_traj_paths = [f'{args.ref_dir}/{traj_id}/traj.xtc' for traj_id in md_traj_ids]
    ref_pdb_paths = [f'{args.ref_dir}/{traj_id}/system.pdb' for traj_id in ref_traj_ids]
    md_pdb_paths = [f'{args.ref_dir}/{traj_id}/system.pdb' for traj_id in md_traj_ids]
    mol_path = "/".join(ref_traj_paths[0].split("/")[:-1])+'/mol.pkl'  # Only get atoms from mol, thus all traj of the same molecule share the same mol info
    mol = pickle.load(open(mol_path, 'rb'))
    # out['rdmol'] = mol
    
    torsion_index = np.array(get_torsions_idx_mol(mol)[0]+get_torsions_idx_mol(mol)[2]).reshape(-1,4)   # 0 is non-ring torsion, 2 is ring but non-aromatic torsion

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
    ####### TICA #############
    tica = ref_data[smiles]['tica']
    # ref_tica = tica.get_output()   # Cannot call get_output() when load tica from the .pkl file
    md_tica = tica.transform(md_torsion_tensor)

    ###### Markov state model stuff #################
    clusters = ref_data[smiles]['clusters']
    ref_clusters = clusters.dtrajs   # A list of array: (500,)
    md_clusters = clusters.transform(md_tica)
    if 'cmsm' not in ref_data[smiles].keys():
        return smiles, {}
    msm = ref_data[smiles]['msm']
    nstates = 10
    msm.pcca(nstates)
    assert len(msm.metastable_assignments) == 100
    coarse_traj = msm.metastable_assignments[ref_clusters]   # A list of array: (500,)
    coarse_traj = [i for i in coarse_traj]
    assert len(coarse_traj) == 120, f'The coarse traj should have length 100, but it has length {len(coarse_traj)}'
    assert coarse_traj[0].shape == (500,), f'The coarse traj should have shape (500, ), but it has shape {coarse_traj[0].shape}'
    cmsm = ref_data[smiles]['cmsm']
    flux_mat = cmsm.transition_matrix * cmsm.pi[None, :]
    highest_prob_state = cmsm.active_set[np.argmax(cmsm.pi)]
    allidx_to_activeidx = {value: idx for idx, value in enumerate(cmsm.active_set)}
        
    for limit in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        flux_mat[flux_mat < limit] = np.inf  # set 0 flux to inf so we do not choose that as the argmin
        start_state, end_state = np.unravel_index(np.argmin(flux_mat, axis=None), flux_mat.shape)
        # ref_tp = sample_tp(trans=cmsm.transition_matrix, start_state=start_state,
        #                                         end_state=end_state, traj_len=1+100//args.msm_lag, n_samples=1000)
        # ref_stateprobs = np.bincount(ref_tp.reshape(-1), minlength=10)
        # ref_stateprobs = ref_stateprobs / ref_stateprobs.sum()
        out[limit] = {}
        rep_nums = [500, 400, 300, 200, 100, 80, 50]
        rep_names = ['500points', '400points', '300points', '200points', '100points', '80points', '50points']
        for i in range(len(rep_nums)):
            try:
                md_clusters_small = [np.squeeze(j[:rep_nums[i]], axis=-1) for j in md_clusters]
                coarse_traj_md = msm.metastable_assignments[md_clusters_small]   # A list of array: (500,)
                coarse_traj_md = [i for i in coarse_traj_md]
                md_msm = pyemma.msm.estimate_markov_model(coarse_traj_md, lag=args.msm_lag)

                idx_to_repidx = {value: idx for idx, value in enumerate(md_msm.active_set)}
                repidx_to_idx = {idx: value for idx, value in enumerate(md_msm.active_set)}
                if (start_state not in idx_to_repidx.keys()) or (end_state not in idx_to_repidx.keys()):
                    out[limit][f'{rep_names[i]}_rep_prob'] = 0      
                    continue

                repidx_start_state = idx_to_repidx[start_state]
                repidx_end_state = idx_to_repidx[end_state]

                repidx_tp = sample_tp(trans=md_msm.transition_matrix, start_state=repidx_start_state,
                                                        end_state=repidx_end_state, traj_len=1+100//args.msm_lag, n_samples=1000)
                rep_tp = np.vectorize(repidx_to_idx.get)(repidx_tp)
                assert rep_tp[0, 0] == start_state
                assert rep_tp[0, -1] == end_state

                rep_probs = get_tp_likelihood(np.vectorize(allidx_to_activeidx.get)(rep_tp, highest_prob_state),
                                                                cmsm.transition_matrix)
                rep_prob = rep_probs.prod(-1)
                out[limit][f'{rep_names[i]}_rep_prob'] = np.mean(rep_prob)
            except Exception as e:
                print('ERROR', e, smiles, flush=True)
                out[limit][f'{rep_names[i]}_rep_prob'] = 0
                continue

    return smiles, out

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=float, default=5.2)
parser.add_argument('--ref_dir', type=str, default=os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/test')
parser.add_argument('--ref_saved_data', type=str, required=True,
                    help='Path to pre-saved reference TICA/MSM cache pickle.')
parser.add_argument('--smile_match', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'qm9_test_smile_match.txt'))
parser.add_argument('--gener_smiles', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'eval_qm9_unconditional_generation_smiles.txt'))
parser.add_argument('--save', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save_name', type=str, default='evalout')
parser.add_argument('--truncate', type=int, default=None)
parser.add_argument('--tica_lag', type=int, default=2)
parser.add_argument('--msm_lag', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()

with open(args.gener_smiles, 'r') as f:
    gener_smiles = [line.strip() for line in f.readlines()]

with open(args.ref_saved_data, 'rb') as f:
    ref_data = pickle.load(f)

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
    with open(f"{args.save_name}.pkl", 'wb') as f:
        f.write(pickle.dumps(out))