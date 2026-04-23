import os, sys
_PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PIPELINE_DIR)

import mdtraj as md
import types

# Monkey‑patch so that PyEMMA can see mdtraj.version.version
md.version = types.SimpleNamespace(version=md.__version__)

import argparse
import pyemma, tqdm, os, pickle
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
from utils import get_torsions_idx_mol, tica_on_ref, tica_projection, kmeans, downsample_traj
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
    ref_mol_paths = [f'{args.ref_dir}/{traj_id}/mol.pkl' for traj_id in ref_traj_ids]
    mol_path = "/".join(md_traj_paths[0].split("/")[:-1])+'/mol.pkl'  # Only get atoms from mol, thus all traj of the same molecule share the same mol info
    mol = pickle.load(open(mol_path, 'rb'))
    # Sanity check that all mol are the same
    # mols = []
    # for mol_path in ref_mol_paths:
    #     mol = pickle.load(open(mol_path, 'rb'))
    #     mols.append(mol)
    # for i in range(len(mols)):
    #     mol_i = mols[i]
    #     for a1, a2 in zip(mol.GetAtoms(), mol_i.GetAtoms()):
    #         if a1.GetAtomicNum() != a2.GetAtomicNum() or a1.GetDegree() != a2.GetDegree() or \
    #         a1.GetFormalCharge() != a2.GetFormalCharge() or a1.GetHybridization() != a2.GetHybridization() or a1.GetIsAromatic() != a2.GetIsAromatic():
    #             raise ValueError(f"Atoms in mol and mol_{i} do not match, smiles: {smiles}")
    #         for neighbor1, neighbor2 in zip(a1.GetNeighbors(), a2.GetNeighbors()):
    #             if neighbor1.GetAtomicNum() != neighbor2.GetAtomicNum() or neighbor1.GetDegree() != neighbor2.GetDegree() or neighbor1.GetIdx() != neighbor2.GetIdx() or \
    #             neighbor1.GetFormalCharge() != neighbor2.GetFormalCharge() or neighbor1.GetHybridization() != neighbor2.GetHybridization() or neighbor1.GetIsAromatic() != neighbor2.GetIsAromatic():
    #                 raise ValueError(f"Atoms in mol and mol_{i} do not match, smiles: {smiles}")

    # Load ref traj
    ref_traj = []
    for i in range(len(ref_pdb_paths)):
        traj = md.load(ref_traj_paths[i], top=ref_pdb_paths[i])
        assert traj.xyz.shape[0] == 12500, f'The ref traj should have shape (12500, N, 3), but it has shape {traj.xyz.shape}'
        downsampled_traj_xyz = downsample_traj(traj.xyz, num_new_traj=30, num_points=500, stepsize=args.time_step)
        assert len(downsampled_traj_xyz) == 30, f'The ref bond angles should have length 30, but it has length {len(downsampled_traj_xyz)}'
        assert downsampled_traj_xyz[0].shape[0] == 500, f'The downsampled ref bond angles should have shape (500, N, 3), but it has shape {downsampled_traj_xyz[0].shape}'
        if np.isnan(downsampled_traj_xyz).any():
            print(f"Nan encountered in ref traj at index {i} for molecule {smiles}")
            raise ValueError(f"Nan encountered in ref traj at index {i} for molecule {smiles}")
        ref_traj += downsampled_traj_xyz
    if len(ref_traj) != 120:
        # Save the smiles to a file for debugging
        with open('ref_smaller_than_four_smiles.txt', 'a') as f:
            f.write(f'{smiles}\n')
        return smiles, {}
    assert len(ref_traj) == 120, f'The ref traj should have length 120, but it has length {len(ref_traj)}, smiles is {smiles}'

    torsion_index = np.array(get_torsions_idx_mol(mol)[0]+get_torsions_idx_mol(mol)[2]).reshape(-1,4)   # 0 is non-ring torsion, 2 is ring but non-aromatic torsion

    # Reload the ref traj and feat_dihedral, because now we need to use sin and cos of torsion instead of the original torsion angle.
    feat_dihedral = pyemma.coordinates.featurizer(ref_pdb_paths[0])
    feat_dihedral.add_dihedrals(torsion_index, cossin=True, periodic=False)
    ref_torsion_tensor = pyemma.coordinates.load(ref_traj_paths, features=feat_dihedral)  # A list of all torsion traj. Each traj should have (N, T)
    downsampled_ref_torsion = []
    for traj in ref_torsion_tensor:
        downsampled_traj = downsample_traj(traj, num_new_traj=30, num_points=500, stepsize=args.time_step)
        downsampled_ref_torsion += downsampled_traj
    assert len(downsampled_ref_torsion) == 120, f'The ref torsion should have length 120, but it has length {len(downsampled_ref_torsion)}'
    assert downsampled_ref_torsion[0].shape[0] == 500, f'The downsampled ref torsion should have shape (500, N), but it has shape {downsampled_ref_torsion[0].shape}'
    ref_torsion_tensor = downsampled_ref_torsion

    # Reload md torsion
    # feat_dihedral = pyemma.coordinates.featurizer(md_pdb_paths[0])
    # feat_dihedral.add_dihedrals(torsion_index, cossin=True, periodic=False)
    # md_torsion_tensor = pyemma.coordinates.load(md_traj_paths, features=feat_dihedral)
    # if not isinstance(md_torsion_tensor, list):
    #     md_torsion_tensor = [md_torsion_tensor]
    # downsampled_md_torsion = []
    # for traj in md_torsion_tensor:
    #     downsampled_traj = downsample_traj(traj, num_new_traj=10, num_points=500, stepsize=args.time_step)
    #     downsampled_md_torsion += downsampled_traj
    # assert len(downsampled_md_torsion) == 10, f'The md torsion should have length 10, but it has length {len(downsampled_md_torsion)}'
    # assert downsampled_md_torsion[0].shape[0] == 500, f'The downsampled md torsion should have shape (500, N), but it has shape {downsampled_md_torsion[0].shape}'
    # md_torsion_tensor = downsampled_md_torsion
    ####### TICA #############
    tica, ref_tica , _ = tica_on_ref(ref_torsion_tensor, lag=args.tica_lag, plot=False)
    # md_tica = tica_projection(md_torsion_tensor, tica, plot_dir=None, name=name, plot=False, num_points_to_plot=100)
    ###### Markov state model stuff #################
    clusters, k = kmeans(tica, k=100)
    ref_clusters = clusters.dtrajs   # A list of array: (500,)
    # md_clusters = clusters.transform(md_tica)
    out['tica'] = tica
    out['ref_tica'] = ref_tica
    out['clusters'] = clusters
    try:
        msm = pyemma.msm.estimate_markov_model(ref_clusters, lag=args.msm_lag)
        out['msm'] = msm
        nstates = 10
        msm.pcca(nstates)
        assert len(msm.metastable_assignments) == 100
        coarse_traj = msm.metastable_assignments[ref_clusters]   # A list of array: (500,)
        coarse_traj = [i for i in coarse_traj]
        assert len(coarse_traj) == 120, f'The coarse traj should have length 120, but it has length {len(coarse_traj)}'
        assert coarse_traj[0].shape == (500,), f'The coarse traj should have shape (500, ), but it has shape {coarse_traj[0].shape}'
        cmsm = pyemma.msm.estimate_markov_model(coarse_traj, lag=args.msm_lag)
        out['cmsm'] = cmsm
    except Exception as e:
        print('ERROR', e, smiles, flush=True)
        return smiles, {}
    # flux_mat = cmsm.transition_matrix * cmsm.pi[None, :]
    # flux_mat[flux_mat < 0.0001] = np.inf  # set 0 flux to inf so we do not choose that as the argmin
    # start_state, end_state = np.unravel_index(np.argmin(flux_mat, axis=None), flux_mat.shape)
    # start_idxs = np.where(coarse_traj == start_state)
    # end_idxs = np.where(coarse_traj == end_state)
    # if (coarse_traj == start_state).sum() == 0 or (coarse_traj == end_state).sum() == 0:
    #     print('No start or end state found for ', name, 'skipping...')
    #     return smiles, out
    # sample_indices = np.random.choice(len(start_idxs[0]), size=1000, replace=True)
    # sampled_start_idxs = np.array([(start_idxs[0][i], start_idxs[1][i]) for i in sample_indices])
    # sample_indices = np.random.choice(len(end_idxs[0]), size=1000, replace=True)
    # sampled_end_idxs = np.array([(end_idxs[0][i], end_idxs[1][i]) for i in sample_indices])

    # start_frames = []
    # end_frames = []
    # for i in range(sampled_start_idxs.shape[0]):
    #     start_idx = sampled_start_idxs[i]
    #     end_idx = sampled_end_idxs[i]
    #     start_frame = ref_traj[start_idx[0]][start_idx[1]] * 10  # (N,3)
    #     end_frame = ref_traj[end_idx[0]][end_idx[1]] * 10  # convert to angstrom
    #     assert start_frame.shape[1] == 3, f'The start frame should have shape (N, 3), but it has shape {start_frame.shape}'
    #     assert end_frame.shape[1] == 3, f'The end frame should have shape (N, 3), but it has shape {end_frame.shape}'
    #     start_frames.append(start_frame)
    #     end_frames.append(end_frame)
    # out['start_frames'] = torch.tensor(np.array(start_frames), dtype=torch.float32)  # (1000, N, 3)
    # out['end_frames'] = torch.tensor(np.array(end_frames), dtype=torch.float32)
    # out['rdmol'] = mol
    return smiles, out

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=float, default=5.2)
parser.add_argument('--ref_dir', type=str, default=os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-DRUGS/4fs_HMR15_5ns_actual/test')
parser.add_argument('--smile_match', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'drugs_test_smile_match.txt'))
parser.add_argument('--gener_smiles', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'drugs_test_smiles.txt'))
parser.add_argument('--save_name', type=str, default='drug_tica_msm_data')
parser.add_argument('--tica_lag', type=int, default=10)
parser.add_argument('--msm_lag', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=32)
args = parser.parse_args()

with open(args.gener_smiles, 'r') as f:
    gener_smiles = [line.strip() for line in f.readlines()]

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

with open(f"{args.save_name}.pkl", 'wb') as f:
    f.write(pickle.dumps(out))