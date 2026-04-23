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
from scipy.stats import wasserstein_distance
from utils import get_torsions_idx_mol, tica_on_ref, downsample_traj, sample_tp, get_tp_likelihood
import mdtraj as md
import torch
from rdkit import Chem
from utils import get_torsions_in_gen, get_torsion_index_noH
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def main(smiles):
    # print(f'\nThe evaluated molecules is {smiles}')
    out = {}
    ref_traj_ids = []
    md_traj_ids = []
    gener_traj_ids = []
    with open(args.smile_match, 'r') as f:    # Get the full name of ref trajectory from smile_match.txt file
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
    if name != "Cc1ccc_NC_O_CSC2_NN3CCCC_O_N_C3S2_c_C_c1_11" and name != 'Cn1c_C_O_NCCN2CCOCC2_cc2c_O_n_C_c3ccccc3c21_28' and name != 'O_C_CCCSc1nc2ccccc2_nH_1_NCc1ccccc1F_11':
        return smiles, out
    
    ### Read pickle file from gener_dir.
    with open(args.gener_dir, 'rb') as f:
        gener_data = pickle.load(f)

        # if the keys are 2‑tuples, reduce to just the SMILES string
        if isinstance(next(iter(gener_data)), tuple):
            gener_data = {k[0]: gener_data[k] for k in gener_data}

        gen_mol = gener_data[smiles]['rdmol']  # It already has Hs removed.
        gen_mol = Chem.RemoveHs(gen_mol)  # Double check to remove Hs from the molecule
        gener_traj_coords = gener_data[smiles]['coords']  # (900, 9, 3, 101), 900 trajs of 101 length.
    assert gener_traj_coords[0].shape[2] == 101, f'The generated coords should have shape (N, 3, 101), but it has shape {gener_traj_coords[0].shape}'
    assert len(gener_traj_coords) == 900, f'The generated coords should have length 900, but it has length {len(gener_traj_coords)}'
    for i in range(len(gener_traj_coords)):
        # check if any value from this traj is not nan.
        if np.isnan(gener_traj_coords[i]).any():
            # raise ValueError(f"Nan encountered in ref traj at index {i} for molecule {smiles}")
            return smiles, {}

    ref_traj_paths = [f'{args.ref_dir}/{traj_id}/traj.xtc' for traj_id in ref_traj_ids]
    md_traj_paths = [f'{args.ref_dir}/{traj_id}/traj.xtc' for traj_id in md_traj_ids]
    ref_pdb_paths = [f'{args.ref_dir}/{traj_id}/system.pdb' for traj_id in ref_traj_ids]
    md_pdb_paths = [f'{args.ref_dir}/{traj_id}/system.pdb' for traj_id in md_traj_ids]

    mol_path = "/".join(ref_traj_paths[0].split("/")[:-1])+'/mol.pkl'  # Only get atoms from mol, thus all traj of the same molecule share the same mol info
    mol = pickle.load(open(mol_path, 'rb'))
    torsion_index = np.array(get_torsions_idx_mol(mol)[0]+get_torsions_idx_mol(mol)[2]).reshape(-1,4)   # 0 is non-ring torsion, 2 is ring but non-aromatic torsion
   
    torsion_index_noH = get_torsion_index_noH(torsion_index, mol)  # used for gener mol without Hs.
    ### Check whether torsion_index has the same atoms as torsion_index_noH
    for i in range(len(torsion_index_noH)):
        torsion_noH = torsion_index_noH[i]
        torsion = torsion_index[i]
        for idx_noH, idx in zip(torsion_noH, torsion):  # Get atom info from mol and gen_mol and compare
            atom_noH = gen_mol.GetAtomWithIdx(int(idx_noH))
            atom = mol.GetAtomWithIdx(int(idx))
            assert atom.GetSymbol() == atom_noH.GetSymbol(), f"Atom mismatch: {atom.GetSymbol()} vs {atom_noH.GetSymbol()} for molecule {smiles}"

    ### Load gen traj
    gener_traj_tensor = []
    for i in range(len(gener_traj_coords)):
        gener_traj_tensor.append(torch.permute(torch.from_numpy(gener_traj_coords[i] / 10), (2,0,1)))      # /10 convert to nm
    # assert len(gener_traj_tensor) == 10, f'The gener traj should have length 10, but it has length {len(gener_traj_tensor)}'
    assert gener_traj_tensor[0].shape[0] == 101, f'The gener traj should have shape (101, N, 3), but it has shape {gener_traj_tensor[0].shape}'

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
    md_torsion_tensor = downsampled_md_torsion

    # Reload gener torsion
    gener_torsion_tensor = []    # (num_traj, 101, 2N)
    for i in range(len(gener_traj_tensor)):
        torsions = get_torsions_in_gen(gener_traj_tensor[i], torsion_index_noH)  # (101,N)
        torsions_2dim = []
        for j in range(torsions.shape[1]):
            torsions_cossin = np.concatenate([np.cos(torsions[:,j])[:,None], np.sin(torsions[:,j])[:,None]], axis=1)
            assert torsions_cossin.shape == (101, 2), f'The torsion should have shape (960, 2), but it has shape {torsions_cossin.shape}'
            torsions_2dim.append(torsions_cossin)
        torsions_2dim = np.concatenate(torsions_2dim,axis=1)
        assert torsions_2dim.shape[0] == 101, f'The torsion should have shape (101, 2N), but it has shape {torsions_2dim.shape}'
        assert torsions_2dim.shape[1] % 2 == 0, f'The torsion should have shape (101, 2N), but it has shape {torsions_2dim.shape}'
        gener_torsion_tensor.append(torsions_2dim)
    
    # tica, ref_tica, ref_concatenate = tica_on_ref(ref_torsion_tensor, lag=args.tica_lag, plot=False)
    if 'tica' not in ref_data[smiles].keys():
        return smiles, out
    tica = ref_data[smiles]['tica']
    ref_tica = tica.transform(ref_torsion_tensor)
    tica_concatenated = np.concatenate(ref_tica)  # (5*960, 2N)
    gen_tica = tica.transform(gener_torsion_tensor)
    md_tica = tica.transform(md_torsion_tensor)
 
    ###### Markov state model stuff #################

    clusters = ref_data[smiles]['clusters']
    k = 100
    ref_clusters = clusters.transform(ref_tica)  # A list of array: (500,)
    ref_clusters = [np.squeeze(i) for i in ref_clusters]
    try:
        gen_clusters = clusters.transform(gen_tica)
        md_clusters = clusters.transform(md_tica)
    except Exception as e:
        print('ERROR', e, smiles, flush=True)
        return smiles, out
    if 'cmsm' not in ref_data[smiles].keys():
        return smiles, out
    msm = ref_data[smiles]['msm']
    nstates = 10
    msm.pcca(nstates)
    cmsm = ref_data[smiles]['cmsm']
    
    ref_coarse_traj = msm.metastable_assignments[ref_clusters]   # A list of array: (960,)
    ref_coarse_traj = [i for i in ref_coarse_traj]
    gen_coarse_traj = msm.metastable_assignments[gen_clusters]   # A list of array: (101,)
    gen_coarse_traj = [i for i in gen_coarse_traj]

    if name == "Cc1ccc_NC_O_CSC2_NN3CCCC_O_N_C3S2_c_C_c1_11":
        traj_to_plot = 1
    elif name == 'Cn1c_C_O_NCCN2CCOCC2_cc2c_O_n_C_c3ccccc3c21_28':
        traj_to_plot = 8
    elif name == 'O_C_CCCSc1nc2ccccc2_nH_1_NCc1ccccc1F_11':
        traj_to_plot = 6
    if tica_concatenated.shape[1] >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        pyemma.plots.plot_free_energy(
                *tica_concatenated[:, :2].T, ax=axes[0], legacy=False, cbar=True)

        _, _,misc = pyemma.plots.plot_state_map(
            *tica_concatenated[:, :2].T, np.concatenate(ref_coarse_traj), ax=axes[1])
        misc['cbar'].set_ticklabels([f'S{j+1}' for j in range(nstates)])

        # Plot the traj projection on the FES
        axes[1].scatter(gen_tica[traj_to_plot][:, 0], 
                gen_tica[traj_to_plot][:, 1], 
                color='black', label='Points', alpha=0.5, s=10) 
        axes[1].scatter(gen_tica[traj_to_plot][0, 0],
                    gen_tica[traj_to_plot][0, 1], color='red', label='Start', alpha=1, s=25)
        axes[1].scatter(gen_tica[traj_to_plot][-1, 0],
                    gen_tica[traj_to_plot][-1, 1], color='orange', label='End', alpha=1, s=25)
        axes[1].plot(gen_tica[traj_to_plot][:, 0],
                    gen_tica[traj_to_plot][:, 1], color='black', label='Connections', alpha=0.5)
        
        axes[0].set_title('FES (Reference)')
        axes[1].set_title('Generated interpolation trajectory')
        xlims = [axes[0].get_xlim(), axes[1].get_xlim()]
        ylims = [axes[0].get_ylim(), axes[1].get_ylim()]
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
        
        if name == "O_C_CCCSc1nc2ccccc2_nH_1_NCc1ccccc1F_11":
            frames = []
            assert np.array(gener_traj_tensor[traj_to_plot]).shape[0] == 101 # (101, N, 3)
            assert np.array(gener_traj_tensor[traj_to_plot]).shape[2] == 3 # (101, 9, 3)
            # print(np.array(gen_coarse_traj[traj_to_plot]).flatten())
            print('=======================')
            frames_to_save = [11,12,14,19,24,27]
            for i in frames_to_save:
                frames.append(np.array(gener_traj_tensor[traj_to_plot])[i, :, :])
            # save frames as npy
            frames = np.array(frames)

        
        # fig, axes = plt.subplots(1, 10, figsize=(25, 3))
        # pyemma.plots.plot_free_energy(
        #         *tica_concatenated[:, :2].T, ax=axes[0], legacy=False, cbar=False)
        # for i in range(1, 10):
        #     pyemma.plots.plot_free_energy(
        #         *tica_concatenated[:, :2].T, ax=axes[i])
        #     _, _, _ = pyemma.plots.plot_state_map(
        #         *tica_concatenated[:, :2].T, np.concatenate(ref_coarse_traj), ax=axes[i])
        #     axes[i].set_xlabel('TICA 1')
        #     axes[i].set_ylabel('TICA 2')

        #     # Plot the traj projection on the FES
        #     axes[i].scatter(gen_tica[i][:, 0], 
        #             gen_tica[i][:, 1], 
        #             color='black', label='Points', alpha=0.5, s=10) 
        #     axes[i].scatter(gen_tica[i][0, 0],
        #                 gen_tica[i][0, 1], color='red', label='Start', alpha=1, s=25)
        #     axes[i].scatter(gen_tica[i][-1, 0],
        #                 gen_tica[i][-1, 1], color='orange', label='End', alpha=1, s=25)
        #     axes[i].plot(gen_tica[i][:, 0],
        #                 gen_tica[i][:, 1], color='black', label='Connections', alpha=0.5)
            
        #     axes[0].set_title('FES (Reference)')
        #     axes[1].set_title('Projection of generated trajectory on reference FES')
        # xlims = [axes[0].get_xlim(), axes[1].get_xlim()]
        # ylims = [axes[0].get_ylim(), axes[1].get_ylim()]
        # x_min = min(x[0] for x in xlims)
        # x_max = max(x[1] for x in xlims)
        # y_min = min(y[0] for y in ylims)
        # y_max = max(y[1] for y in ylims)
        # for ax in axes:
        #     ax.set_xlim(x_min, x_max)
        #     ax.set_ylim(y_min, y_max)
        # for ax in axes:
        #     ax.set_xlabel('TICA 0')
        #     ax.set_ylabel('TICA 1')
        #     # no ticks
        #     ax.set_xticks([])
        #     ax.set_yticks([])
            
        # fig.savefig(f'{plot_dir}/{name}_tica_projection.png', bbox_inches='tight')
        # plt.close()

    assert len(ref_coarse_traj) == 20, f'The coarse traj should have length 20, but it has length {len(ref_coarse_traj)}'
    assert ref_coarse_traj[0].shape == (960,), f'The coarse traj should have shape (960, ), but it has shape {ref_coarse_traj[0].shape}'
    highest_prob_state = cmsm.active_set[np.argmax(cmsm.pi)]
    allidx_to_activeidx = {value: idx for idx, value in enumerate(cmsm.active_set)}
    flux_mat = cmsm.transition_matrix * cmsm.pi[None, :]
    flux_mat[flux_mat < 0.0001] = np.inf  # set 0 flux to inf so we do not choose that as the argmin
    start_state, end_state = np.unravel_index(np.argmin(flux_mat, axis=None), flux_mat.shape)
    start_idxs = np.where(ref_coarse_traj == start_state)
    end_idxs = np.where(ref_coarse_traj == end_state)
    if (ref_coarse_traj == start_state).sum() == 0 or (ref_coarse_traj == end_state).sum() == 0:
        print('No start or end state found for ', name, 'skipping...')
        return smiles, out
    # sample_indices = np.random.choice(len(start_idxs[0]), size=900, replace=True)
    # sampled_start_idxs = np.array([(start_idxs[0][i], start_idxs[1][i]) for i in sample_indices])
    # sample_indices = np.random.choice(len(end_idxs[0]), size=900, replace=True)
    # sampled_end_idxs = np.array([(end_idxs[0][i], end_idxs[1][i]) for i in sample_indices])

    ref_tp = sample_tp(trans=cmsm.transition_matrix, start_state=start_state,
                                            end_state=end_state, traj_len=1+100//args.msm_lag, n_samples=900)
    ref_stateprobs = np.bincount(ref_tp.reshape(-1), minlength=10)
    ref_stateprobs = ref_stateprobs / ref_stateprobs.sum()
    ### Gen analysis
    gen_stateprobs = np.bincount(np.array(gen_coarse_traj).flatten(), minlength=10)
    gen_stateprobs = gen_stateprobs / gen_stateprobs.sum()
    out['jsd_msm_gen'] = jensenshannon(ref_stateprobs, gen_stateprobs)
    gen_tp_all = np.squeeze(np.array(gen_coarse_traj), axis=-1)
    assert gen_tp_all.shape == (900,101), f'The gen tp_all should have shape (900, 101), but it has shape {gen_tp_all.shape}'
    gen_tp = gen_tp_all[:,::args.msm_lag]
    assert np.all(gen_tp[:, -1] == gen_tp_all[:, -1]), f'The last frame of gen tp should be the same as gen tp_all, but they are different'
    assert gen_tp.shape == ref_tp.shape, f'The gen tp should have shape {ref_tp.shape}, but it has shape {gen_tp.shape}'
    assert gen_tp.shape[0] == 900, f'The gen tp should have length 900, but it has length {gen_tp.shape[0]}'
    gen_probs = get_tp_likelihood(np.vectorize(allidx_to_activeidx.get)(gen_tp, highest_prob_state),
                                                        cmsm.transition_matrix)
    gen_prob = gen_probs.prod(-1)
    out['gen_prob'] = np.mean(gen_prob)
    out[f'gen_valid_prob'] = gen_prob[gen_prob > 0].mean()
    out[f'gen_valid_rate'] = (gen_prob > 0).mean()
    ### replicate MD analysis
    rep_stateprobs_list = []
    rep_nums = [960, 768, 576, 384, 192, 100, 50]
    rep_names = ['960points', '768points', '576points', '384points', '192points', '100points', '50points']
    for i in range(len(rep_nums)):
        try:
            md_clusters_small = [np.squeeze(j[:rep_nums[i]], axis=-1) for j in md_clusters]
            coarse_traj_md = msm.metastable_assignments[md_clusters_small]   # A list of array: (960,)
            coarse_traj_md = [i for i in coarse_traj_md]
            md_msm = pyemma.msm.estimate_markov_model(coarse_traj_md, lag=args.msm_lag)

            idx_to_repidx = {value: idx for idx, value in enumerate(md_msm.active_set)}
            repidx_to_idx = {idx: value for idx, value in enumerate(md_msm.active_set)}
            if (start_state not in idx_to_repidx.keys()) or (end_state not in idx_to_repidx.keys()):
                out[f'{rep_names[i]}_rep_prob'] = 0      
                out[f'{rep_names[i]}_rep_valid_prob'] = 0
                out[f'{rep_names[i]}_rep_valid_rate'] = 0
                out[f'{rep_names[i]}_rep_JSD'] = 1
                continue

            repidx_start_state = idx_to_repidx[start_state]
            repidx_end_state = idx_to_repidx[end_state]

            repidx_tp = sample_tp(trans=md_msm.transition_matrix, start_state=repidx_start_state,
                                                    end_state=repidx_end_state, traj_len=1+100//args.msm_lag, n_samples=900)
            rep_tp = np.vectorize(repidx_to_idx.get)(repidx_tp)
            assert rep_tp[0, 0] == start_state
            assert rep_tp[0, -1] == end_state
            assert rep_tp.shape[0] == 900, f'The rep tp should have shape (900, 26 or 6), but it has shape {rep_tp.shape}'
            assert rep_tp.shape == gen_tp.shape, f'The rep tp should be the same as gen_tp {gen_tp.shape}, but it has shape {rep_tp.shape}'
            rep_probs = get_tp_likelihood(np.vectorize(allidx_to_activeidx.get)(rep_tp, highest_prob_state),
                                                            cmsm.transition_matrix)
            rep_prob = rep_probs.prod(-1)
            rep_stateprobs = np.bincount(rep_tp.reshape(-1), minlength=10)
            rep_stateprobs = rep_stateprobs / rep_stateprobs.sum()
            rep_stateprobs_list.append(rep_stateprobs)
            out[f'{rep_names[i]}_rep_prob'] = rep_prob.mean()
            out[f'{rep_names[i]}_rep_valid_prob'] = rep_prob[rep_prob > 0].mean()
            out[f'{rep_names[i]}_rep_valid_rate'] = (rep_prob > 0).mean()
            out[f'{rep_names[i]}_rep_JSD'] = jensenshannon(ref_stateprobs, rep_stateprobs)
            out[f'{rep_names[i]}_rep_prob'] = np.mean(rep_prob)
        except Exception as e:
            print('ERROR', e, smiles, flush=True)
            out[f'{rep_names[i]}_rep_prob'] = 0
            out[f'{rep_names[i]}_rep_valid_prob'] = 0
            out[f'{rep_names[i]}_rep_valid_rate'] = 0
            out[f'{rep_names[i]}_rep_JSD'] = 1
            continue


    return smiles, out

parser = argparse.ArgumentParser()
parser.add_argument('--time_step', type=float, default=5.2)
parser.add_argument('--ref_dir', type=str, default=os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-DRUGS/4fs_HMR15_5ns_actual/test')
parser.add_argument('--gener_dir', type=str, default='model_outputs_official/drugs_interpolation/drugs_noH_1000_kabsch_traj_interpolator_pretrain_inter/epoch=399-step=71200-gen.pkl')
parser.add_argument('--ref_saved_data', type=str, required=True,
                    help='Path to pre-saved reference TICA/MSM cache pickle.')
# drug saved data: drug_tica_msm_data.pkl
parser.add_argument('--smile_match', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'drugs_test_smile_match.txt'))
parser.add_argument('--gener_smiles', type=str, default=os.path.join(_PIPELINE_DIR, 'smiles', 'drugs_test_smiles.txt'))
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save_name', type=str, default='drug_interpolation_evalout')
parser.add_argument('--tica_lag', type=int, default=10)
parser.add_argument('--msm_lag', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=1)

args = parser.parse_args()

with open(args.gener_smiles, 'r') as f:
    gener_smiles_tot = [line.strip() for line in f.readlines()]

with open(args.gener_dir, 'rb') as f:
    gen_dict = pickle.load(f)

with open(args.ref_saved_data, 'rb') as f:
    ref_data = pickle.load(f)

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

with open(f"{args.save_name}_evalout.pkl", 'wb') as f:
    f.write(pickle.dumps(out))