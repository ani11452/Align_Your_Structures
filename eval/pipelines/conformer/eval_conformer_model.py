import os, sys
_PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PIPELINE_DIR)

import pickle
import argparse
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import pyemma.plots
import pyemma
from utils import get_torsions_in_gen, get_torsions_idx_mol, get_bond_lengths_in_gen, get_bond_lengths_in_conformer_ref, get_torsion_index_noH, get_bond_angles_in_gen
import torch
import tqdm, pickle
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import urllib.parse
from multiprocessing import Pool
import os

# Replace 'your_file.pkl' with the path to your pickle file
pkl_file_path = 'model_outputs_official/qm9_conformer_official_fixed/qm9_noH_1000_kabsch_conf_basic_es_order_3/epoch=239-step=353760-gen.pkl'
pkl_file_path_539 = 'model_outputs_official/qm9_conformer_official_fixed/qm9_noH_1000_kabsch_conf_basic_es_order_3/epoch=539-step=795960-gen.pkl'
pkl_file_path = 'model_outputs_official/qm9_conformer_official_fixed/qm9_noH_1000_kabsch_conf_basic_es_order_3/epoch=99-step=147400-gen.pkl'
drug_pkl_file_path = 'model_outputs_official/drugs_conformer_official_fixed/drugs_noH_1000_kabsch_conf_basic_es_order_3/epoch=539-step=801900-gen.pkl'
drug_pkl_file_path = 'model_outputs_official/drugs_conformer_official_fixed/drugs_noH_1000_kabsch_conf_basic_es_order_3/epoch=339-step=504900-gen.pkl'

ref_path = os.environ.get('MD_DATA_ROOT', '') + '/processed_input_data/GEOM-QM9/GEOM-QM9_Test_Actual_compat.pkl'
drug_ref_path = os.environ.get('MD_DATA_ROOT', '') + '/processed_input_data/GEOM-DRUGS/GEOM-DRUGS_Test_Actual_compat.pkl'
# Argument parsing
parser = argparse.ArgumentParser(description="Read and process a pickle file.")
parser.add_argument('--use_drug', action='store_true', help="Use the drug pickle file path instead of qm9.")
parser.add_argument('--num_workers', type=int, default=1, help="Number of workers for multiprocessing.")
parser.add_argument('--plot', action='store_true', help="Whether to plot the data.")
parser.add_argument('--save_name', type=str, default='evalout.pkl', help="Name of the output file.")
parser.add_argument('--save', action='store_true', help="Whether to save the output.")
parser.add_argument('--output_dir', type=str, default='.',
                    help="Directory to save the evalout.pkl into.")

args = parser.parse_args()

# Select the appropriate file path
selected_file_path = drug_pkl_file_path if args.use_drug else pkl_file_path
selected_ref_path = drug_ref_path if args.use_drug else ref_path

if args.use_drug:
    plot_dir = f'{args.save_name}_' + selected_file_path.split('/')[-1].split('.')[0]
else:
    plot_dir = f'{args.save_name}_' + selected_file_path.split('/')[-1].split('.')[0]
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
else:
    for f in os.listdir(plot_dir):
        fp = os.path.join(plot_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)
# Load and process the selected file

def main(smiles):
    if smiles == 'O=C(N=C1SC2CS(=O)(=O)CC2N1c1ccc(F)cc1)C1CCCCC1' or smiles == 'COc1ccc(CNCCC2(O)CC(C)[NH+](C)CC2C)cc1OC':     # This molecule has two kinds of mol in the ref data
        return smiles, {}
    out = {}
    with open(selected_file_path, 'rb') as file:
        data = pickle.load(file)
        gen_mol = data[smiles]['rdmol']  # no H
        # load gen traj
        gener_traj_coords = np.array(data[smiles]["coords"]) / 10  # A list of N elements, each element is (N_atom, 3), now (N_conformers, N_atom, 3)
    with open(pkl_file_path_539, 'rb') as file:
        ref_539_data = pickle.load(file)
        gen_mol_539 = ref_539_data[smiles]['rdmol']  # no H
        # load gen traj
        gener_traj_coords_539 = np.array(ref_539_data[smiles]["coords"]) / 10  # A list of N elements, each element is (N_atom, 3), now (N_conformers, N_atom, 3)
    # Load ref traj
    ref_torsions = []
    ref_bond_lengths = []
    ref_bond_angles = []
    torsions_539 = []
    bond_lengths_539 = []
    bond_angles_539 = []
    last_torsion_index_ref = None
    for data_point in ref_data:
        if data_point['smiles'] == smiles:
            ref_traj_coords = data_point['pos'] / 10
            if len(ref_traj_coords.shape) == 2:
                ref_traj_coords = np.expand_dims(ref_traj_coords, axis=0)  # (N_atom, 3) to (1, N_atom, 3)
            ref_mol = data_point['rdmol']  # has H
            torsion_index_ref = np.array(get_torsions_idx_mol(ref_mol)[0]+get_torsions_idx_mol(ref_mol)[2]).reshape(-1,4)
            # Use this to check for all ref mol, whether they give same torsion index
            if last_torsion_index_ref is not None:
                assert (last_torsion_index_ref == torsion_index_ref).all(), f"torsion index ref and last torsion index ref are not equal for {smiles}, they should be the same"
            last_torsion_index_ref = torsion_index_ref

            ref_torsions_one_traj = get_torsions_in_gen(torch.from_numpy(ref_traj_coords), torsion_index_ref)   # (1,N_torsion)
            ref_torsions.append(ref_torsions_one_traj)
            ref_bond_lengths_one_traj = get_bond_lengths_in_conformer_ref(torch.from_numpy(ref_traj_coords), ref_mol)  # (1,N_bond)
            ref_bond_lengths.append(ref_bond_lengths_one_traj)
            ref_bond_angles_one_traj = get_bond_angles_in_gen(torch.from_numpy(ref_traj_coords), ref_mol)  # (1,N_bond)
            ref_bond_angles.append(ref_bond_angles_one_traj)
    # print(gener_traj_coords.shape)  # (N_conformers, N_atom, 3)
    # Load gen torsion
    # torsion_index_noH = get_torsion_index_noH(torsion_index_ref, ref_mol)
    # assert (np.array(torsion_index_noH) == torsion_index_ref).all, f"torsion index noH and torsion index ref are not equal for {smiles}, they should be the same because ref is noH"
    # torsions = get_torsions_in_gen(torch.from_numpy(gener_traj_coords), torsion_index_noH)  # (N_conformers, N_torsion)
    bond_lengths = get_bond_lengths_in_gen(torch.from_numpy(gener_traj_coords), gen_mol)
    bond_angles = get_bond_angles_in_gen(torch.from_numpy(gener_traj_coords), gen_mol)
    # torsions_539 = get_torsions_in_gen(torch.from_numpy(gener_traj_coords_539), torsion_index_noH)  # (N_conformers, N_torsion)
    bond_lengths_539 = get_bond_lengths_in_gen(torch.from_numpy(gener_traj_coords_539), gen_mol_539)
    bond_angles_539 = get_bond_angles_in_gen(torch.from_numpy(gener_traj_coords_539), gen_mol_539)
    # if not isinstance(torsions, np.ndarray):
    #     print(f"Torsions is not a numpy array for molecule {smiles}. Skipping.")
    #     return smiles, out
    
    # if np.isnan(torsions).any() or np.isnan(bond_lengths).any() or np.isnan(bond_angles).any():
    #     print(f"Nan encountered in gen traj for molecule {smiles}")
    #     return smiles, out
    
    # try:
    #     ref_torsions = np.array(ref_torsions).reshape(-1, torsions.shape[-1])  # (N_conformers, N_torsion)
    # except ValueError:
    #     print(smiles)
    #     print(f"ValueError: ref_torsions shape is {np.array(ref_torsions).shape}, torsions shape is {torsions.shape} for molecule {smiles}")
        
    # print(ref_torsions.shape)  # (N_conformers, N_torsion)
    # print(ref_bond_lengths.shape)  # (N_conformers, N_bond)
    ref_bond_lengths = np.array(ref_bond_lengths).reshape(-1, bond_lengths.shape[-1])  # (N_conformers, N_bond)
    ref_bond_angles = np.array(ref_bond_angles).reshape(-1, bond_angles.shape[-1])  # (N_conformers, N_bond)

    # out['JSD_bond_angle'] = []
    # out['W1_bond_angle'] = []
    # for i in range(ref_bond_angles.shape[-1]):
    #     ref_p, ref_bin_edges = np.histogram(ref_bond_angles[:,i],range=(0,np.pi), bins=100)
    #     ref_p = ref_p / np.sum(ref_p)
    #     ref_bin_centers = (ref_bin_edges[:-1] + ref_bin_edges[1:]) / 2
    #     gener_p, gener_bin_edges = np.histogram(bond_angles[:,i],range=(0,np.pi), bins=100)
    #     gener_p = gener_p / np.sum(gener_p)
    #     gener_bin_centers = (gener_bin_edges[:-1] + gener_bin_edges[1:]) / 2
    #     out['JSD_bond_angle'].append(jensenshannon(ref_p, gener_p))
    #     out['W1_bond_angle'].append(wasserstein_distance(ref_bin_centers, gener_bin_centers, u_weights=ref_p, v_weights=gener_p))
    # out['JSD_bond_length'] = []
    # out['W1_bond_length'] = []
    # for i in range(ref_bond_lengths.shape[-1]):
    #     ref_p, ref_bin_edges = np.histogram(ref_bond_lengths[:,i],range=(0.05,0.3), bins=100)
    #     ref_p = ref_p / np.sum(ref_p)
    #     ref_bin_centers = (ref_bin_edges[:-1] + ref_bin_edges[1:]) / 2
    #     gener_p, gener_bin_edges = np.histogram(bond_lengths[:,i],range=(0.05,0.3), bins=100)
    #     gener_p = gener_p / np.sum(gener_p)
    #     gener_bin_centers = (gener_bin_edges[:-1] + gener_bin_edges[1:]) / 2
    #     out['JSD_bond_length'].append(jensenshannon(ref_p, gener_p))
    #     out['W1_bond_length'].append(wasserstein_distance(ref_bin_centers, gener_bin_centers, u_weights=ref_p, v_weights=gener_p))
    # out['JSD_torsion'] = []
    # out['W1_torsion'] = []
    # for i in range(ref_torsions.shape[-1]):
    #     ref_p, ref_bin_edges = np.histogram(ref_torsions[:,i],range=(-np.pi,np.pi), bins=100)
    #     ref_p = ref_p / np.sum(ref_p)
    #     ref_bin_centers = (ref_bin_edges[:-1] + ref_bin_edges[1:]) / 2
    #     gener_p, gener_bin_edges = np.histogram(torsions[:,i],range=(-np.pi,np.pi), bins=100)
    #     gener_p = gener_p / np.sum(gener_p)
    #     gener_bin_centers = (gener_bin_edges[:-1] + gener_bin_edges[1:]) / 2
    #     out['JSD_torsion'].append(jensenshannon(ref_p, gener_p))
    #     out['W1_torsion'].append(wasserstein_distance(ref_bin_centers, gener_bin_centers, u_weights=ref_p, v_weights=gener_p))
    num_bond_angles = min(ref_bond_angles.shape[0], np.array(bond_angles).shape[0], np.array(bond_angles_539).shape[0])
    num_bond_lengths = min(ref_bond_lengths.shape[0], np.array(bond_lengths).shape[0], np.array(bond_lengths_539).shape[0])
    # num_torsions = min(ref_torsions.shape[0], np.array(torsions).shape[0], np.array(torsions_539).shape[0])
    
    if args.plot:
        # fig, ax = plt.subplots(figsize=(12, 8))
        # pyemma.plots.plot_feature_histograms(torsions[:num_torsions], ax=ax, color='blue')
        # pyemma.plots.plot_feature_histograms(ref_torsions[:num_torsions], ax=ax, color='red')
        # pyemma.plots.plot_feature_histograms(torsions_539[:num_torsions], ax=ax, color='green')
        # ax.set_ylabel('Feature No.')
        # ax.set_xlabel('Torsion angle (rad)')
        # fig.tight_layout()
        # plt.savefig(f"{plot_dir}/{urllib.parse.quote(smiles, safe='')}_torsion_hist.png")  # Use a reversible way to encode the smiles, because it contains / and \
        # plt.close()

        fig, ax = plt.subplots(figsize=(12, 8))
        pyemma.plots.plot_feature_histograms(bond_lengths[:num_bond_lengths], ax=ax, color='blue')
        pyemma.plots.plot_feature_histograms(ref_bond_lengths[:num_bond_lengths], ax=ax, color='red')
        pyemma.plots.plot_feature_histograms(bond_lengths_539[:num_bond_lengths], ax=ax, color='green')
        ax.set_ylabel('Feature No.')
        ax.set_xlabel('Bond length (nm)')
        fig.tight_layout()
        plt.savefig(f"{plot_dir}/{urllib.parse.quote(smiles, safe='')}_bond_length.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 8))
        pyemma.plots.plot_feature_histograms(bond_angles[:num_bond_angles], ax=ax, color='blue')
        pyemma.plots.plot_feature_histograms(ref_bond_angles[:num_bond_angles], ax=ax, color='red')
        pyemma.plots.plot_feature_histograms(bond_angles_539[:num_bond_angles], ax=ax, color='green')
        ax.set_ylabel('Feature No.')
        ax.set_xlabel('Bond angle (radian)')
        fig.tight_layout()
        plt.savefig(f"{plot_dir}/{urllib.parse.quote(smiles, safe='')}_bond_angle.png")
        plt.close()
    return smiles, out

with open(selected_file_path, 'rb') as f:
    data = pickle.load(f)
    gener_smiles = data.keys()
print('number of unique molecules with generated conformers', len(gener_smiles))
with open(selected_ref_path, 'rb') as file:
    ref_data = pickle.load(file)


if args.num_workers > 1:
    p = Pool(args.num_workers)
    p.__enter__()
    __map__ = p.imap
else:
    __map__ = map
out = dict(tqdm.tqdm(__map__(main, gener_smiles), total=len(gener_smiles)))
if args.num_workers > 1:
    p.__exit__(None, None, None)

if args.use_drug:
    save_path = os.path.join(args.output_dir, f"conformer_model_drugs_{selected_file_path.split('/')[-1].split('.')[0]}_evalout.pkl")
else:
    save_path = os.path.join(args.output_dir, f"conformer_model_qm9_{selected_file_path.split('/')[-1].split('.')[0]}_evalout.pkl")
if args.save:
    with open(save_path, 'wb') as f:
        f.write(pickle.dumps(out))