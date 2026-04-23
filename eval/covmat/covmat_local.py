import torch
import numpy as np
import pandas as pd
import yaml
import multiprocessing as mp
from torch_geometric.data import Data
from functools import partial
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from ..chem import set_rdmol_positions, get_best_rmsd
import pickle

import sys
from utils.data_filter import filter_data
import argparse

# Global variables for worker processes\REF_DICT = None
GEN_DICT = None
USE_FF = None
REMOVE_HS = None

def _init_worker(ref_dict, gen_dict, use_ff, remove_hs):
    global REF_DICT, GEN_DICT, USE_FF, REMOVE_HS
    REF_DICT = ref_dict
    GEN_DICT = gen_dict
    USE_FF = use_ff
    REMOVE_HS = remove_hs


def _worker(smiles):
    # Retrieve per-SMILES data
    ref_vals = REF_DICT[smiles]
    gen_vals = GEN_DICT[smiles]

    # Get the generated coordinates
    gen_coords = gen_vals['coords']
    # Get the respective numbers
    num_ref, num_gen = len(ref_vals), len(gen_coords)
    # Initialize the confusion matrix
    mat = -1 * np.ones((num_ref, num_gen), dtype=float)

    # Build blank template
    base = Chem.Mol(gen_vals['rdmol'])
    base = Chem.RemoveHs(base)
    conf = base.GetConformer()
    for atom in base.GetAtoms():
        conf.SetAtomPosition(atom.GetIdx(), (0.0, 0.0, 0.0))

    # Compute confusion matrix
    for i, coords in enumerate(gen_coords):
        gen_mol = set_rdmol_positions(base, coords)
        for j, ref_mol in enumerate(ref_vals):
            try:
                mat[j, i] = get_best_rmsd(gen_mol, ref_mol, remove_hs=REMOVE_HS)
            except Exception as e:
                mat[j, i] = np.nan
    return mat


def get_smiles_dict(data):
    smile_keys = {}
    for item in tqdm(data):
        smile_keys.setdefault(item['smiles'], []).append(item['rdmol'])
    return smile_keys


def print_covmat_results(results, print_fn=print):
    # Core COV/ MAT results
    df = pd.DataFrame({
        'COV-R_mean':   np.nanmean(results.CoverageR, 0),
        'COV-R_median': np.nanmedian(results.CoverageR, 0),
        'COV-R_std':    np.nanstd(results.CoverageR, 0),
        'COV-P_mean':   np.nanmean(results.CoverageP, 0),
        'COV-P_median': np.nanmedian(results.CoverageP, 0),
        'COV-P_std':    np.nanstd(results.CoverageP, 0),
    }, index=results.thresholds)
    print_fn('\n' + str(df))

    # MAT-R and MAT-P
    mat_r_mean   = np.nanmean(results.MatchingR)
    mat_r_med    = np.nanmedian(results.MatchingR)
    mat_r_std    = np.nanstd(results.MatchingR)
    mat_p_mean   = np.nanmean(results.MatchingP)
    mat_p_med    = np.nanmedian(results.MatchingP)
    mat_p_std    = np.nanstd(results.MatchingP)
    print_fn(f'MAT-R_mean: {mat_r_mean:.4f} | MAT-R_median: {mat_r_med:.4f} | MAT-R_std {mat_r_std:.4f}')
    print_fn(f'MAT-P_mean: {mat_p_mean:.4f} | MAT-P_median: {mat_p_med:.4f} | MAT-P_std {mat_p_std:.4f}')

    # NaN statistics (per molecule, from results.NanCounts)
    nan_counts = np.array(list(results.NanCounts.values()), dtype=float)
    nan_mean   = np.mean(nan_counts)
    nan_med    = np.median(nan_counts)
    nan_std    = np.std(nan_counts)
    print_fn(f'\nNaN counts per molecule — mean: {nan_mean:.2f}, median: {nan_med:.2f}, std: {nan_std:.2f}')

    # Number of skipped SMILES
    num_skipped = len(results.SkippedSmiles)
    print_fn(f'Number of skipped SMILES (insufficient valid conformers): {num_skipped}')

    # NaN counts directly in COV / MAT arrays
    covr_nans = np.isnan(results.CoverageR).sum()
    covp_nans = np.isnan(results.CoverageP).sum()
    matr_nans = np.isnan(results.MatchingR).sum()
    matp_nans = np.isnan(results.MatchingP).sum()
    total_nans = covr_nans + covp_nans + matr_nans + matp_nans
    print_fn(
        f'\nNaN entries in metrics — '
        f'COV-R: {covr_nans}, COV-P: {covp_nans}, '
        f'MAT-R: {matr_nans}, MAT-P: {matp_nans}, '
        f'Total: {total_nans}'
    )

    return df


class CovMatEvaluator(object):

    def __init__(self,
        num_workers=8,
        use_force_field=False,
        thresholds=np.arange(0.05, 3.05, 0.05),
        ratio=2,
        print_fn=print,
        config=None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.use_force_field = use_force_field
        self.thresholds = np.array(thresholds).flatten()
        self.ratio = ratio
        self.print_fn = print_fn
        self.remove_hs = None
        if 'remove_hs' in config.dataset.transforms[0]:
            self.remove_hs = int(config.dataset.transforms[0].split('|')[1])

    def __call__(self, ref_dict, gen_dict, start_idx=0):
        # Build list of smiles keys to evaluate
        smiles_list = []
        nan_counts = {}
        skipped_smiles = []
        for smiles, gen_vals in gen_dict.items():
            if smiles not in ref_dict:
                continue

            if Chem.RemoveHs(gen_dict[smiles]['rdmol']).GetNumAtoms() != gen_vals['coords'][0].shape[0]:
                continue

            num_ref = len(ref_dict[smiles])
            nan_counts[smiles] = 0
            coords = []
            for coord in gen_vals['coords']:
                if np.isnan(coord).any():
                    nan_counts[smiles] += 1
                    continue
                coords.append(coord)

            # if len(coords) < num_ref * self.ratio:
            #     skipped_smiles.append(smiles)
            #     continue

            gen_vals['coords'] = coords[:num_ref * self.ratio]
            smiles_list.append(smiles)

        smiles_list = smiles_list[start_idx:]
        self.print_fn(f'Filtered: {len(smiles_list)} unique molecules for evaluation')

        # Launch process pool with globals initialized
        pool = mp.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(ref_dict, gen_dict, self.use_force_field, self.remove_hs),
        )

        covr_scores, matr_scores, covp_scores, matp_scores = [], [], [], []
        for mat in tqdm(pool.imap_unordered(_worker, smiles_list), total=len(smiles_list)):
            rmsd_ref_min = mat.min(-1)
            rmsd_gen_min = mat.min(0)
            rmsd_cov_thres = rmsd_ref_min.reshape(-1, 1) <= self.thresholds.reshape(1, -1)
            rmsd_jnk_thres = rmsd_gen_min.reshape(-1, 1) <= self.thresholds.reshape(1, -1)

            matr_scores.append(rmsd_ref_min.mean())
            covr_scores.append(rmsd_cov_thres.mean(0, keepdims=True))
            matp_scores.append(rmsd_gen_min.mean())
            covp_scores.append(rmsd_jnk_thres.mean(0, keepdims=True))

        pool.close()
        pool.join()

        covr_scores = np.vstack(covr_scores)
        matr_scores = np.array(matr_scores)
        covp_scores = np.vstack(covp_scores)
        matp_scores = np.array(matp_scores)

        results = EasyDict({
            'CoverageR': covr_scores,
            'MatchingR': matr_scores,
            'thresholds': self.thresholds,
            'CoverageP': covp_scores,
            'MatchingP': matp_scores,
            'NanCounts': nan_counts,
            'SkippedSmiles': skipped_smiles
        })
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help='Path to config file (see configs_official/)')
    parser.add_argument('--gen_path', type=str, default='test_conformers_multi.pkl',
                        help='Path to generated conformers pickle file')
    parser.add_argument('--output_file', type=str, default='covmat_results_1M.pkl',
                        help='Path to output results pickle file')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    config = EasyDict(config)
    print(config)

    gen_dict = pickle.load(open(args.gen_path, 'rb'))

    # Determine which reference dataset to use.
    if hasattr(config.dataset, 'test_conf_path'):
        test_pkl = config.dataset.test_conf_path
    elif config.dataset.type == 'both':
        test_on_drugs = getattr(config.dataset, 'test_on_drugs', True)
        if test_on_drugs:
            test_pkl = config.dataset.test_conf_path_drugs
        else:
            test_pkl = config.dataset.test_conf_path_qm9
    else:
        raise AttributeError(
            "Could not determine reference test_conf_path for dataset type "
            f"{config.dataset.type}. Please ensure the config specifies "
            "`test_conf_path` or set `test_on_drugs` for 'both' datasets."
        )
    ref_data = pickle.load(open(test_pkl, 'rb'))
    if config.dataset.filter_data:
        ref_data = filter_data(ref_data)[0]
    ref_dict = get_smiles_dict(ref_data)

    evaluator = CovMatEvaluator(num_workers=2, use_force_field=False, ratio=2, config=config)
    results = evaluator(ref_dict, gen_dict)

    print_covmat_results(results, print_fn=print)

    with open(args.output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {args.output_file}")